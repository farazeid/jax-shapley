"""Host callbacks, selective CSV records and atomic Orbax recovery for DQN.

All filesystem and serialization effects live here. The trainer imports host
operations at its boundaries; this module imports the trainer's typed contracts.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from math import isfinite
from pathlib import Path
from time import perf_counter, time_ns
from types import TracebackType
from typing import BinaryIO, Literal, Protocol, Self, cast

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx, struct
from jax import Array
from numpy.typing import NDArray

from jax_shapley.networks import MLPConfig, QNetwork
from jax_shapley.replay import ReplayState, empty_replay
from jax_shapley.train_dqn import (
    AdamConfig,
    DQNRunConfig,
    DQNTrainConfig,
    EnvironmentConfig,
    GridworldConfig,
    HypercubeConfig,
    Mastermind222Config,
    MastermindConfig,
    RunState,
    RunSummary,
    StepMetrics,
    epsilon_at,
)
from jax_shapley.wrappers import EpisodeStatisticsState

REFERENCE_COMMIT = "25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b"


type HostArray = np.ndarray[tuple[int, ...], np.dtype[np.generic]]


@dataclass(frozen=True, slots=True)
class RunDescription[Config: EnvironmentConfig]:
    config: DQNRunConfig[Config]
    network: MLPConfig
    environment: str
    reference_commit: str = REFERENCE_COMMIT


@dataclass(frozen=True, slots=True)
class LoadedAgent:
    model: QNetwork
    target: QNetwork
    epsilon: float
    vector_step: int
    replay: ReplayState | None


@dataclass(frozen=True, slots=True)
class MetricRow:
    kind: Literal["episode", "update", "evaluation"]
    step: int
    epsilon: float | None = None
    loss: float | None = None
    episode_return: float | None = None
    episode_length: int | None = None
    sps: int | None = None


METRIC_COLUMNS = ("kind", "step", "epsilon", "loss", "episode_return", "episode_length", "sps")


@struct.dataclass
class Snapshot:
    agent: tuple[Array, ...]
    replay: ReplayState
    optimizer: tuple[Array, ...]
    runtime: tuple[Array, ...]
    vector_step: Array
    phase: Array
    evaluation_episodes: Array
    epsilon: Array
    key_implementation: str = struct.field(pytree_node=False, default="threefry2x32")


type CallbackToken = NDArray[np.int32]


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    settings: str
    vector_step: int
    phase: int
    evaluation_episodes: int
    csv_bytes: int = 0
    csv_digest: str = ""
    elapsed_seconds: float = 0.0
    training_seconds: float = 0.0
    key_implementation: str = "threefry2x32"
    schema: int = 1
    reference_commit: str = REFERENCE_COMMIT

    def validate(self) -> None:
        if self.schema != 1 or self.reference_commit != REFERENCE_COMMIT:
            raise ValueError("unsupported checkpoint schema or reference")
        if self.vector_step < 0 or self.phase not in (0, 1, 2) or self.evaluation_episodes < 0:
            raise ValueError("invalid checkpoint counters or phase")
        if self.csv_bytes < 0 or not isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("invalid checkpoint log boundary or timing")
        if not isfinite(self.training_seconds) or not 0 <= self.training_seconds <= self.elapsed_seconds:
            raise ValueError("invalid training timing")
        if self.key_implementation not in ("threefry2x32", "rbg", "unsafe_rbg"):
            raise ValueError("unsupported checkpoint PRNG implementation")
        config = parse_settings(self.settings)
        _validate_phase(self, config.train.eval_episodes, config.train.num_envs, config.train.vector_steps)
        if self.vector_step > config.train.vector_steps:
            raise ValueError("checkpoint exceeds saved training duration")


class EventSink(Protocol):
    def training_complete(self) -> CallbackToken: ...

    def training(self, metrics: tuple[Array, ...], count: Array) -> CallbackToken: ...

    def evaluation(self, returns: Array, completed: Array, first_index: Array) -> CallbackToken: ...

    def checkpoint(self, snapshot: Snapshot) -> CallbackToken: ...


class CallbackRouter:
    """Bind the writer only after compilation; preflight cannot perform host I/O."""

    def __init__(self) -> None:
        self.sink: EventSink | None = None

    def training_complete(self) -> CallbackToken:
        return self._sink().training_complete()

    def training(self, metrics: tuple[Array, ...], count: Array) -> CallbackToken:
        return self._sink().training(metrics, count)

    def evaluation(self, returns: Array, completed: Array, first_index: Array) -> CallbackToken:
        return self._sink().evaluation(returns, completed, first_index)

    def checkpoint(self, snapshot: Snapshot) -> CallbackToken:
        return self._sink().checkpoint(snapshot)

    def _sink(self) -> EventSink:
        if self.sink is None:
            raise RuntimeError("callback invoked before writer binding")
        return self.sink


class RunWriter[Config: EnvironmentConfig]:
    """Host-only writer; numerical code sees only the EventSink protocol."""

    def __init__(
        self,
        folder: Path,
        config: DQNRunConfig[Config],
        *,
        metadata: CheckpointMetadata | None = None,
        checkpoint_sequence: int | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.folder, self.config, self.clock = folder, config, clock
        self.lock = acquire_writer_lock(folder)
        try:
            if metadata is not None:
                # Preflight restoration/compilation runs without the writer lock.
                # Reject stale state before rolling back any committed log rows.
                store = CheckpointStore(folder, read_only=True)
                try:
                    if checkpoint_sequence is None or store.sequence != checkpoint_sequence:
                        raise RuntimeError("checkpoint changed during resume preflight; retry resume")
                finally:
                    store.close()
                verify_log(folder / "metrics.csv", metadata)
                with (folder / "metrics.csv").open("r+b") as log:
                    log.truncate(metadata.csv_bytes)
                    log.flush()
                    os.fsync(log.fileno())
            self.store = CheckpointStore(folder)
        except BaseException:
            self.lock.close()
            raise
        self.started = clock()
        self.training_seconds = 0.0 if metadata is None else metadata.training_seconds
        self.prior_elapsed = 0.0 if metadata is None else metadata.elapsed_seconds

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def training_complete(self) -> CallbackToken:
        self.training_seconds = self.elapsed()
        return np.asarray(0, np.int32)

    def training(self, metrics: tuple[Array, ...], count: Array) -> CallbackToken:
        active = int(host_array(count))
        rows = StepMetrics(*(value[:active] for value in metrics))
        record_metrics(self.folder / "metrics.csv", rows, self.config.train.num_envs, self.elapsed())
        return np.asarray(0, np.int32)

    def evaluation(self, returns: Array, completed: Array, first_index: Array) -> CallbackToken:
        values, flags = host_array(returns), host_array(completed)
        first = int(host_array(first_index))
        selected = [float(value) for value, flag in zip(values, flags, strict=True) if bool(flag)]
        write_metric_rows(
            self.folder / "metrics.csv",
            [MetricRow("evaluation", first + index, episode_return=value) for index, value in enumerate(selected)],
        )
        return np.asarray(0, np.int32)

    def checkpoint(self, snapshot: Snapshot) -> CallbackToken:
        self.store.wait()
        path = self.folder / "metrics.csv"
        with path.open("a+b") as log:
            log.flush()
            os.fsync(log.fileno())
            boundary = log.tell()
        metadata = CheckpointMetadata(
            describe(self.config),
            int(host_array(snapshot.vector_step)),
            int(host_array(snapshot.phase)),
            int(host_array(snapshot.evaluation_episodes)),
            boundary,
            prefix_digest(path, boundary),
            self.elapsed(),
            training_seconds=self.training_seconds,
            key_implementation=snapshot.key_implementation,
        )
        self.store.save(snapshot, metadata)
        # The pre-evaluation snapshot must commit before evaluation can emit rows.
        if metadata.phase == 1:
            self.store.wait()
        return np.asarray(0, np.int32)

    def elapsed(self) -> float:
        return self.prior_elapsed + self.clock() - self.started

    def close(self) -> None:
        try:
            self.store.close()
        finally:
            self.lock.close()


class OrbaxStore:
    """One serialized producer; Orbax owns its asynchronous writer and commit protocol.

    Items are named tuples of arrays plus one JSON metadata item, saved atomically
    as a numbered composite. Only the latest committed checkpoint is kept.
    """

    def __init__(self, folder: Path, *, read_only: bool = False) -> None:
        options = ocp.CheckpointManagerOptions(
            read_only=read_only,
            create=not read_only,
            enable_async_checkpointing=not read_only,
            preservation_policy=ocp.v1.training.preservation_policies.LatestN(n=1),
        )
        self.manager = ocp.CheckpointManager(folder.resolve() / "checkpoints", options=options)
        self.sequence = self.manager.latest_step()

    def save_items(self, items: dict[str, tuple[Array, ...]], metadata: dict[str, object]) -> None:
        self.manager.wait_until_finished()
        sequence = 0 if self.sequence is None else self.sequence + 1
        arguments: dict[str, object] = {name: ocp.args.StandardSave(_pack(values)) for name, values in items.items()}
        arguments["metadata"] = ocp.args.JsonSave(metadata)
        if not self.manager.save(sequence, args=ocp.args.Composite(**arguments)):
            raise OSError("Orbax declined a required checkpoint")
        self.sequence = sequence

    def metadata_json(self) -> dict[str, object]:
        result = self._restore({"metadata": ocp.args.JsonRestore()})
        return _mapping(result["metadata"])

    def arrays(self, item: str, template: tuple[Array, ...]) -> tuple[Array, ...]:
        # Read the saved NumPy dtypes before destination placement. Target-driven
        # restoration could silently coerce a corrupt/wrong-precision checkpoint.
        result = self._restore({item: ocp.args.StandardRestore()})
        data = cast(dict[str, object], result[item])  # Orbax mapping boundary, narrowed below.
        names = tuple(f"array_{index:04d}" for index in range(len(template)))
        if set(data) != set(names):
            raise ValueError(f"invalid checkpoint {item} topology")
        values: list[Array] = []
        for name, expected in zip(names, template, strict=True):
            actual = data[name]
            if not isinstance(actual, np.ndarray) or actual.shape != expected.shape or actual.dtype != expected.dtype:
                raise ValueError(f"invalid checkpoint {item} shape or dtype")
            if np.issubdtype(actual.dtype, np.floating) and not bool(np.all(np.isfinite(actual))):
                raise ValueError(f"nonfinite checkpoint {item}")
            values.append(jax.device_put(actual, expected.sharding))
        return tuple(values)

    def wait(self) -> None:
        self.manager.wait_until_finished()

    def close(self) -> None:
        self.manager.close()

    def _restore(self, items: dict[str, object]) -> dict[str, object]:
        if self.sequence is None:
            raise ValueError("no committed checkpoint")
        result = self.manager.restore(self.sequence, args=ocp.args.Composite(**items))
        return {name: result[name] for name in items}  # Dynamic values end at this adapter.


class CheckpointStore(OrbaxStore):
    """DQN snapshots: agent, replay and complete run state with validated DQN metadata."""

    def save(self, snapshot: Snapshot, metadata: CheckpointMetadata) -> None:
        metadata.validate()
        if metadata.key_implementation != snapshot.key_implementation:
            raise ValueError("checkpoint PRNG metadata disagrees with snapshot")
        if not snapshot.optimizer or not snapshot.runtime:
            raise ValueError("checkpoint requires complete optimizer and run state")
        self.save_items(
            {
                "agent": (*snapshot.agent, snapshot.epsilon, snapshot.vector_step),
                "replay": _leaves(snapshot.replay),
                "run_state": (*snapshot.optimizer, *snapshot.runtime),
            },
            asdict(metadata),
        )

    def metadata(self) -> CheckpointMetadata:
        value = self.metadata_json()  # JSON is checked field-by-field below.
        metadata = CheckpointMetadata(
            settings=_string(value, "settings"),
            vector_step=_integer(value, "vector_step"),
            phase=_integer(value, "phase"),
            evaluation_episodes=_integer(value, "evaluation_episodes"),
            csv_bytes=_integer(value, "csv_bytes"),
            csv_digest=_string(value, "csv_digest"),
            elapsed_seconds=_number(value, "elapsed_seconds"),
            training_seconds=_number(value, "training_seconds"),
            key_implementation=_string(value, "key_implementation"),
            schema=_integer(value, "schema"),
            reference_commit=_string(value, "reference_commit"),
        )
        metadata.validate()
        return metadata


def create_run[Config: EnvironmentConfig](config: DQNRunConfig[Config], environment_name: str) -> Path:
    """Create fresh artifacts only after successful numerical compilation."""
    folder = config.output / f"{environment_name}-{config.seed}-{time_ns()}"
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "config.json").write_text(describe(config))
    initialize_metrics(folder / "metrics.csv")
    return folder


def load_agent(folder: Path, network: MLPConfig, *, load_replay: bool = True) -> LoadedAgent:
    store = CheckpointStore(folder, read_only=True)
    try:
        metadata = store.metadata()
        if json.loads(metadata.settings)["network"] != json.loads(json.dumps(asdict(network))):
            raise ValueError("checkpoint architecture does not match requested network")
        model, target = QNetwork(network, rngs=nnx.Rngs(0)), QNetwork(network, rngs=nnx.Rngs(0))
        graph_state = nnx.state((model, target))
        template = (*_leaves(graph_state), jnp.asarray(0.0, jnp.float32), jnp.asarray(0, jnp.int32))
        restored = store.arrays("agent", template)
        nnx.update((model, target), jax.tree.unflatten(jax.tree.structure(graph_state), restored[:-2]))
        epsilon, step = float(restored[-2]), int(restored[-1])
        if not 0 <= epsilon <= 1 or step != metadata.vector_step:
            raise ValueError("invalid checkpoint epsilon or step")
        replay = load_replay_state(store, network.input_features, _replay_capacity(metadata)) if load_replay else None
        return LoadedAgent(model, target, epsilon, step, replay)
    finally:
        store.close()


def load_replay(folder: Path) -> ReplayState:
    """Read replay independently; neither parameters nor optimizer arrays are loaded."""
    store = CheckpointStore(folder, read_only=True)
    try:
        config = parse_settings(store.metadata().settings)
        return load_replay_state(store, config.network.input_features, config.train.buffer_size)
    finally:
        store.close()


def restore_run[State, Config: EnvironmentConfig](
    store: CheckpointStore, template: RunState[State], config: DQNRunConfig[Config]
) -> tuple[RunState[State], CheckpointMetadata]:
    metadata = store.metadata()
    if describe(parse_settings(metadata.settings)) != describe(config):
        raise ValueError("resume requires identical saved settings and a run checkpoint")
    agent = nnx.state((template.model, template.target))
    values = store.arrays("agent", (*_leaves(agent), jnp.asarray(0.0, jnp.float32), template.vector_step))
    if int(values[-1]) != metadata.vector_step or not 0 <= float(values[-2]) <= 1:
        raise ValueError("invalid checkpoint agent step or epsilon")
    nnx.update((template.model, template.target), jax.tree.unflatten(jax.tree.structure(agent), values[:-2]))
    expected_epsilon = float(
        epsilon_at(config.train, jnp.asarray(max(metadata.vector_step - 1, 0) * config.train.num_envs, jnp.int32))
    )
    if not np.isclose(float(values[-2]), expected_epsilon, rtol=0, atol=1e-7):
        raise ValueError("checkpoint epsilon disagrees with training progress")
    optimizer = nnx.state(template.optimizer)
    runtime = (
        template.observations,
        template.env_state,
        template.vector_step,
        template.updates,
        template.target_updates,
        # Key shape belongs to the saved implementation, not the host's current default.
        jax.device_put(
            jax.random.key_data(jax.random.key(0, impl=metadata.key_implementation)),
            jax.random.key_data(template.key).sharding,
        ),
    )
    optimizer_count = len(_leaves(optimizer))
    values = store.arrays("run_state", (*_leaves(optimizer), *_leaves(runtime)))
    nnx.update(template.optimizer, jax.tree.unflatten(jax.tree.structure(optimizer), values[:optimizer_count]))
    observations, env_state, step, updates, targets, key_data = cast(
        tuple[Array, object, Array, Array, Array, Array],
        jax.tree.unflatten(jax.tree.structure(runtime), values[optimizer_count:]),
    )  # Fixed runtime topology, checked by StandardRestore and arrays().
    if int(step) != metadata.vector_step or int(step) > config.train.vector_steps:
        raise ValueError("invalid checkpoint vector step")
    if not 0 <= int(updates) <= int(step) or not 0 <= int(targets) <= int(step):
        raise ValueError("invalid checkpoint update counters")
    replay = _restore_replay(store, template.replay)
    # Preserve the environment's concrete generic structure through the validated tree restoration.
    restored_env = cast(EpisodeStatisticsState[State], env_state)
    restored = RunState(
        template.model,
        template.target,
        template.optimizer,
        replay,
        observations,
        restored_env,
        step,
        updates,
        targets,
        jax.random.wrap_key_data(key_data, impl=metadata.key_implementation),
    )
    if int(template.optimizer.step[...]) != int(updates):
        raise ValueError("optimizer step disagrees with update counter")
    inserted = int(step) * min(config.train.num_envs, config.train.buffer_size)
    if int(replay.pointer) != inserted % replay.capacity or int(replay.size) != min(inserted, replay.capacity):
        raise ValueError("replay counters disagree with training progress")
    return restored, metadata


def load_settings(path: Path) -> DQNRunConfig[EnvironmentConfig]:
    return parse_settings(path.read_text())


def parse_settings(text: str) -> DQNRunConfig[EnvironmentConfig]:
    envelope = _mapping(json.loads(text))
    config = _mapping(envelope["config"])
    # Older full-run settings carried a mode selector. Normalize that one field;
    # partial/no-checkpoint formats remain unsupported.
    if "checkpoint_mode" in config and config.pop("checkpoint_mode") != "run":
        raise ValueError("saved settings require a full run checkpoint")
    if set(config) != {field.name for field in fields(DQNRunConfig)}:
        raise ValueError("saved run settings have an invalid schema")
    train = _training(_mapping(config["train"]))
    environment = _mapping(config["env"])
    name = _string(envelope, "environment")
    env: EnvironmentConfig
    if name == "GridworldConfig" and not environment:
        env = GridworldConfig()
    elif name == "Mastermind222Config" and not environment:
        env = Mastermind222Config()
    elif name == "MastermindConfig" and set(environment) == {"code_size", "num_guesses", "num_pegs"}:
        env = MastermindConfig(
            _integer(environment, "code_size"), _integer(environment, "num_guesses"), _integer(environment, "num_pegs")
        )
    elif name == "HypercubeConfig" and set(environment) == {"side_length", "dimension"}:
        env = HypercubeConfig(_integer(environment, "side_length"), _integer(environment, "dimension"))
    else:
        raise ValueError("unsupported saved environment")
    widths = config["hidden_widths"]
    if not isinstance(widths, list) or any(type(width) is not int for width in widths):
        raise ValueError("invalid saved hidden widths")
    result = DQNRunConfig(
        env,
        train,
        tuple(cast(list[int], widths)),
        _integer(config, "seed"),
        _integer(config, "chunk_steps"),
        Path(_string(config, "output")),
        _boolean(config, "validate_only"),
        _boolean(config, "compile_only"),
    )
    if result.validate_only or result.compile_only or json.loads(describe(result)) != envelope:
        raise ValueError("saved settings or network/reference metadata are inconsistent")
    return result


def describe[Config: EnvironmentConfig](config: DQNRunConfig[Config]) -> str:
    return (
        json.dumps(
            asdict(RunDescription(config, config.network, type(config.env).__name__)), default=_json_default, indent=2
        )
        + "\n"
    )


def initialize_metrics(path: Path) -> None:
    with path.open("w", newline="") as output:
        csv.writer(output).writerow(METRIC_COLUMNS)


def record_metrics(path: Path, metrics: StepMetrics, num_envs: int, elapsed_seconds: float = 1.0) -> None:
    steps, active = host_array(metrics.global_step), host_array(metrics.active)
    epsilon, losses, updated = host_array(metrics.epsilon), host_array(metrics.loss), host_array(metrics.updated)
    completed, returns, lengths = (
        host_array(metrics.completed),
        host_array(metrics.episode_return),
        host_array(metrics.episode_length),
    )
    rows: list[MetricRow] = []
    for index in range(len(steps)):
        if not bool(active[index]):
            continue
        step = int(steps[index])
        rows.extend(
            MetricRow(
                "episode",
                step,
                epsilon=float(epsilon[index]),
                episode_return=float(returns[index, env]),
                episode_length=int(lengths[index, env]),
            )
            for env in range(num_envs)
            if bool(completed[index, env])
        )
        if bool(updated[index]) and step % 100 < num_envs:
            rows.append(
                MetricRow("update", step, loss=float(losses[index]), sps=int(step / max(elapsed_seconds, 1e-9)))
            )
    write_metric_rows(path, rows)


def write_metric_rows(path: Path, rows: list[MetricRow]) -> None:
    with path.open("a", newline="") as output:
        writer = csv.writer(output)
        for row in rows:
            writer.writerow((
                row.kind,
                row.step,
                row.epsilon,
                row.loss,
                row.episode_return,
                row.episode_length,
                row.sps,
            ))


def write_summary(folder: Path, summary: RunSummary) -> None:
    staged = folder / "summary.json.tmp"
    staged.write_text(json.dumps(asdict(summary), indent=2) + "\n")
    staged.replace(folder / "summary.json")


def verify_log(path: Path, metadata: CheckpointMetadata) -> None:
    verify_prefix(path, metadata.csv_bytes, metadata.csv_digest)


def verify_prefix(path: Path, size: int, digest: str) -> None:
    if path.stat().st_size < size or prefix_digest(path, size) != digest:
        raise ValueError("metrics prefix does not match committed checkpoint")


def acquire_writer_lock(folder: Path) -> BinaryIO:
    """Exclusive non-blocking lock: one active writer per run folder."""
    lock = (folder / ".writer.lock").open("a+b")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        raise RuntimeError("run folder already has an active writer") from None
    return lock


def prefix_digest(path: Path, boundary: int) -> str:
    digest = hashlib.sha256()
    remaining = boundary
    with path.open("rb") as source:
        while remaining:
            block = source.read(min(remaining, 65536))
            if not block:
                raise ValueError("metrics file is shorter than committed boundary")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def load_replay_state(store: CheckpointStore, features: int, capacity: int) -> ReplayState:
    return _restore_replay(store, empty_replay(capacity, features))


def _restore_replay(store: CheckpointStore, template: ReplayState) -> ReplayState:
    replay = cast(
        ReplayState, jax.tree.unflatten(jax.tree.structure(template), store.arrays("replay", _leaves(template)))
    )
    replay.data.validate()
    if not 0 <= int(replay.pointer) < replay.capacity or not 0 <= int(replay.size) <= replay.capacity:
        raise ValueError("invalid replay pointer/size")
    return replay


def _leaves(tree: object) -> tuple[Array, ...]:
    return tuple(cast(list[Array], jax.tree.leaves(tree)))  # Numerical payload construction only.


def _pack(values: tuple[Array, ...]) -> dict[str, np.ndarray[tuple[int, ...], np.dtype[np.generic]]]:
    return {f"array_{index:04d}": host_array(value).copy() for index, value in enumerate(values)}


def _replay_capacity(metadata: CheckpointMetadata) -> int:
    value = json.loads(metadata.settings)["config"]["train"]["buffer_size"]
    if not isinstance(value, int) or value < 1:
        raise ValueError("invalid saved replay capacity")
    return value


def _training(value: dict[str, object]) -> DQNTrainConfig:
    if set(value) != {field.name for field in fields(DQNTrainConfig)}:
        raise ValueError("invalid saved training schema")
    adam = _mapping(value["adam"])
    if set(adam) != {field.name for field in fields(AdamConfig)}:
        raise ValueError("invalid saved Adam schema")
    return DQNTrainConfig(
        num_envs=_integer(value, "num_envs"),
        total_timesteps=_integer(value, "total_timesteps"),
        learning_rate=_number(value, "learning_rate"),
        buffer_size=_integer(value, "buffer_size"),
        gamma=_number(value, "gamma"),
        tau=_number(value, "tau"),
        target_network_frequency=_integer(value, "target_network_frequency"),
        batch_size=_integer(value, "batch_size"),
        start_e=_number(value, "start_e"),
        end_e=_number(value, "end_e"),
        exploration_fraction=_number(value, "exploration_fraction"),
        learning_starts=_integer(value, "learning_starts"),
        train_frequency=_integer(value, "train_frequency"),
        eval_episodes=_integer(value, "eval_episodes"),
        adam=AdamConfig(_number(adam, "beta1"), _number(adam, "beta2"), _number(adam, "epsilon")),
    )


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError("expected a JSON object")
    return cast(dict[str, object], value)  # Validated JSON boundary; fields narrowed before constructing configs.


def _boolean(value: dict[str, object], name: str) -> bool:
    field = value[name]
    if not isinstance(field, bool):
        raise ValueError(f"invalid boolean setting {name}")
    return field


def _string(value: dict[str, object], name: str) -> str:
    field = value[name]
    if not isinstance(field, str):
        raise ValueError(f"invalid metadata {name}")
    return field


def _integer(value: dict[str, object], name: str) -> int:
    field = value[name]
    if type(field) is not int:
        raise ValueError(f"invalid metadata {name}")
    return field


def _number(value: dict[str, object], name: str) -> float:
    field = value[name]
    if type(field) not in (int, float):
        raise ValueError(f"invalid numerical field {name}")
    return cast(float, field)  # Preserve integer JSON values for exact saved-settings round trips.


def _validate_phase(metadata: CheckpointMetadata, episodes: int, num_envs: int, vector_steps: int) -> None:
    if metadata.phase != 0 and metadata.vector_step != vector_steps:
        raise ValueError("checkpoint phase disagrees with training progress")
    limit = episodes + num_envs - 1 if episodes else 0
    if metadata.phase != 2 and metadata.evaluation_episodes != 0:
        raise ValueError("checkpoint phase disagrees with evaluation progress")
    if metadata.phase == 2 and not episodes <= metadata.evaluation_episodes <= limit:
        raise ValueError("invalid completed evaluation count")


def host_array(value: Array) -> HostArray:
    return cast(HostArray, np.asarray(jax.device_get(value)))  # Explicit JAX-to-NumPy host boundary.


def _json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported JSON value {type(value)}")
