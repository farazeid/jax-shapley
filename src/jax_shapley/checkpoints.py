"""Typed host boundary for atomic Orbax composites and portable partial loading."""

import json
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from jax import Array

from jax_shapley.algorithms.dqn import RunState, epsilon_at
from jax_shapley.artifacts import REFERENCE_COMMIT, LoadedAgent, describe, host_array
from jax_shapley.config import DQNRunConfig, EnvironmentConfig
from jax_shapley.execution import Snapshot
from jax_shapley.networks import MLPConfig, QNetwork
from jax_shapley.replay import ReplayState, empty_replay
from jax_shapley.settings import parse_settings
from jax_shapley.wrappers import EpisodeStatisticsState


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


class CheckpointStore:
    """One serialized producer; Orbax owns its asynchronous writer and commit protocol."""

    def __init__(self, folder: Path, *, read_only: bool = False) -> None:
        options = ocp.CheckpointManagerOptions(
            read_only=read_only,
            create=not read_only,
            enable_async_checkpointing=not read_only,
            preservation_policy=ocp.v1.training.preservation_policies.LatestN(n=1),
        )
        self.manager = ocp.CheckpointManager(folder.resolve() / "checkpoints", options=options)
        self.sequence = self.manager.latest_step()

    def save(self, snapshot: Snapshot, metadata: CheckpointMetadata) -> None:
        metadata.validate()
        if metadata.key_implementation != snapshot.key_implementation:
            raise ValueError("checkpoint PRNG metadata disagrees with snapshot")
        self.manager.wait_until_finished()
        sequence = 0 if self.sequence is None else self.sequence + 1
        items: dict[str, object] = {
            "agent": ocp.args.StandardSave(_pack((*snapshot.agent, snapshot.epsilon, snapshot.vector_step))),
            "metadata": ocp.args.JsonSave(asdict(metadata)),
        }
        if snapshot.replay is not None:
            items["replay"] = ocp.args.StandardSave(_pack(_leaves(snapshot.replay)))
        if snapshot.runtime:
            items["run_state"] = ocp.args.StandardSave(_pack((*snapshot.optimizer, *snapshot.runtime)))
        if not self.manager.save(sequence, args=ocp.args.Composite(**items)):
            raise OSError("Orbax declined a required checkpoint")
        self.sequence = sequence

    def metadata(self) -> CheckpointMetadata:
        result = self._restore({"metadata": ocp.args.JsonRestore()})
        value = cast(dict[str, object], result["metadata"])  # JSON is checked field-by-field below.
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


def load_replay_state(store: CheckpointStore, features: int, capacity: int) -> ReplayState:
    return _restore_replay(store, empty_replay(capacity, features))


def restore_run[State, Config: EnvironmentConfig](
    store: CheckpointStore, template: RunState[State], config: DQNRunConfig[Config]
) -> tuple[RunState[State], CheckpointMetadata]:
    metadata = store.metadata()
    if config.checkpoint_mode != "run" or metadata.settings != describe(config):
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
    if not isinstance(field, (int, float)):
        raise ValueError(f"invalid metadata {name}")
    return float(field)


def _validate_phase(metadata: CheckpointMetadata, episodes: int, num_envs: int, vector_steps: int) -> None:
    if metadata.phase != 0 and metadata.vector_step != vector_steps:
        raise ValueError("checkpoint phase disagrees with training progress")
    limit = episodes + num_envs - 1 if episodes else 0
    if metadata.phase != 2 and metadata.evaluation_episodes != 0:
        raise ValueError("checkpoint phase disagrees with evaluation progress")
    if metadata.phase == 2 and not episodes <= metadata.evaluation_episodes <= limit:
        raise ValueError("invalid completed evaluation count")
