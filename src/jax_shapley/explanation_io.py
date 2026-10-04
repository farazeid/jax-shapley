"""Host context, callbacks, CSV metrics and atomic Orbax recovery for explanation runs.

All filesystem and serialization effects live here. Saved settings are validated
field by field into typed configuration; the explained agent and characteristic
are identified by array digests so a resumed run explains the same models.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import Field, asdict, dataclass, fields
from math import inf, isfinite
from pathlib import Path
from time import perf_counter, time_ns
from types import TracebackType
from typing import Literal, Protocol, Self, cast

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx, struct
from jax import Array

from jax_shapley.characteristics import (
    CharacteristicConfig,
    CharacteristicModel,
    Kind,
    Policy,
    characteristic_network,
    max_unique_states,
)
from jax_shapley.outcome import ConditionedReplay, OutcomeNetworks, Variant, outcome_network
from jax_shapley.replay import ReplayState
from jax_shapley.run_io import (
    REFERENCE_COMMIT,
    CallbackToken,
    HostArray,
    OrbaxStore,
    _boolean,
    _integer,
    _mapping,
    _number,
    _string,
    acquire_writer_lock,
    host_array,
    load_agent,
    load_settings,
    prefix_digest,
    verify_prefix,
)
from jax_shapley.train_dqn import AdamConfig, EnvironmentConfig, GridworldConfig, Mastermind222Config, MastermindConfig
from jax_shapley.train_explanations import (
    AnyState,
    Buffers,
    BufferSource,
    CharacteristicRunConfig,
    ExplainedOutcome,
    ExplanationPhase,
    ExplanationState,
    ExplanationTrainConfig,
    FrozenAgent,
    OnPolicyState,
    OutcomeInputs,
    OutcomeRunConfig,
    OutcomeTrainConfig,
    ReplayRows,
    RunConfig,
    ShapleyRunConfig,
    buffer_template,
    replay_prefix,
)
from jax_shapley.wrappers import EnvironmentProtocol, EpisodeStatisticsState

type Stage = Literal["characteristic", "shapley", "outcome"]

# Characteristic settings added with replay data, filled in for settings saved before them.
REPLAY_DEFAULTS: dict[str, object] = {"data": "rollouts", "off_policy": False, "weighting": True, "clip_coef": inf}

METRIC_COLUMNS = ("epoch", "step", "train_loss", "validation_loss", "exact_loss", "best_epoch", "seconds")


@dataclass(frozen=True, slots=True)
class RunSettings:
    config: RunConfig
    agent_sha256: str
    characteristic_sha256: str | None


@dataclass(frozen=True, slots=True)
class RunContext:
    """Host-loaded inputs: the frozen agent (and characteristic), static shapes and saved settings."""

    environment: str
    env_config: EnvironmentConfig
    agent: FrozenAgent
    agent_sha256: str
    characteristic: CharacteristicModel | ExplainedOutcome | None
    characteristic_sha256: str | None
    inherited: Buffers | OutcomeInputs | ReplayRows | None
    explained: CharacteristicConfig
    features: int
    actions: int
    exact_rows: int
    max_unique: int
    settings: str
    outcome: OutcomeRunConfig | None = None
    """The explained outcome run's settings, when a Shapley run explains an outcome characteristic."""


@dataclass(frozen=True, slots=True)
class LoadedCharacteristic:
    config: CharacteristicRunConfig
    agent_sha256: str
    model: CharacteristicModel
    buffers: Buffers
    sha256: str


@dataclass(frozen=True, slots=True)
class LoadedOutcome:
    """A completed outcome run's best networks; `characteristic_sha256` identifies its behaviour characteristic."""

    config: OutcomeRunConfig
    agent_sha256: str
    characteristic_sha256: str
    networks: OutcomeNetworks
    sha256: str


@dataclass(frozen=True, slots=True)
class _Sources:
    """What a run explains, as loaded for its context."""

    explained: CharacteristicConfig
    characteristic: CharacteristicModel | ExplainedOutcome | None
    characteristic_sha256: str | None
    inherited: Buffers | OutcomeInputs | ReplayRows | None
    exact_rows: int
    outcome: OutcomeRunConfig | None = None


@dataclass(frozen=True, slots=True)
class ExplanationMetadata:
    settings: str
    epoch: int
    phase: int
    csv_bytes: int = 0
    csv_digest: str = ""
    elapsed_seconds: float = 0.0
    key_implementation: str = "threefry2x32"
    schema: int = 1
    reference_commit: str = REFERENCE_COMMIT

    def validate(self) -> None:
        if self.schema != 1 or self.reference_commit != REFERENCE_COMMIT:
            raise ValueError("unsupported checkpoint schema or reference")
        if self.phase not in (ExplanationPhase.TRAINING, ExplanationPhase.COMPLETE) or self.csv_bytes < 0:
            raise ValueError("invalid checkpoint phase or log boundary")
        if not isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("invalid checkpoint timing")
        if self.key_implementation not in ("threefry2x32", "rbg", "unsafe_rbg"):
            raise ValueError("unsupported checkpoint PRNG implementation")
        config = parse_run_settings(self.settings).config
        if not 1 <= self.epoch <= config.progress_limit:
            raise ValueError("checkpoint epoch outside the configured training duration")


@dataclass(frozen=True, slots=True)
class ExplanationSummary:
    folder: str
    stage: Stage
    epochs: int
    best_epoch: int
    best_validation_loss: float | None
    """None for on-policy outcome runs, which neither validate nor stop early on validation."""
    stopped_early: bool
    seconds: float
    backend: str


@struct.dataclass
class ExplanationSnapshot:
    model: tuple[Array, ...]
    best: tuple[Array, ...]
    optimizer: tuple[Array, ...]
    buffers: tuple[Array, ...]
    runtime: tuple[Array, ...]
    epoch: Array
    phase: Array
    key_implementation: str = struct.field(pytree_node=False, default="threefry2x32")


class ExplanationSink(Protocol):
    """Compiled-code callbacks; every leaf carries the seed axis (after the chunk axis for metrics)."""

    def epochs(self, metrics: tuple[Array, ...]) -> CallbackToken: ...

    def checkpoint(self, snapshot: ExplanationSnapshot, due: Array) -> CallbackToken: ...


class CallbackRouter:
    """Bind the writer only after compilation; preflight cannot perform host I/O."""

    def __init__(self) -> None:
        self.sink: ExplanationSink | None = None

    def epochs(self, metrics: tuple[Array, ...]) -> CallbackToken:
        return self._sink().epochs(metrics)

    def checkpoint(self, snapshot: ExplanationSnapshot, due: Array) -> CallbackToken:
        return self._sink().checkpoint(snapshot, due)

    def _sink(self) -> ExplanationSink:
        if self.sink is None:
            raise RuntimeError("callback invoked before writer binding")
        return self.sink


class ExplanationStore(OrbaxStore):
    """Explanation snapshots: current and best models, optimizer, buffers and loop state."""

    def save(self, snapshot: ExplanationSnapshot, metadata: ExplanationMetadata) -> None:
        metadata.validate()
        if metadata.key_implementation != snapshot.key_implementation:
            raise ValueError("checkpoint PRNG metadata disagrees with snapshot")
        self.save_items(
            {
                "model": snapshot.model,
                "best": snapshot.best,
                "optimizer": snapshot.optimizer,
                "buffers": snapshot.buffers,
                "runtime": (*snapshot.runtime, snapshot.epoch, snapshot.phase),
            },
            asdict(metadata),
        )

    def metadata(self) -> ExplanationMetadata:
        value = self.metadata_json()  # JSON is checked field-by-field below.
        if set(value) != {item.name for item in fields(ExplanationMetadata)}:
            raise ValueError("invalid checkpoint metadata schema")
        metadata = ExplanationMetadata(
            settings=_string(value, "settings"),
            epoch=_integer(value, "epoch"),
            phase=_integer(value, "phase"),
            csv_bytes=_integer(value, "csv_bytes"),
            csv_digest=_string(value, "csv_digest"),
            elapsed_seconds=_number(value, "elapsed_seconds"),
            key_implementation=_string(value, "key_implementation"),
            schema=_integer(value, "schema"),
            reference_commit=_string(value, "reference_commit"),
        )
        metadata.validate()
        return metadata


class SweepWriter:
    """The compiled run's sink: fans seed-batched callbacks out to one writer per seed folder."""

    def __init__(self, writers: list[ExplanationWriter]) -> None:
        self.writers = writers

    def epochs(self, metrics: tuple[Array, ...]) -> CallbackToken:
        values = tuple(host_array(value) for value in metrics)
        for seed, writer in enumerate(self.writers):
            # EpochMetrics leaves are (chunk, seeds); the last field is the active flag.
            if bool(np.any(values[-1][:, seed])):
                writer.epochs(tuple(value[:, seed] for value in values))
        return np.asarray(0, np.int32)

    def checkpoint(self, snapshot: ExplanationSnapshot, due: Array) -> CallbackToken:
        flags = host_array(due)
        for seed, writer in enumerate(self.writers):
            if bool(flags[seed]):
                writer.checkpoint(seed_snapshot(snapshot, seed))
        return np.asarray(0, np.int32)

    def wait(self) -> None:
        for writer in self.writers:
            writer.store.wait()


class ExplanationWriter:
    """One run folder's metrics and checkpoints; SweepWriter routes each seed's callbacks here."""

    def __init__(
        self,
        folder: Path,
        settings: str,
        *,
        metadata: ExplanationMetadata | None = None,
        checkpoint_sequence: int | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.folder, self.settings, self.clock = folder, settings, clock
        self.lock = acquire_writer_lock(folder)
        try:
            if metadata is not None:
                # Preflight restoration/compilation runs without the writer lock.
                store = ExplanationStore(folder, read_only=True)
                try:
                    if checkpoint_sequence is None or store.sequence != checkpoint_sequence:
                        raise RuntimeError("checkpoint changed during resume preflight; retry resume")
                finally:
                    store.close()
                verify_prefix(folder / "metrics.csv", metadata.csv_bytes, metadata.csv_digest)
                with (folder / "metrics.csv").open("r+b") as log:
                    log.truncate(metadata.csv_bytes)
                    log.flush()
                    os.fsync(log.fileno())
            self.store = ExplanationStore(folder)
        except BaseException:
            self.lock.close()
            raise
        self.started = clock()
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

    def epochs(self, metrics: tuple[HostArray, ...]) -> CallbackToken:
        write_epochs(self.folder / "metrics.csv", metrics, self.elapsed())
        return np.asarray(0, np.int32)

    def checkpoint(self, snapshot: ExplanationSnapshot) -> CallbackToken:
        self.store.wait()
        path = self.folder / "metrics.csv"
        with path.open("a+b") as log:
            log.flush()
            os.fsync(log.fileno())
            boundary = log.tell()
        metadata = ExplanationMetadata(
            self.settings,
            int(host_array(snapshot.epoch)),
            int(host_array(snapshot.phase)),
            boundary,
            prefix_digest(path, boundary),
            self.elapsed(),
            key_implementation=snapshot.key_implementation,
        )
        self.store.save(snapshot, metadata)
        return np.asarray(0, np.int32)

    def elapsed(self) -> float:
        return self.prior_elapsed + self.clock() - self.started

    def close(self) -> None:
        try:
            self.store.close()
        finally:
            self.lock.close()


def prepare_context(config: RunConfig) -> RunContext:
    """Load and identify the explained models; derive static shapes and the saved settings."""
    agent_settings = load_settings(config.agent / "config.json")
    replay = (isinstance(config, OutcomeRunConfig) and config.variant == "off_policy") or (
        isinstance(config, CharacteristicRunConfig) and config.data == "replay"
    )
    loaded = load_agent(config.agent, agent_settings.network, load_replay=replay)
    if loaded.vector_step != agent_settings.train.vector_steps:
        raise ValueError("explanations require a completed DQN training run")
    agent = FrozenAgent(loaded.model, jnp.asarray(loaded.epsilon, jnp.float32))
    agent_sha256 = array_sha256((*jax.tree.leaves(nnx.state(loaded.model)), agent.epsilon))
    environment, actions, low, high = _environment(agent_settings.env)
    features = agent_settings.network.input_features
    if isinstance(config, CharacteristicRunConfig):
        sources = _Sources(
            config.characteristic, None, None, _replay_rows(config, loaded.replay), config.train.exact_size
        )
    elif isinstance(config, OutcomeRunConfig):
        sources = _outcome_sources(config, loaded.replay, agent_sha256)
    elif isinstance(load_run_settings(config.characteristic / "config.json").config, OutcomeRunConfig):
        sources = _outcome_shapley_sources(config, agent_sha256)
    else:
        sources = _shapley_sources(config, agent_sha256)
    max_unique = max_unique_states(low, high, features, sources.exact_rows)
    settings = describe(config, environment, agent_sha256, sources.characteristic_sha256)
    return RunContext(
        environment,
        agent_settings.env,
        agent,
        agent_sha256,
        sources.characteristic,
        sources.characteristic_sha256,
        sources.inherited,
        sources.explained,
        features,
        actions,
        sources.exact_rows,
        max_unique,
        settings,
        sources.outcome,
    )


def load_characteristic(folder: Path) -> LoadedCharacteristic:
    """A completed characteristic run's best model and buffers (train, validation, exact)."""
    settings = load_run_settings(folder / "config.json")
    config = settings.config
    if not isinstance(config, CharacteristicRunConfig):
        raise ValueError("expected a characteristic run")
    agent = load_settings(config.agent / "config.json")
    features, actions = agent.network.input_features, agent.env.make().num_actions
    model = CharacteristicModel(characteristic_network(features, actions, config.characteristic), rngs=nnx.Rngs(0))
    template = buffer_template(config, features, config.train.exact_size)
    store = ExplanationStore(folder, read_only=True)
    try:
        if store.metadata().phase != ExplanationPhase.COMPLETE:
            raise ValueError("the characteristic run has not completed")
        state = nnx.state(model)
        values = store.arrays("model", _leaves(state))
        nnx.update(model, jax.tree.unflatten(jax.tree.structure(state), values))
        buffers = _buffers(store, template)
    finally:
        store.close()
    return LoadedCharacteristic(config, settings.agent_sha256, model, buffers, array_sha256(values))


def load_outcome(folder: Path) -> LoadedOutcome:
    """A completed outcome run's networks (the best-validation model and its target)."""
    settings = load_run_settings(folder / "config.json")
    config = settings.config
    if not isinstance(config, OutcomeRunConfig) or settings.characteristic_sha256 is None:
        raise ValueError("expected an outcome run")
    agent = load_settings(config.agent / "config.json")
    features, actions = agent.network.input_features, agent.env.make().num_actions
    networks = OutcomeNetworks(outcome_network(features, actions, config.outcome), rngs=nnx.Rngs(0))
    store = ExplanationStore(folder, read_only=True)
    try:
        if store.metadata().phase != ExplanationPhase.COMPLETE:
            raise ValueError("the outcome run has not completed")
        state = nnx.state(networks)
        values = store.arrays("model", _leaves(state))
        nnx.update(networks, jax.tree.unflatten(jax.tree.structure(state), values))
    finally:
        store.close()
    return LoadedOutcome(config, settings.agent_sha256, settings.characteristic_sha256, networks, array_sha256(values))


def restore_state[Model: nnx.Module](
    store: ExplanationStore, template: ExplanationState[Model], config: RunConfig, context: RunContext
) -> tuple[ExplanationState[Model], ExplanationMetadata]:
    metadata = store.metadata()
    if parse_run_settings(metadata.settings) != parse_run_settings(context.settings):
        raise ValueError("resume requires identical saved settings")
    for item, module in (("model", template.model), ("best", template.best), ("optimizer", template.optimizer)):
        state = nnx.state(module)
        nnx.update(module, jax.tree.unflatten(jax.tree.structure(state), store.arrays(item, _leaves(state))))
    buffers = _buffers(store, template.buffers)
    key_data = jax.random.key_data(jax.random.key(0, impl=metadata.key_implementation))
    runtime = store.arrays(
        "runtime",
        (template.best_loss, template.best_epoch, template.stopped, key_data, template.epoch, template.phase),
    )
    best_loss, best_epoch, stopped, key, epoch, phase = runtime
    if int(epoch) != metadata.epoch or int(phase) != metadata.phase:
        raise ValueError("checkpoint counters disagree with metadata")
    if not 1 <= int(best_epoch) <= int(epoch):
        raise ValueError("invalid best epoch")
    if int(template.optimizer.step[...]) != int(epoch) * config.train.batches:
        raise ValueError("optimizer step disagrees with completed epochs")
    if int(phase) == ExplanationPhase.TRAINING and bool(stopped):
        raise ValueError("a stopped run must be complete")
    restored = ExplanationState(
        template.model,
        template.best,
        template.optimizer,
        buffers,
        epoch,
        best_loss,
        best_epoch,
        stopped,
        phase,
        jax.random.wrap_key_data(key, impl=metadata.key_implementation),
    )
    return restored, metadata


def restore_on_policy[State](
    store: ExplanationStore, template: OnPolicyState[State], config: OutcomeRunConfig, context: RunContext
) -> tuple[OnPolicyState[State], ExplanationMetadata]:
    """Restore an on-policy outcome checkpoint into a template with the run's static shapes."""
    metadata = store.metadata()
    if metadata.settings != context.settings:
        raise ValueError("resume requires identical saved settings")
    for item, module in (("model", template.networks), ("optimizer", template.optimizer)):
        state = nnx.state(module)
        nnx.update(module, jax.tree.unflatten(jax.tree.structure(state), store.arrays(item, _leaves(state))))
    layout = (template.replay, template.observations, template.env_state, template.explained, template.coalitions)
    restored = store.arrays("buffers", _leaves(layout))
    restored_layout = cast(
        tuple[ConditionedReplay, Array, EpisodeStatisticsState[State], Array, Array],
        jax.tree.unflatten(jax.tree.structure(layout), restored),
    )  # Unflattening with the template's own structure returns its topology.
    replay, observations, env_state, explained, coalitions = restored_layout
    key_data = jax.random.key_data(jax.random.key(0, impl=metadata.key_implementation))
    stopped, key, step, phase = store.arrays("runtime", (template.stopped, key_data, template.step, template.phase))
    if int(step) != metadata.epoch or int(phase) != metadata.phase:
        raise ValueError("checkpoint counters disagree with metadata")
    training = config.train
    updates = sum(
        1 for index in range(int(step)) if index * training.num_envs % training.train_frequency < training.num_envs
    )
    if int(template.optimizer.step[...]) != updates:
        raise ValueError("optimizer step disagrees with completed vector steps")
    if int(phase) == ExplanationPhase.TRAINING and bool(stopped):
        raise ValueError("a stopped run must be complete")
    state = OnPolicyState(
        template.networks,
        template.optimizer,
        replay,
        observations,
        env_state,
        explained,
        coalitions,
        step,
        stopped,
        phase,
        jax.random.wrap_key_data(key, impl=metadata.key_implementation),
    )
    return state, metadata


def create_run(config: RunConfig, context: RunContext) -> Path:
    """Create fresh artifacts only after successful numerical compilation."""
    folder = config.output / f"{context.environment}-{_label(config, context)}-{config.seed}-{time_ns()}"
    _initialize_folder(folder, context.settings)
    return folder


def create_sweep(config: RunConfig, contexts: list[RunContext]) -> list[Path]:
    """A sweep folder with one ordinary run folder per seed and a `sweep.json` index for resume."""
    root = config.output / f"{contexts[0].environment}-{_label(config, contexts[0])}-sweep-{time_ns()}"
    folders = [root / f"seed-{seed}" for seed in config.sweep]
    for folder, context in zip(folders, contexts, strict=True):
        _initialize_folder(folder, context.settings)
    index = {"seeds": list(config.sweep), "folders": [folder.name for folder in folders]}
    (root / "sweep.json").write_text(json.dumps(index, indent=2) + "\n")
    return folders


def sweep_folders(folder: Path) -> list[Path]:
    """Seed folders of a sweep, or the folder itself for a single run."""
    index_path = folder / "sweep.json"
    if not index_path.exists():
        return [folder]
    index = _mapping(json.loads(index_path.read_text()))
    names = index["folders"]
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        raise ValueError("invalid sweep index")
    return [folder / name for name in cast(list[str], names)]  # Element types checked above.


def seed_snapshot(snapshot: ExplanationSnapshot, seed: int) -> ExplanationSnapshot:
    """One seed's arrays from a seed-batched snapshot."""

    def pick(values: tuple[Array, ...]) -> tuple[Array, ...]:
        return tuple(value[seed] for value in values)

    return ExplanationSnapshot(
        pick(snapshot.model),
        pick(snapshot.best),
        pick(snapshot.optimizer),
        pick(snapshot.buffers),
        pick(snapshot.runtime),
        snapshot.epoch[seed],
        snapshot.phase[seed],
        snapshot.key_implementation,
    )


def _initialize_folder(folder: Path, settings: str) -> None:
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "config.json").write_text(settings)
    with (folder / "metrics.csv").open("w", newline="") as output:
        csv.writer(output).writerow(METRIC_COLUMNS)


def _replay_rows(config: CharacteristicRunConfig, replay: ReplayState | None) -> ReplayRows | None:
    """The agent's filled replay rows, for characteristic runs that learn from it."""
    if config.data != "replay":
        return None
    rows = config.train.train_size + config.train.validation_size
    if replay is None or int(replay.size) < rows:
        raise ValueError(f"replay characteristics need at least {rows} transitions in the agent's replay")
    size = int(replay.size)
    data = replay.data
    return ReplayRows(data.observations[:size], data.actions[:size], data.log_probabilities[:size])


def _outcome_sources(config: OutcomeRunConfig, replay: ReplayState | None, agent_sha256: str) -> _Sources:
    source = load_characteristic(config.characteristic)
    if source.agent_sha256 != agent_sha256:
        raise ValueError("the behaviour characteristic explains a different agent")
    if source.config.kind != "behaviour":
        raise ValueError("outcome characteristics condition on a behaviour characteristic")
    if config.variant == "on_policy":
        # On-policy runs draw e_obs from the behaviour characteristic's train buffer and interact.
        return _Sources(
            source.config.characteristic, source.model, source.sha256, source.buffers, source.config.train.exact_size
        )
    rows = max(config.train.train_size, config.train.validation_size)
    if replay is None or int(replay.size) < rows:
        raise ValueError(f"outcome runs need at least {rows} transitions in the agent's replay")
    inherited = OutcomeInputs(
        source.buffers,
        replay_prefix(replay.data, config.train.train_size),
        replay_prefix(replay.data, config.train.validation_size),
    )
    return _Sources(
        source.config.characteristic, source.model, source.sha256, inherited, source.config.train.exact_size
    )


def _outcome_shapley_sources(config: ShapleyRunConfig, agent_sha256: str) -> _Sources:
    explained = load_outcome(config.characteristic)
    behaviour = load_characteristic(explained.config.characteristic)
    if explained.agent_sha256 != agent_sha256:
        raise ValueError("the outcome characteristic explains a different agent")
    if behaviour.sha256 != explained.characteristic_sha256:
        raise ValueError("the outcome characteristic's behaviour characteristic changed")
    if config.buffers != "fresh":
        raise ValueError("outcome Shapley runs collect fresh buffers")
    return _Sources(
        behaviour.config.characteristic,
        ExplainedOutcome(explained.networks, behaviour.model),
        explained.sha256,
        behaviour.buffers,
        behaviour.config.train.exact_size,
        explained.config,
    )


def _shapley_sources(config: ShapleyRunConfig, agent_sha256: str) -> _Sources:
    source = load_characteristic(config.characteristic)
    if source.agent_sha256 != agent_sha256:
        raise ValueError("the characteristic explains a different agent")
    if config.buffers == "characteristic" and (
        config.train.train_size != source.config.train.train_size
        or config.train.validation_size != source.config.train.validation_size
    ):
        raise ValueError("reused characteristic buffers require identical train/validation sizes")
    return _Sources(
        source.config.characteristic, source.model, source.sha256, source.buffers, source.config.train.exact_size
    )


def _label(config: RunConfig, context: RunContext) -> str:
    if isinstance(config, CharacteristicRunConfig):
        return f"characteristic-{config.kind}"
    if isinstance(config, OutcomeRunConfig):
        return f"outcome-{config.variant.replace('_', '-')}"
    return "shapley-outcome" if context.outcome is not None else f"shapley-{context.explained.kind}"


def write_epochs(path: Path, metrics: tuple[HostArray, ...], seconds: float) -> None:
    """Active epoch rows, in EpochMetrics field order; a loss a run does not compute is left empty.

    On-policy outcome rows have no validation loss, and no exact loss when it is disabled.
    """
    epoch, step, train_loss, validation_loss, exact_loss, best_epoch, active = metrics
    rows = [
        (
            int(epoch[index]),
            int(step[index]),
            _cell(train_loss[index]),
            _cell(validation_loss[index]),
            _cell(exact_loss[index]),
            int(best_epoch[index]),
            round(seconds, 6),
        )
        for index in np.flatnonzero(active)
    ]
    with path.open("a", newline="") as output:
        csv.writer(output).writerows(rows)


def summarize(
    folder: Path,
    config: RunConfig,
    result: AnyState,
    seconds: float,
) -> ExplanationSummary:
    """`epochs` counts progress units: epochs, or vector steps for on-policy outcome runs."""
    if isinstance(result, OnPolicyState):
        steps = int(result.step)
        return ExplanationSummary(
            str(folder),
            _stage(config),
            steps,
            steps,
            None,
            steps < config.progress_limit,
            seconds,
            jax.default_backend(),
        )
    return ExplanationSummary(
        str(folder),
        _stage(config),
        int(result.epoch),
        int(result.best_epoch),
        float(result.best_loss),
        int(result.epoch) < config.progress_limit,
        seconds,
        jax.default_backend(),
    )


def _cell(value: HostArray) -> float | str:
    number = float(value)
    return "" if np.isnan(number) else number


def write_summary(folder: Path, summary: ExplanationSummary) -> None:
    staged = folder / "summary.json.tmp"
    staged.write_text(json.dumps(asdict(summary), indent=2) + "\n")
    staged.replace(folder / "summary.json")


def describe(config: RunConfig, environment: str, agent_sha256: str, characteristic_sha256: str | None) -> str:
    envelope = {
        "stage": _stage(config),
        "config": asdict(config),
        "environment": environment,
        "agent_sha256": agent_sha256,
        "characteristic_sha256": characteristic_sha256,
        "reference_commit": REFERENCE_COMMIT,
    }
    return json.dumps(envelope, default=_json_default, indent=2) + "\n"


def describe_options(config: RunConfig) -> str:
    """Validated CLI settings only; no agent or characteristic artifacts are read."""
    return json.dumps({"stage": _stage(config), "config": asdict(config)}, default=_json_default, indent=2) + "\n"


def load_run_settings(path: Path) -> RunSettings:
    return parse_run_settings(path.read_text())


def parse_run_settings(text: str) -> RunSettings:
    envelope = _mapping(json.loads(text))
    if set(envelope) != {"stage", "config", "environment", "agent_sha256", "characteristic_sha256", "reference_commit"}:
        raise ValueError("saved explanation settings have an invalid schema")
    if _string(envelope, "reference_commit") != REFERENCE_COMMIT:
        raise ValueError("saved settings reference a different FastSVERL commit")
    value = _mapping(envelope["config"])
    stage = _string(envelope, "stage")
    config: RunConfig
    if stage == "characteristic":
        config = _characteristic_config(value)
    elif stage == "shapley":
        config = _shapley_config(value)
    elif stage == "outcome":
        config = _outcome_config(value)
    else:
        raise ValueError("unsupported explanation stage")
    characteristic = envelope["characteristic_sha256"]
    if characteristic is not None and not isinstance(characteristic, str):
        raise ValueError("invalid characteristic digest")
    settings = RunSettings(config, _string(envelope, "agent_sha256"), characteristic)
    if config.validate_only or config.compile_only:
        raise ValueError("saved settings cannot select an inspection mode")
    expected = describe(config, _string(envelope, "environment"), settings.agent_sha256, characteristic)
    if json.loads(expected) != envelope:
        raise ValueError("saved settings are inconsistent")
    return settings


def array_sha256(arrays: tuple[Array, ...] | list[Array]) -> str:
    """Digest of dtypes, shapes and bytes; identifies frozen models across runs."""
    digest = hashlib.sha256()
    for value in arrays:
        host = np.ascontiguousarray(host_array(value))
        digest.update(f"{host.dtype.str}{host.shape}".encode())
        digest.update(host.tobytes())
    return digest.hexdigest()


def _characteristic_config(value: dict[str, object]) -> CharacteristicRunConfig:
    for name, default in REPLAY_DEFAULTS.items():
        value.setdefault(name, default)  # Settings saved before replay data existed.
    _schema(value, fields(CharacteristicRunConfig))
    data = _string(value, "data")
    if data not in ("rollouts", "replay"):
        raise ValueError("invalid saved characteristic data")
    return CharacteristicRunConfig(
        agent=Path(_string(value, "agent")),
        kind=_kind(_string(value, "kind")),
        policy=_policy(_string(value, "policy")),
        hidden_widths=_widths(value),
        data="replay" if data == "replay" else "rollouts",
        off_policy=_boolean(value, "off_policy"),
        weighting=_boolean(value, "weighting"),
        clip_coef=_number(value, "clip_coef"),
        train=_train_config(_mapping(value["train"])),
        seed=_integer(value, "seed"),
        seeds=_seeds(value),
        chunk_epochs=_integer(value, "chunk_epochs"),
        output=Path(_string(value, "output")),
        validate_only=_boolean(value, "validate_only"),
        compile_only=_boolean(value, "compile_only"),
    )


def _buffers(store: ExplanationStore, template: Buffers) -> Buffers:
    restored = store.arrays("buffers", _leaves(template))
    return cast(Buffers, jax.tree.unflatten(jax.tree.structure(template), restored))  # The template's own layout.


def _shapley_config(value: dict[str, object]) -> ShapleyRunConfig:
    _schema(value, fields(ShapleyRunConfig))
    buffers: BufferSource = "fresh" if _string(value, "buffers") == "fresh" else "characteristic"
    if _string(value, "buffers") != buffers:
        raise ValueError("invalid saved buffer source")
    return ShapleyRunConfig(
        Path(_string(value, "agent")),
        Path(_string(value, "characteristic")),
        _boolean(value, "with_exact_char"),
        buffers,
        _widths(value),
        _train_config(_mapping(value["train"])),
        _integer(value, "seed"),
        _seeds(value),
        _integer(value, "chunk_epochs"),
        Path(_string(value, "output")),
        _boolean(value, "validate_only"),
        _boolean(value, "compile_only"),
    )


def _outcome_config(value: dict[str, object]) -> OutcomeRunConfig:
    _schema(value, fields(OutcomeRunConfig))
    return OutcomeRunConfig(
        Path(_string(value, "agent")),
        Path(_string(value, "characteristic")),
        _variant(_string(value, "variant")),
        _boolean(value, "with_exact_char"),
        _number(value, "tau"),
        _number(value, "gamma"),
        _widths(value),
        _outcome_train_config(_mapping(value["train"])),
        _integer(value, "seed"),
        _seeds(value),
        _integer(value, "chunk_epochs"),
        Path(_string(value, "output")),
        _boolean(value, "validate_only"),
        _boolean(value, "compile_only"),
    )


def _outcome_train_config(value: dict[str, object]) -> OutcomeTrainConfig:
    _schema(value, fields(OutcomeTrainConfig))
    return OutcomeTrainConfig(
        epochs=_integer(value, "epochs"),
        patience=None if value["patience"] is None else _integer(value, "patience"),
        learning_rate=_number(value, "learning_rate"),
        batch_size=_integer(value, "batch_size"),
        train_size=_integer(value, "train_size"),
        validation_size=_integer(value, "validation_size"),
        exact_loss=_boolean(value, "exact_loss"),
        exact_condition=None if value["exact_condition"] is None else _number(value, "exact_condition"),
        track_batch=_boolean(value, "track_batch"),
        total_timesteps=_integer(value, "total_timesteps"),
        num_envs=_integer(value, "num_envs"),
        train_frequency=_integer(value, "train_frequency"),
        target_network_frequency=_integer(value, "target_network_frequency"),
        adam=_adam(_mapping(value["adam"])),
    )


def _train_config(value: dict[str, object]) -> ExplanationTrainConfig:
    _schema(value, fields(ExplanationTrainConfig))
    return ExplanationTrainConfig(
        epochs=_integer(value, "epochs"),
        patience=None if value["patience"] is None else _integer(value, "patience"),
        learning_rate=_number(value, "learning_rate"),
        batch_size=_integer(value, "batch_size"),
        buffer_size=_integer(value, "buffer_size"),
        val_fraction=_number(value, "val_fraction"),
        num_envs=_integer(value, "num_envs"),
        exact_loss=_boolean(value, "exact_loss"),
        exact_buffer_size=None if value["exact_buffer_size"] is None else _integer(value, "exact_buffer_size"),
        exact_condition=None if value["exact_condition"] is None else _number(value, "exact_condition"),
        track_batch=_boolean(value, "track_batch"),
        adam=_adam(_mapping(value["adam"])),
    )


def _adam(value: dict[str, object]) -> AdamConfig:
    _schema(value, fields(AdamConfig))
    return AdamConfig(_number(value, "beta1"), _number(value, "beta2"), _number(value, "epsilon"))


def _schema(value: dict[str, object], expected: tuple[Field[object], ...]) -> None:
    if set(value) != {item.name for item in expected}:
        raise ValueError("saved settings have an invalid schema")


def _kind(value: str) -> Kind:
    if value == "behaviour":
        return "behaviour"
    if value == "prediction":
        return "prediction"
    raise ValueError("invalid saved characteristic kind")


def _policy(value: str) -> Policy:
    if value == "greedy":
        return "greedy"
    if value == "explore":
        return "explore"
    raise ValueError("invalid saved explained policy")


def _variant(value: str) -> Variant:
    if value == "off_policy":
        return "off_policy"
    if value == "on_policy":
        return "on_policy"
    raise ValueError("invalid saved outcome variant")


def _environment(config: EnvironmentConfig) -> tuple[str, int, HostArray, HostArray]:
    """Name, action count and observation bounds, preserving each environment's parameter type."""
    if isinstance(config, GridworldConfig):
        return _space(config.make())
    if isinstance(config, Mastermind222Config | MastermindConfig):
        return _space(config.make())
    return _space(config.make())


def _space[State, Params](env: EnvironmentProtocol[State, Params]) -> tuple[str, int, HostArray, HostArray]:
    space = env.observation_space(env.default_params)
    return env.name, env.num_actions, np.asarray(space.low), np.asarray(space.high)


def _seeds(value: dict[str, object]) -> tuple[int, ...]:
    seeds = value["seeds"]
    if not isinstance(seeds, list) or any(type(seed) is not int for seed in seeds):
        raise ValueError("invalid saved sweep seeds")
    return tuple(cast(list[int], seeds))  # Element types checked above.


def _widths(value: dict[str, object]) -> tuple[int, ...]:
    widths = value["hidden_widths"]
    if not isinstance(widths, list) or any(type(width) is not int for width in widths):
        raise ValueError("invalid saved hidden widths")
    return tuple(cast(list[int], widths))  # Element types checked above.


def _stage(config: RunConfig) -> Stage:
    if isinstance(config, CharacteristicRunConfig):
        return "characteristic"
    return "outcome" if isinstance(config, OutcomeRunConfig) else "shapley"


def _leaves(tree: object) -> tuple[Array, ...]:
    return tuple(cast(list[Array], jax.tree.leaves(tree)))  # Numerical payload construction only.


def _json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported JSON value {type(value)}")
