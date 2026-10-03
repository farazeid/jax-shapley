"""FastSVERL behaviour/prediction explanations: configuration, compiled training and CLI.

A characteristic run explains a trained DQN agent; a Shapley run explains that
characteristic. Each run collects its observation buffers with the agent, trains
for whole epochs with early stopping, and checkpoints complete state for exact
resume. Shell recipes provide experiment values; explanation_io.py owns host I/O.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from functools import partial
from math import isfinite, nan
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, NamedTuple, Protocol, cast

import jax
import jax.numpy as jnp
import tyro
from flax import nnx, struct
from jax import Array
from jax.experimental import io_callback

from jax_shapley.characteristics import (
    CharacteristicConfig,
    CharacteristicModel,
    ExactCharacteristic,
    Kind,
    Policy,
    characteristic_loss,
    characteristic_network,
    exact_characteristic,
    exact_characteristic_loss,
    explained_values,
    update_characteristic,
)
from jax_shapley.coalitions import all_coalitions, kernel_coalitions, uniform_coalitions
from jax_shapley.envs import Gridworld, Mastermind222
from jax_shapley.networks import QNetwork
from jax_shapley.shapley import (
    CharacteristicFunction,
    MaskedCharacteristic,
    ShapleyLayout,
    ShapleyModel,
    characteristic_table,
    exact_shapley,
    exact_shapley_loss,
    shapley_loss,
    shapley_network,
    update_shapley,
)
from jax_shapley.train_dqn import (
    AdamConfig,
    GridworldConfig,
    Mastermind222Config,
    ResumeConfig,
    choose_action,
    environment_step,
    make_adam,
)
from jax_shapley.wrappers import EnvironmentProtocol, EpisodeStatisticsState, EpisodeStatisticsWrapper

if TYPE_CHECKING:
    from jax_shapley.explanation_io import ExplanationSink, ExplanationSnapshot, ExplanationSummary, RunContext

type BufferSource = Literal["fresh", "characteristic"]


class ExplanationPhase(IntEnum):
    TRAINING = 0
    COMPLETE = 1


@dataclass(frozen=True, slots=True)
class ExplanationTrainConfig:
    """Source train_models settings; buffer sizes count observations across all environments."""

    epochs: int = 50
    patience: int | None = None
    learning_rate: float = 2.5e-4
    batch_size: int = 128
    buffer_size: int = 10_000
    val_fraction: float = 0.2
    num_envs: int = 16
    exact_loss: bool = True
    exact_buffer_size: int | None = None
    exact_condition: float | None = None
    track_batch: bool = False
    adam: AdamConfig = field(default_factory=AdamConfig)

    def __post_init__(self) -> None:
        for name, value in (
            ("epochs", self.epochs),
            ("batch_size", self.batch_size),
            ("buffer_size", self.buffer_size),
            ("num_envs", self.num_envs),
        ):
            _positive(name, value)
        if self.patience is not None:
            _positive("patience", self.patience)
        if not isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 < self.val_fraction < 1:
            raise ValueError("val_fraction must be in (0, 1)")
        self._validate_buffers()
        if self.exact_condition is not None:
            if not self.exact_loss:
                raise ValueError("exact_condition requires exact_loss")
            if not isfinite(self.exact_condition) or self.exact_condition <= 0:
                raise ValueError("exact_condition must be positive")

    def _validate_buffers(self) -> None:
        sizes = [("train", self.train_size), ("validation", self.validation_size)]
        if self.exact_buffer_size is not None:
            sizes.append(("exact", self.exact_buffer_size))
        for name, size in sizes:
            if size < self.num_envs or size % self.num_envs:
                raise ValueError(f"{name} buffer size {size} must be a positive multiple of num_envs")
        if self.batches < 1:
            raise ValueError("the train buffer must hold at least one batch")

    @property
    def train_size(self) -> int:
        # Source arithmetic: int(buffer_size * (1 - val_fraction)).
        return int(self.buffer_size * (1 - self.val_fraction))

    @property
    def validation_size(self) -> int:
        return int(self.buffer_size * self.val_fraction)

    @property
    def exact_size(self) -> int:
        """Without a separate exact buffer, exact values use the train buffer."""
        return self.train_size if self.exact_buffer_size is None else self.exact_buffer_size

    @property
    def batches(self) -> int:
        return self.train_size // self.batch_size

    @property
    def patience_epochs(self) -> int:
        return self.epochs if self.patience is None else self.patience

    @property
    def checkpoint_epochs(self) -> int:
        return max(1, self.epochs // 10)


@dataclass(frozen=True, slots=True)
class CharacteristicRunConfig:
    agent: Path
    kind: Kind = "behaviour"
    policy: Policy = "greedy"
    hidden_widths: tuple[int, ...] = (64, 64)
    train: ExplanationTrainConfig = field(default_factory=ExplanationTrainConfig)
    seed: int = 1
    chunk_epochs: int = 1
    output: Path = Path("runs")
    validate_only: bool = False
    compile_only: bool = False

    def __post_init__(self) -> None:
        _run_options(self.seed, self.chunk_epochs, self.validate_only, self.compile_only)
        CharacteristicConfig(self.kind, self.policy, self.hidden_widths)

    @property
    def characteristic(self) -> CharacteristicConfig:
        return CharacteristicConfig(self.kind, self.policy, self.hidden_widths)


@dataclass(frozen=True, slots=True)
class ShapleyRunConfig:
    """Explains a completed characteristic run; kind and policy come from that run."""

    agent: Path
    characteristic: Path
    with_exact_char: bool = True
    buffers: BufferSource = "fresh"
    hidden_widths: tuple[int, ...] = (64, 64)
    train: ExplanationTrainConfig = field(default_factory=ExplanationTrainConfig)
    seed: int = 1
    chunk_epochs: int = 1
    output: Path = Path("runs")
    validate_only: bool = False
    compile_only: bool = False

    def __post_init__(self) -> None:
        _run_options(self.seed, self.chunk_epochs, self.validate_only, self.compile_only)
        if self.buffers not in ("fresh", "characteristic"):
            raise ValueError("buffers must be fresh or characteristic")
        if any(width < 1 for width in self.hidden_widths):
            raise ValueError("hidden widths must be positive")


type RunConfig = CharacteristicRunConfig | ShapleyRunConfig


@struct.dataclass
class Buffers:
    """int32 observations recorded before each environment step."""

    train: Array
    validation: Array
    exact: Array


@struct.dataclass
class ExplanationState[Model]:
    model: Model
    best: Model
    optimizer: nnx.Optimizer[Model]
    buffers: Buffers
    epoch: Array
    best_loss: Array
    best_epoch: Array
    stopped: Array
    phase: Array
    key: Array


@struct.dataclass
class EpochMetrics:
    epoch: Array
    step: Array
    train_loss: Array
    validation_loss: Array
    exact_loss: Array
    best_epoch: Array
    active: Array


@struct.dataclass
class FrozenAgent:
    """The explained agent: Q-network and the epsilon its explore policy uses."""

    model: QNetwork
    epsilon: Array


class Draw(NamedTuple):
    """One batch's sampled buffer rows and coalition(s)."""

    rows: Array
    coalitions: Array


class Learner[Model](NamedTuple):
    """Model-specific steps closed over precomputed targets and frozen models.

    Sampling is separate from learning: `train_draw(key, epoch, batch)` and
    `validation_draw(key, epoch)` default to the source's random draws, and
    parity tests replay recorded upstream draws through them instead.
    """

    train_batch: Callable[[Model, nnx.Optimizer[Model], Buffers, Draw], Array]
    validation_loss: Callable[[Model, Buffers, Draw], Array]
    exact_loss: Callable[[Model], Array]
    train_draw: Callable[[Array, Array, Array], Draw]
    validation_draw: Callable[[Array, Array], Draw]


class Runner(Protocol):
    def __call__(self, *args: object) -> ExplanationState[CharacteristicModel] | ExplanationState[ShapleyModel]: ...


type CLIConfig = (
    Annotated[CharacteristicRunConfig, tyro.conf.subcommand(name="characteristic")]
    | Annotated[ShapleyRunConfig, tyro.conf.subcommand(name="shapley")]
    | Annotated[ResumeConfig, tyro.conf.subcommand(name="resume")]
)


def main(args: list[str] | None = None) -> ExplanationSummary | None:
    """Validate, restore, compile, execute once and finalize host artifacts."""
    from jax_shapley.explanation_io import (
        CallbackRouter,
        ExplanationStore,
        ExplanationWriter,
        create_run,
        describe_options,
        load_run_settings,
        prepare_context,
        restore_state,
        summarize,
        write_summary,
    )

    options = parse_args(args)
    resume = options if isinstance(options, ResumeConfig) else None
    if resume is not None:
        config = load_run_settings(resume.run / "config.json").config
    else:
        assert not isinstance(options, ResumeConfig)
        config = options
    if config.validate_only:
        print(describe_options(config), end="")
        return None
    context = prepare_context(config)
    restored: ExplanationState[CharacteristicModel] | ExplanationState[ShapleyModel] | None = None
    sequence: int | None = None
    metadata = None
    if resume is not None:
        saved = load_run_settings(resume.run / "config.json")
        if saved.agent_sha256 != context.agent_sha256 or saved.characteristic_sha256 != context.characteristic_sha256:
            raise ValueError("explained agent or characteristic changed since the run was created")
        store = ExplanationStore(resume.run, read_only=True)
        try:
            restored, metadata = restore_state(store, template_state(context, config), config, context)
            sequence = store.sequence
        finally:
            store.close()
        if resume.validate_only:
            print(context.settings, end="")
            return None
    router = CallbackRouter()
    runner = nnx.jit(_bind(context, config, router))
    key = jax.random.key(config.seed)
    abstract = nnx.eval_shape(lambda state: state, restored) if restored is not None else None
    arguments = (
        jax.ShapeDtypeStruct((), key.dtype),
        context.agent,
        context.characteristic,
        context.inherited,
        abstract,
    )
    compiled = cast(Runner, runner.trace(*arguments).lower().compile())
    if config.compile_only or (resume is not None and resume.compile_only):
        print(f"Compiled {context.environment} {_stage(config)} on {jax.default_backend()}; no run artifacts created.")
        return None
    folder = resume.run.resolve() if resume is not None else create_run(config, context)
    try:
        with ExplanationWriter(folder, context.settings, metadata=metadata, checkpoint_sequence=sequence) as sink:
            router.sink = sink
            result = compiled(key, context.agent, context.characteristic, context.inherited, restored)
            jax.block_until_ready(result.epoch)
            jax.effects_barrier()
            sink.store.wait()
            summary = summarize(folder, config, result, sink.elapsed())
            write_summary(folder, summary)
    finally:
        router.sink = None
    print(f"Trained {context.environment} {_stage(config)}: {summary.epochs} epochs. Artifacts: {folder}")
    return summary


def run_characteristic[State, Params](
    key: Array,
    agent: FrozenAgent,
    characteristic: CharacteristicModel | None,
    inherited: Buffers | None,
    state: ExplanationState[CharacteristicModel] | None = None,
    *,
    environment_factory: Callable[[], EnvironmentProtocol[State, Params]],
    config: CharacteristicRunConfig,
    max_unique: int,
    sink: ExplanationSink,
) -> ExplanationState[CharacteristicModel]:
    """Collect buffers (fresh runs), then train the characteristic to completion."""
    del characteristic, inherited
    env = EpisodeStatisticsWrapper(environment_factory())
    explained = config.characteristic
    training = config.train
    features, actions = _dimensions(env)
    if state is None:
        key, buffers = collect_buffers(env, agent, explained.policy, training, key, None)
        model_key, key = jax.random.split(key)
        model = CharacteristicModel(characteristic_network(features, actions, explained), rngs=nnx.Rngs(model_key))
        state = initialize_state(model, buffers, training, key)
    learner = characteristic_learner(state.buffers, agent, explained, training, features, max_unique)
    return train(state, learner, training, config.chunk_epochs, sink)


def run_shapley[State, Params](
    key: Array,
    agent: FrozenAgent,
    characteristic: CharacteristicModel | None,
    inherited: Buffers | None,
    state: ExplanationState[ShapleyModel] | None = None,
    *,
    environment_factory: Callable[[], EnvironmentProtocol[State, Params]],
    config: ShapleyRunConfig,
    explained: CharacteristicConfig,
    max_unique: int,
    sink: ExplanationSink,
) -> ExplanationState[ShapleyModel]:
    """Explain a frozen characteristic; exact values always use its exact buffer."""
    if characteristic is None or inherited is None:
        raise ValueError("Shapley runs need the trained characteristic and its buffers")
    env = EpisodeStatisticsWrapper(environment_factory())
    training = config.train
    features, actions = _dimensions(env)
    layout = ShapleyLayout(features, actions if explained.kind == "behaviour" else 1)
    if state is None:
        if config.buffers == "characteristic":
            buffers = inherited
        else:
            key, collected = collect_buffers(env, agent, explained.policy, training, key, inherited.exact)
            buffers = collected
        model_key, key = jax.random.split(key)
        model = ShapleyModel(shapley_network(layout, config.hidden_widths), layout, rngs=nnx.Rngs(model_key))
        state = initialize_state(model, buffers, training, key)
    learner = shapley_learner(
        state.buffers, agent, MaskedCharacteristic(characteristic), explained, config, features, max_unique
    )
    return train(state, learner, training, config.chunk_epochs, sink)


def characteristic_learner(
    buffers: Buffers,
    agent: FrozenAgent,
    explained: CharacteristicConfig,
    training: ExplanationTrainConfig,
    features: int,
    max_unique: int,
) -> Learner[CharacteristicModel]:
    """Targets are the frozen agent's explained outputs; exact values use the exact buffer."""

    def targets(observations: Array) -> Array:
        return explained_values(agent.model(observations).q_values, explained, agent.epsilon)

    train_targets, validation_targets = targets(buffers.train), targets(buffers.validation)
    exact = (
        exact_characteristic(buffers.exact, targets(buffers.exact), all_coalitions(features), max_unique=max_unique)
        if training.exact_loss
        else None
    )

    def train_draw(key: Array, epoch: Array, batch: Array) -> Draw:
        del epoch, batch
        index_key, coalition_key = jax.random.split(key)
        rows = jax.random.randint(index_key, (training.batch_size,), 0, training.train_size)
        return Draw(rows, uniform_coalitions(coalition_key, training.batch_size, features))

    def validation_draw(key: Array, epoch: Array) -> Draw:
        del epoch
        index_key, coalition_key = jax.random.split(key)
        # Source validation resamples the whole validation size with replacement.
        rows = jax.random.randint(index_key, (training.validation_size,), 0, training.validation_size)
        return Draw(rows, uniform_coalitions(coalition_key, training.validation_size, features))

    def train_batch(
        model: CharacteristicModel, optimizer: nnx.Optimizer[CharacteristicModel], buffers: Buffers, draw: Draw
    ) -> Array:
        rows = draw.rows
        return update_characteristic(model, optimizer, buffers.train[rows], train_targets[rows], draw.coalitions).loss

    def validation_loss(model: CharacteristicModel, buffers: Buffers, draw: Draw) -> Array:
        rows = draw.rows
        return characteristic_loss(model, buffers.validation[rows], validation_targets[rows], draw.coalitions)[0]

    def exact_loss(model: CharacteristicModel) -> Array:
        assert exact is not None
        return exact_characteristic_loss(model, exact)

    return Learner(train_batch, validation_loss, exact_loss, train_draw, validation_draw)


def shapley_learner(
    buffers: Buffers,
    agent: FrozenAgent,
    characteristic: CharacteristicFunction,
    explained: CharacteristicConfig,
    config: ShapleyRunConfig,
    features: int,
    max_unique: int,
) -> Learner[ShapleyModel]:
    """One Shapley-kernel coalition per batch; exact values use the characteristic's exact buffer."""
    training = config.train
    exact_values: Array | None = None
    exact: ExactCharacteristic | None = None
    if training.exact_loss:
        observations = buffers.exact
        targets = explained_values(agent.model(observations).q_values, explained, agent.epsilon)
        coalitions = all_coalitions(features)
        exact = exact_characteristic(observations, targets, coalitions, max_unique=max_unique)
        # with_exact_char only selects the ground truth; training always uses the learnt characteristic.
        values = (
            exact.at_states
            if config.with_exact_char
            else characteristic_table(characteristic, exact.states, coalitions)
        )
        exact_values = exact_shapley(values)

    def train_draw(key: Array, epoch: Array, batch: Array) -> Draw:
        del epoch, batch
        index_key, coalition_key = jax.random.split(key)
        rows = jax.random.randint(index_key, (training.batch_size,), 0, training.train_size)
        return Draw(rows, kernel_coalitions(coalition_key, 1, features)[0])

    def validation_draw(key: Array, epoch: Array) -> Draw:
        del epoch
        index_key, coalition_key = jax.random.split(key)
        rows = jax.random.randint(index_key, (training.validation_size,), 0, training.validation_size)
        return Draw(rows, kernel_coalitions(coalition_key, 1, features)[0])

    def train_batch(model: ShapleyModel, optimizer: nnx.Optimizer[ShapleyModel], buffers: Buffers, draw: Draw) -> Array:
        return update_shapley(model, optimizer, characteristic, buffers.train[draw.rows], draw.coalitions).loss

    def validation_loss(model: ShapleyModel, buffers: Buffers, draw: Draw) -> Array:
        return shapley_loss(model, characteristic, buffers.validation[draw.rows], draw.coalitions)[0]

    def exact_loss(model: ShapleyModel) -> Array:
        assert exact is not None and exact_values is not None
        return exact_shapley_loss(model, characteristic, exact.states, exact.state_valid, exact_values)

    return Learner(train_batch, validation_loss, exact_loss, train_draw, validation_draw)


def train[Model: nnx.Module](
    state: ExplanationState[Model],
    learner: Learner[Model],
    config: ExplanationTrainConfig,
    chunk_epochs: int,
    sink: ExplanationSink,
) -> ExplanationState[Model]:
    """Whole epochs in compiled blocks; metrics flush per block, checkpoints every tenth of the epochs."""

    def running(state: ExplanationState[Model]) -> Array:
        return (state.phase == ExplanationPhase.TRAINING) & (state.epoch < config.epochs) & ~state.stopped

    def block(state: ExplanationState[Model]) -> ExplanationState[Model]:
        start = state.epoch

        def step(state: ExplanationState[Model], index: Array) -> tuple[ExplanationState[Model], EpochMetrics]:
            del index
            result = nnx.cond(
                running(state),
                lambda value: epoch_step(value, learner, config),
                lambda value: (value, _inactive_metrics()),
                state,
            )
            return cast(tuple[ExplanationState[Model], EpochMetrics], result)  # NNX cond keeps the carry topology.

        state, metrics = nnx.scan(step, in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))(
            state, jnp.arange(chunk_epochs, dtype=jnp.int32)
        )
        _flush(metrics, sink)
        due = (state.epoch // config.checkpoint_epochs > start // config.checkpoint_epochs) & running(state)
        jax.lax.cond(due, lambda: _emit_checkpoint(state, sink), lambda: jnp.asarray(0, jnp.int32))
        return state

    trained = nnx.while_loop(running, block, state)
    result = nnx.cond(
        trained.phase == ExplanationPhase.TRAINING,
        lambda value: finalize(value, sink),
        lambda value: value,
        trained,
    )
    return cast(ExplanationState[Model], result)  # NNX cond keeps the carry topology.


def epoch_step[Model: nnx.Module](
    state: ExplanationState[Model], learner: Learner[Model], config: ExplanationTrainConfig
) -> tuple[ExplanationState[Model], EpochMetrics]:
    """One source epoch: shuffled-with-replacement batches, validation, exact loss, early stopping."""
    key, batch_key, validation_key = jax.random.split(state.key, 3)

    def batch(
        carry: tuple[Model, nnx.Optimizer[Model]], inputs: tuple[Array, Array]
    ) -> tuple[tuple[Model, nnx.Optimizer[Model]], Array]:
        model, optimizer = carry
        key, index = inputs
        draw = learner.train_draw(key, state.epoch, index)
        return (model, optimizer), learner.train_batch(model, optimizer, state.buffers, draw)

    batches = (jax.random.split(batch_key, config.batches), jnp.arange(config.batches, dtype=jnp.int32))
    _, losses = nnx.scan(batch, in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))(
        (state.model, state.optimizer), batches
    )
    validation = learner.validation_loss(
        state.model, state.buffers, learner.validation_draw(validation_key, state.epoch)
    )
    exact = learner.exact_loss(state.model) if config.exact_loss else jnp.asarray(nan, jnp.float32)
    epoch = state.epoch + 1
    improved = validation < state.best_loss
    nnx.cond(improved, _copy_parameters, lambda source, destination: None, state.model, state.best)
    best_loss = jnp.where(improved, validation, state.best_loss)
    best_epoch = jnp.where(improved, epoch, state.best_epoch)
    stopped = epoch - best_epoch >= config.patience_epochs
    if config.exact_condition is not None:
        stopped = stopped | (exact < config.exact_condition)
    # Source logging step: the epoch, or (epoch + 1) * batches under track_batch.
    step = (epoch + 1) * config.batches if config.track_batch else epoch
    next_state = ExplanationState(
        state.model,
        state.best,
        state.optimizer,
        state.buffers,
        epoch,
        best_loss,
        best_epoch,
        stopped,
        state.phase,
        key,
    )
    metrics = EpochMetrics(epoch, step, jnp.mean(losses), validation, exact, best_epoch, jnp.asarray(True))
    return next_state, metrics


def finalize[Model: nnx.Module](state: ExplanationState[Model], sink: ExplanationSink) -> ExplanationState[Model]:
    """Restore the best-validation model, mark completion and commit the final checkpoint."""
    _copy_parameters(state.best, state.model)
    complete = ExplanationState(
        state.model,
        state.best,
        state.optimizer,
        state.buffers,
        state.epoch,
        state.best_loss,
        state.best_epoch,
        state.stopped,
        jnp.asarray(ExplanationPhase.COMPLETE, jnp.int32),
        state.key,
    )
    _emit_checkpoint(complete, sink)
    return complete


def collect_buffers[State, Params](
    env: EpisodeStatisticsWrapper[State, Params],
    agent: FrozenAgent,
    policy: Policy,
    config: ExplanationTrainConfig,
    key: Array,
    exact: Array | None,
) -> tuple[Array, Buffers]:
    """Source fill_buffer for train, validation and (optionally) a separate exact buffer.

    Each buffer starts from a fresh reset; without a separate exact buffer the
    train buffer is used, unless `exact` is supplied by an earlier run.
    """
    epsilon = agent.epsilon if policy == "explore" else jnp.asarray(0.0, jnp.float32)
    keys: Array = jax.random.split(key, 4)
    key, train_key, validation_key, exact_key = keys[0], keys[1], keys[2], keys[3]
    train_rows = collect_observations(env, agent.model, epsilon, train_key, config.train_size, config.num_envs)
    validation = collect_observations(
        env, agent.model, epsilon, validation_key, config.validation_size, config.num_envs
    )
    if exact is None:
        exact = (
            train_rows
            if config.exact_buffer_size is None
            else collect_observations(env, agent.model, epsilon, exact_key, config.exact_buffer_size, config.num_envs)
        )
    return key, Buffers(train_rows, validation, exact)


def collect_observations[State, Params](
    env: EpisodeStatisticsWrapper[State, Params],
    model: QNetwork,
    epsilon: Array,
    key: Array,
    rows: int,
    num_envs: int,
) -> Array:
    """Observations before each vector step, step-major then environment, with Gymnax auto-reset."""
    params = env.default_params
    keys: Array = jax.random.split(key, num_envs + 1)
    observations, env_state = jax.vmap(env.reset, in_axes=(0, None))(keys[1:], params)

    def step(
        carry: tuple[Array, EpisodeStatisticsState[State], Array], index: Array
    ) -> tuple[tuple[Array, EpisodeStatisticsState[State], Array], Array]:
        del index
        current, state, key = carry
        actions, key = choose_action(model, current, epsilon, key)
        step_keys: Array = jax.random.split(key, num_envs + 1)
        transition = jax.vmap(partial(environment_step, env), in_axes=(0, 0, 0, None))(
            step_keys[1:], state, actions.actions, params
        )
        return (transition.observations, transition.state, step_keys[0]), current

    _, recorded = jax.lax.scan(step, (observations, env_state, keys[0]), jnp.arange(rows // num_envs))
    return recorded.reshape(rows, recorded.shape[-1]).astype(jnp.int32)


def initialize_state[Model: nnx.Module](
    model: Model, buffers: Buffers, config: ExplanationTrainConfig, key: Array
) -> ExplanationState[Model]:
    return ExplanationState(
        model,
        nnx.clone(model),
        make_adam(model, config.learning_rate, config.adam),
        buffers,
        jnp.asarray(0, jnp.int32),
        jnp.asarray(jnp.inf, jnp.float32),
        jnp.asarray(0, jnp.int32),
        jnp.asarray(False),
        jnp.asarray(ExplanationPhase.TRAINING, jnp.int32),
        key,
    )


def template_state(
    context: RunContext, config: RunConfig
) -> ExplanationState[CharacteristicModel] | ExplanationState[ShapleyModel]:
    """Host template with the run's static shapes, used only to validate and place restored arrays."""
    training = config.train
    features = context.features
    buffers = Buffers(
        jnp.zeros((training.train_size, features), jnp.int32),
        jnp.zeros((training.validation_size, features), jnp.int32),
        jnp.zeros((context.exact_rows, features), jnp.int32),
    )
    key = jax.random.key(0)
    if isinstance(config, CharacteristicRunConfig):
        network = characteristic_network(features, context.actions, config.characteristic)
        return initialize_state(CharacteristicModel(network, rngs=nnx.Rngs(0)), buffers, training, key)
    layout = ShapleyLayout(features, context.actions if context.explained.kind == "behaviour" else 1)
    model = ShapleyModel(shapley_network(layout, config.hidden_widths), layout, rngs=nnx.Rngs(0))
    return initialize_state(model, buffers, training, key)


def make_snapshot[Model: nnx.Module](state: ExplanationState[Model]) -> ExplanationSnapshot:
    """Extract numerical state only; NNX graph definitions stay outside callbacks."""
    from jax_shapley.explanation_io import ExplanationSnapshot

    return ExplanationSnapshot(
        _array_leaves(nnx.state(state.model)),
        _array_leaves(nnx.state(state.best)),
        _array_leaves(nnx.state(state.optimizer)),
        (state.buffers.train, state.buffers.validation, state.buffers.exact),
        (state.best_loss, state.best_epoch, state.stopped, jax.random.key_data(state.key)),
        state.epoch,
        state.phase,
        str(jax.random.key_impl(state.key)),
    )


def parse_args(args: list[str] | None = None) -> CLIConfig:
    return tyro.cli(CLIConfig, args=args)


def _bind(context: RunContext, config: RunConfig, sink: ExplanationSink) -> Callable[..., object]:
    """Bind static configuration, preserving each environment's state/parameter types."""
    env = context.env_config
    if isinstance(config, CharacteristicRunConfig):
        options = {"config": config, "max_unique": context.max_unique, "sink": sink}
        if isinstance(env, GridworldConfig):
            return partial(run_characteristic, environment_factory=Gridworld, **options)
        if isinstance(env, Mastermind222Config):
            return partial(run_characteristic, environment_factory=Mastermind222, **options)
        return partial(run_characteristic, environment_factory=env.make, **options)
    options = {"config": config, "explained": context.explained, "max_unique": context.max_unique, "sink": sink}
    if isinstance(env, GridworldConfig):
        return partial(run_shapley, environment_factory=Gridworld, **options)
    if isinstance(env, Mastermind222Config):
        return partial(run_shapley, environment_factory=Mastermind222, **options)
    return partial(run_shapley, environment_factory=env.make, **options)


def _flush(metrics: EpochMetrics, sink: ExplanationSink) -> None:
    jax.lax.cond(
        jnp.any(metrics.active),
        lambda: io_callback(sink.epochs, jax.ShapeDtypeStruct((), jnp.int32), _array_leaves(metrics), ordered=True),
        lambda: jnp.asarray(0, jnp.int32),
    )


def _emit_checkpoint[Model: nnx.Module](state: ExplanationState[Model], sink: ExplanationSink) -> Array:
    return cast(
        Array, io_callback(sink.checkpoint, jax.ShapeDtypeStruct((), jnp.int32), make_snapshot(state), ordered=True)
    )  # Callback result has the declared scalar int32 specification.


def _copy_parameters(source: nnx.Module, destination: nnx.Module) -> None:
    nnx.update(destination, nnx.state(source, nnx.Param))


def _inactive_metrics() -> EpochMetrics:
    zero, nothing = jnp.asarray(0, jnp.int32), jnp.asarray(0.0, jnp.float32)
    return EpochMetrics(zero, zero, nothing, nothing, nothing, zero, jnp.asarray(False))


def _dimensions[State, Params](env: EpisodeStatisticsWrapper[State, Params]) -> tuple[int, int]:
    shape = env.observation_space(env.default_params).shape
    if len(shape) != 1:
        raise ValueError("explanations require flat observations")
    return int(shape[0]), env.num_actions


def _stage(config: RunConfig) -> str:
    return "characteristic" if isinstance(config, CharacteristicRunConfig) else "Shapley model"


def _array_leaves(tree: object) -> tuple[Array, ...]:
    return tuple(cast(list[Array], jax.tree.leaves(tree)))  # Only numerical structs and NNX state cross this boundary.


def _run_options(seed: int, chunk_epochs: int, validate_only: bool, compile_only: bool) -> None:
    _positive("chunk_epochs", chunk_epochs)
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be in [0, 2**32 - 1]")
    if validate_only and compile_only:
        raise ValueError("validate_only and compile_only are mutually exclusive")


def _positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be positive")


if __name__ == "__main__":
    # Run the canonical module, so host I/O and this entry point share one set of config classes.
    from jax_shapley import train_explanations

    train_explanations.main()
