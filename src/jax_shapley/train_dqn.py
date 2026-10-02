"""FastSVERL DQN: configuration, compiled training, evaluation and CLI.

Shell recipes provide experiment values. Numerical state and learning live here;
replay.py owns buffer storage, networks.py owns the models, and run_io.py owns
host callbacks and artifacts.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from functools import partial
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NamedTuple, Protocol, cast

import jax
import jax.numpy as jnp
import optax
import tyro
from flax import nnx, struct
from jax import Array
from jax.experimental import io_callback

from jax_shapley.envs import (
    Gridworld,
    GridworldState,
    Hypercube,
    HypercubeState,
    Mastermind222,
    Mastermind222State,
)
from jax_shapley.networks import MLPConfig, ModelInput, QNetwork, dqn_config
from jax_shapley.replay import ReplayState, TransitionBatch, append, empty_replay, sample
from jax_shapley.wrappers import EnvironmentProtocol, EpisodeStatisticsState, EpisodeStatisticsWrapper

if TYPE_CHECKING:
    from jax_shapley.run_io import CheckpointMetadata, EventSink, Snapshot


@dataclass(frozen=True, slots=True)
class AdamConfig:
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        for name, value in (("beta1", self.beta1), ("beta2", self.beta2)):
            _probability(name, value)
            if value == 1:
                raise ValueError(f"{name} must be less than 1")
        _finite("epsilon", self.epsilon)
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")


@dataclass(frozen=True, slots=True)
class DQNTrainConfig:
    """Gridworld baseline; timestep/frequency units count all environment transitions."""

    num_envs: int = 1
    total_timesteps: int = 50_000
    learning_rate: float = 2.5e-4
    buffer_size: int = 50_000
    gamma: float = 1.0
    tau: float = 0.01
    target_network_frequency: int = 1
    batch_size: int = 64
    start_e: float = 1.0
    end_e: float = 0.01
    exploration_fraction: float = 0.5
    learning_starts: int = 10_000
    train_frequency: int = 10
    eval_episodes: int = 0
    adam: AdamConfig = field(default_factory=AdamConfig)

    def __post_init__(self) -> None:
        for name, value in (
            ("num_envs", self.num_envs),
            ("total_timesteps", self.total_timesteps),
            ("buffer_size", self.buffer_size),
            ("batch_size", self.batch_size),
            ("train_frequency", self.train_frequency),
            ("target_network_frequency", self.target_network_frequency),
        ):
            _positive(name, value)
        if self.vector_steps < 1:
            raise ValueError("total_timesteps must include at least one vector step")
        for name, value in (("learning_starts", self.learning_starts), ("eval_episodes", self.eval_episodes)):
            if not 0 <= value <= 2**31 - 1:
                raise ValueError(f"{name} must be a nonnegative int32 value")
        for name, value in (("gamma", self.gamma), ("tau", self.tau), ("start_e", self.start_e), ("end_e", self.end_e)):
            _probability(name, value)
        for name, value in (("learning_rate", self.learning_rate), ("exploration_fraction", self.exploration_fraction)):
            _finite(name, value)
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def vector_steps(self) -> int:
        return self.total_timesteps // self.num_envs


@dataclass(frozen=True, slots=True)
class GridworldConfig:
    def make(self) -> Gridworld:
        return Gridworld()


@dataclass(frozen=True, slots=True)
class Mastermind222Config:
    def make(self) -> Mastermind222:
        return Mastermind222()


@dataclass(frozen=True, slots=True)
class HypercubeConfig:
    side_length: int = 2
    dimension: int = 2

    def __post_init__(self) -> None:
        if self.side_length not in range(2, 7) or self.dimension not in range(2, 6):
            raise ValueError("Hypercube requires side_length 2..6 and dimension 2..5")

    def make(self) -> Hypercube:
        return Hypercube(self.side_length, self.dimension)


class ExecutionPhase(IntEnum):
    TRAINING = 0
    EVALUATION = 1
    COMPLETE = 2


@dataclass(frozen=True, slots=True)
class ResumeConfig:
    run: Path
    validate_only: bool = False
    compile_only: bool = False

    def __post_init__(self) -> None:
        if self.validate_only and self.compile_only:
            raise ValueError("validate_only and compile_only are mutually exclusive")


type EnvironmentConfig = GridworldConfig | Mastermind222Config | HypercubeConfig


@dataclass(frozen=True, slots=True)
class DQNRunConfig[EnvConfig: EnvironmentConfig]:
    env: EnvConfig
    train: DQNTrainConfig = field(default_factory=DQNTrainConfig)
    hidden_widths: tuple[int, ...] = (64, 64)
    seed: int = 1
    chunk_steps: int = 64
    output: Path = Path("runs")
    validate_only: bool = False
    compile_only: bool = False

    def __post_init__(self) -> None:
        _positive("chunk_steps", self.chunk_steps)
        for width in self.hidden_widths:
            _positive("hidden_width", width)
        if not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be in [0, 2**32 - 1]")
        if self.validate_only and self.compile_only:
            raise ValueError("validate_only and compile_only are mutually exclusive")

    @property
    def network(self) -> MLPConfig:
        return network_config(self.env, self.hidden_widths)


@struct.dataclass
class RunState[EnvState]:
    model: QNetwork
    target: QNetwork
    optimizer: nnx.Optimizer[QNetwork]
    replay: ReplayState
    observations: Array
    env_state: EpisodeStatisticsState[EnvState]
    vector_step: Array
    updates: Array
    target_updates: Array
    key: Array


@struct.dataclass
class UpdateState:
    model: QNetwork
    target: QNetwork
    optimizer: nnx.Optimizer[QNetwork]
    batch: TransitionBatch


@struct.dataclass
class StepMetrics:
    global_step: Array
    epsilon: Array
    loss: Array
    updated: Array
    target_updated: Array
    episode_return: Array
    episode_length: Array
    completed: Array
    active: Array


@struct.dataclass
class WholeRunState[EnvState]:
    run: RunState[EnvState]
    pending: StepMetrics
    pending_count: Array
    phase: Array
    evaluation_episodes: Array


@struct.dataclass
class EvaluationState[EnvState]:
    observations: Array
    env_state: EpisodeStatisticsState[EnvState]
    returns: Array
    count: Array
    key: Array


@struct.dataclass
class EvaluationResult:
    returns: Array
    count: Array
    key: Array


@struct.dataclass
class EnvironmentStep[EnvState]:
    observations: Array
    state: EpisodeStatisticsState[EnvState]
    raw_observations: Array
    rewards: Array
    terminated: Array
    truncated: Array
    episode_return: Array
    episode_length: Array
    completed: Array


@struct.dataclass
class LossMetrics:
    loss: Array
    target_mean: Array
    value_mean: Array


@dataclass(frozen=True, slots=True)
class RunSummary:
    folder: str
    vector_steps: int
    environment_steps: int
    updates: int
    target_updates: int
    evaluation_episodes: int
    training_seconds: float
    backend: str


type CLIConfig = (
    Annotated[
        DQNRunConfig[GridworldConfig], tyro.conf.subcommand(name="gridworld", default=DQNRunConfig(GridworldConfig()))
    ]
    | Annotated[
        DQNRunConfig[Mastermind222Config],
        tyro.conf.subcommand(name="mastermind222", default=DQNRunConfig(Mastermind222Config())),
    ]
    | Annotated[
        DQNRunConfig[HypercubeConfig], tyro.conf.subcommand(name="hypercube", default=DQNRunConfig(HypercubeConfig()))
    ]
    | Annotated[ResumeConfig, tyro.conf.subcommand(name="resume")]
)


type SupportedRunState = RunState[GridworldState] | RunState[Mastermind222State] | RunState[HypercubeState]
type SupportedRunResult = (
    WholeRunState[GridworldState] | WholeRunState[Mastermind222State] | WholeRunState[HypercubeState]
)


class Runner(Protocol):
    def __call__(
        self,
        key: Array,
        state: SupportedRunState | None,
        phase: Array | None,
        evaluation_episodes: Array | None,
    ) -> SupportedRunResult: ...


class ActionOutput(NamedTuple):
    actions: Array
    log_probabilities: Array


def main(args: list[str] | None = None) -> RunSummary | None:
    """Validate, restore, compile, execute once and finalize host artifacts."""
    from jax_shapley.run_io import (
        CallbackRouter,
        CheckpointStore,
        RunWriter,
        create_run,
        describe,
        load_settings,
        restore_run,
        verify_log,
        write_summary,
    )

    options = parse_args(args)
    resume = options if isinstance(options, ResumeConfig) else None
    config = load_settings(resume.run / "config.json") if resume is not None else options
    assert isinstance(config, DQNRunConfig)
    if config.validate_only:
        print(describe(config), end="")
        return None
    factory = config.env.make
    metadata: CheckpointMetadata | None = None
    checkpoint_sequence: int | None = None
    restored: SupportedRunState | None = None
    phase: Array | None = None
    episodes: Array | None = None
    if resume is not None:
        store = CheckpointStore(resume.run, read_only=True)
        try:
            # This host template validates destination graph/shapes; run() never
            # reinitializes restored state during numerical execution.
            env = EpisodeStatisticsWrapper(factory())
            template = initialize_run(
                env, env.default_params, config.network, config.train, jax.random.key(config.seed)
            )
            restored, metadata = restore_run(store, template, config)
            checkpoint_sequence = store.sequence
            verify_log(resume.run / "metrics.csv", metadata)
        finally:
            store.close()
        if resume.validate_only:
            print(describe(config), end="")
            return None
        phase = jnp.asarray(metadata.phase, jnp.int32)
        episodes = jnp.asarray(metadata.evaluation_episodes, jnp.int32)
    router = CallbackRouter()
    # Preserve each environment's state/parameter types when binding the transform.
    if isinstance(config.env, GridworldConfig):
        runner = nnx.jit(
            partial(
                run,
                environment_factory=Gridworld,
                config=config.train,
                hidden_widths=config.hidden_widths,
                chunk_steps=config.chunk_steps,
                sink=router,
            )
        )
    elif isinstance(config.env, Mastermind222Config):
        runner = nnx.jit(
            partial(
                run,
                environment_factory=Mastermind222,
                config=config.train,
                hidden_widths=config.hidden_widths,
                chunk_steps=config.chunk_steps,
                sink=router,
            )
        )
    else:
        runner = nnx.jit(
            partial(
                run,
                environment_factory=config.env.make,
                config=config.train,
                hidden_widths=config.hidden_widths,
                chunk_steps=config.chunk_steps,
                sink=router,
            )
        )
    key = jax.random.key(config.seed)
    key_spec = jax.ShapeDtypeStruct((), key.dtype)
    # Retain restored optimizer static functions. None selects the fresh graph;
    # arrays select the resume graph, with all phases compiled before execution.
    abstract = nnx.eval_shape(lambda state: state, restored) if restored is not None else None
    scalar = jax.ShapeDtypeStruct((), jnp.int32) if restored is not None else None
    compiled = cast(Runner, runner.trace(key_spec, abstract, scalar, scalar).lower().compile())
    environment_name = factory().name  # Host metadata only; run() owns numerical initialization.
    if config.compile_only or (resume is not None and resume.compile_only):
        print(f"Compiled {environment_name} DQN on {jax.default_backend()}; no run artifacts created.")
        return None
    folder = resume.run.resolve() if resume is not None else create_run(config, environment_name)
    try:
        with RunWriter(folder, config, metadata=metadata, checkpoint_sequence=checkpoint_sequence) as sink:
            router.sink = sink
            result = compiled(key, restored, phase, episodes)
            jax.block_until_ready(result.run.vector_step)
            jax.effects_barrier()
            sink.store.wait()
            summary = RunSummary(
                str(folder),
                int(result.run.vector_step),
                int(result.run.vector_step) * config.train.num_envs,
                int(result.run.updates),
                int(result.run.target_updates),
                int(result.evaluation_episodes),
                sink.training_seconds,
                jax.default_backend(),
            )
            write_summary(folder, summary)
    finally:
        router.sink = None
    print(
        f"Trained {environment_name}: {summary.environment_steps} transitions, {summary.updates} updates. Artifacts: {folder}"
    )
    return summary


def run[State, Params](
    key: Array,
    state: RunState[State] | None = None,
    phase: Array | None = None,
    evaluation_episodes: Array | None = None,
    *,
    environment_factory: Callable[[], EnvironmentProtocol[State, Params]],
    config: DQNTrainConfig,
    hidden_widths: tuple[int, ...],
    chunk_steps: int,
    sink: EventSink,
) -> WholeRunState[State]:
    """Construct static environment metadata and initialize or resume one compiled run."""
    env = EpisodeStatisticsWrapper(environment_factory())
    params = env.default_params
    if state is None:
        network = _network_for_env(env.env, hidden_widths)
        state = initialize_run(env, params, network, config, key)
        phase = jnp.asarray(ExecutionPhase.TRAINING, jnp.int32)
        evaluation_episodes = jnp.asarray(0, jnp.int32)
    elif phase is None or evaluation_episodes is None:
        raise ValueError("restored state requires saved phase and evaluation counters")
    assert phase is not None and evaluation_episodes is not None
    _, empty = _inactive_step(state, config)
    pending = cast(StepMetrics, jax.tree.map(lambda value: jnp.broadcast_to(value, (chunk_steps, *value.shape)), empty))
    initial = WholeRunState(state, pending, jnp.asarray(0, jnp.int32), phase, evaluation_episodes)

    def block(carry: WholeRunState[State]) -> WholeRunState[State]:
        def step(current: WholeRunState[State], index: Array) -> tuple[WholeRunState[State], Array]:
            del index
            next_state = nnx.cond(
                current.run.vector_step < config.vector_steps,
                lambda value: _whole_training_step(value, env, params, config, sink),
                lambda value: value,
                current,
            )
            return cast(WholeRunState[State], next_state), jnp.asarray(
                0, jnp.int32
            )  # NNX cond retains the carry topology.

        result, _ = nnx.scan(step, in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))(
            carry, jnp.arange(chunk_steps, dtype=jnp.int32)
        )
        return result

    trained = nnx.while_loop(
        lambda carry: (carry.phase == ExecutionPhase.TRAINING) & (carry.run.vector_step < config.vector_steps),
        block,
        initial,
    )
    prepared = nnx.cond(
        trained.phase == ExecutionPhase.TRAINING,
        lambda carry: _finish_training(carry, config, sink),
        lambda carry: carry,
        trained,
    )
    evaluated = nnx.cond(
        prepared.phase == ExecutionPhase.EVALUATION,
        lambda carry: _whole_evaluation(carry, env, params, config, sink),
        lambda carry: carry,
        prepared,
    )
    jax.lax.cond(
        phase != ExecutionPhase.COMPLETE,
        lambda: _emit_checkpoint(evaluated, config, sink),
        lambda: jnp.asarray(0, jnp.int32),
    )
    return cast(WholeRunState[State], evaluated)  # NNX transforms retain the explicit carry topology.


def initialize_run[State, Params](
    env: EpisodeStatisticsWrapper[State, Params], params: Params, network: MLPConfig, config: DQNTrainConfig, key: Array
) -> RunState[State]:
    model, target, optimizer, key = initialize_models(network, config, key)
    keys: Array = jax.random.split(key, config.num_envs + 1)
    observations, state = jax.vmap(env.reset, in_axes=(0, None))(keys[1:], params)
    return RunState(
        model,
        target,
        optimizer,
        empty_replay(config.buffer_size, network.input_features),
        observations,
        state,
        jnp.asarray(0, jnp.int32),
        jnp.asarray(0, jnp.int32),
        jnp.asarray(0, jnp.int32),
        keys[0],
    )


def run_chunk[State, Params](
    state: RunState[State],
    active_steps: Array,
    *,
    env: EpisodeStatisticsWrapper[State, Params],
    params: Params,
    config: DQNTrainConfig,
    chunk_steps: int,
) -> tuple[RunState[State], StepMetrics]:
    """Fixed-size compiled chunk; inactive tail steps neither learn nor consume keys."""

    def step(state: RunState[State], index: Array) -> tuple[RunState[State], StepMetrics]:
        return nnx.cond(
            index < active_steps,
            lambda carry: training_step(carry, env, params, config),
            lambda carry: _inactive_step(carry, config),
            state,
        )

    result = nnx.scan(step, in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))(
        state, jnp.arange(chunk_steps, dtype=jnp.int32)
    )
    return result


def evaluate_dqn[State, Params](
    model: QNetwork,
    key: Array,
    *,
    env: EpisodeStatisticsWrapper[State, Params],
    params: Params,
    num_envs: int,
    episodes: int,
) -> EvaluationResult:
    """Collect whole vector-step episode events, including upstream's final overshoot."""
    if episodes < 1:
        raise ValueError("evaluation episodes must be positive")
    keys: Array = jax.random.split(key, num_envs + 1)
    observations, env_state = jax.vmap(env.reset, in_axes=(0, None))(keys[1:], params)
    initial = EvaluationState(
        observations, env_state, jnp.zeros(episodes + num_envs - 1, jnp.float32), jnp.asarray(0, jnp.int32), keys[0]
    )

    def step(state: EvaluationState[State]) -> EvaluationState[State]:
        transition, key = evaluation_step(model, state.observations, state.env_state, state.key, env=env, params=params)

        def record(index: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
            returns, count = carry
            returns = cast(
                Array,
                jax.lax.cond(
                    transition.completed[index],
                    lambda data: data.at[count].set(transition.episode_return[index]),
                    lambda data: data,
                    returns,
                ),
            )  # JAX conditional preserves this array boundary.
            return returns, count + transition.completed[index].astype(jnp.int32)

        returns, count = jax.lax.fori_loop(0, num_envs, record, (state.returns, state.count))
        return EvaluationState(transition.observations, transition.state, returns, count, key)

    final = jax.lax.while_loop(lambda state: state.count < episodes, step, initial)
    return EvaluationResult(final.returns, final.count, final.key)


def parse_args(args: list[str] | None = None) -> CLIConfig:
    return tyro.cli(CLIConfig, args=args)


def checkpoint_due(vector_step: int | Array, num_envs: int, total_timesteps: int) -> bool | Array:
    return (vector_step * num_envs) % max(1, total_timesteps // 10) < num_envs


def network_config(config: EnvironmentConfig, widths: tuple[int, ...]) -> MLPConfig:
    if isinstance(config, GridworldConfig):
        return _network_for_env(config.make(), widths)
    if isinstance(config, Mastermind222Config):
        return _network_for_env(config.make(), widths)
    return _network_for_env(config.make(), widths)


def _network_for_env[State, Params](env: EnvironmentProtocol[State, Params], widths: tuple[int, ...]) -> MLPConfig:
    shape = env.observation_space(env.default_params).shape
    if len(shape) != 1:
        raise ValueError("DQN requires flat observations")
    return dqn_config(shape[0], env.num_actions, hidden_widths=widths)


def initialize_models(
    config: MLPConfig, training: DQNTrainConfig, key: Array
) -> tuple[QNetwork, QNetwork, nnx.Optimizer[QNetwork], Array]:
    keys: Array = jax.random.split(key)
    model = QNetwork(config, rngs=nnx.Rngs(keys[1]))
    target = nnx.clone(model)
    return model, target, make_optimizer(model, training), keys[0]


def training_step[State, Params](
    state: RunState[State], env: EpisodeStatisticsWrapper[State, Params], params: Params, config: DQNTrainConfig
) -> tuple[RunState[State], StepMetrics]:
    global_step = state.vector_step * config.num_envs
    epsilon = epsilon_at(config, global_step)
    actions, key = choose_action(state.model, state.observations, epsilon, state.key)
    keys: Array = jax.random.split(key, config.num_envs + 1)
    transition = jax.vmap(partial(environment_step, env), in_axes=(0, 0, 0, None))(
        keys[1:], state.env_state, actions.actions, params
    )
    replay = append(
        state.replay,
        TransitionBatch(
            state.observations,
            actions.actions,
            transition.rewards,
            transition.raw_observations,
            transition.terminated,
            actions.log_probabilities,
        ),
    )
    ready = global_step > config.learning_starts
    should_learn = ready & (global_step % config.train_frequency < config.num_envs)
    should_target = ready & (global_step % config.target_network_frequency < config.num_envs)
    # Sampling outside the condition is safe after insertion. Advance its stream
    # only on actual updates, so grouping steps into chunks cannot change samples.
    batch, sampled_key = sample(replay, config.batch_size, keys[0])
    key = jnp.where(should_learn, sampled_key, keys[0])
    learning = UpdateState(state.model, state.target, state.optimizer, batch)
    loss = nnx.cond(
        should_learn,
        lambda value, gamma: update(value.model, value.target, value.optimizer, value.batch, gamma),
        _skip_learn,
        learning,
        config.gamma,
    )
    nnx.cond(should_target, update_target, lambda model, target, tau: None, state.model, state.target, config.tau)
    next_state = RunState(
        state.model,
        state.target,
        state.optimizer,
        replay,
        transition.observations,
        transition.state,
        state.vector_step + 1,
        state.updates + should_learn.astype(jnp.int32),
        state.target_updates + should_target.astype(jnp.int32),
        key,
    )
    metrics = StepMetrics(
        global_step,
        epsilon,
        loss.loss,
        should_learn,
        should_target,
        transition.episode_return,
        transition.episode_length,
        transition.completed,
        jnp.asarray(True),
    )
    return next_state, metrics


def evaluation_step[State, Params](
    model: QNetwork,
    observations: Array,
    state: EpisodeStatisticsState[State],
    key: Array,
    *,
    env: EpisodeStatisticsWrapper[State, Params],
    params: Params,
) -> tuple[EnvironmentStep[State], Array]:
    actions, key = choose_action(model, observations, jnp.asarray(0.0, jnp.float32), key)
    keys: Array = jax.random.split(key, observations.shape[0] + 1)
    transition = jax.vmap(partial(environment_step, env), in_axes=(0, 0, 0, None))(
        keys[1:], state, actions.actions, params
    )
    return transition, keys[0]


def environment_step[State, Params](
    env: EpisodeStatisticsWrapper[State, Params],
    key: Array,
    state: EpisodeStatisticsState[State],
    action: Array,
    params: Params,
) -> EnvironmentStep[State]:
    """Keep Gymnax auto-reset and expose its pre-reset observation with the same step key.

    This narrow adapter calls pure step_env with Gymnax's key_step. The repeated
    raw transition is eligible for compiler common-subexpression elimination.
    """
    keys: Array = jax.random.split(key)
    raw_observations, _, _, _, _ = env.env.step_env(keys[0], state.env_state, action, params)
    observation, next_state, reward, _, info = env.step(key, state, action, params)
    episode = info["episode"]
    return EnvironmentStep(
        observation,
        next_state,
        raw_observations,
        reward,
        info["terminated"],
        info["truncated"],
        episode["return"],
        episode["length"],
        episode["completed"],
    )


def choose_action(model: QNetwork, observations: ModelInput, epsilon: Array, key: Array) -> tuple[ActionOutput, Array]:
    probabilities = exploration_probabilities(model(observations).q_values, epsilon)
    keys: Array = jax.random.split(key)
    # Sampling uses true zero-probability logits; the clamp applies only to reported log probabilities.
    actions: Array = jax.random.categorical(keys[1], jnp.log(probabilities), axis=-1).astype(jnp.int32)
    return ActionOutput(actions, action_log_probabilities(probabilities, actions)), keys[0]


def make_optimizer(model: QNetwork, config: DQNTrainConfig) -> nnx.Optimizer[QNetwork]:
    return make_adam(model, config.learning_rate, config.adam)


def make_adam[Model: nnx.Module](model: Model, learning_rate: float, adam: AdamConfig) -> nnx.Optimizer[Model]:
    """Torch Adam: constant rate, epsilon outside the square root, no root epsilon."""
    return nnx.Optimizer(
        model,
        optax.adam(learning_rate, b1=adam.beta1, b2=adam.beta2, eps=adam.epsilon, eps_root=0.0),
        wrt=nnx.Param,
    )


def update(
    model: QNetwork, target: QNetwork, optimizer: nnx.Optimizer[QNetwork], batch: TransitionBatch, gamma: float
) -> LossMetrics:
    # NNX transformation has a dynamic return signature; assert its known loss/state boundary.
    result = cast(
        tuple[tuple[Array, LossMetrics], nnx.State[str | int, nnx.Param[Array]]],
        nnx.value_and_grad(dqn_loss, has_aux=True)(model, target, batch, gamma),
    )
    (_, metrics), gradients = result
    optimizer.update(model, gradients)
    return metrics


def update_target(model: QNetwork, target: QNetwork, tau: float) -> None:
    for source, destination in zip(model.mlp.layers, target.mlp.layers, strict=True):
        destination.kernel[...] = tau * source.kernel[...] + (1 - tau) * destination.kernel[...]
        destination.bias[...] = tau * source.bias[...] + (1 - tau) * destination.bias[...]


def epsilon_at(config: DQNTrainConfig, global_step: Array) -> Array:
    fraction = global_step.astype(jnp.float32) / (config.exploration_fraction * config.total_timesteps)
    return jnp.maximum(config.start_e + (config.end_e - config.start_e) * fraction, config.end_e).astype(jnp.float32)


def exploration_probabilities(q_values: Array, epsilon: Array) -> Array:
    return epsilon / q_values.shape[-1] + (1 - epsilon) * greedy_probabilities(q_values)


def action_log_probabilities(probabilities: Array, actions: Array) -> Array:
    # Torch Categorical renormalizes probabilities and clamps before converting to logits.
    probabilities = probabilities / jnp.sum(probabilities, axis=-1, keepdims=True)
    epsilon = jnp.finfo(jnp.float32).eps
    logits = jnp.log(jnp.clip(probabilities, epsilon, 1 - epsilon))
    return jnp.take_along_axis(logits, actions[..., None], axis=-1)[..., 0]


def dqn_loss(model: QNetwork, target: QNetwork, batch: TransitionBatch, gamma: float) -> tuple[Array, LossMetrics]:
    batch.validate()
    targets = td_targets(target, batch, gamma)
    values = jnp.take_along_axis(model(batch.observations).q_values, batch.actions[:, None], axis=-1)[:, 0]
    loss = jnp.mean(jnp.square(targets - values))
    return loss, LossMetrics(loss, jnp.mean(targets), jnp.mean(values))


def greedy_probabilities(q_values: Array) -> Array:
    ties = q_values == jnp.max(q_values, axis=-1, keepdims=True)
    return ties.astype(jnp.float32) / jnp.sum(ties, axis=-1, keepdims=True)


def td_targets(target: QNetwork, batch: TransitionBatch, gamma: float) -> Array:
    values = jnp.max(target(batch.next_observations).q_values, axis=-1)
    return jax.lax.stop_gradient(batch.rewards + gamma * values * (~batch.terminated).astype(jnp.float32))


def _whole_training_step[State, Params](
    carry: WholeRunState[State],
    env: EpisodeStatisticsWrapper[State, Params],
    params: Params,
    config: DQNTrainConfig,
    sink: EventSink,
) -> WholeRunState[State]:
    run, metrics = training_step(carry.run, env, params, config)
    pending = cast(
        StepMetrics,
        jax.tree.map(lambda storage, value: storage.at[carry.pending_count].set(value), carry.pending, metrics),
    )
    current = WholeRunState(run, pending, carry.pending_count + 1, carry.phase, carry.evaluation_episodes)
    checkpoint = checkpoint_due(run.vector_step, config.num_envs, config.total_timesteps)
    selected_loss = metrics.updated & (metrics.global_step % 100 < config.num_envs)
    flush = (current.pending_count == pending.active.shape[0]) | selected_loss | checkpoint
    current = nnx.cond(flush, lambda value: _flush_training(value, sink), lambda value: value, current)
    jax.lax.cond(checkpoint, lambda: _emit_checkpoint(current, config, sink), lambda: jnp.asarray(0, jnp.int32))
    return cast(WholeRunState[State], current)  # NNX cond retains the explicit carry topology.


def _flush_training[State](carry: WholeRunState[State], sink: EventSink) -> WholeRunState[State]:
    jax.lax.cond(
        carry.pending_count > 0,
        lambda: io_callback(
            sink.training,
            jax.ShapeDtypeStruct((), jnp.int32),
            _array_leaves(carry.pending),
            carry.pending_count,
            ordered=True,
        ),
        lambda: jnp.asarray(0, jnp.int32),
    )
    return WholeRunState(carry.run, carry.pending, jnp.asarray(0, jnp.int32), carry.phase, carry.evaluation_episodes)


def _finish_training[State](
    carry: WholeRunState[State], config: DQNTrainConfig, sink: EventSink
) -> WholeRunState[State]:
    flushed = _flush_training(carry, sink)
    io_callback(sink.training_complete, jax.ShapeDtypeStruct((), jnp.int32), ordered=True)
    prepared = WholeRunState(
        flushed.run,
        flushed.pending,
        flushed.pending_count,
        jnp.asarray(ExecutionPhase.EVALUATION, jnp.int32),
        flushed.evaluation_episodes,
    )
    _emit_checkpoint(prepared, config, sink)
    return prepared


def _whole_evaluation[State, Params](
    carry: WholeRunState[State],
    env: EpisodeStatisticsWrapper[State, Params],
    params: Params,
    config: DQNTrainConfig,
    sink: EventSink,
) -> WholeRunState[State]:
    run = carry.run
    count = jnp.asarray(0, jnp.int32)
    key = run.key
    if config.eval_episodes:
        keys: Array = jax.random.split(key, config.num_envs + 1)
        observations, env_state = jax.vmap(env.reset, in_axes=(0, None))(keys[1:], params)
        initial = EvaluationState(observations, env_state, jnp.zeros(config.num_envs, jnp.float32), count, keys[0])

        def step(state: EvaluationState[State]) -> EvaluationState[State]:
            transition, next_key = evaluation_step(
                run.model, state.observations, state.env_state, state.key, env=env, params=params
            )
            jax.lax.cond(
                jnp.any(transition.completed),
                lambda: io_callback(
                    sink.evaluation,
                    jax.ShapeDtypeStruct((), jnp.int32),
                    transition.episode_return,
                    transition.completed,
                    state.count,
                    ordered=True,
                ),
                lambda: jnp.asarray(0, jnp.int32),
            )
            return EvaluationState(
                transition.observations,
                transition.state,
                state.returns,
                state.count + jnp.sum(transition.completed, dtype=jnp.int32),
                next_key,
            )

        final = jax.lax.while_loop(lambda state: state.count < config.eval_episodes, step, initial)
        count, key = final.count, final.key
    final_run = RunState(
        run.model,
        run.target,
        run.optimizer,
        run.replay,
        run.observations,
        run.env_state,
        run.vector_step,
        run.updates,
        run.target_updates,
        key,
    )
    return WholeRunState(
        final_run, carry.pending, carry.pending_count, jnp.asarray(ExecutionPhase.COMPLETE, jnp.int32), count
    )


def _emit_checkpoint[State](carry: WholeRunState[State], config: DQNTrainConfig, sink: EventSink) -> Array:
    return cast(
        Array,
        io_callback(
            sink.checkpoint,
            jax.ShapeDtypeStruct((), jnp.int32),
            make_snapshot(carry.run, carry.phase, carry.evaluation_episodes, config),
            ordered=True,
        ),
    )  # Callback result has the declared scalar int32 specification.


def make_snapshot[State](
    state: RunState[State],
    phase: Array,
    evaluation_episodes: Array,
    config: DQNTrainConfig,
) -> Snapshot:
    """Extract numerical state only; NNX graph definitions stay outside callbacks."""
    from jax_shapley.run_io import Snapshot

    agent = _array_leaves(nnx.state((state.model, state.target)))
    optimizer = _array_leaves(nnx.state(state.optimizer))
    runtime = _array_leaves((
        state.observations,
        state.env_state,
        state.vector_step,
        state.updates,
        state.target_updates,
        jax.random.key_data(state.key),
    ))
    epsilon = epsilon_at(config, jnp.maximum(state.vector_step - 1, 0) * config.num_envs)
    return Snapshot(
        agent,
        state.replay,
        optimizer,
        runtime,
        state.vector_step,
        phase,
        evaluation_episodes,
        epsilon,
        str(jax.random.key_impl(state.key)),
    )


def _array_leaves(tree: object) -> tuple[Array, ...]:
    return tuple(cast(list[Array], jax.tree.leaves(tree)))  # Only numerical structs and NNX state cross this boundary.


def _inactive_step[State](state: RunState[State], config: DQNTrainConfig) -> tuple[RunState[State], StepMetrics]:
    zero = jnp.asarray(0.0, jnp.float32)
    flags = jnp.zeros(config.num_envs, jnp.bool_)
    return state, StepMetrics(
        state.vector_step * config.num_envs,
        epsilon_at(config, state.vector_step * config.num_envs),
        zero,
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.zeros(config.num_envs, jnp.float32),
        jnp.zeros(config.num_envs, jnp.int32),
        flags,
        jnp.asarray(False),
    )


def _skip_learn(state: UpdateState, gamma: float) -> LossMetrics:
    del state, gamma
    zero = jnp.asarray(0.0, jnp.float32)
    return LossMetrics(zero, zero, zero)


def _positive(name: str, value: int) -> None:
    if value < 1 or value > 2**31 - 1:
        raise ValueError(f"{name} must be a positive int32 value")


def _finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


def _probability(name: str, value: float) -> None:
    _finite(name, value)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1]")


if __name__ == "__main__":
    # Use the canonical module so host adapters share the CLI's dataclass identities.
    from jax_shapley import train_dqn

    train_dqn.main()
