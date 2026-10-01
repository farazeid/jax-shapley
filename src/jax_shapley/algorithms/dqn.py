"""FastSVERL DQN policies and differentiable learning units."""

from functools import partial
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import optax
from flax import nnx, struct
from jax import Array
from jax.experimental import io_callback

from jax_shapley.config import CheckpointMode, DQNTrainConfig, ExecutionPhase
from jax_shapley.execution import EventSink, Snapshot
from jax_shapley.networks import MLPConfig, ModelInput, QNetwork
from jax_shapley.replay import ReplayState, TransitionBatch, append, empty_replay, sample
from jax_shapley.wrappers import EpisodeStatisticsState, EpisodeStatisticsWrapper


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


class ActionOutput(NamedTuple):
    actions: Array
    log_probabilities: Array


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


def whole_run[State, Params](
    key: Array,
    *,
    env: EpisodeStatisticsWrapper[State, Params],
    params: Params,
    network: MLPConfig,
    config: DQNTrainConfig,
    chunk_steps: int,
    checkpoint_mode: CheckpointMode,
    sink: EventSink,
) -> WholeRunState[State]:
    """Initialization, training, evaluation and ordered events in one numerical call."""
    state = initialize_run(env, params, network, config, key)
    return continue_run(
        state,
        jnp.asarray(ExecutionPhase.TRAINING, jnp.int32),
        jnp.asarray(0, jnp.int32),
        env=env,
        params=params,
        config=config,
        chunk_steps=chunk_steps,
        checkpoint_mode=checkpoint_mode,
        sink=sink,
    )


def continue_run[State, Params](
    state: RunState[State],
    phase: Array,
    evaluation_episodes: Array,
    *,
    env: EpisodeStatisticsWrapper[State, Params],
    params: Params,
    config: DQNTrainConfig,
    chunk_steps: int,
    checkpoint_mode: CheckpointMode,
    sink: EventSink,
) -> WholeRunState[State]:
    """Continue a validated snapshot; inactive tail steps preserve numerical state."""
    _, empty = _inactive_step(state, config)
    pending = cast(StepMetrics, jax.tree.map(lambda value: jnp.broadcast_to(value, (chunk_steps, *value.shape)), empty))
    initial = WholeRunState(state, pending, jnp.asarray(0, jnp.int32), phase, evaluation_episodes)

    def block(carry: WholeRunState[State]) -> WholeRunState[State]:
        def step(current: WholeRunState[State], index: Array) -> tuple[WholeRunState[State], Array]:
            del index
            next_state = nnx.cond(
                current.run.vector_step < config.vector_steps,
                lambda value: _whole_training_step(value, env, params, config, checkpoint_mode, sink),
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
        lambda carry: _finish_training(carry, config, checkpoint_mode, sink),
        lambda carry: carry,
        trained,
    )
    evaluated = nnx.cond(
        prepared.phase == ExecutionPhase.EVALUATION,
        lambda carry: _whole_evaluation(carry, env, params, config, sink),
        lambda carry: carry,
        prepared,
    )
    if checkpoint_mode != "none":
        jax.lax.cond(
            phase != ExecutionPhase.COMPLETE,
            lambda: _emit_checkpoint(evaluated, config, checkpoint_mode, sink),
            lambda: jnp.asarray(0, jnp.int32),
        )
    return cast(WholeRunState[State], evaluated)  # NNX transforms retain the explicit carry topology.


def make_snapshot[State](
    state: RunState[State],
    phase: Array,
    evaluation_episodes: Array,
    config: DQNTrainConfig,
    mode: CheckpointMode,
) -> Snapshot:
    """Extract numerical state only; NNX graph definitions stay outside callbacks."""
    agent = _array_leaves(nnx.state((state.model, state.target)))
    optimizer = _array_leaves(nnx.state(state.optimizer)) if mode == "run" else ()
    runtime = (
        _array_leaves((
            state.observations,
            state.env_state,
            state.vector_step,
            state.updates,
            state.target_updates,
            jax.random.key_data(state.key),
        ))
        if mode == "run"
        else ()
    )
    replay = state.replay if mode in ("agent-replay", "run") else None
    epsilon = epsilon_at(config, jnp.maximum(state.vector_step - 1, 0) * config.num_envs)
    return Snapshot(
        agent,
        replay,
        optimizer,
        runtime,
        state.vector_step,
        phase,
        evaluation_episodes,
        epsilon,
        str(jax.random.key_impl(state.key)),
    )


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
    loss = nnx.cond(should_learn, _learn, _skip_learn, learning, config.gamma)
    nnx.cond(should_target, _target_update, _target_skip, state.model, state.target, config.tau)
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
    adam = config.adam
    return nnx.Optimizer(
        model,
        optax.adam(config.learning_rate, b1=adam.beta1, b2=adam.beta2, eps=adam.epsilon, eps_root=0.0),
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
    mode: CheckpointMode,
    sink: EventSink,
) -> WholeRunState[State]:
    run, metrics = training_step(carry.run, env, params, config)
    pending = cast(
        StepMetrics,
        jax.tree.map(lambda storage, value: storage.at[carry.pending_count].set(value), carry.pending, metrics),
    )
    current = WholeRunState(run, pending, carry.pending_count + 1, carry.phase, carry.evaluation_episodes)
    checkpoint = (run.vector_step * config.num_envs) % max(1, config.total_timesteps // 10) < config.num_envs
    selected_loss = metrics.updated & (metrics.global_step % 100 < config.num_envs)
    flush = (current.pending_count == pending.active.shape[0]) | selected_loss
    if mode != "none":
        flush = flush | checkpoint
    current = nnx.cond(flush, lambda value: _flush_training(value, sink), lambda value: value, current)
    if mode != "none":
        jax.lax.cond(
            checkpoint, lambda: _emit_checkpoint(current, config, mode, sink), lambda: jnp.asarray(0, jnp.int32)
        )
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
    carry: WholeRunState[State], config: DQNTrainConfig, mode: CheckpointMode, sink: EventSink
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
    if mode != "none":
        _emit_checkpoint(prepared, config, mode, sink)
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


def _emit_checkpoint[State](
    carry: WholeRunState[State], config: DQNTrainConfig, mode: CheckpointMode, sink: EventSink
) -> Array:
    return cast(
        Array,
        io_callback(
            sink.checkpoint,
            jax.ShapeDtypeStruct((), jnp.int32),
            make_snapshot(carry.run, carry.phase, carry.evaluation_episodes, config, mode),
            ordered=True,
        ),
    )  # Callback result has the declared scalar int32 specification.


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


def _learn(state: UpdateState, gamma: float) -> LossMetrics:
    return update(state.model, state.target, state.optimizer, state.batch, gamma)


def _skip_learn(state: UpdateState, gamma: float) -> LossMetrics:
    del state, gamma
    zero = jnp.asarray(0.0, jnp.float32)
    return LossMetrics(zero, zero, zero)


def _target_update(model: QNetwork, target: QNetwork, tau: float) -> None:
    update_target(model, target, tau)


def _target_skip(model: QNetwork, target: QNetwork, tau: float) -> None:
    del model, target, tau
