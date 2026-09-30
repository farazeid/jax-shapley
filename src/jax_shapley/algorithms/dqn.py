"""FastSVERL DQN policies and differentiable learning units."""

from functools import partial
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import optax
from flax import nnx, struct
from jax import Array

from jax_shapley.config import DQNTrainConfig
from jax_shapley.networks import MLPConfig, ModelInput, QNetwork
from jax_shapley.replay import ReplayState, TransitionBatch, append, empty_replay, sample
from jax_shapley.wrappers import EpisodeStatisticsState, EpisodeStatisticsWrapper


class ActionOutput(NamedTuple):
    actions: Array
    log_probabilities: Array


@struct.dataclass
class LossMetrics:
    loss: Array
    target_mean: Array
    value_mean: Array


def greedy_probabilities(q_values: Array) -> Array:
    ties = q_values == jnp.max(q_values, axis=-1, keepdims=True)
    return ties.astype(jnp.float32) / jnp.sum(ties, axis=-1, keepdims=True)


def exploration_probabilities(q_values: Array, epsilon: Array) -> Array:
    return epsilon / q_values.shape[-1] + (1 - epsilon) * greedy_probabilities(q_values)


def epsilon_at(config: DQNTrainConfig, global_step: Array) -> Array:
    fraction = global_step.astype(jnp.float32) / (config.exploration_fraction * config.total_timesteps)
    return jnp.maximum(config.start_e + (config.end_e - config.start_e) * fraction, config.end_e).astype(jnp.float32)


def action_log_probabilities(probabilities: Array, actions: Array) -> Array:
    # Torch Categorical renormalizes probabilities and clamps before converting to logits.
    probabilities = probabilities / jnp.sum(probabilities, axis=-1, keepdims=True)
    epsilon = jnp.finfo(jnp.float32).eps
    logits = jnp.log(jnp.clip(probabilities, epsilon, 1 - epsilon))
    return jnp.take_along_axis(logits, actions[..., None], axis=-1)[..., 0]


def choose_action(model: QNetwork, observations: ModelInput, epsilon: Array, key: Array) -> tuple[ActionOutput, Array]:
    probabilities = exploration_probabilities(model(observations).q_values, epsilon)
    keys: Array = jax.random.split(key)
    # Sampling uses true zero-probability logits; the clamp applies only to reported log probabilities.
    actions: Array = jax.random.categorical(keys[1], jnp.log(probabilities), axis=-1).astype(jnp.int32)
    return ActionOutput(actions, action_log_probabilities(probabilities, actions)), keys[0]


def td_targets(target: QNetwork, batch: TransitionBatch, gamma: float) -> Array:
    values = jnp.max(target(batch.next_observations).q_values, axis=-1)
    return jax.lax.stop_gradient(batch.rewards + gamma * values * (~batch.terminated).astype(jnp.float32))


def dqn_loss(model: QNetwork, target: QNetwork, batch: TransitionBatch, gamma: float) -> tuple[Array, LossMetrics]:
    batch.validate()
    targets = td_targets(target, batch, gamma)
    values = jnp.take_along_axis(model(batch.observations).q_values, batch.actions[:, None], axis=-1)[:, 0]
    loss = jnp.mean(jnp.square(targets - values))
    return loss, LossMetrics(loss, jnp.mean(targets), jnp.mean(values))


def make_optimizer(model: QNetwork, config: DQNTrainConfig) -> nnx.Optimizer[QNetwork]:
    adam = config.adam
    return nnx.Optimizer(
        model,
        optax.adam(config.learning_rate, b1=adam.beta1, b2=adam.beta2, eps=adam.epsilon, eps_root=0.0),
        wrt=nnx.Param,
    )


def initialize_models(
    config: MLPConfig, training: DQNTrainConfig, key: Array
) -> tuple[QNetwork, QNetwork, nnx.Optimizer[QNetwork], Array]:
    keys: Array = jax.random.split(key)
    model = QNetwork(config, rngs=nnx.Rngs(keys[1]))
    target = nnx.clone(model)
    return model, target, make_optimizer(model, training), keys[0]


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
