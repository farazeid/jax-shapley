"""State-buffer collection: a frozen agent's policy run in vectorised environments.

The policy is the agent's epsilon-greedy one (epsilon 0 for greedy). Gymnax auto-reset
is preserved by `dqn.environment_step`.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from new_jax_shapley.algorithms.dqn import EnvParams, EnvState, QNetwork, choose_action, environment_step
from new_jax_shapley.common.supervised_learning import StateBuffers
from new_jax_shapley.envs.wrappers import EpisodeStatisticsState, EpisodeStatisticsWrapper


def collect_buffers(
    env: EpisodeStatisticsWrapper[EnvState, EnvParams],
    model: QNetwork,
    epsilon: Array,
    key: Array,
    *,
    train_size: int,
    validation_size: int,
    exact_size: int | None,
    num_envs: int,
    exact: Array | None = None,
) -> tuple[Array, StateBuffers]:
    """Source fill_buffer for train, validation and (optionally) a separate exact buffer.

    Each buffer starts from a fresh reset and records the observations before each vector
    step, step-major then environment. Without a separate exact buffer exact values use
    the train buffer, unless `exact` is supplied by an earlier step. Returns the key
    advanced past the collection.
    """
    keys: Array = jax.random.split(key, 4)
    train = collect_observations(env, model, epsilon, keys[1], train_size, num_envs)
    validation = collect_observations(env, model, epsilon, keys[2], validation_size, num_envs)
    if exact is None:
        exact = (
            train if exact_size is None else collect_observations(env, model, epsilon, keys[3], exact_size, num_envs)
        )
    return keys[0], StateBuffers(train, validation, exact)


def collect_observations(
    env: EpisodeStatisticsWrapper[EnvState, EnvParams],
    model: QNetwork,
    epsilon: Array,
    key: Array,
    rows: int,
    num_envs: int,
) -> Array:
    """One buffer from a fresh reset: the observations before each vector step, step-major then environment."""
    params = env.default_params

    def step(
        env: EpisodeStatisticsWrapper[EnvState, EnvParams],
        params: EnvParams,
        num_envs: int,
        model: QNetwork,
        epsilon: Array,
        carry: tuple[Array, EpisodeStatisticsState[EnvState], Array],
        index: Array,
    ) -> tuple[tuple[Array, EpisodeStatisticsState[EnvState], Array], Array]:
        del index
        current, state, key = carry
        actions, _, key = choose_action(model, current, epsilon, key)
        step_keys: Array = jax.random.split(key, num_envs + 1)
        transition = jax.vmap(partial(environment_step, env), in_axes=(0, 0, 0, None))(
            step_keys[1:], state, actions, params
        )
        return (transition.observations, transition.state, step_keys[0]), current

    reset_keys: Array = jax.random.split(key, num_envs + 1)
    observations, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys[1:], params)
    _, recorded = jax.lax.scan(
        partial(step, env, params, num_envs, model, epsilon),
        (observations, env_state, reset_keys[0]),
        jnp.arange(rows // num_envs),
    )
    return recorded.reshape(rows, recorded.shape[-1]).astype(jnp.int32)
