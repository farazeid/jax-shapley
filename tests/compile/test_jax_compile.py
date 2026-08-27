"""Compile-focused regression tests for JAX environments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypedDict, cast

import jax
import jax.numpy as jnp
import pytest

from jax_shapley.envs import Gridworld, Hypercube, Mastermind222
from jax_shapley.wrappers import (
    EnvironmentProtocol,
    EpisodeInfo,
    EpisodeStatisticsWrapper,
)


class _Space(Protocol):
    def contains(self, value: object) -> bool: ...


class _CompileEnv(Protocol):
    @property
    def default_params(self) -> object: ...

    def reset_env(self, key: jax.Array, params: object) -> tuple[jax.Array, object]: ...
    def reset(self, key: jax.Array, params: object) -> tuple[jax.Array, object]: ...
    def step_env(
        self, key: jax.Array, state: object, action: jax.Array, params: object
    ) -> tuple[jax.Array, object, jax.Array, jax.Array, EpisodeInfo]: ...
    def observation_space(self, params: object) -> _Space: ...


class _WrappedInfo(TypedDict):
    episode: EpisodeInfo


class _BatchedStep(Protocol):
    def __call__(
        self, keys: jax.Array, states: object, actions: jax.Array, params: object
    ) -> tuple[jax.Array, object, jax.Array, jax.Array, _WrappedInfo]: ...


type EnvFactory = Callable[[], _CompileEnv]


@pytest.mark.compile
@pytest.mark.parametrize(
    "name,factory",
    [
        ("gridworld", lambda: cast(_CompileEnv, Gridworld())),
        ("hypercube-3x2", lambda: cast(_CompileEnv, Hypercube(3, 2))),
        ("mastermind-222", lambda: cast(_CompileEnv, Mastermind222())),
    ],
    ids=["gridworld", "hypercube-3x2", "mastermind-222"],
)
def test_env_reset_and_step_compile(name: str, factory: EnvFactory) -> None:
    del name
    env = factory()
    params = env.default_params

    jax.jit(env.reset_env).lower(jax.random.key(0), params).compile()
    initial_obs, initial_state = env.reset(jax.random.key(1), params)

    action = jnp.asarray(0, dtype=jnp.int32)

    step = jax.jit(env.step_env)
    step.lower(jax.random.key(2), initial_state, action, params).compile()

    assert env.observation_space(params).contains(initial_obs)


@pytest.mark.compile
@pytest.mark.parametrize(
    "side_length,dimension",
    [
        (2, 2),
        (2, 5),
        (6, 2),
        (6, 5),
        (4, 3),
    ],
    ids=["2x2", "2x5", "6x2", "6x5", "4x3"],
)
def test_hypercube_shape_variants_compile(side_length: int, dimension: int) -> None:
    env = Hypercube(side_length=side_length, dimension=dimension)
    params = env.default_params

    jax.jit(env.reset_env).lower(jax.random.key(10), params).compile()
    _, state = env.reset(jax.random.key(11), params)

    action = jnp.asarray(0, dtype=jnp.int32)
    jax.jit(env.step_env).lower(jax.random.key(12), state, action, params).compile()
    assert state.position.shape == (dimension,)


@pytest.mark.compile
@pytest.mark.parametrize(
    "environment",
    [Gridworld(), Hypercube(4, 5), Mastermind222()],
    ids=["gridworld", "hypercube-4x5", "mastermind-222"],
)
def test_vectorized_rollout_scan_compile(
    environment: Gridworld | Hypercube | Mastermind222,
) -> None:
    wrapped = EpisodeStatisticsWrapper(cast(EnvironmentProtocol[object, object], environment))
    params = wrapped.default_params
    batch = 16
    horizon = 8

    keys = jax.random.split(jax.random.key(20), batch)
    _, states = jax.vmap(wrapped.reset, in_axes=(0, None))(keys, params)
    actions = (jnp.arange(batch * horizon, dtype=jnp.int32) % wrapped.num_actions).reshape(horizon, batch)
    batched_step = cast(_BatchedStep, jax.vmap(wrapped.step, in_axes=(0, 0, 0, None)))

    def rollout(
        current_keys: jax.Array,
        current_states: object,
        rollout_actions: jax.Array,
        current_params: object,
    ) -> tuple[jax.Array, jax.Array, EpisodeInfo]:
        def scan_step(
            carry: tuple[jax.Array, object],
            action: jax.Array,
        ) -> tuple[
            tuple[jax.Array, object],
            tuple[jax.Array, jax.Array, EpisodeInfo],
        ]:
            step_keys, current_env_states = carry
            split = jax.vmap(jax.random.split)(step_keys)
            next_keys = split[:, 0]
            action_keys = split[:, 1]
            _, next_env_states, rewards, done, info = batched_step(
                action_keys,
                current_env_states,
                action,
                current_params,
            )
            return (next_keys, next_env_states), (rewards, done, info["episode"])

        _, outputs = jax.lax.scan(scan_step, (current_keys, current_states), rollout_actions)
        return outputs

    compiled = jax.jit(rollout).lower(keys, states, actions, params).compile()
    rewards, done, episode = compiled(keys, states, actions, params)
    assert rewards.shape == (horizon, batch)
    assert done.shape == (horizon, batch)
    assert episode["completed"].shape == (horizon, batch)
