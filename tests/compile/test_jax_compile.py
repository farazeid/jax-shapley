"""Compile abstract tensor inputs before exercising numerical environment paths."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

import jax
import jax.numpy as jnp
import pytest
from einops import rearrange

from jax_shapley.envs import Gridworld, Hypercube, Mastermind222
from jax_shapley.envs.contracts import EpisodeInfo, EpisodeStepInfo, StepInfo
from jax_shapley.wrappers import EnvironmentProtocol, EpisodeStatisticsState, EpisodeStatisticsWrapper


class _BatchedStep[State, Params](Protocol):
    def __call__(
        self,
        keys: jax.Array,
        states: EpisodeStatisticsState[State],
        actions: jax.Array,
        params: Params,
    ) -> tuple[jax.Array, EpisodeStatisticsState[State], jax.Array, jax.Array, EpisodeStepInfo]: ...


def _abstract(tree: object) -> object:
    """Use only shapes/dtypes when tracing; values are not compile-time fixtures."""
    return jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), tree)


def _assert_env_compiles[State, Params](env: EnvironmentProtocol[State, Params]) -> None:
    params = env.default_params
    key_spec = jax.ShapeDtypeStruct((), jax.random.key(0).dtype)
    reset = jax.jit(env.reset_env).trace(key_spec, params).lower().compile()
    initial_obs, initial_state = reset(jax.random.key(1), params)
    action_spec = jax.ShapeDtypeStruct((), jnp.int32)
    step = jax.jit(env.step_env).trace(key_spec, _abstract(initial_state), action_spec, params).lower().compile()
    observation, _, reward, done, info = step(jax.random.key(2), initial_state, jnp.asarray(0, jnp.int32), params)
    assert env.observation_space(params).contains(initial_obs)
    assert observation.shape == initial_obs.shape
    assert observation.dtype == jnp.int32
    assert reward.shape == done.shape == ()
    assert reward.dtype == jnp.float32
    assert done.dtype == jnp.bool_
    _assert_step_info(info, ())


def _assert_step_info(info: StepInfo, shape: tuple[int, ...]) -> None:
    assert set(info) == {"discount", "terminated", "truncated"}
    assert info["discount"].shape == shape
    assert info["discount"].dtype == jnp.float32
    for flag in (info["terminated"], info["truncated"]):
        assert flag.shape == shape
        assert flag.dtype == jnp.bool_


@pytest.mark.compile
@pytest.mark.parametrize(
    "check",
    [
        lambda: _assert_env_compiles(Gridworld()),
        lambda: _assert_env_compiles(Hypercube(3, 2)),
        lambda: _assert_env_compiles(Mastermind222()),
    ],
    ids=["gridworld", "hypercube-3x2", "mastermind-222"],
)
def test_env_reset_and_step_compile(check: Callable[[], None]) -> None:
    check()


@pytest.mark.compile
@pytest.mark.parametrize(
    "side_length,dimension",
    [(2, 2), (2, 5), (6, 2), (6, 5), (4, 3)],
    ids=["2x2", "2x5", "6x2", "6x5", "4x3"],
)
def test_hypercube_shape_variants_compile(side_length: int, dimension: int) -> None:
    _assert_env_compiles(Hypercube(side_length, dimension))


def _assert_rollout_compiles[State, Params](env: EnvironmentProtocol[State, Params]) -> None:
    wrapped = EpisodeStatisticsWrapper(env)
    params = wrapped.default_params
    batch = 16
    horizon = 8
    keys = jax.random.split(jax.random.key(20), batch)
    _, states = jax.vmap(wrapped.reset, in_axes=(0, None))(keys, params)
    actions = rearrange(jnp.arange(batch * horizon, dtype=jnp.int32), "(time batch) -> time batch", batch=batch)
    actions %= wrapped.num_actions
    # JAX vmap preserves the state/info pytree but its library type erases that relationship.
    batched_step = cast(_BatchedStep[State, Params], jax.vmap(wrapped.step, in_axes=(0, 0, 0, None)))

    def rollout(
        current_keys: jax.Array,
        current_states: EpisodeStatisticsState[State],
        rollout_actions: jax.Array,
        current_params: Params,
    ) -> tuple[jax.Array, jax.Array, EpisodeInfo]:
        def scan_step(
            carry: tuple[jax.Array, EpisodeStatisticsState[State]], action: jax.Array
        ) -> tuple[tuple[jax.Array, EpisodeStatisticsState[State]], tuple[jax.Array, jax.Array, EpisodeInfo]]:
            step_keys, current_env_states = carry
            split = jax.vmap(jax.random.split)(step_keys)
            next_keys, action_keys = split[:, 0], split[:, 1]
            _, next_states, rewards, done, info = batched_step(action_keys, current_env_states, action, current_params)
            return (next_keys, next_states), (rewards, done, info["episode"])

        _, outputs = jax.lax.scan(scan_step, (current_keys, current_states), rollout_actions)
        return outputs

    compiled = jax.jit(rollout).trace(_abstract(keys), _abstract(states), _abstract(actions), params).lower().compile()
    rewards, done, episode = compiled(keys, states, actions, params)
    assert rewards.shape == done.shape == (horizon, batch)
    assert rewards.dtype == episode["return"].dtype == jnp.float32
    assert done.dtype == episode["completed"].dtype == jnp.bool_
    assert episode["length"].dtype == jnp.int32
    for value in episode.values():
        assert value.shape == (horizon, batch)


@pytest.mark.compile
@pytest.mark.parametrize(
    "check",
    [
        lambda: _assert_rollout_compiles(Gridworld()),
        lambda: _assert_rollout_compiles(Hypercube(4, 5)),
        lambda: _assert_rollout_compiles(Mastermind222()),
    ],
    ids=["gridworld", "hypercube-4x5", "mastermind-222"],
)
def test_vectorized_rollout_scan_compile(check: Callable[[], None]) -> None:
    check()
