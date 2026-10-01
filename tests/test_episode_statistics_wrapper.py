"""JIT, auto-reset, vectorization, and scan tests for episode statistics."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_shapley import EpisodeStatisticsState, EpisodeStatisticsWrapper
from jax_shapley.envs import (
    Gridworld,
    GridworldState,
    Hypercube,
    Mastermind222,
    Mastermind222Params,
    Mastermind222State,
)
from jax_shapley.envs.contracts import EpisodeInfo, EpisodeStepInfo
from jax_shapley.wrappers import EnvironmentProtocol


class _BatchedStep(Protocol):
    def __call__(
        self,
        keys: jax.Array,
        states: EpisodeStatisticsState[Mastermind222State],
        actions: jax.Array,
        params: Mastermind222Params,
    ) -> tuple[jax.Array, EpisodeStatisticsState[Mastermind222State], jax.Array, jax.Array, EpisodeStepInfo]: ...


def test_reset_and_jit_step_contract(wrapper: GenericWrapper) -> None:
    params = wrapper.default_params
    reset_key, step_key = jax.random.split(jax.random.key(0))
    observation, state = wrapper.reset(reset_key, params)

    assert wrapper.observation_space(params).contains(observation)
    assert np.asarray(state.episode_return).dtype == np.float32
    assert np.asarray(state.episode_length).dtype == np.int32
    assert np.asarray(state.episode_count).dtype == np.int32
    np.testing.assert_equal(float(state.episode_return), 0.0)
    assert int(state.episode_length) == 0
    assert int(state.episode_count) == 0

    eager = wrapper.step(step_key, state, jnp.asarray(0, dtype=jnp.int32), params)
    compiled = jax.jit(wrapper.step)(step_key, state, jnp.asarray(0, dtype=jnp.int32), params)
    for eager_leaf, compiled_leaf in zip(jax.tree.leaves(eager), jax.tree.leaves(compiled), strict=True):
        np.testing.assert_array_equal(eager_leaf, compiled_leaf)

    _, _, _, done, info = eager
    assert set(info) == {"discount", "terminated", "truncated", "episode"}
    assert set(info["episode"]) == {"return", "length", "completed"}
    assert np.asarray(info["episode"]["return"]).dtype == np.float32
    assert np.asarray(info["episode"]["length"]).dtype == np.int32
    assert np.asarray(info["episode"]["completed"]).dtype == np.bool_
    assert bool(info["episode"]["completed"]) == bool(done)


def test_completed_episode_is_emitted_before_accumulators_reset() -> None:
    wrapper = EpisodeStatisticsWrapper(Gridworld())
    params = wrapper.default_params
    state = EpisodeStatisticsState(
        env_state=GridworldState(
            position=jnp.asarray([0, 2], dtype=jnp.int32),
            terminated=jnp.asarray(False),
            time=jnp.asarray(3, dtype=jnp.int32),
        ),
        episode_return=jnp.asarray(-3.0, dtype=jnp.float32),
        episode_length=jnp.asarray(3, dtype=jnp.int32),
        episode_count=jnp.asarray(2, dtype=jnp.int32),
    )

    observation, next_state, reward, done, info = wrapper.step(
        jax.random.key(0), state, jnp.asarray(0, dtype=jnp.int32), params
    )

    assert bool(done)
    np.testing.assert_equal(float(reward), 9.0)
    assert bool(info["episode"]["completed"])
    np.testing.assert_equal(float(info["episode"]["return"]), 6.0)
    assert int(info["episode"]["length"]) == 4
    np.testing.assert_equal(float(next_state.episode_return), 0.0)
    assert int(next_state.episode_length) == 0
    assert int(next_state.episode_count) == 3
    assert tuple(np.asarray(observation)) in {(0, 0), (1, 0)}
    assert not bool(next_state.env_state.terminated)


def test_incomplete_episode_emits_masked_zero_values() -> None:
    wrapper = EpisodeStatisticsWrapper(Gridworld())
    params = wrapper.default_params
    state = EpisodeStatisticsState(
        env_state=GridworldState(
            position=jnp.asarray([0, 0], dtype=jnp.int32),
            terminated=jnp.asarray(False),
            time=jnp.asarray(0, dtype=jnp.int32),
        ),
        episode_return=jnp.asarray(0.0, dtype=jnp.float32),
        episode_length=jnp.asarray(0, dtype=jnp.int32),
        episode_count=jnp.asarray(0, dtype=jnp.int32),
    )

    _, next_state, reward, done, info = wrapper.step(jax.random.key(0), state, jnp.asarray(0, dtype=jnp.int32), params)

    assert not bool(done)
    np.testing.assert_equal(float(reward), -1.0)
    assert not bool(info["episode"]["completed"])
    np.testing.assert_equal(float(info["episode"]["return"]), 0.0)
    assert int(info["episode"]["length"]) == 0
    np.testing.assert_equal(float(next_state.episode_return), -1.0)
    assert int(next_state.episode_length) == 1
    assert int(next_state.episode_count) == 0


@pytest.mark.slow
def test_jitted_vmapped_scan_emits_every_completed_episode() -> None:
    batch_size = 512
    horizon = 128
    wrapper = EpisodeStatisticsWrapper(Mastermind222())
    params = wrapper.default_params
    keys = jax.random.split(jax.random.key(123), batch_size)
    reset = jax.vmap(wrapper.reset, in_axes=(0, None))
    # JAX vmap erases the callable's generic state/info relationships in library typing.
    step = cast(_BatchedStep, jax.vmap(wrapper.step, in_axes=(0, 0, 0, None)))
    observation, state = reset(keys, params)
    actions = jnp.arange(horizon * batch_size, dtype=jnp.int32).reshape(horizon, batch_size) % wrapper.num_actions

    def rollout(carry: RolloutCarry, rollout_actions: jax.Array) -> tuple[RolloutCarry, RolloutOutput]:
        def scan_step(current: RolloutCarry, action: jax.Array) -> tuple[RolloutCarry, RolloutOutput]:
            _, current_state, current_keys = current
            split_keys = jax.vmap(jax.random.split)(current_keys)
            next_keys, step_keys = split_keys[:, 0], split_keys[:, 1]
            next_observation, next_state, reward, done, info = step(step_keys, current_state, action, params)
            outputs = (reward, done, info["episode"])
            return (next_observation, next_state, next_keys), outputs

        return jax.lax.scan(scan_step, carry, rollout_actions)

    final, outputs = jax.jit(rollout)((observation, state, keys), actions)
    _, final_state, _ = final
    rewards, dones, episode = jax.device_get(outputs)
    completed = episode["completed"]

    assert rewards.shape == (horizon, batch_size)
    assert dones.shape == (horizon, batch_size)
    assert episode["return"].shape == (horizon, batch_size)
    assert episode["length"].shape == (horizon, batch_size)
    assert completed.shape == (horizon, batch_size)
    np.testing.assert_array_equal(completed, dones)
    np.testing.assert_array_equal(episode["return"][~completed], 0.0)
    np.testing.assert_array_equal(episode["length"][~completed], 0)
    np.testing.assert_array_equal(final_state.episode_count, completed.sum(axis=0))
    assert np.all(episode["length"][completed] > 0)


type GenericWrapper = EpisodeStatisticsWrapper[object, object]
type RolloutCarry = tuple[jax.Array, EpisodeStatisticsState[Mastermind222State], jax.Array]
type RolloutOutput = tuple[jax.Array, jax.Array, EpisodeInfo]


@pytest.fixture(
    params=[Gridworld(), Hypercube(2, 2), Hypercube(6, 5), Mastermind222()],
    ids=["gridworld", "hypercube-2x2", "hypercube-6x5", "mastermind-222"],
)
def wrapper(request: pytest.FixtureRequest) -> Iterator[GenericWrapper]:
    yield EpisodeStatisticsWrapper(cast(EnvironmentProtocol[object, object], request.param))
