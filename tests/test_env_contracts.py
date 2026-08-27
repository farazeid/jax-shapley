"""Gymnax API, reset support, JIT, vectorization, and rollout tests."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_shapley.envs import (
    Gridworld,
    GridworldState,
    Hypercube,
    Mastermind222,
    Mastermind222State,
)


class _Space(Protocol):
    shape: tuple[int, ...]

    def contains(self, value: object) -> bool: ...


class _State(Protocol):
    time: jax.Array


class _Environment(Protocol):
    @property
    def default_params(self) -> object: ...

    @property
    def num_actions(self) -> int: ...

    def reset(self, key: jax.Array, params: object) -> tuple[jax.Array, _State]: ...
    def step_env(
        self, key: jax.Array, state: _State, action: jax.Array, params: object
    ) -> tuple[jax.Array, _State, jax.Array, jax.Array, dict[str, jax.Array]]: ...
    def step(
        self, key: jax.Array, state: _State, action: jax.Array, params: object
    ) -> tuple[jax.Array, _State, jax.Array, jax.Array, dict[str, jax.Array]]: ...
    def get_obs(self, state: _State, params: object) -> jax.Array: ...
    def observation_space(self, params: object) -> _Space: ...


@pytest.fixture(
    params=[Gridworld(), Hypercube(2, 2), Hypercube(6, 5), Mastermind222()],
    ids=["gridworld", "hypercube-2x2", "hypercube-6x5", "mastermind-222"],
)
def env(request: pytest.FixtureRequest) -> Iterator[_Environment]:
    yield cast(_Environment, request.param)


def test_reset_and_jit_step_contract(env: _Environment) -> None:
    params = env.default_params
    key_reset, key_step = jax.random.split(jax.random.key(0))
    obs, state = env.reset(key_reset, params)
    action = jnp.asarray(0, dtype=jnp.int32)
    eager = env.step_env(key_step, state, action, params)
    compiled = jax.jit(env.step_env)(key_step, state, action, params)

    np.testing.assert_array_equal(obs, env.get_obs(state, params))
    assert env.observation_space(params).contains(obs)
    for eager_leaf, compiled_leaf in zip(jax.tree.leaves(eager), jax.tree.leaves(compiled), strict=True):
        np.testing.assert_array_equal(eager_leaf, compiled_leaf)

    next_obs, next_state, reward, done, info = eager
    assert np.asarray(next_obs).dtype == np.int32
    assert np.asarray(next_state.time).dtype == np.int32
    assert np.asarray(reward).dtype == np.float32
    assert np.asarray(done).dtype == np.bool_
    assert set(info) == {"discount", "terminated", "truncated"}
    assert np.asarray(info["discount"]).dtype == np.float32
    assert np.asarray(info["terminated"]).dtype == np.bool_
    assert np.asarray(info["truncated"]).dtype == np.bool_
    assert bool(info["terminated"]) == bool(done)
    assert not bool(info["truncated"])
    np.testing.assert_equal(float(info["discount"]), 0.0 if bool(done) else 1.0)


def test_reset_support() -> None:
    keys = jax.random.split(jax.random.key(7), 256)

    grid_obs, _ = jax.vmap(Gridworld().reset, in_axes=(0, None))(keys, Gridworld().default_params)
    starts = {tuple(row) for row in np.asarray(grid_obs)}
    assert starts == {(0, 0), (1, 0)}

    mastermind = Mastermind222()
    _, states = jax.vmap(mastermind.reset, in_axes=(0, None))(keys, mastermind.default_params)
    goals = {tuple(row) for row in np.asarray(states.goal)}
    assert goals == {(1, 1), (2, 1), (1, 2), (2, 2)}

    cube = Hypercube(6, 5)
    cube_obs, _ = jax.vmap(cube.reset, in_axes=(0, None))(keys, cube.default_params)
    np.testing.assert_array_equal(cube_obs, np.zeros((256, 5), dtype=np.int32))


def test_constructor_validation() -> None:
    with pytest.raises(ValueError):
        Hypercube(1, 2)
    with pytest.raises(ValueError):
        Hypercube(7, 2)
    with pytest.raises(ValueError):
        Hypercube(2, 1)
    with pytest.raises(ValueError):
        Hypercube(2, 6)


def test_gridworld_raw_terminal_and_public_autoreset() -> None:
    env = Gridworld()
    params = env.default_params
    state = GridworldState(
        position=jnp.asarray([0, 2], dtype=jnp.int32),
        terminated=jnp.asarray(False),
        time=jnp.asarray(3, dtype=jnp.int32),
    )
    raw_obs, raw_state, reward, done, _ = env.step_env(jax.random.key(0), state, jnp.asarray(0), params)
    np.testing.assert_array_equal(raw_obs, [0, 0])
    np.testing.assert_array_equal(raw_state.position, [0, 0])
    np.testing.assert_equal(float(reward), 9.0)
    assert bool(done)

    obs, next_state, reward, done, _ = env.step(jax.random.key(0), state, jnp.asarray(0), params)
    assert tuple(np.asarray(obs)) in {(0, 0), (1, 0)}
    assert not bool(next_state.terminated)
    np.testing.assert_equal(float(reward), 9.0)
    assert bool(done)


def test_mastermind_clues_and_termination() -> None:
    env = Mastermind222()
    params = env.default_params
    empty = jnp.full((2, 4), -1, dtype=jnp.int32)
    state = Mastermind222State(
        board=empty,
        goal=jnp.asarray([1, 2], dtype=jnp.int32),
        guess_index=jnp.asarray(0, dtype=jnp.int32),
        terminated=jnp.asarray(False),
        time=jnp.asarray(0, dtype=jnp.int32),
    )

    # Action 1 decodes to [2, 1]: both symbols are misplaced.
    obs, state, reward, done, _ = env.step_env(jax.random.key(0), state, jnp.asarray(1), params)
    np.testing.assert_array_equal(obs[:4], [2, 2, 1, 0])
    np.testing.assert_equal(float(reward), -1.0)
    assert not bool(done)

    # Action 2 decodes to [1, 2]: exact on the final allowed guess.
    obs, _, reward, done, _ = env.step_env(jax.random.key(1), state, jnp.asarray(2), params)
    np.testing.assert_array_equal(obs[4:], [0, 1, 2, 2])
    np.testing.assert_equal(float(reward), 1.0)
    assert bool(done)


@pytest.mark.slow
def test_jitted_vmapped_scans(env: _Environment) -> None:
    batch_size = 512
    horizon = 128
    params = env.default_params
    keys = jax.random.split(jax.random.key(123), batch_size)
    batched_reset = jax.vmap(env.reset, in_axes=(0, None))
    batched_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    obs, state = batched_reset(keys, params)
    actions = jnp.arange(horizon * batch_size, dtype=jnp.int32).reshape(horizon, batch_size) % env.num_actions

    def rollout(
        initial: tuple[jax.Array, _State, jax.Array], rollout_actions: jax.Array
    ) -> tuple[tuple[jax.Array, _State, jax.Array], tuple[jax.Array, jax.Array]]:
        def scan_step(
            carry: tuple[jax.Array, _State, jax.Array], action: jax.Array
        ) -> tuple[tuple[jax.Array, _State, jax.Array], tuple[jax.Array, jax.Array]]:
            current_obs, current_state, current_keys = carry
            del current_obs
            split = jax.vmap(jax.random.split)(current_keys)
            next_keys, step_keys = split[:, 0], split[:, 1]
            next_obs, next_state, reward, done, _ = batched_step(step_keys, current_state, action, params)
            return (next_obs, next_state, next_keys), (reward, done)

        return jax.lax.scan(scan_step, initial, rollout_actions)

    final, outputs = jax.jit(rollout)((obs, state, keys), actions)
    final_obs, _, _ = final
    rewards, dones = outputs
    jax.block_until_ready(final_obs)
    assert final_obs.shape == (batch_size, *env.observation_space(params).shape)
    assert rewards.shape == (horizon, batch_size)
    assert dones.shape == (horizon, batch_size)
