"""Integration checks for transition collection, timing, and execution grouping."""

from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx, struct
from gymnax.environments import spaces
from jax import Array

from jax_shapley.algorithms.dqn import (
    RunState,
    environment_step,
    evaluate_dqn,
    initialize_run,
    run_chunk,
    td_targets,
    training_step,
)
from jax_shapley.config import DQNTrainConfig
from jax_shapley.envs import Gridworld, Mastermind222
from jax_shapley.envs.contracts import StepInfo
from jax_shapley.networks import dqn_config
from jax_shapley.wrappers import EpisodeStatisticsWrapper


@struct.dataclass
class ToyState:
    time: Array


@struct.dataclass
class ToyParams:
    terminal: bool = True


class OneStepEnv:
    """Distinct reset/raw observations; terminates or truncates every step."""

    @property
    def name(self) -> str:
        return "OneStep"

    @property
    def num_actions(self) -> int:
        return 2

    def observation_space(self, params: ToyParams) -> spaces.Box:
        del params
        return spaces.Box(0, 7, (1,), dtype=jnp.int32)

    def action_space(self, params: ToyParams | None = None) -> spaces.Discrete:
        del params
        return spaces.Discrete(2)

    @property
    def default_params(self) -> ToyParams:
        return ToyParams()

    def reset_env(self, key: Array, params: ToyParams) -> tuple[Array, ToyState]:
        del key, params
        return jnp.array([0], jnp.int32), ToyState(jnp.asarray(0, jnp.int32))

    def reset(self, key: Array, params: ToyParams) -> tuple[Array, ToyState]:
        return self.reset_env(key, params)

    def step_env(
        self, key: Array, state: ToyState, action: float | Array, params: ToyParams
    ) -> tuple[Array, ToyState, Array, Array, StepInfo]:
        del key, action
        terminated, truncated = jnp.asarray(params.terminal), jnp.logical_not(params.terminal)
        info: StepInfo = {
            "terminated": terminated,
            "truncated": truncated,
            "discount": (~terminated).astype(jnp.float32),
        }
        return jnp.array([7], jnp.int32), ToyState(state.time + 1), jnp.asarray(2, jnp.float32), jnp.asarray(True), info

    def step(
        self, key: Array, state: ToyState, action: float | Array, params: ToyParams
    ) -> tuple[Array, ToyState, Array, Array, StepInfo]:
        keys: Array = jax.random.split(key)
        raw, next_state, reward, done, info = self.step_env(keys[0], state, action, params)
        reset_obs, reset_state = self.reset_env(keys[1], params)
        return (
            jnp.where(done, reset_obs, raw),
            ToyState(jnp.where(done, reset_state.time, next_state.time)),
            reward,
            done,
            info,
        )


def test_upstream_vector_timing_and_target_order() -> None:
    config = tiny_config()
    env = EpisodeStatisticsWrapper(Gridworld())
    state = initialize_run(env, env.default_params, dqn_config(2, 4, hidden_widths=(4,)), config, jax.random.key(0))
    initial = state.model.mlp.layers[0].kernel[...].copy()
    run = nnx.jit(partial(run_chunk, env=env, params=env.default_params, config=config, chunk_steps=8))
    state, metrics = run(state, jnp.asarray(8, jnp.int32))
    steps = np.arange(8) * 2
    np.testing.assert_array_equal(metrics.global_step, steps)
    np.testing.assert_array_equal(metrics.updated, (steps > 2) & (steps % 3 < 2))
    np.testing.assert_array_equal(metrics.target_updated, (steps > 2) & (steps % 5 < 2))
    assert int(state.updates) == int(np.count_nonzero(np.asarray(metrics.updated)))
    assert int(state.replay.size) == 7
    assert not jnp.array_equal(initial, state.model.mlp.layers[0].kernel[...])
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(nnx.state(state.model)))


@pytest.mark.parametrize("terminal", [True, False])
def test_raw_observation_reset_and_bootstrap_mask(terminal: bool) -> None:
    toy = OneStepEnv()
    env = EpisodeStatisticsWrapper(toy)
    params = ToyParams(terminal)
    config = DQNTrainConfig(num_envs=2, total_timesteps=12, buffer_size=4, batch_size=2, learning_starts=0)
    state = initialize_run(env, params, dqn_config(1, 2, hidden_widths=()), config, jax.random.key(1))
    state.target.mlp.layers[0].kernel[...] = jnp.ones((1, 2), jnp.float32)
    state.target.mlp.layers[0].bias[...] = jnp.zeros(2, jnp.float32)
    state, metrics = nnx.jit(partial(training_step, env=env, params=params, config=config))(state)
    np.testing.assert_array_equal(td_targets(state.target, state.replay.data, 1.0)[:2], [2 if terminal else 9] * 2)
    np.testing.assert_array_equal(state.observations, [[0], [0]])
    np.testing.assert_array_equal(state.replay.data.next_observations[:2], [[7], [7]])
    np.testing.assert_array_equal(state.replay.data.terminated[:2], [terminal, terminal])
    np.testing.assert_array_equal(metrics.episode_return, [2, 2])
    np.testing.assert_array_equal(metrics.episode_length, [1, 1])
    assert not bool(metrics.updated)  # Strict greater-than, even learning_starts=0.


def test_collection_matches_gymnax_same_key() -> None:
    env = EpisodeStatisticsWrapper(Mastermind222())
    key = jax.random.key(0)
    observation, state = env.reset(key)
    del observation
    action = jnp.asarray(0, jnp.int32)
    expected = env.step(key, state, action)
    actual = environment_step(env, key, state, action, env.default_params)
    np.testing.assert_array_equal(actual.observations, expected[0])
    for left, right in zip(jax.tree.leaves(actual.state), jax.tree.leaves(expected[1]), strict=True):
        np.testing.assert_array_equal(left, right)


def test_chunk_equivalence_and_repeated_seed() -> None:
    baseline = _run_chunks(1)
    for other in (_run_chunks(5), _run_chunks(5)):
        for left, right in zip(
            jax.tree.leaves(nnx.state(baseline.model)), jax.tree.leaves(nnx.state(other.model)), strict=True
        ):
            np.testing.assert_array_equal(left, right)
        for left, right in zip(jax.tree.leaves(baseline.replay), jax.tree.leaves(other.replay), strict=True):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(jax.random.key_data(baseline.key), jax.random.key_data(other.key))
        assert int(other.vector_step) == int(baseline.vector_step)
        assert int(other.updates) == int(baseline.updates)


def test_target_update_independent_of_learning_frequency() -> None:
    config = DQNTrainConfig(
        total_timesteps=12,
        buffer_size=4,
        batch_size=2,
        learning_starts=0,
        train_frequency=10,
        target_network_frequency=1,
        tau=0.5,
    )
    env = EpisodeStatisticsWrapper(Gridworld())
    state = initialize_run(env, env.default_params, dqn_config(2, 4, hidden_widths=()), config, jax.random.key(0))
    state.model.mlp.layers[0].bias[...] += 2
    initial = state.target.mlp.layers[0].bias[...].copy()
    state, _ = training_step(state, env, env.default_params, config)
    state, metrics = training_step(state, env, env.default_params, config)
    assert not bool(metrics.updated)
    assert bool(metrics.target_updated)
    np.testing.assert_allclose(state.target.mlp.layers[0].bias[...], initial + 1, atol=1e-7)


def test_evaluation_collects_entire_final_vector_step() -> None:
    env = EpisodeStatisticsWrapper(OneStepEnv())
    config = DQNTrainConfig(num_envs=3)
    state = initialize_run(env, env.default_params, dqn_config(1, 2), config, jax.random.key(0))
    evaluate = nnx.jit(partial(evaluate_dqn, env=env, params=env.default_params, num_envs=3, episodes=2))
    result = evaluate(state.model, state.key)
    assert int(result.count) == 3
    np.testing.assert_array_equal(result.returns[:3], [2, 2, 2])


def tiny_config(num_envs: int = 2) -> DQNTrainConfig:
    return DQNTrainConfig(
        num_envs=num_envs,
        total_timesteps=32,
        buffer_size=7,
        batch_size=3,
        learning_starts=2,
        train_frequency=3,
        target_network_frequency=5,
    )


def _run_chunks(chunk_steps: int) -> RunState[object]:
    config = tiny_config()
    env = EpisodeStatisticsWrapper(Gridworld())
    state = initialize_run(env, env.default_params, dqn_config(2, 4, hidden_widths=(4,)), config, jax.random.key(12))
    run = nnx.jit(partial(run_chunk, env=env, params=env.default_params, config=config, chunk_steps=chunk_steps))
    remaining = config.vector_steps
    while remaining:
        active = min(remaining, chunk_steps)
        state, _ = run(state, jnp.asarray(active, jnp.int32))
        remaining -= active
    return cast(RunState[object], state)  # Test compares arrays without inspecting the concrete environment state.
