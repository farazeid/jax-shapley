"""DQN collection, replay insertion, timing and evaluation semantics on small runs."""

from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from jax_shapley.algorithms import dqn
from jax_shapley.envs import Gridworld, Mastermind
from jax_shapley.envs.wrappers import EpisodeStatisticsWrapper

TINY = dqn.DQNConfig(
    env=Gridworld(),
    hidden_widths=(4,),
    num_envs=2,
    total_timesteps=32,
    buffer_size=7,
    batch_size=3,
    learning_starts=2,
    train_frequency=3,
    target_network_frequency=5,
)


def test_upstream_vector_timing_and_target_order() -> None:
    initial = dqn.initialize(TINY, jax.random.key(0)).model.mlp.layers[0].kernel[...].copy()
    state, metrics = nnx.jit(partial(dqn.train, TINY, None, None))(jax.random.key(0), jnp.asarray(0))
    steps = np.arange(TINY.vector_steps) * TINY.num_envs
    np.testing.assert_array_equal(metrics.global_step, steps)
    np.testing.assert_array_equal(metrics.updated, (steps > 2) & (steps % 3 < 2))
    np.testing.assert_array_equal(metrics.target_updated, (steps > 2) & (steps % 5 < 2))
    assert int(state.updates) == int(np.count_nonzero(np.asarray(metrics.updated)))
    assert int(state.target_updates) == int(np.count_nonzero(np.asarray(metrics.target_updated)))
    assert int(state.vector_step) == TINY.vector_steps
    assert int(state.replay.size) == 7
    assert not jnp.array_equal(initial, state.model.mlp.layers[0].kernel[...])
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(nnx.state(state.model)))


def test_repeated_seed_is_deterministic() -> None:
    program = nnx.jit(partial(dqn.train, TINY, None, None))
    first, _ = program(jax.random.key(12), jnp.asarray(0))
    second, _ = program(jax.random.key(12), jnp.asarray(0))
    for left, right in zip(
        jax.tree.leaves((nnx.state(first.model), nnx.state(first.target), first.replay, first.observations)),
        jax.tree.leaves((nnx.state(second.model), nnx.state(second.target), second.replay, second.observations)),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(jax.random.key_data(first.key), jax.random.key_data(second.key))


def test_partial_wraparound_and_oversized_insertion() -> None:
    """Upstream stores only the last `capacity` items of an insertion, starting at the old pointer."""
    env = EpisodeStatisticsWrapper(TINY.env)
    for num_envs, capacity, pointers in ((3, 5, [3, 1, 4]), (8, 5, [0, 0, 0])):
        config = replace(TINY, num_envs=num_envs, buffer_size=capacity, learning_starts=1_000)
        state = dqn.initialize(config, jax.random.key(1))
        for expected_pointer in pointers:
            pointer, observations = int(state.replay.pointer), np.asarray(state.observations)
            state, _ = dqn.step(config, env, None, jnp.asarray(0), state, jnp.asarray(0))
            count = min(num_envs, capacity)
            rows = (pointer + np.arange(count)) % capacity
            np.testing.assert_array_equal(state.replay.data.observations[rows], observations[-count:])
            assert int(state.replay.pointer) == expected_pointer
        assert int(state.replay.size) == capacity
        assert state.replay.pointer.dtype == state.replay.size.dtype == jnp.int32


def test_replay_stores_raw_next_observations_while_acting_from_resets() -> None:
    """Mastermind-222 episodes end within two guesses: the replay keeps each final board, while
    the next decision starts from the auto-reset empty board."""
    config = dqn.DQNConfig(env=Mastermind(2, 2, 2), hidden_widths=(4,), num_envs=4, total_timesteps=8)
    env = EpisodeStatisticsWrapper(config.env)
    state = dqn.initialize(config, jax.random.key(3))
    reset = np.asarray(state.observations[0])
    ended = np.zeros(config.num_envs, bool)
    for index in range(2):
        state, metrics = dqn.step(config, env, None, jnp.asarray(0), state, jnp.asarray(index))
        completed = np.asarray(metrics.completed)
        rows = index * config.num_envs + np.flatnonzero(completed)
        np.testing.assert_array_equal(np.asarray(state.observations)[completed], np.broadcast_to(reset, (len(rows), 8)))
        assert not (np.asarray(state.replay.data.next_observations)[rows] == reset).all(axis=1).any()
        assert (np.asarray(metrics.episode_length)[completed] <= index + 1).all()  # Correct first guesses end early.
        ended |= completed
    assert ended.all()


def test_collection_matches_gymnax_same_key() -> None:
    env = EpisodeStatisticsWrapper(dqn.DQNConfig(env=Mastermind(2, 2, 2)).env)
    key = jax.random.key(0)
    _, state = env.reset(key, env.default_params)
    action = jnp.asarray(0, jnp.int32)
    expected = env.step(key, state, action, env.default_params)
    actual = dqn.environment_step(env, key, state, action, env.default_params)
    np.testing.assert_array_equal(actual.observations, expected[0])
    for left, right in zip(jax.tree.leaves(actual.state), jax.tree.leaves(expected[1]), strict=True):
        np.testing.assert_array_equal(left, right)
    raw, _, _, _, _ = env.env.step_env(jax.random.split(key)[0], state.env_state, action, env.default_params)
    np.testing.assert_array_equal(actual.raw_observations, raw)


def test_target_update_independent_of_learning_frequency() -> None:
    config = replace(
        TINY,
        hidden_widths=(),
        num_envs=1,
        total_timesteps=12,
        buffer_size=4,
        batch_size=2,
        learning_starts=0,
        train_frequency=10,
        target_network_frequency=1,
        tau=0.5,
    )
    env = EpisodeStatisticsWrapper(config.env)
    state = dqn.initialize(config, jax.random.key(0))
    state.model.mlp.layers[0].bias[...] += 2
    initial = state.target.mlp.layers[0].bias[...].copy()
    state, metrics = dqn.step(config, env, None, jnp.asarray(0), state, jnp.asarray(0))
    assert not bool(metrics.updated) and not bool(metrics.target_updated)  # Strict warmup, even at zero.
    state, metrics = dqn.step(config, env, None, jnp.asarray(0), state, jnp.asarray(1))
    assert not bool(metrics.updated)
    assert bool(metrics.target_updated)
    np.testing.assert_allclose(state.target.mlp.layers[0].bias[...], initial + 1, atol=1e-7)


def test_evaluation_collects_entire_final_vector_step() -> None:
    """Every Mastermind-222 episode ends on its second step, so asking for 2 of 3 environments'
    episodes records all 3, as the source does."""
    config = dqn.DQNConfig(env=Mastermind(2, 2, 2), hidden_widths=(4,), num_envs=3, eval_episodes=2)
    state = dqn.initialize(config, jax.random.key(0))
    recorder = Recorder()
    log = partial(dqn.log_evaluation, [recorder])
    evaluated = nnx.jit(partial(dqn.evaluate, config, log))(state, jnp.asarray(0))
    jax.effects_barrier()
    assert sorted(recorder.steps) == [0, 1, 2]
    assert not jnp.array_equal(jax.random.key_data(evaluated.key), jax.random.key_data(state.key))


class Recorder:
    """A run that keeps the steps it was logged at."""

    def __init__(self) -> None:
        self.steps: list[int] = []

    def log(self, data: dict[str, float], /, step: int | None = None) -> None:
        assert step is not None and set(data) == {"agent/eval_episodic_return"}
        self.steps.append(step)
