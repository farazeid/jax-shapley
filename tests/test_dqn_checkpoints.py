"""Orbax commits, typed partial loading and complete numerical continuation."""

from functools import partial
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax import Array

from jax_shapley.envs import Gridworld
from jax_shapley.run_io import (
    CheckpointMetadata,
    CheckpointStore,
    RunWriter,
    Snapshot,
    describe,
    initialize_metrics,
    load_agent,
    restore_run,
)
from jax_shapley.train_dqn import (
    DQNRunConfig,
    DQNTrainConfig,
    GridworldConfig,
    epsilon_at,
    initialize_run,
    make_snapshot,
    run_chunk,
)
from jax_shapley.wrappers import EpisodeStatisticsWrapper


def test_full_restoration_and_latest_retention(tmp_path: Path) -> None:
    config = DQNRunConfig(
        GridworldConfig(),
        train=DQNTrainConfig(total_timesteps=8, buffer_size=3, batch_size=2, learning_starts=0, train_frequency=1),
        hidden_widths=(4,),
    )
    env = EpisodeStatisticsWrapper(Gridworld())
    state = initialize_run(env, env.default_params, config.network, config.train, jax.random.key(0))
    state, _ = nnx.jit(partial(run_chunk, env=env, params=env.default_params, config=config.train, chunk_steps=4))(
        state, jnp.asarray(4, jnp.int32)
    )

    def leaves(tree: object) -> tuple[Array, ...]:
        return tuple(cast(list[Array], jax.tree.leaves(tree)))

    snapshot = Snapshot(
        leaves(nnx.state((state.model, state.target))),
        state.replay,
        leaves(nnx.state(state.optimizer)),
        leaves((
            state.observations,
            state.env_state,
            state.vector_step,
            state.updates,
            state.target_updates,
            jax.random.key_data(state.key),
        )),
        state.vector_step,
        jnp.asarray(0, jnp.int32),
        jnp.asarray(0, jnp.int32),
        epsilon_at(config.train, (state.vector_step - 1) * config.train.num_envs),
    )
    store = CheckpointStore(tmp_path)
    store.save(snapshot, CheckpointMetadata(describe(config), 4, 0, 0))
    store.wait()
    store.save(snapshot, CheckpointMetadata(describe(config), 4, 0, 0))
    store.wait()
    assert store.manager.all_steps() == [1]
    store.close()
    store = CheckpointStore(tmp_path, read_only=True)
    try:
        template = initialize_run(env, env.default_params, config.network, config.train, jax.random.key(77))
        restored, metadata = restore_run(store, template, config)
        assert metadata.vector_step == 4
        for left, right in zip(
            leaves(nnx.state((state.model, state.target, state.optimizer))),
            leaves(nnx.state((restored.model, restored.target, restored.optimizer))),
            strict=True,
        ):
            np.testing.assert_array_equal(left, right)
        for left, right in zip(leaves(state.replay), leaves(restored.replay), strict=True):
            np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(jax.random.key_data(state.key), jax.random.key_data(restored.key))
        assert all(value.device == jax.devices()[0] for value in leaves(restored.replay))
    finally:
        store.close()
    agent = load_agent(tmp_path, config.network)
    assert agent.replay is not None
    assert agent.vector_step == 4
    assert load_agent(tmp_path, config.network, load_replay=False).replay is None
    with pytest.raises(ValueError, match="architecture"):
        load_agent(tmp_path, DQNRunConfig(GridworldConfig()).network)
    # A staged/incomplete directory is not a committed checkpoint.
    (tmp_path / "checkpoints" / "999.orbax-checkpoint-tmp").mkdir()
    assert load_agent(tmp_path, config.network, load_replay=False).vector_step == 4


def test_metadata_validation_and_empty_store(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="schema"):
        CheckpointMetadata("{}", 0, 0, 0, schema=2).validate()
    with pytest.raises(ValueError, match="timing"):
        CheckpointMetadata("{}", 0, 0, 0, elapsed_seconds=float("nan")).validate()
    with pytest.raises(ValueError, match="PRNG"):
        CheckpointMetadata("{}", 0, 0, 0, key_implementation="unknown").validate()
    store = CheckpointStore(tmp_path)
    try:
        with pytest.raises(ValueError, match="committed"):
            store.metadata()
    finally:
        store.close()


@pytest.mark.parametrize("implementation", ["threefry2x32", "rbg", "unsafe_rbg"])
def test_restoration_preserves_prng_across_different_defaults(tmp_path: Path, implementation: str) -> None:
    config = DQNRunConfig(
        GridworldConfig(),
        train=DQNTrainConfig(total_timesteps=8, buffer_size=3, batch_size=2, learning_starts=0, train_frequency=1),
        hidden_widths=(4,),
    )
    env = EpisodeStatisticsWrapper(config.env.make())
    state = initialize_run(
        env, env.default_params, config.network, config.train, jax.random.key(1, impl=implementation)
    )
    advance = nnx.jit(partial(run_chunk, env=env, params=env.default_params, config=config.train, chunk_steps=2))
    state, _ = advance(state, jnp.asarray(2, jnp.int32))
    zero = jnp.asarray(0, jnp.int32)
    snapshot = make_snapshot(state, zero, zero, config.train, "run")
    assert snapshot.key_implementation == implementation
    initialize_metrics(tmp_path / "metrics.csv")
    sink = RunWriter(tmp_path, config)
    try:
        sink.checkpoint(snapshot)
    finally:
        sink.close()
    other_implementation = "rbg" if implementation == "threefry2x32" else "threefry2x32"
    with jax.default_prng_impl(other_implementation):
        template = initialize_run(env, env.default_params, config.network, config.train, jax.random.key(99))
        assert str(jax.random.key_impl(template.key)) == other_implementation
        store = CheckpointStore(tmp_path, read_only=True)
        try:
            restored, metadata = restore_run(store, template, config)
            assert metadata.key_implementation == implementation
        finally:
            store.close()
        assert str(jax.random.key_impl(restored.key)) == implementation
        np.testing.assert_array_equal(jax.random.key_data(restored.key), jax.random.key_data(state.key))
        expected, _ = advance(state, jnp.asarray(2, jnp.int32))
        actual, _ = advance(restored, jnp.asarray(2, jnp.int32))
    for left, right in zip(
        jax.tree.leaves(nnx.state((expected.model, expected.target, expected.optimizer))),
        jax.tree.leaves(nnx.state((actual.model, actual.target, actual.optimizer))),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)
    for left, right in zip(
        jax.tree.leaves((expected.replay, expected.observations, expected.env_state)),
        jax.tree.leaves((actual.replay, actual.observations, actual.env_state)),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(jax.random.key_data(expected.key), jax.random.key_data(actual.key))


def test_save_rejects_inconsistent_prng_metadata(tmp_path: Path) -> None:
    config = DQNRunConfig(GridworldConfig(), train=DQNTrainConfig(total_timesteps=8, buffer_size=3), hidden_widths=(4,))
    env = EpisodeStatisticsWrapper(config.env.make())
    state = initialize_run(env, env.default_params, config.network, config.train, jax.random.key(1, impl="rbg"))
    zero = jnp.asarray(0, jnp.int32)
    store = CheckpointStore(tmp_path)
    try:
        with pytest.raises(ValueError, match="PRNG metadata"):
            store.save(
                make_snapshot(state, zero, zero, config.train, "run"), CheckpointMetadata(describe(config), 0, 0, 0)
            )
        assert store.sequence is None
    finally:
        store.close()


@pytest.mark.parametrize("corruption", ["shape", "dtype", "nonfinite"])
def test_agent_loading_rejects_corrupt_arrays(tmp_path: Path, corruption: str) -> None:
    from jax_shapley.train_dqn import make_snapshot

    config = DQNRunConfig(GridworldConfig(), train=DQNTrainConfig(total_timesteps=8, buffer_size=3), hidden_widths=(4,))
    env = EpisodeStatisticsWrapper(Gridworld())
    state = initialize_run(env, env.default_params, config.network, config.train, jax.random.key(0))
    zero = jnp.asarray(0, jnp.int32)
    snapshot = make_snapshot(state, zero, zero, config.train, "run")
    value = snapshot.agent[0]
    if corruption == "shape":
        value = value[:-1]
    elif corruption == "dtype":
        value = value.astype(jnp.int32)
    else:
        value = value.at[0].set(jnp.nan)
    snapshot = Snapshot(
        (value, *snapshot.agent[1:]),
        snapshot.replay,
        snapshot.optimizer,
        snapshot.runtime,
        snapshot.vector_step,
        snapshot.phase,
        snapshot.evaluation_episodes,
        snapshot.epsilon,
    )
    store = CheckpointStore(tmp_path)
    try:
        store.save(snapshot, CheckpointMetadata(describe(config), 0, 0, 0))
        store.wait()
    finally:
        store.close()
    with pytest.raises(ValueError, match=r"shape or dtype|nonfinite"):
        load_agent(tmp_path, config.network, load_replay=False)
