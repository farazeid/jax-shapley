"""Destination placement with two CPU devices in an isolated JAX process."""

import os
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- isolated local regression test argv.
import sys
from functools import partial
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import Array

from jax_shapley.algorithms.dqn import initialize_run, make_snapshot, run_chunk
from jax_shapley.artifacts import describe
from jax_shapley.checkpoints import CheckpointMetadata, CheckpointStore, load_agent, load_replay, restore_run
from jax_shapley.config import DQNRunConfig, DQNTrainConfig, GridworldConfig
from jax_shapley.wrappers import EpisodeStatisticsWrapper


def test_checkpoint_loaders_preserve_destination_sharding(tmp_path: Path) -> None:
    # Device count must be set before backend initialization; leave the test
    # suite's own JAX backend and environment unchanged.
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] -- fixed local Python module argv.
        [sys.executable, "-m", "tests.test_dqn_checkpoint_devices", str(tmp_path)],
        env=environment,
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _verify_placement(folder: Path) -> None:
    source, destination = jax.devices("cpu")
    config = DQNRunConfig(
        GridworldConfig(),
        train=DQNTrainConfig(total_timesteps=8, buffer_size=3, batch_size=2, learning_starts=0, train_frequency=1),
        hidden_widths=(4,),
    )
    with jax.default_device(source):
        env = EpisodeStatisticsWrapper(config.env.make())
        state = initialize_run(env, env.default_params, config.network, config.train, jax.random.key(1, impl="rbg"))
        state, _ = nnx.jit(partial(run_chunk, env=env, params=env.default_params, config=config.train, chunk_steps=2))(
            state, jnp.asarray(2, jnp.int32)
        )
        zero = jnp.asarray(0, jnp.int32)
        snapshot = make_snapshot(state, zero, zero, config.train, "run")
    store = CheckpointStore(folder)
    try:
        store.save(snapshot, CheckpointMetadata(describe(config), 2, 0, 0, key_implementation="rbg"))
        store.wait()
    finally:
        store.close()

    with jax.default_device(destination):
        agent = load_agent(folder, config.network)
        replay = load_replay(folder)
        target_env = EpisodeStatisticsWrapper(config.env.make())
        # Different PRNG shapes must still preserve the template key's destination.
        template = initialize_run(
            target_env, target_env.default_params, config.network, config.train, jax.random.key(99, impl="threefry2x32")
        )
    # Restoration must follow the supplied template, not this default context.
    with jax.default_device(source):
        store = CheckpointStore(folder, read_only=True)
        try:
            restored, _ = restore_run(store, template, config)
        finally:
            store.close()
    assert agent.replay is not None
    _assert_placement(nnx.state((agent.model, agent.target)), destination)
    _assert_placement((agent.replay, replay), destination)
    _assert_placement(nnx.state((restored.model, restored.target, restored.optimizer)), destination)
    _assert_placement(
        (
            restored.replay,
            restored.observations,
            restored.env_state,
            restored.vector_step,
            restored.updates,
            restored.target_updates,
            jax.random.key_data(restored.key),
        ),
        destination,
    )
    assert str(jax.random.key_impl(restored.key)) == "rbg"
    for left, right in zip(_arrays(state.replay), _arrays(restored.replay), strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(jax.random.key_data(state.key), jax.random.key_data(restored.key))
    observations = jax.device_put(state.observations, destination)
    prediction = nnx.jit(lambda model, obs: model(obs))
    compiled = prediction.trace(agent.model, observations).lower().compile()
    actual = compiled(agent.model, observations).q_values
    assert actual.device == destination
    np.testing.assert_array_equal(actual, state.model(state.observations).q_values)


def _assert_placement(tree: object, device: jax.Device) -> None:
    assert all(value.sharding == jax.sharding.SingleDeviceSharding(device) for value in _arrays(tree))


def _arrays(tree: object) -> tuple[Array, ...]:
    return tuple(cast(list[Array], jax.tree.leaves(tree)))  # Numerical NNX state and struct payloads only.


if __name__ == "__main__":
    _verify_placement(Path(sys.argv[1]))
