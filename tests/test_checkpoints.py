"""Stage checkpoints restore the whole training state exactly, and refuse different settings."""

import os
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- isolated local regression test argv.
import sys
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jax_shapley.algorithms import dqn
from jax_shapley.common import checkpoints
from jax_shapley.envs import Gridworld
from jax_shapley.envs.wrappers import EpisodeStatisticsWrapper

CONFIG = dqn.DQNConfig(
    env=Gridworld(),
    hidden_widths=(4,),
    num_envs=2,
    total_timesteps=8,
    buffer_size=3,
    batch_size=2,
    learning_starts=0,
    train_frequency=1,
)
SETTINGS = {"stage": "algorithm", "env": "Gridworld", "config": {"hidden_widths": [4]}}


@pytest.mark.parametrize("implementation", ["threefry2x32", "rbg", "unsafe_rbg"])
def test_restored_runs_continue_exactly_across_prng_defaults(tmp_path: Path, implementation: str) -> None:
    """Every leaf restores, the key keeps its saved PRNG implementation under another default,
    and training continues from the restored state exactly as from the saved one."""
    advance = nnx.jit(partial(_steps, CONFIG, 2))
    state = advance(dqn.initialize(CONFIG, jax.random.key(1, impl=implementation)))
    checkpoints.save(tmp_path / "run", state, SETTINGS)
    other = "rbg" if implementation == "threefry2x32" else "threefry2x32"
    with jax.default_prng_impl(other):
        template = dqn.initialize(CONFIG, jax.random.key(99))
        assert str(jax.random.key_impl(template.key)) == other
        restored = checkpoints.load(tmp_path / "run", template, SETTINGS)
    assert str(jax.random.key_impl(restored.key)) == implementation
    _assert_equal(state, restored)
    _assert_equal(advance(state), advance(restored))


def test_loading_refuses_different_settings_and_shapes(tmp_path: Path) -> None:
    state = dqn.initialize(CONFIG, jax.random.key(0))
    checkpoints.save(tmp_path / "run", state, SETTINGS)
    with pytest.raises(ValueError, match="different settings"):
        checkpoints.load(tmp_path / "run", state, {**SETTINGS, "config": {"hidden_widths": [8]}})
    wider = dqn.initialize(dqn.DQNConfig(env=Gridworld(), hidden_widths=(8,), buffer_size=3), jax.random.key(0))
    with pytest.raises(ValueError, match="not compatible with the stored shape"):
        checkpoints.load(tmp_path / "run", wider, SETTINGS)
    with pytest.raises(FileExistsError):
        checkpoints.save(tmp_path / "run", state, SETTINGS)  # A checkpoint is written once, into a new folder.


def test_restored_arrays_follow_the_destination_device(tmp_path: Path) -> None:
    # The device count must be set before backend initialization: run in an isolated process.
    environment = {**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": "--xla_force_host_platform_device_count=2"}
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] -- fixed local Python module argv.
        [sys.executable, "-m", "tests.test_checkpoints", str(tmp_path)],
        env=environment,
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _steps(config: dqn.DQNConfig, count: int, state: dqn.RunState) -> dqn.RunState:
    env = EpisodeStatisticsWrapper(config.env)
    for index in range(count):
        state, _ = dqn.step(config, env, None, jnp.asarray(0), state, jnp.asarray(index))
    return state


def _assert_equal(left: dqn.RunState, right: dqn.RunState) -> None:
    pairs = zip(
        jax.tree.leaves((nnx.state((left.model, left.target, left.optimizer)), left.replay, left.observations)),
        jax.tree.leaves((nnx.state((right.model, right.target, right.optimizer)), right.replay, right.observations)),
        strict=True,
    )
    for a, b in pairs:
        np.testing.assert_array_equal(a, b)
    for a, b in zip(jax.tree.leaves(left.env_state), jax.tree.leaves(right.env_state), strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(jax.random.key_data(left.key), jax.random.key_data(right.key))
    assert (int(left.vector_step), int(left.updates)) == (int(right.vector_step), int(right.updates))


def _verify_placement(folder: Path) -> None:
    source, destination = jax.devices("cpu")
    with jax.default_device(source):
        state = nnx.jit(partial(_steps, CONFIG, 2))(dqn.initialize(CONFIG, jax.random.key(1, impl="rbg")))
        checkpoints.save(folder / "run", state, SETTINGS)
    with jax.default_device(destination):
        restored = checkpoints.load(folder / "run", dqn.initialize(CONFIG, jax.random.key(2)), SETTINGS)
    leaves = jax.tree.leaves((nnx.state(restored.model), restored.replay, jax.random.key_data(restored.key)))
    assert all(leaf.devices() == {destination} for leaf in leaves), "restored arrays left the destination device"
    _assert_equal(state, restored)


if __name__ == "__main__":
    _verify_placement(Path(sys.argv[1]))
