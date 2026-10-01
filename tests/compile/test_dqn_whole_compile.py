"""Abstract compilation of production fresh/resumed paths and optional effects."""

from functools import partial

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from jax_shapley.algorithms.dqn import continue_run, initialize_run, whole_run
from jax_shapley.config import CheckpointMode, DQNTrainConfig
from jax_shapley.envs import Mastermind222
from jax_shapley.networks import dqn_config
from jax_shapley.wrappers import EpisodeStatisticsWrapper
from tests.dqn_support import RecordingSink


@pytest.mark.compile
@pytest.mark.parametrize("mode", ["none", "agent", "agent-replay", "run"])
@pytest.mark.parametrize("episodes", [0, 2])
def test_complete_run_abstract_compile(mode: CheckpointMode, episodes: int) -> None:
    _compile(mode, episodes)


@pytest.mark.compile
@pytest.mark.gpu
def test_complete_run_gpu_compile() -> None:
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        devices = []
    if not devices:
        pytest.skip("whole-run DQN GPU compilation requires an available JAX GPU backend")
    with jax.default_device(devices[0]):
        for mode in ("none", "agent", "agent-replay", "run"):
            _compile(mode, 2)


def _compile(mode: CheckpointMode, episodes: int) -> None:
    sink = RecordingSink()
    config = DQNTrainConfig(
        num_envs=2,
        total_timesteps=16,
        buffer_size=7,
        batch_size=3,
        learning_starts=0,
        train_frequency=1,
        eval_episodes=episodes,
    )
    env = EpisodeStatisticsWrapper(Mastermind222())
    network = dqn_config(8, 4, hidden_widths=(4,))
    key_spec = jax.ShapeDtypeStruct((), jax.random.key(1).dtype)
    fresh = nnx.jit(
        partial(
            whole_run,
            env=env,
            params=env.default_params,
            network=network,
            config=config,
            chunk_steps=3,
            checkpoint_mode=mode,
            sink=sink,
        )
    )
    fresh.trace(key_spec).lower().compile()
    abstract = nnx.eval_shape(partial(initialize_run, env, env.default_params, network, config), key_spec)
    resumed = nnx.jit(
        partial(
            continue_run,
            env=env,
            params=env.default_params,
            config=config,
            chunk_steps=3,
            checkpoint_mode=mode,
            sink=sink,
        )
    )
    scalar = jax.ShapeDtypeStruct((), jnp.int32)
    resumed.trace(abstract, scalar, scalar).lower().compile()
    assert not sink.training_steps and not sink.checkpoints and not sink.evaluation_returns
