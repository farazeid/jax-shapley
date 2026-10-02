"""Abstract compilation of production fresh/resumed paths and optional effects."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from jax_shapley.envs import Gridworld, Hypercube, Mastermind222
from jax_shapley.networks import dqn_config
from jax_shapley.train_dqn import DQNTrainConfig, initialize_run, run
from jax_shapley.wrappers import EnvironmentProtocol, EpisodeStatisticsWrapper
from tests.dqn_support import RecordingSink


@pytest.mark.compile
@pytest.mark.parametrize("episodes", [0, 2])
def test_complete_run_abstract_compile(episodes: int) -> None:
    _compile(Mastermind222, episodes)


@pytest.mark.compile
def test_gridworld_complete_run_abstract_compile() -> None:
    _compile(Gridworld, 2)


@pytest.mark.compile
def test_hypercube_complete_run_abstract_compile() -> None:
    _compile(partial(Hypercube, side_length=3, dimension=3), 2)


@pytest.mark.compile
@pytest.mark.parametrize("implementation", ["rbg", "unsafe_rbg"])
def test_complete_run_prng_compile(implementation: str) -> None:
    _compile(Mastermind222, 2, implementation=implementation)


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
        _compile(Mastermind222, 2)


def _compile[State, Params](
    factory: Callable[[], EnvironmentProtocol[State, Params]],
    episodes: int,
    *,
    implementation: str = "threefry2x32",
) -> None:
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
    env = EpisodeStatisticsWrapper(factory())
    network = dqn_config(env.observation_space(env.default_params).shape[0], env.num_actions, hidden_widths=(4,))
    key_spec = jax.ShapeDtypeStruct((), jax.random.key(1, impl=implementation).dtype)
    fresh = nnx.jit(
        partial(
            run,
            environment_factory=factory,
            hidden_widths=(4,),
            config=config,
            chunk_steps=3,
            sink=sink,
        )
    )
    fresh.trace(key_spec, None, None, None).lower().compile()
    abstract = nnx.eval_shape(partial(initialize_run, env, env.default_params, network, config), key_spec)
    resumed = nnx.jit(
        partial(
            run,
            environment_factory=factory,
            hidden_widths=(4,),
            config=config,
            chunk_steps=3,
            sink=sink,
        )
    )
    scalar = jax.ShapeDtypeStruct((), jnp.int32)
    resumed.trace(key_spec, abstract, scalar, scalar).lower().compile()
    assert not sink.training_steps and not sink.checkpoints and not sink.evaluation_returns
