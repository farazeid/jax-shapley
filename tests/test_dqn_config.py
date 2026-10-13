"""Static DQN settings fail on construction, before allocating replay or compiling a run."""

from dataclasses import FrozenInstanceError, replace

import jax
import pytest

from jax_shapley.algorithms import dqn
from jax_shapley.common.networks import AdamConfig
from jax_shapley.envs import Gridworld, Hypercube, Mastermind

GRIDWORLD = dqn.DQNConfig(env=Gridworld())


def test_upstream_gridworld_defaults_and_floor_division() -> None:
    config = GRIDWORLD
    assert (config.learning_rate, config.gamma, config.tau) == (2.5e-4, 1.0, 0.01)
    assert (config.buffer_size, config.total_timesteps, config.batch_size) == (50_000, 50_000, 64)
    assert (config.start_e, config.end_e, config.exploration_fraction) == (1, 0.01, 0.5)
    assert (config.learning_starts, config.train_frequency, config.target_network_frequency) == (10_000, 10, 1)
    assert config.adam == AdamConfig(0.9, 0.999, 1e-8)
    assert replace(config, num_envs=16, total_timesteps=35).vector_steps == 2
    assert replace(config, buffer_size=1, batch_size=64).buffer_size == 1
    with pytest.raises(FrozenInstanceError):
        config.__setattr__("gamma", 0.5)


@pytest.mark.parametrize("value", [0, -1, 2**31])
def test_invalid_counts(value: int) -> None:
    for name in ("buffer_size", "num_envs", "total_timesteps", "batch_size", "train_frequency"):
        with pytest.raises(ValueError):
            replace(GRIDWORLD, **{name: value})


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_probabilities(value: float) -> None:
    for name in ("gamma", "tau", "start_e", "end_e"):
        with pytest.raises(ValueError):
            replace(GRIDWORLD, **{name: value})


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_positive_settings(value: float) -> None:
    with pytest.raises(ValueError):
        replace(GRIDWORLD, learning_rate=value)
    with pytest.raises(ValueError):
        replace(GRIDWORLD, exploration_fraction=value)
    with pytest.raises(ValueError):
        AdamConfig(epsilon=value)


def test_invalid_optimizer_and_run_settings() -> None:
    with pytest.raises(ValueError):
        AdamConfig(beta1=1)
    with pytest.raises(ValueError):
        replace(GRIDWORLD, num_envs=16, total_timesteps=1)
    with pytest.raises(ValueError):
        replace(GRIDWORLD, learning_starts=-1)
    with pytest.raises(ValueError):
        replace(GRIDWORLD, eval_episodes=-1)
    with pytest.raises(ValueError):
        replace(GRIDWORLD, hidden_widths=(0,))
    with pytest.raises(ValueError):
        replace(GRIDWORLD, seeds=())
    with pytest.raises(ValueError):
        replace(GRIDWORLD, seeds=(1, 1))
    with pytest.raises(ValueError):
        Hypercube(7, 2)


def test_network_dimensions_come_from_environments() -> None:
    for env, features, actions in ((Gridworld(), 2, 4), (Mastermind(2, 2, 2), 8, 4), (Hypercube(3, 5), 5, 10)):
        config = dqn.DQNConfig(env=env, hidden_widths=(16, 8))
        network = dqn.initialize(config, jax.random.key(0)).model.mlp.config
        assert network.input_features == features and network.output_features == actions
        assert [layer.features for layer in network.layers] == [16, 8, actions]
        assert [layer.activation for layer in network.layers] == ["relu", "relu", "identity"]
