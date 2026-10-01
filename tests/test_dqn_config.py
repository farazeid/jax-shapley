"""Static host contracts fail before allocating replay or compiling a run."""

from dataclasses import FrozenInstanceError, replace

import pytest

from jax_shapley.config import (
    AdamConfig,
    DQNRunConfig,
    DQNTrainConfig,
    GridworldConfig,
    HypercubeConfig,
    Mastermind222Config,
)


def test_upstream_gridworld_defaults_and_floor_division() -> None:
    config = DQNTrainConfig()
    assert (config.learning_rate, config.gamma, config.tau) == (2.5e-4, 1.0, 0.01)
    assert (config.buffer_size, config.total_timesteps, config.batch_size) == (50_000, 50_000, 64)
    assert (config.start_e, config.end_e, config.exploration_fraction) == (1, 0.01, 0.5)
    assert config.adam == AdamConfig(0.9, 0.999, 1e-8)
    assert replace(config, num_envs=16, total_timesteps=35).vector_steps == 2
    assert replace(config, buffer_size=1, batch_size=64).buffer_size == 1
    with pytest.raises(FrozenInstanceError):
        config.__setattr__("gamma", 0.5)


@pytest.mark.parametrize("value", [0, -1, 2**31])
def test_invalid_counts(value: int) -> None:
    with pytest.raises(ValueError):
        DQNTrainConfig(buffer_size=value)
    with pytest.raises(ValueError):
        DQNRunConfig(GridworldConfig(), chunk_steps=value)


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_probabilities(value: float) -> None:
    with pytest.raises(ValueError):
        DQNTrainConfig(gamma=value)
    with pytest.raises(ValueError):
        DQNTrainConfig(tau=value)
    with pytest.raises(ValueError):
        DQNTrainConfig(start_e=value)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_positive_settings(value: float) -> None:
    with pytest.raises(ValueError):
        DQNTrainConfig(learning_rate=value)
    with pytest.raises(ValueError):
        DQNTrainConfig(exploration_fraction=value)
    with pytest.raises(ValueError):
        AdamConfig(epsilon=value)


def test_invalid_optimizer_and_run_settings() -> None:
    with pytest.raises(ValueError):
        AdamConfig(beta1=1)
    with pytest.raises(ValueError):
        DQNTrainConfig(num_envs=16, total_timesteps=1)
    with pytest.raises(ValueError):
        DQNTrainConfig(learning_starts=-1)
    with pytest.raises(ValueError):
        DQNTrainConfig(eval_episodes=-1)
    with pytest.raises(ValueError):
        DQNRunConfig(GridworldConfig(), hidden_widths=(0,))
    with pytest.raises(ValueError):
        DQNRunConfig(GridworldConfig(), seed=-1)
    with pytest.raises(ValueError):
        DQNRunConfig(GridworldConfig(), validate_only=True, compile_only=True)
    with pytest.raises(ValueError):
        HypercubeConfig(7, 2)


def test_network_dimensions_come_from_environments() -> None:
    grid = DQNRunConfig(GridworldConfig()).network
    mastermind = DQNRunConfig(Mastermind222Config()).network
    cube = DQNRunConfig(HypercubeConfig(6, 5), hidden_widths=(128, 128)).network
    assert (grid.input_features, grid.output_features) == (2, 4)
    assert (mastermind.input_features, mastermind.output_features) == (8, 4)
    assert (cube.input_features, cube.output_features) == (5, 10)
    assert tuple(layer.features for layer in cube.layers) == (128, 128, 10)


def test_checkpoint_modes_and_resume_contracts() -> None:
    from pathlib import Path

    from jax_shapley.config import ResumeConfig

    assert DQNRunConfig(GridworldConfig()).checkpoint_mode == "run"
    for mode in ("none", "agent", "agent-replay", "run"):
        assert replace(DQNRunConfig(GridworldConfig()), checkpoint_mode=mode).checkpoint_mode == mode
    with pytest.raises(ValueError, match="mutually exclusive"):
        ResumeConfig(Path("runs/example"), validate_only=True, compile_only=True)
    resume = ResumeConfig(Path("runs/example"))
    with pytest.raises(FrozenInstanceError):
        resume.__setattr__("run", Path("other"))
