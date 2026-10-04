"""Continual runs traced, lowered and compiled at the source's parallel-driver shapes."""

from functools import partial

import jax
import pytest
from flax import nnx

from jax_shapley.tabular import transition_model
from jax_shapley.train_continual import ContinualConfig, ContinualKind, ContinualRunConfig, run_continual
from jax_shapley.train_dqn import DQNTrainConfig, GridworldConfig, Mastermind222Config

SEEDS = 3
KEYS = jax.ShapeDtypeStruct((SEEDS,), jax.random.key(0).dtype)


@pytest.mark.compile
@pytest.mark.parametrize("case", ["gridworld", "mastermind222"])
@pytest.mark.parametrize("kind", ["behaviour", "prediction", "outcome"])
@pytest.mark.parametrize("update_ratio", [1, 50])
def test_continual_runs_compile(case: str, kind: ContinualKind, update_ratio: int) -> None:
    """experiments/parallel: 100k steps over 16 environments, batch 64, exact buffers of 10,000."""
    training = DQNTrainConfig(
        num_envs=16,
        total_timesteps=100_000,
        buffer_size=100_000,
        exploration_fraction=0.5 if case == "gridworld" else 0.25,
        learning_starts=0,
    )
    explain = ContinualConfig(kind, update_ratio, exact_frequency=5_000 if kind == "outcome" else 1_000)
    environment = GridworldConfig() if case == "gridworld" else Mastermind222Config()
    config = ContinualRunConfig(environment, training, explain, seeds=tuple(range(1, SEEDS + 1)))
    run = partial(run_continual, config=config, tabular=transition_model(environment))
    nnx.jit(run).trace(KEYS).lower().compile()
