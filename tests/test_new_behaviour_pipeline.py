"""The refactored behaviour pipeline (new_jax_shapley) reproduces the current jax_shapley CLIs bitwise.

A scaled-down Mastermind-222 behaviour experiment runs through both: the agent's weights
and every logged DQN, characteristic and Shapley metric must agree.
"""

import csv
from pathlib import Path

import jax
import numpy as np
import pytest
from flax import nnx

from jax_shapley import train_dqn, train_explanations
from jax_shapley.run_io import load_agent, load_settings
from new_jax_shapley.algorithms.dqn import DQNConfig
from new_jax_shapley.characteristic_models.behaviour import CharacteristicConfig
from new_jax_shapley.common.supervised_learning import EpochMetrics, SupervisedConfig, seed_slice
from new_jax_shapley.envs import Mastermind
from new_jax_shapley.experiment import ExperimentConfig
from new_jax_shapley.main import ExperimentResult, run
from new_jax_shapley.shapley_models.behaviour import ShapleyConfig

SEEDS = (1, 2)
WIDTHS = (16, 16)
TRAINING = SupervisedConfig(epochs=3, batch_size=16, buffer_size=160, num_envs=4)
ENV = Mastermind(2, 2, 2)
CONFIG = ExperimentConfig(
    name="test_behaviour",
    algorithm=DQNConfig(
        env=ENV,
        seeds=(1,),
        hidden_widths=WIDTHS,
        num_envs=4,
        total_timesteps=1024,
        buffer_size=1024,
        learning_starts=128,
        train_frequency=10,
        batch_size=16,
        exploration_fraction=0.25,
    ),
    characteristic=CharacteristicConfig(env=ENV, hidden_widths=WIDTHS, seeds=SEEDS, train=TRAINING),
    shapley=tuple(
        ShapleyConfig(env=ENV, hidden_widths=WIDTHS, with_exact_char=exact, seeds=SEEDS, train=TRAINING)
        for exact in (True, False)
    ),
)
EXPLANATION_ARGUMENTS = [
    "--hidden-widths",
    *map(str, WIDTHS),
    "--seeds",
    *map(str, SEEDS),
    "--train.epochs",
    "3",
    "--train.batch-size",
    "16",
    "--train.buffer-size",
    "160",
    "--train.num-envs",
    "4",
]


def test_agent_matches_the_current_dqn(pipelines: tuple[ExperimentResult, Path]) -> None:
    result, root = pipelines
    agent = _single(root / "dqn")
    loaded = load_agent(agent, load_settings(agent / "config.json").network, load_replay=False)
    old = jax.tree.leaves(nnx.state(loaded.model))
    new = jax.tree.leaves(nnx.state(seed_slice(result.dqn_state, 0).model))
    for left, right in zip(old, new, strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    rows = list(csv.DictReader((agent / "metrics.csv").open()))
    old_returns = [float(row["episode_return"]) for row in rows if row["kind"] == "episode"]
    assert result.dqn_metrics is not None
    metrics = seed_slice(result.dqn_metrics, 0)
    new_returns = np.asarray(metrics.episode_return)[np.asarray(metrics.completed)].tolist()
    assert old_returns == new_returns


def test_characteristic_matches_the_current_sweep(pipelines: tuple[ExperimentResult, Path]) -> None:
    result, root = pipelines
    _assert_epochs(_single(root / "characteristic"), result.characteristic_metrics)


@pytest.mark.parametrize(("index", "exact"), [(0, "with"), (1, "without")])
def test_shapley_matches_the_current_sweep(pipelines: tuple[ExperimentResult, Path], index: int, exact: str) -> None:
    result, root = pipelines
    _assert_epochs(_single(root / f"shapley-{exact}"), result.shapley[index][1])


@pytest.fixture(scope="module")
def pipelines(tmp_path_factory: pytest.TempPathFactory) -> tuple[ExperimentResult, Path]:
    """The new pipeline in process, and the current CLIs with the same settings."""
    root = tmp_path_factory.mktemp("old")
    dqn = CONFIG.algorithm
    train_dqn.main([
        "mastermind222",
        "--seed",
        str(dqn.seeds[0]),
        "--hidden-widths",
        *map(str, WIDTHS),
        "--train.num-envs",
        str(dqn.num_envs),
        "--train.total-timesteps",
        str(dqn.total_timesteps),
        "--train.buffer-size",
        str(dqn.buffer_size),
        "--train.learning-starts",
        str(dqn.learning_starts),
        "--train.train-frequency",
        str(dqn.train_frequency),
        "--train.batch-size",
        str(dqn.batch_size),
        "--train.exploration-fraction",
        str(dqn.exploration_fraction),
        "--train.eval-episodes",
        "0",
        "--output",
        str(root / "dqn"),
    ])
    agent = _single(root / "dqn")
    train_explanations.main([
        "characteristic",
        "--agent",
        str(agent),
        "--kind",
        "behaviour",
        "--policy",
        "greedy",
        *EXPLANATION_ARGUMENTS,
        "--output",
        str(root / "characteristic"),
    ])
    characteristic = _single(root / "characteristic") / "seed-1"
    for exact in ("with", "without"):
        train_explanations.main([
            "shapley",
            "--agent",
            str(agent),
            "--characteristic",
            str(characteristic),
            "--buffers",
            "fresh",
            f"--{'' if exact == 'with' else 'no-'}with-exact-char",
            *EXPLANATION_ARGUMENTS,
            "--output",
            str(root / f"shapley-{exact}"),
        ])
    return run(CONFIG), root


def _assert_epochs(sweep: Path, metrics: EpochMetrics | None) -> None:
    assert metrics is not None
    names = ("epoch", "step", "train_loss", "validation_loss", "exact_loss", "best_epoch")
    active = np.asarray(metrics.active)
    for index, seed in enumerate(SEEDS):
        rows = list(csv.DictReader((sweep / f"seed-{seed}" / "metrics.csv").open()))
        old = np.asarray([[float(row[name]) for name in names] for row in rows])
        epochs = np.flatnonzero(active[:, index])
        new = np.stack([np.asarray(getattr(metrics, name))[epochs, index] for name in names], axis=1)
        np.testing.assert_array_equal(new, old)


def _single(folder: Path) -> Path:
    (run_folder,) = [path for path in folder.iterdir() if path.is_dir()]
    return run_folder
