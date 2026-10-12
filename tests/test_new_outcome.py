"""Off-policy outcome characteristics and their Shapley models (new_jax_shapley) reproduce jax_shapley bitwise.

Mastermind-222 at small scale: an agent, a behaviour characteristic sweep, off-policy outcome
characteristics conditioned on behaviour seed 1 (exact values conditioned on the exact or the
learnt behaviour characteristic), and Shapley models explaining outcome seed 1 against the
exact or the learnt outcome values.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from jax_shapley import train_dqn, train_explanations
from new_jax_shapley.algorithms.dqn import DQNConfig
from new_jax_shapley.characteristic_models.behaviour import CharacteristicConfig
from new_jax_shapley.characteristic_models.outcome import OutcomeConfig, OutcomeTrainConfig
from new_jax_shapley.common.supervised_learning import EpochMetrics, SupervisedConfig
from new_jax_shapley.envs import Mastermind
from new_jax_shapley.experiment import ExperimentConfig
from new_jax_shapley.main import ExperimentResult, run
from new_jax_shapley.shapley_models.behaviour import ShapleyConfig

SEEDS = (1, 2)
WIDTHS = (16, 16)
ENV = Mastermind(2, 2, 2)
TRAINING = SupervisedConfig(epochs=3, batch_size=16, buffer_size=160, num_envs=4)
OUTCOME = OutcomeTrainConfig(epochs=3, batch_size=16, train_size=512, validation_size=40)


def test_outcome_matches_the_current_cli(pipelines: tuple[ExperimentResult, Path]) -> None:
    result, root = pipelines
    _assert_epochs(_single(root / "outcome"), result.outcome_metrics)


@pytest.mark.parametrize(("index", "exact"), [(0, "with"), (1, "without")])
def test_outcome_shapley_matches_the_current_cli(
    pipelines: tuple[ExperimentResult, Path], index: int, exact: str
) -> None:
    result, root = pipelines
    _assert_epochs(_single(root / f"shapley-{exact}"), result.shapley[index][1])


@pytest.fixture(scope="module", params=[True, False], ids=["exact-behaviour", "learnt-behaviour"])
def pipelines(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[ExperimentResult, Path]:
    """The new pipeline in process, and the current CLIs with the same settings."""
    with_exact_char: bool = request.param
    config = ExperimentConfig(
        name="test_outcome",
        algorithm=DQNConfig(
            env=ENV,
            hidden_widths=WIDTHS,
            num_envs=4,
            total_timesteps=1024,
            buffer_size=1024,
            learning_starts=128,
            batch_size=16,
            exploration_fraction=0.25,
        ),
        characteristic=CharacteristicConfig(env=ENV, hidden_widths=WIDTHS, seeds=SEEDS, train=TRAINING),
        outcome=OutcomeConfig(
            env=ENV, with_exact_char=with_exact_char, tau=0.01, hidden_widths=WIDTHS, seeds=SEEDS, train=OUTCOME
        ),
        shapley=tuple(
            ShapleyConfig(env=ENV, hidden_widths=WIDTHS, with_exact_char=exact, seeds=SEEDS, train=TRAINING)
            for exact in (True, False)
        ),
    )
    root = tmp_path_factory.mktemp("old")
    dqn = config.algorithm
    train_dqn.main([
        "mastermind222",
        "--seed",
        "1",
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
    explanation = [
        "--seeds",
        *map(str, SEEDS),
        "--hidden-widths",
        *map(str, WIDTHS),
        "--train.epochs",
        "3",
        "--train.batch-size",
        "16",
        "--train.buffer-size",
        "160",
        "--train.num-envs",
        "4",
    ]
    train_explanations.main([
        "characteristic",
        "--agent",
        str(agent),
        "--kind",
        "behaviour",
        *explanation,
        "--output",
        str(root / "characteristic"),
    ])
    train_explanations.main([
        "outcome",
        "--agent",
        str(agent),
        "--characteristic",
        str(_single(root / "characteristic") / "seed-1"),
        "--variant",
        "off_policy",
        f"--{'' if with_exact_char else 'no-'}with-exact-char",
        "--tau",
        "0.01",
        "--seeds",
        *map(str, SEEDS),
        "--hidden-widths",
        *map(str, WIDTHS),
        "--train.epochs",
        str(OUTCOME.epochs),
        "--train.batch-size",
        str(OUTCOME.batch_size),
        "--train.train-size",
        str(OUTCOME.train_size),
        "--train.validation-size",
        str(OUTCOME.validation_size),
        "--output",
        str(root / "outcome"),
    ])
    for exact in ("with", "without"):
        train_explanations.main([
            "shapley",
            "--agent",
            str(agent),
            "--characteristic",
            str(_single(root / "outcome") / "seed-1"),
            "--buffers",
            "fresh",
            f"--{'' if exact == 'with' else 'no-'}with-exact-char",
            *explanation,
            "--output",
            str(root / f"shapley-{exact}"),
        ])
    return run(config), root


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
