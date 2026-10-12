"""Outcome characteristics and their Shapley models (new_jax_shapley) reproduce jax_shapley bitwise.

At small scale: an agent, a behaviour characteristic sweep, off-policy (Mastermind-222) or
on-policy (Mastermind-222, Gridworld) outcome characteristics conditioned on behaviour seed 1
(exact values conditioned on the exact or the learnt behaviour characteristic), and Shapley
models explaining outcome seed 1 against the exact or the learnt outcome values.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from jax_shapley import train_dqn, train_explanations
from new_jax_shapley.algorithms.dqn import DQNConfig
from new_jax_shapley.characteristic_models.behaviour import CharacteristicConfig
from new_jax_shapley.characteristic_models.outcome import OutcomeConfig, OutcomeTrainConfig, Variant
from new_jax_shapley.common.supervised_learning import EpochMetrics, SupervisedConfig
from new_jax_shapley.envs import Gridworld, Mastermind
from new_jax_shapley.experiment import ExperimentConfig
from new_jax_shapley.main import ExperimentResult, run
from new_jax_shapley.shapley_models.behaviour import ShapleyConfig

SEEDS = (1, 2)
WIDTHS = (16, 16)
TRAINING = SupervisedConfig(epochs=3, batch_size=16, buffer_size=160, num_envs=4)
OUTCOME = OutcomeTrainConfig(
    epochs=3, batch_size=16, train_size=512, validation_size=40, total_timesteps=480, num_envs=4
)
# Environment, outcome variant and whether exact outcome values condition on the exact behaviour.
CASES: list[tuple[str, Variant, bool]] = [
    ("mastermind222", "off_policy", True),
    ("mastermind222", "off_policy", False),
    ("mastermind222", "on_policy", True),
    ("gridworld", "on_policy", False),
]


def test_outcome_matches_the_current_cli(pipelines: tuple[ExperimentResult, Path]) -> None:
    result, root = pipelines
    _assert_epochs(_single(root / "outcome"), result.outcome_metrics)


@pytest.mark.parametrize(("index", "exact"), [(0, "with"), (1, "without")])
def test_outcome_shapley_matches_the_current_cli(
    pipelines: tuple[ExperimentResult, Path], index: int, exact: str
) -> None:
    result, root = pipelines
    _assert_epochs(_single(root / f"shapley-{exact}"), result.shapley[index][1])


@pytest.fixture(scope="module", params=CASES, ids=["-".join(map(str, case)) for case in CASES])
def pipelines(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[ExperimentResult, Path]:
    """The new pipeline in process, and the current CLIs with the same settings."""
    environment, variant, with_exact_char = request.param
    env = Mastermind(2, 2, 2) if environment == "mastermind222" else Gridworld()
    config = ExperimentConfig(
        name="test_outcome",
        algorithm=DQNConfig(
            env=env,
            hidden_widths=WIDTHS,
            num_envs=4,
            total_timesteps=1024,
            buffer_size=1024,
            learning_starts=128,
            batch_size=16,
            exploration_fraction=0.25,
        ),
        characteristic=CharacteristicConfig(env=env, hidden_widths=WIDTHS, seeds=SEEDS, train=TRAINING),
        outcome=OutcomeConfig(
            env=env,
            variant=variant,
            with_exact_char=with_exact_char,
            tau=0.01,
            hidden_widths=WIDTHS,
            seeds=SEEDS,
            train=OUTCOME,
        ),
        shapley=tuple(
            ShapleyConfig(env=env, hidden_widths=WIDTHS, with_exact_char=exact, seeds=SEEDS, train=TRAINING)
            for exact in (True, False)
        ),
    )
    root = tmp_path_factory.mktemp("old")
    dqn = config.algorithm
    train_dqn.main([
        environment,
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
        variant,
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
        "--train.total-timesteps",
        str(OUTCOME.total_timesteps),
        "--train.num-envs",
        str(OUTCOME.num_envs),
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
        # On-policy rows leave the losses the source does not log empty.
        old = np.asarray([[float(row[name] or "nan") for name in names] for row in rows])
        epochs = np.flatnonzero(active[:, index])
        new = np.stack([np.asarray(getattr(metrics, name))[epochs, index] for name in names], axis=1)
        np.testing.assert_array_equal(new, old)


def _single(folder: Path) -> Path:
    (run_folder,) = [path for path in folder.iterdir() if path.is_dir()]
    return run_folder
