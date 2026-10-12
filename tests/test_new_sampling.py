"""Sampled characteristics (new_jax_shapley) reproduce the current jax_shapley sampling CLIs bitwise.

The paper's sampling groups at small scale on Mastermind-222, every seed paired: Shapley
models explaining a learnt characteristic on fresh buffers (with and without exact
characteristic values), Shapley models explaining a sampled characteristic on its own
buffers, and on-policy outcome characteristics conditioned on a sampled behaviour.
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
from new_jax_shapley.main import run
from new_jax_shapley.shapley_models.behaviour import ShapleyConfig

SEEDS = (1, 2)
WIDTHS = (16, 16)
ENV = Mastermind(2, 2, 2)
TRAINING = SupervisedConfig(epochs=3, batch_size=16, buffer_size=160, num_envs=4)
OUTCOME = OutcomeTrainConfig(epochs=3, batch_size=16, total_timesteps=480, num_envs=4)
AGENT = DQNConfig(
    env=ENV,
    hidden_widths=WIDTHS,
    num_envs=4,
    total_timesteps=1024,
    buffer_size=1024,
    learning_starts=128,
    batch_size=16,
    exploration_fraction=0.25,
)
EXPLANATION = [
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


def test_model_groups_match_the_current_cli(agent: Path, tmp_path: Path) -> None:
    config = ExperimentConfig(
        name="test_sampling",
        algorithm=AGENT,
        characteristic=CharacteristicConfig(env=ENV, hidden_widths=WIDTHS, seeds=SEEDS, train=TRAINING),
        shapley=tuple(
            ShapleyConfig(env=ENV, hidden_widths=WIDTHS, with_exact_char=exact, seeds=SEEDS, train=TRAINING)
            for exact in (True, False)
        ),
        explained_seed=None,
    )
    characteristic = _characteristic(agent, tmp_path / "model", [])
    result = run(config)
    _assert_epochs(characteristic, result.characteristic_metrics)
    for index, exact in enumerate(("with", "without")):
        folder = tmp_path / f"shapley-{exact}"
        flags = ["--buffers", "fresh", f"--{'' if exact == 'with' else 'no-'}with-exact-char"]
        _shapley(agent, characteristic, folder, flags)
        _assert_epochs(_single(folder), result.shapley[index][1])


def test_sampled_group_matches_the_current_cli(agent: Path, tmp_path: Path) -> None:
    sampled = CharacteristicConfig(
        env=ENV,
        hidden_widths=WIDTHS,
        estimator="sampled",
        seeds=SEEDS,
        train=SupervisedConfig(epochs=3, batch_size=16, buffer_size=160, num_envs=4, exact_loss=False),
    )
    config = ExperimentConfig(
        name="test_sampled",
        algorithm=AGENT,
        characteristic=sampled,
        shapley=(ShapleyConfig(env=ENV, hidden_widths=WIDTHS, buffers="characteristic", seeds=SEEDS, train=TRAINING),),
        explained_seed=None,
    )
    characteristic = _characteristic(agent, tmp_path / "sampled", ["--estimator", "sampled", "--train.no-exact-loss"])
    _shapley(agent, characteristic, tmp_path / "shapley", ["--buffers", "characteristic", "--with-exact-char"])
    result = run(config)
    assert result.characteristic_metrics is None
    _assert_epochs(_single(tmp_path / "shapley"), result.shapley[0][1])
    outcome = ExperimentConfig(
        name="test_sampled_outcome",
        algorithm=AGENT,
        characteristic=sampled,
        outcome=OutcomeConfig(env=ENV, variant="on_policy", tau=0.01, hidden_widths=WIDTHS, seeds=SEEDS, train=OUTCOME),
        shapley=(),
        explained_seed=None,
    )
    train_explanations.main([
        "outcome",
        "--agent",
        str(agent),
        "--characteristic",
        str(characteristic),
        "--variant",
        "on_policy",
        "--with-exact-char",
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
        "--train.total-timesteps",
        str(OUTCOME.total_timesteps),
        "--train.num-envs",
        str(OUTCOME.num_envs),
        "--output",
        str(tmp_path / "outcome"),
    ])
    _assert_epochs(_single(tmp_path / "outcome"), run(outcome).outcome_metrics)


@pytest.fixture(scope="module")
def agent(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The agent trained by the current DQN CLI."""
    folder = tmp_path_factory.mktemp("agent")
    train_dqn.main([
        "mastermind222",
        "--seed",
        "1",
        "--hidden-widths",
        *map(str, WIDTHS),
        "--train.num-envs",
        str(AGENT.num_envs),
        "--train.total-timesteps",
        str(AGENT.total_timesteps),
        "--train.buffer-size",
        str(AGENT.buffer_size),
        "--train.learning-starts",
        str(AGENT.learning_starts),
        "--train.batch-size",
        str(AGENT.batch_size),
        "--train.exploration-fraction",
        str(AGENT.exploration_fraction),
        "--train.eval-episodes",
        "0",
        "--output",
        str(folder),
    ])
    return _single(folder)


def _characteristic(agent: Path, folder: Path, flags: list[str]) -> Path:
    """A characteristic sweep by the current CLI; Shapley and outcome runs pair seed s with its seed s."""
    train_explanations.main([
        "characteristic",
        "--agent",
        str(agent),
        "--kind",
        "behaviour",
        *flags,
        *EXPLANATION,
        "--output",
        str(folder),
    ])
    return _single(folder)


def _shapley(agent: Path, characteristic: Path, folder: Path, flags: list[str]) -> None:
    train_explanations.main([
        "shapley",
        "--agent",
        str(agent),
        "--characteristic",
        str(characteristic),
        *flags,
        *EXPLANATION,
        "--output",
        str(folder),
    ])


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
