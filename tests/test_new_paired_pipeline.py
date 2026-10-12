"""Per-seed agents and paired seeds (new_jax_shapley) reproduce the current jax_shapley sweeps bitwise.

The Hypercube protocol at small scale: agent s trained with seed s, characteristic seed s
explaining agent s (separate exact buffer, exact-loss stopping, batch-step logging), and
Shapley seed s explaining characteristic seed s on its buffers.
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
from new_jax_shapley.envs import Hypercube
from new_jax_shapley.experiment import ExperimentConfig
from new_jax_shapley.main import ExperimentResult, run
from new_jax_shapley.shapley_models.behaviour import ShapleyConfig

SEEDS = (1, 2)
WIDTHS = (16, 16)
TRAINING = SupervisedConfig(
    epochs=6, batch_size=16, buffer_size=320, val_fraction=0.1, num_envs=4, exact_loss=True, track_batch=True
)
# Exact-loss stopping conditions (characteristic, Shapley) that stop some seeds partway.
KINDS = [("behaviour", "0.1", "0.01"), ("prediction", "0.01", "0.02")]


def test_agents_match_the_current_dqn(pipelines: tuple[ExperimentResult, Path]) -> None:
    result, root = pipelines
    for index, seed in enumerate(SEEDS):
        (agent,) = [path for path in (root / "agents").iterdir() if load_settings(path / "config.json").seed == seed]
        loaded = load_agent(agent, load_settings(agent / "config.json").network, load_replay=False)
        old = jax.tree.leaves(nnx.state(loaded.model))
        new = jax.tree.leaves(nnx.state(seed_slice(result.dqn_state, index).model))
        for left, right in zip(old, new, strict=True):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_characteristic_matches_the_current_sweep(pipelines: tuple[ExperimentResult, Path]) -> None:
    result, root = pipelines
    _assert_epochs(_single(root / "characteristic"), result.characteristic_metrics)
    assert result.characteristic_metrics is not None
    # The exact-loss condition stops some seed early (no patience is set), exercising that path.
    assert not np.asarray(result.characteristic_metrics.active).all()


def test_shapley_matches_the_current_sweep(pipelines: tuple[ExperimentResult, Path]) -> None:
    result, root = pipelines
    metrics = result.shapley[0][1]
    _assert_epochs(_single(root / "shapley"), metrics)
    assert metrics is not None
    assert not np.asarray(metrics.active).all()


@pytest.fixture(scope="module", params=KINDS, ids=[kind for kind, _, _ in KINDS])
def pipelines(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[ExperimentResult, Path]:
    """The new pipeline in process, and the current CLIs with the same settings."""
    kind, characteristic_condition, shapley_condition = request.param
    env = Hypercube(2, 3)
    config = ExperimentConfig(
        name="test_paired",
        algorithm=DQNConfig(
            env=env,
            seeds=SEEDS,
            hidden_widths=WIDTHS,
            num_envs=4,
            total_timesteps=1024,
            buffer_size=1024,
            learning_starts=64,
            batch_size=16,
            end_e=0.05,
            exploration_fraction=0.5,
        ),
        characteristic=CharacteristicConfig(
            env=env,
            kind=kind,
            hidden_widths=WIDTHS,
            seeds=SEEDS,
            train=SupervisedConfig(
                epochs=6,
                batch_size=16,
                buffer_size=320,
                val_fraction=0.1,
                num_envs=4,
                exact_loss=True,
                track_batch=True,
                exact_buffer_size=640,
                exact_condition=float(characteristic_condition),
            ),
        ),
        shapley=(
            ShapleyConfig(
                env=env,
                hidden_widths=WIDTHS,
                with_exact_char=False,
                buffers="characteristic",
                seeds=SEEDS,
                train=SupervisedConfig(
                    epochs=6,
                    batch_size=16,
                    buffer_size=320,
                    val_fraction=0.1,
                    num_envs=4,
                    exact_loss=True,
                    track_batch=True,
                    exact_condition=float(shapley_condition),
                ),
            ),
        ),
        explained_seed=None,
    )
    root = tmp_path_factory.mktemp(f"old-{kind}")
    dqn = config.algorithm
    for seed in SEEDS:
        train_dqn.main([
            "hypercube",
            "--env.side-length",
            "2",
            "--env.dimension",
            "3",
            "--seed",
            str(seed),
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
            "--train.end-e",
            str(dqn.end_e),
            "--train.exploration-fraction",
            str(dqn.exploration_fraction),
            "--train.eval-episodes",
            "0",
            "--output",
            str(root / "agents"),
        ])
    common = [
        "--seeds",
        *map(str, SEEDS),
        "--hidden-widths",
        *map(str, WIDTHS),
        "--train.epochs",
        "6",
        "--train.batch-size",
        "16",
        "--train.buffer-size",
        "320",
        "--train.val-fraction",
        "0.1",
        "--train.num-envs",
        "4",
        "--train.exact-loss",
        "--train.track-batch",
    ]
    train_explanations.main([
        "characteristic",
        "--agent",
        str(root / "agents"),
        "--kind",
        kind,
        "--policy",
        "greedy",
        "--train.exact-buffer-size",
        "640",
        "--train.exact-condition",
        characteristic_condition,
        *common,
        "--output",
        str(root / "characteristic"),
    ])
    train_explanations.main([
        "shapley",
        "--agent",
        str(root / "agents"),
        "--characteristic",
        str(_single(root / "characteristic")),
        "--no-with-exact-char",
        "--buffers",
        "characteristic",
        "--train.exact-condition",
        shapley_condition,
        *common,
        "--output",
        str(root / "shapley"),
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
