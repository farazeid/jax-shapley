"""Off-policy characteristics (new_jax_shapley) reproduce the current jax_shapley CLIs bitwise.

The paper's off-policy groups at small scale: each seed's agent, then characteristic seed s
explaining agent s from rollouts (exact), from its replay without importance weights
(wo_weight), with normalised or unnormalised weights, and with clipped weights.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from jax_shapley import train_dqn, train_explanations
from new_jax_shapley.algorithms.dqn import DQNConfig
from new_jax_shapley.characteristic_models.behaviour import CharacteristicConfig, Data, Kind
from new_jax_shapley.common.supervised_learning import SupervisedConfig
from new_jax_shapley.envs import Gridworld, Mastermind
from new_jax_shapley.experiment import ExperimentConfig
from new_jax_shapley.main import run

SEEDS = (1, 2)
WIDTHS = (16, 16)
TRAINING = SupervisedConfig(
    epochs=4, batch_size=16, buffer_size=640, val_fraction=0.2, num_envs=4, exact_loss=True, exact_buffer_size=320
)
# Group: (data, off_policy, weighting, clip_coef) and the old CLI's flags for it.
GROUPS: dict[str, tuple[tuple[Data, bool, bool, float], list[str]]] = {
    "exact": (("rollouts", False, True, float("inf")), []),
    "wo_weight": (("replay", False, True, float("inf")), ["--data", "replay"]),
    "normalised": (("replay", True, True, float("inf")), ["--data", "replay", "--off-policy"]),
    "unnormalised": (("replay", True, False, float("inf")), ["--data", "replay", "--off-policy", "--no-weighting"]),
    "clip_0.99": (("replay", True, True, 0.99), ["--data", "replay", "--off-policy", "--clip-coef", "0.99"]),
}
KINDS: tuple[Kind, ...] = ("behaviour", "prediction")
CASES = [*(("gridworld", kind, group) for kind in KINDS for group in GROUPS), ("mastermind222", KINDS[0], "normalised")]


@pytest.mark.parametrize(("environment", "kind", "group"), CASES)
def test_characteristic_matches_the_current_cli(
    agents: dict[str, Path], environment: str, kind: Kind, group: str, tmp_path: Path
) -> None:
    (data, off_policy, weighting, clip_coef), flags = GROUPS[group]
    env = Gridworld() if environment == "gridworld" else Mastermind(2, 2, 2)
    config = ExperimentConfig(
        name="test_offpolicy",
        algorithm=_agent_config(env),
        characteristic=CharacteristicConfig(
            env=env,
            kind=kind,
            hidden_widths=WIDTHS,
            data=data,
            off_policy=off_policy,
            weighting=weighting,
            clip_coef=clip_coef,
            seeds=SEEDS,
            train=TRAINING,
        ),
        shapley=(),
    )
    train_explanations.main([
        "characteristic",
        "--agent",
        str(agents[environment]),
        "--kind",
        kind,
        "--policy",
        "greedy",
        *flags,
        "--seeds",
        *map(str, SEEDS),
        "--hidden-widths",
        *map(str, WIDTHS),
        "--train.epochs",
        str(TRAINING.epochs),
        "--train.batch-size",
        str(TRAINING.batch_size),
        "--train.buffer-size",
        str(TRAINING.buffer_size),
        "--train.val-fraction",
        str(TRAINING.val_fraction),
        "--train.num-envs",
        str(TRAINING.num_envs),
        "--train.exact-loss",
        "--train.exact-buffer-size",
        str(TRAINING.exact_buffer_size),
        "--output",
        str(tmp_path),
    ])
    metrics = run(config).characteristic_metrics
    assert metrics is not None
    (sweep,) = [path for path in tmp_path.iterdir() if path.is_dir()]
    names = ("epoch", "step", "train_loss", "validation_loss", "exact_loss", "best_epoch")
    active = np.asarray(metrics.active)
    for index, seed in enumerate(SEEDS):
        rows = list(csv.DictReader((sweep / f"seed-{seed}" / "metrics.csv").open()))
        old = np.asarray([[float(row[name]) for name in names] for row in rows])
        epochs = np.flatnonzero(active[:, index])
        new = np.stack([np.asarray(getattr(metrics, name))[epochs, index] for name in names], axis=1)
        np.testing.assert_array_equal(new, old)


@pytest.fixture(scope="module")
def agents(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Each environment's per-seed agents trained by the current DQN CLI."""
    folders = {}
    for environment, env in (("gridworld", Gridworld()), ("mastermind222", Mastermind(2, 2, 2))):
        config = _agent_config(env)
        folder = tmp_path_factory.mktemp(f"agents-{environment}")
        for seed in SEEDS:
            train_dqn.main([
                environment,
                "--seed",
                str(seed),
                "--hidden-widths",
                *map(str, WIDTHS),
                "--train.num-envs",
                str(config.num_envs),
                "--train.total-timesteps",
                str(config.total_timesteps),
                "--train.buffer-size",
                str(config.buffer_size),
                "--train.learning-starts",
                str(config.learning_starts),
                "--train.batch-size",
                str(config.batch_size),
                "--train.exploration-fraction",
                str(config.exploration_fraction),
                "--train.eval-episodes",
                "0",
                "--output",
                str(folder),
            ])
        folders[environment] = folder
    return folders


def _agent_config(env: Gridworld | Mastermind) -> DQNConfig:
    return DQNConfig(
        env=env,
        seeds=SEEDS,
        hidden_widths=WIDTHS,
        num_envs=4,
        total_timesteps=1024,
        buffer_size=1024,
        learning_starts=128,
        batch_size=16,
        exploration_fraction=0.25,
    )
