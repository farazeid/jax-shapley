"""Continual learning: the agent trains as a plain DQN run, explanations follow the source's cadence."""

import csv
from pathlib import Path

import numpy as np
import pytest

from jax_shapley import train_continual, train_dqn
from jax_shapley.train_continual import ContinualSummary

NUM_ENVS, STEPS, EXACT_FREQUENCY = 4, 2000, 500


@pytest.mark.parametrize("kind", ["behaviour", "prediction", "outcome"])
def test_agent_trains_as_a_plain_dqn_run(tmp_path: Path, kind: str) -> None:
    summaries = _summaries(train_continual.main(_continual(tmp_path / "continual", kind)))
    plain = train_dqn.main(_dqn(tmp_path / "dqn"))
    assert plain is not None
    episodes = _rows(Path(plain.folder) / "metrics.csv", "episode")
    continual = _csv(Path(summaries[0].folder) / "episodes.csv")
    assert [(int(row["step"]), float(row["episode_return"]), int(row["episode_length"])) for row in continual] == [
        (int(row["step"]), float(row["episode_return"]), int(row["episode_length"])) for row in episodes
    ]
    # The DQN loss at each logged step is the plain run's update loss at that step.
    updates = {int(row["step"]): float(row["loss"]) for row in _rows(Path(plain.folder) / "metrics.csv", "update")}
    logged = _csv(Path(summaries[0].folder) / "metrics.csv")
    shared = [row for row in logged if int(row["step"]) in updates]
    assert shared
    for row in shared:
        assert float(row["dqn_loss"]) == pytest.approx(updates[int(row["step"])], rel=1e-6)


def test_logging_and_exact_cadence_follow_the_source(tmp_path: Path) -> None:
    summaries = _summaries(train_continual.main(_continual(tmp_path, "outcome")))
    assert [summary.seed for summary in summaries] == [1, 2]
    rows = _csv(Path(summaries[0].folder) / "metrics.csv")
    steps = [int(row["step"]) for row in rows]
    # Learning starts after step 0; losses log every 100 environment steps, exact losses every 500.
    learning = [step for step in range(0, STEPS, NUM_ENVS) if step > 0]
    assert steps == [step for step in learning if step % 100 < NUM_ENVS or step % EXACT_FREQUENCY < NUM_ENVS]
    for row in rows:
        exact = int(row["step"]) % EXACT_FREQUENCY < NUM_ENVS
        for name in ("characteristic_exact_loss", "outcome_exact_loss", "shapley_exact_loss"):
            assert (row[name] != "") == exact
        assert all(np.isfinite(float(row[name])) for name in ("characteristic_loss", "outcome_loss", "shapley_loss"))
    other = _csv(Path(summaries[1].folder) / "metrics.csv")
    assert [row["characteristic_loss"] for row in other] != [row["characteristic_loss"] for row in rows]


def test_update_ratio_changes_only_the_explanations(tmp_path: Path) -> None:
    one = _summaries(train_continual.main(_continual(tmp_path / "one", "behaviour")))[0]
    many = _summaries(
        train_continual.main([*_continual(tmp_path / "many", "behaviour"), "--explain.update-ratio", "5"])
    )[0]
    assert _csv(Path(one.folder) / "episodes.csv") == _csv(Path(many.folder) / "episodes.csv")
    first, second = _csv(Path(one.folder) / "metrics.csv"), _csv(Path(many.folder) / "metrics.csv")
    assert [row["dqn_loss"] for row in first] == [row["dqn_loss"] for row in second]
    assert [row["shapley_loss"] for row in first] != [row["shapley_loss"] for row in second]


def test_continual_validation() -> None:
    with pytest.raises(SystemExit):
        train_continual.parse_args(["gridworld", "--explain.update-ratio", "0"])
    with pytest.raises(SystemExit):
        train_continual.parse_args(["gridworld", "--train.num-envs", "3", "--explain.exact-buffer-size", "10"])
    with pytest.raises(SystemExit):
        train_continual.parse_args(["gridworld", "--seeds", "1", "1"])


def _summaries(result: list[ContinualSummary] | None) -> list[ContinualSummary]:
    assert result is not None
    return result


def _continual(output: Path, kind: str) -> list[str]:
    return [
        "gridworld",
        "--seeds",
        "1",
        "2",
        *_training(),
        "--explain.kind",
        kind,
        "--explain.update-ratio",
        "2",
        "--explain.exact-frequency",
        str(EXACT_FREQUENCY),
        "--explain.exact-buffer-size",
        "400",
        "--explain.hidden-widths",
        "16",
        "16",
        "--output",
        str(output),
    ]


def _dqn(output: Path) -> list[str]:
    return ["gridworld", "--seed", "1", *_training(), "--output", str(output)]


def _training() -> list[str]:
    return [
        "--hidden-widths",
        "16",
        "16",
        "--train.num-envs",
        str(NUM_ENVS),
        "--train.total-timesteps",
        str(STEPS),
        "--train.buffer-size",
        str(STEPS),
        "--train.learning-starts",
        "0",
        "--train.exploration-fraction",
        "0.5",
    ]


def _rows(path: Path, kind: str) -> list[dict[str, str]]:
    return [row for row in _csv(path) if row["kind"] == kind]


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))
