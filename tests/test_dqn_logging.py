"""Selected-event parity with the pinned trainer and committed CSV boundaries."""

import csv
import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from jax_shapley.algorithms.dqn import StepMetrics
from jax_shapley.artifacts import MetricRow, initialize_metrics, record_metrics, write_metric_rows
from jax_shapley.checkpoints import CheckpointMetadata
from jax_shapley.logging import prefix_digest, verify_log


def test_selected_logging_matches_pinned_trainer(tmp_path: Path) -> None:
    fixture = json.loads((Path(__file__).parent / "fixtures/fastsverl_logging.json").read_text())
    assert fixture["reference_commit"] == "25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b"
    path = tmp_path / "metrics.csv"
    initialize_metrics(path)
    # Several flush widths must deliver the same selected source records.
    indices = jnp.arange(54, dtype=jnp.int32)
    steps = indices * 2
    metrics = StepMetrics(
        steps,
        jnp.full(54, 0.5, jnp.float32),
        steps.astype(jnp.float32) + 0.25,
        (steps > 0) & (steps % 3 < 2),
        steps > 0,
        jnp.stack((indices, indices + 1), axis=1).astype(jnp.float32),
        jnp.stack((indices + 1, indices + 2), axis=1),
        jnp.stack((indices % 7 == 0, indices % 11 == 0), axis=1),
        jnp.ones(54, jnp.bool_),
    )
    record_metrics(path, metrics, 2, elapsed_seconds=2.0)
    write_metric_rows(
        path, [MetricRow("evaluation", index, episode_return=value) for index, value in enumerate((4.0, 5.0, 6.0))]
    )
    actual: list[dict[str, str | float | int]] = []
    with path.open() as source:
        for row in csv.DictReader(source):
            step = int(row["step"])
            names = {
                "episode": (
                    ("episode_return", "episodic_return"),
                    ("episode_length", "episodic_length"),
                    ("epsilon", "epsilon"),
                ),
                "update": (("loss", "DQN_loss"), ("sps", "SPS")),
                "evaluation": (("episode_return", "eval_episodic_return"),),
            }
            actual.extend(
                {"name": f"agent/{name}", "value": float(row[field]), "step": step}
                for field, name in names[row["kind"]]
            )
    assert actual == fixture["records"]


def test_log_prefix_rejects_corruption_and_allows_uncommitted_tail(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    path.write_text("committed\n")
    boundary = path.stat().st_size
    metadata = CheckpointMetadata("{}", 0, 0, 0, boundary, prefix_digest(path, boundary))
    with path.open("a") as output:
        output.write("uncommitted\n")
    verify_log(path, metadata)
    path.write_text("corrupted\n")
    with pytest.raises(ValueError, match="prefix"):
        verify_log(path, metadata)
