"""Selected-event parity with the pinned trainer and committed CSV boundaries."""

import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from jax_shapley.run_io import (
    CheckpointMetadata,
    MetricRow,
    RunWriter,
    initialize_metrics,
    prefix_digest,
    record_metrics,
    verify_log,
    write_metric_rows,
)
from jax_shapley.train_dqn import (
    DQNRunConfig,
    DQNTrainConfig,
    GridworldConfig,
    StepMetrics,
    initialize_run,
    make_snapshot,
    training_step,
)
from jax_shapley.wrappers import EpisodeStatisticsWrapper


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


def test_stale_resume_preserves_newer_checkpoint_and_committed_log(tmp_path: Path) -> None:
    config = DQNRunConfig(GridworldConfig(), train=DQNTrainConfig(total_timesteps=4, buffer_size=3), hidden_widths=(4,))
    env = EpisodeStatisticsWrapper(config.env.make())
    state = initialize_run(env, env.default_params, config.network, config.train, jax.random.key(1))
    zero = jnp.asarray(0, jnp.int32)
    path = tmp_path / "metrics.csv"
    initialize_metrics(path)
    sink = RunWriter(tmp_path, config)
    try:
        sink.checkpoint(make_snapshot(state, zero, zero, config.train))
        sink.store.wait()
        preflight_metadata, preflight_sequence = sink.store.metadata(), sink.store.sequence
        # A writer commits while the other resume is still compiling.
        write_metric_rows(path, [MetricRow("episode", 0, episode_return=1.0, episode_length=1, epsilon=1.0)])
        state, _ = training_step(state, env, env.default_params, config.train)
        sink.checkpoint(make_snapshot(state, zero, zero, config.train))
        sink.store.wait()
        latest_metadata, latest_sequence = sink.store.metadata(), sink.store.sequence
    finally:
        sink.close()
    committed = path.read_bytes()
    verify_log(path, preflight_metadata)  # Prefix validation alone cannot detect staleness.
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        RunWriter(tmp_path, config, metadata=preflight_metadata, checkpoint_sequence=preflight_sequence)
    assert path.read_bytes() == committed
    verify_log(path, latest_metadata)
    # Rejection releases the lock, and a current resume still rolls back only its tail.
    with path.open("a") as output:
        output.write("uncommitted tail\n")
    current = RunWriter(tmp_path, config, metadata=latest_metadata, checkpoint_sequence=latest_sequence)
    current.close()
    assert path.read_bytes() == committed
