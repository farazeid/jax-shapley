"""Selected-event parity with the pinned trainer's logging.

The source logs one record per finished episode; a tracking backend holds one value per tag
and step, so episodes finishing on the same step log their mean. Losses log on the source's
cadence and evaluation returns by episode number, as the source does.
"""

import json
from collections import defaultdict
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from jax_shapley.algorithms import dqn


def test_selected_logging_matches_pinned_trainer() -> None:
    fixture = json.loads((Path(__file__).parent / "fixtures/fastsverl_logging.json").read_text())
    assert fixture["reference_commit"] == "25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b"
    indices = jnp.arange(54, dtype=jnp.int32)
    steps = indices * 2
    metrics = dqn.StepMetrics(
        steps,
        jnp.full(54, 0.5, jnp.float32),
        steps.astype(jnp.float32) + 0.25,
        (steps > 0) & (steps % 3 < 2),
        steps > 0,
        jnp.stack((indices, indices + 1), axis=1).astype(jnp.float32),
        jnp.stack((indices + 1, indices + 2), axis=1),
        jnp.stack((indices % 7 == 0, indices % 11 == 0), axis=1),
    )
    run = Recorder()
    for row in range(54):
        step_metrics = dqn.StepMetrics(*[
            np.asarray(leaf)[row]
            for leaf in (
                metrics.global_step,
                metrics.epsilon,
                metrics.loss,
                metrics.updated,
                metrics.target_updated,
                metrics.episode_return,
                metrics.episode_length,
                metrics.completed,
            )
        ])
        dqn.log_step([run], fixture["num_envs"], jnp.asarray(0), step_metrics)
    evaluation = dqn.EvaluationMetrics(jnp.asarray([4.0, 5.0, 6.0]), jnp.asarray([True, True, True]), jnp.asarray(0))
    dqn.log_evaluation([run], jnp.asarray(0), evaluation)
    expected: dict[tuple[str, int], list[float]] = defaultdict(list)
    for record in fixture["records"]:
        if record["name"] != "agent/SPS":  # Throughput is not an experiment result.
            expected[record["name"], record["step"]].append(record["value"])
    assert run.records == {key: float(np.mean(values)) for key, values in expected.items()}


class Recorder:
    """A run that keeps every logged value by (tag, step)."""

    def __init__(self) -> None:
        self.records: dict[tuple[str, int], float] = {}

    def log(self, data: dict[str, float], /, step: int | None = None) -> None:
        assert step is not None
        for name, value in data.items():
            assert (name, step) not in self.records
            self.records[name, step] = value
