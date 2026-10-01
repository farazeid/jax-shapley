"""Typed host-only JSON and selective CSV artifact boundaries."""

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import jax
import numpy as np
from jax import Array

from jax_shapley.algorithms.dqn import StepMetrics
from jax_shapley.config import DQNRunConfig, EnvironmentConfig
from jax_shapley.networks import MLPConfig, QNetwork
from jax_shapley.replay import ReplayState

REFERENCE_COMMIT = "25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b"
type HostArray = np.ndarray[tuple[int, ...], np.dtype[np.generic]]


@dataclass(frozen=True, slots=True)
class RunDescription[Config: EnvironmentConfig]:
    config: DQNRunConfig[Config]
    network: MLPConfig
    environment: str
    reference_commit: str = REFERENCE_COMMIT


@dataclass(frozen=True, slots=True)
class LoadedAgent:
    model: QNetwork
    target: QNetwork
    epsilon: float
    vector_step: int
    replay: ReplayState | None


@dataclass(frozen=True, slots=True)
class MetricRow:
    kind: Literal["episode", "update", "evaluation"]
    step: int
    epsilon: float | None = None
    loss: float | None = None
    episode_return: float | None = None
    episode_length: int | None = None
    sps: int | None = None


METRIC_COLUMNS = ("kind", "step", "epsilon", "loss", "episode_return", "episode_length", "sps")


def describe[Config: EnvironmentConfig](config: DQNRunConfig[Config]) -> str:
    return (
        json.dumps(
            asdict(RunDescription(config, config.network, type(config.env).__name__)), default=_json_default, indent=2
        )
        + "\n"
    )


def initialize_metrics(path: Path) -> None:
    with path.open("w", newline="") as output:
        csv.writer(output).writerow(METRIC_COLUMNS)


def record_metrics(path: Path, metrics: StepMetrics, num_envs: int, elapsed_seconds: float = 1.0) -> None:
    steps, active = host_array(metrics.global_step), host_array(metrics.active)
    epsilon, losses, updated = host_array(metrics.epsilon), host_array(metrics.loss), host_array(metrics.updated)
    completed, returns, lengths = (
        host_array(metrics.completed),
        host_array(metrics.episode_return),
        host_array(metrics.episode_length),
    )
    rows: list[MetricRow] = []
    for index in range(len(steps)):
        if not bool(active[index]):
            continue
        step = int(steps[index])
        rows.extend(
            MetricRow(
                "episode",
                step,
                epsilon=float(epsilon[index]),
                episode_return=float(returns[index, env]),
                episode_length=int(lengths[index, env]),
            )
            for env in range(num_envs)
            if bool(completed[index, env])
        )
        if bool(updated[index]) and step % 100 < num_envs:
            rows.append(
                MetricRow("update", step, loss=float(losses[index]), sps=int(step / max(elapsed_seconds, 1e-9)))
            )
    write_metric_rows(path, rows)


def write_metric_rows(path: Path, rows: list[MetricRow]) -> None:
    with path.open("a", newline="") as output:
        writer = csv.writer(output)
        for row in rows:
            writer.writerow((
                row.kind,
                row.step,
                row.epsilon,
                row.loss,
                row.episode_return,
                row.episode_length,
                row.sps,
            ))


def host_array(value: Array) -> HostArray:
    return cast(HostArray, np.asarray(jax.device_get(value)))  # Explicit JAX-to-NumPy host boundary.


def _json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported JSON value {type(value)}")
