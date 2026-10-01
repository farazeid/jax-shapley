"""Ordered, bounded event delivery and checkpoint-committed CSV boundaries."""

import fcntl
import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import BinaryIO

import numpy as np
from jax import Array

from jax_shapley.algorithms.dqn import StepMetrics
from jax_shapley.artifacts import MetricRow, describe, host_array, record_metrics, write_metric_rows
from jax_shapley.checkpoints import CheckpointMetadata, CheckpointStore
from jax_shapley.config import DQNRunConfig, EnvironmentConfig
from jax_shapley.execution import CallbackToken, Snapshot


class ArtifactSink[Config: EnvironmentConfig]:
    """Host-only writer; numerical code sees only the EventSink protocol."""

    def __init__(
        self,
        folder: Path,
        config: DQNRunConfig[Config],
        *,
        metadata: CheckpointMetadata | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self.folder, self.config, self.clock = folder, config, clock
        self.lock: BinaryIO = (folder / ".writer.lock").open("a+b")
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.lock.close()
            raise RuntimeError("run folder already has an active writer") from None
        self.store: CheckpointStore | None = None
        try:
            if metadata is not None:
                verify_log(folder / "metrics.csv", metadata)
                with (folder / "metrics.csv").open("r+b") as log:
                    log.truncate(metadata.csv_bytes)
                    log.flush()
                    os.fsync(log.fileno())
            self.store = CheckpointStore(folder) if config.checkpoint_mode != "none" else None
        except BaseException:
            self.lock.close()
            raise
        self.started = clock()
        self.training_seconds = 0.0 if metadata is None else metadata.training_seconds
        self.prior_elapsed = 0.0 if metadata is None else metadata.elapsed_seconds

    def training_complete(self) -> CallbackToken:
        self.training_seconds = self.elapsed()
        return np.asarray(0, np.int32)

    def training(self, metrics: tuple[Array, ...], count: Array) -> CallbackToken:
        active = int(host_array(count))
        rows = StepMetrics(*(value[:active] for value in metrics))
        record_metrics(self.folder / "metrics.csv", rows, self.config.train.num_envs, self.elapsed())
        return np.asarray(0, np.int32)

    def evaluation(self, returns: Array, completed: Array, first_index: Array) -> CallbackToken:
        values, flags = host_array(returns), host_array(completed)
        first = int(host_array(first_index))
        selected = [float(value) for value, flag in zip(values, flags, strict=True) if bool(flag)]
        write_metric_rows(
            self.folder / "metrics.csv",
            [MetricRow("evaluation", first + index, episode_return=value) for index, value in enumerate(selected)],
        )
        return np.asarray(0, np.int32)

    def checkpoint(self, snapshot: Snapshot) -> CallbackToken:
        if self.store is not None:
            self.store.wait()
            path = self.folder / "metrics.csv"
            with path.open("a+b") as log:
                log.flush()
                os.fsync(log.fileno())
                boundary = log.tell()
            metadata = CheckpointMetadata(
                describe(self.config),
                int(host_array(snapshot.vector_step)),
                int(host_array(snapshot.phase)),
                int(host_array(snapshot.evaluation_episodes)),
                boundary,
                prefix_digest(path, boundary),
                self.elapsed(),
                training_seconds=self.training_seconds,
            )
            self.store.save(snapshot, metadata)
            # The pre-evaluation snapshot must commit before evaluation can emit rows.
            if metadata.phase == 1:
                self.store.wait()
        return np.asarray(0, np.int32)

    def elapsed(self) -> float:
        return self.prior_elapsed + self.clock() - self.started

    def close(self) -> None:
        try:
            if self.store is not None:
                self.store.close()
        finally:
            self.lock.close()


def verify_log(path: Path, metadata: CheckpointMetadata) -> None:
    if path.stat().st_size < metadata.csv_bytes or prefix_digest(path, metadata.csv_bytes) != metadata.csv_digest:
        raise ValueError("metrics prefix does not match committed checkpoint")


def prefix_digest(path: Path, boundary: int) -> str:
    digest = hashlib.sha256()
    remaining = boundary
    with path.open("rb") as source:
        while remaining:
            block = source.read(min(remaining, 65536))
            if not block:
                raise ValueError("metrics file is shorter than committed boundary")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()
