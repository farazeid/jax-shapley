"""Inert/recording callback adapters; never create files, writers or workers."""

from dataclasses import dataclass, field

import numpy as np
from jax import Array

from jax_shapley.run_io import CallbackToken, Snapshot


@dataclass
class RecordingSink:
    training_steps: list[int] = field(default_factory=list)
    evaluation_returns: list[float] = field(default_factory=list)
    checkpoints: list[tuple[int, int]] = field(default_factory=list)
    largest_batch: int = 0

    def training_complete(self) -> CallbackToken:
        return np.asarray(0, np.int32)

    def training(self, metrics: tuple[Array, ...], count: Array) -> CallbackToken:
        size = int(count)
        self.largest_batch = max(self.largest_batch, size)
        self.training_steps.extend(int(step) for step in metrics[0][:size])
        return np.asarray(0, np.int32)

    def evaluation(self, returns: Array, completed: Array, first_index: Array) -> CallbackToken:
        assert int(first_index) == len(self.evaluation_returns)
        self.evaluation_returns.extend(
            float(value) for value, flag in zip(returns, completed, strict=True) if bool(flag)
        )
        return np.asarray(0, np.int32)

    def checkpoint(self, snapshot: Snapshot) -> CallbackToken:
        self.checkpoints.append((int(snapshot.vector_step), int(snapshot.phase)))
        return np.asarray(0, np.int32)
