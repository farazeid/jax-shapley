"""Numerical payloads and narrow host callback contracts for complete DQN runs."""

from typing import Protocol

from flax import struct
from jax import Array

from jax_shapley.replay import ReplayState


@struct.dataclass
class Snapshot:
    agent: tuple[Array, ...]
    replay: ReplayState | None
    optimizer: tuple[Array, ...]
    runtime: tuple[Array, ...]
    vector_step: Array
    phase: Array
    evaluation_episodes: Array
    epsilon: Array


class EventSink(Protocol):
    def training(self, metrics: tuple[Array, ...], count: Array) -> Array: ...

    def evaluation(self, returns: Array, completed: Array, first_index: Array) -> Array: ...

    def checkpoint(self, snapshot: Snapshot) -> Array: ...
