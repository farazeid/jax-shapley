"""Fixed-shape JAX replay matching FastSVERL Buffer insertion and sampling."""

from typing import cast

import jax
import jax.numpy as jnp
from flax import struct
from jax import Array
from jaxtyping import Bool, Float, Int

type Observations = Int[Array, "batch features"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type Actions = Int[Array, "batch"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.
type Scalars = Float[Array, "batch"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.
type Terminations = Bool[Array, "batch"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.


@struct.dataclass
class TransitionBatch:
    observations: Observations
    actions: Actions
    rewards: Scalars
    next_observations: Observations
    terminated: Terminations
    log_probabilities: Scalars

    def validate(self) -> None:
        size = self.observations.shape[0]
        if self.observations.ndim != 2 or self.next_observations.shape != self.observations.shape:
            raise ValueError("replay observations must have matching (batch, features) shapes")
        for array in (self.actions, self.rewards, self.terminated, self.log_probabilities):
            if array.shape != (size,):
                raise ValueError("replay scalar fields must have shape (batch,)")
        for array in (self.observations, self.next_observations, self.actions):
            if array.dtype != jnp.int32:
                raise TypeError("replay observations and actions must be int32")
        for array in (self.rewards, self.log_probabilities):
            if array.dtype != jnp.float32:
                raise TypeError("replay rewards and log probabilities must be float32")
        if self.terminated.dtype != jnp.bool_:
            raise TypeError("replay termination flags must be bool")


@struct.dataclass
class ReplayState:
    data: TransitionBatch
    pointer: Array
    size: Array

    @property
    def capacity(self) -> int:
        return self.data.observations.shape[0]


def empty_replay(capacity: int, features: int) -> ReplayState:
    if capacity < 1 or features < 1:
        raise ValueError("replay capacity and features must be positive")
    observations = jnp.zeros((capacity, features), dtype=jnp.int32)
    scalars = jnp.zeros(capacity, dtype=jnp.float32)
    return ReplayState(
        TransitionBatch(
            observations, jnp.zeros(capacity, jnp.int32), scalars, observations, jnp.zeros(capacity, jnp.bool_), scalars
        ),
        jnp.asarray(0, jnp.int32),
        jnp.asarray(0, jnp.int32),
    )


def append(replay: ReplayState, batch: TransitionBatch) -> ReplayState:
    batch.validate()
    if batch.observations.shape[1] != replay.data.observations.shape[1]:
        raise ValueError("replay observation width does not match batch")
    # Upstream discards the prefix of an oversized insertion before advancing its pointer.
    count = min(batch.observations.shape[0], replay.capacity)
    indices = (replay.pointer + jnp.arange(count, dtype=jnp.int32)) % replay.capacity
    data = cast(
        TransitionBatch,
        jax.tree.map(lambda storage, values: storage.at[indices].set(values[-count:]), replay.data, batch),
    )  # JAX tree.map preserves this struct topology.
    return ReplayState(
        data,
        (replay.pointer + count) % replay.capacity,
        jnp.minimum(replay.size + count, replay.capacity).astype(jnp.int32),
    )


def sample(replay: ReplayState, batch_size: int, key: Array) -> tuple[TransitionBatch, Array]:
    """Sample a nonempty replay; callers gate learning until after insertion."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    keys: Array = jax.random.split(key)
    key, sample_key = keys[0], keys[1]
    indices = jax.random.randint(sample_key, (batch_size,), 0, replay.size, dtype=jnp.int32)
    return gather(replay, indices), key


def gather(replay: ReplayState, indices: Array) -> TransitionBatch:
    if indices.ndim != 1 or indices.dtype != jnp.int32:
        raise TypeError("replay indices must be a rank-one int32 array")
    return cast(
        TransitionBatch, jax.tree.map(lambda array: array[indices], replay.data)
    )  # JAX tree.map preserves this struct topology.
