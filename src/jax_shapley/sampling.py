"""Sampled behaviour and prediction characteristics (FastSVERL `ValuePolicyCharacteristicSample`).

The source estimates v(s, C) with one draw (`num_samples=1` in every paper run): a
pool row that agrees with s on every feature in C, chosen uniformly among the
matching rows, then v_F of that row. The pool is the behaviour or Shapley model's
train buffer, and v(empty) is the mean of v_F over the pool. Like the source, which
reads `Cs[0]`, every row of a call uses the first coalition: Shapley batches and the
on-policy outcome loop share one coalition across rows.

The source permutes the pool, scores each (row, pool row) pair with `torch.rand`
where they match and MASK where they do not, and takes the top score. With no match
every score ties at MASK and `torch.topk` picks a position of the freshly permuted
pool, so one uniform random row. Here one uniform score per pair is raised by 1 for
matches, so the top score is uniform among the matches, or among all rows when
nothing matches.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import struct
from jax import Array
from jaxtyping import Float, Int

from jax_shapley.characteristics import Output, RowObservations, RowValues, Values
from jax_shapley.coalitions import Coalition, Coalitions, Observations

# Rows matched against the pool at once; bounds the (rows, pool, features) comparison.
SAMPLE_CHUNK = 256

type PoolScores = Float[Array, "rows pool"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type PoolRows = Int[Array, "rows"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.


@struct.dataclass
class SampledPool:
    """The rows a sampled characteristic draws from, with v_F precomputed for each."""

    observations: RowObservations
    values: RowValues
    null: Output


@dataclass(frozen=True, slots=True)
class SampledCharacteristic:
    """A CharacteristicFunction over a pool, with the draw's key bound in."""

    pool: SampledPool
    key: Array

    def values(self, observations: Observations, coalitions: Coalitions) -> Values:
        """Single draws for every row, matched on the first coalition (the source's `Cs[0]`)."""
        features = observations.shape[-1]
        rows = observations.reshape(-1, features)
        indices = sample_rows(self.key, rows, coalitions.reshape(-1, features)[0], self.pool.observations)
        return self.pool.values[indices].reshape(*observations.shape[:-1], self.pool.values.shape[-1])

    def null(self, observations: Observations) -> Values:
        """The source's null: v_F averaged over the pool, for every row."""
        return jnp.broadcast_to(self.pool.null, (*observations.shape[:-1], self.pool.null.shape[-1]))


def sampled_pool(observations: RowObservations, values: RowValues) -> SampledPool:
    return SampledPool(observations, values, jnp.mean(values, axis=0))


def sample_rows(key: Array, observations: RowObservations, coalition: Coalition, pool: RowObservations) -> PoolRows:
    """One pool row per observation, uniform among the rows matching it on the coalition."""
    count = observations.shape[0]
    size = min(SAMPLE_CHUNK, count)
    chunks = -(-count // size)
    padded = jnp.pad(observations, ((0, chunks * size - count), (0, 0)))

    def chunk(inputs: tuple[Array, Array]) -> PoolRows:
        chunk_key, rows = inputs
        scores = jax.random.uniform(chunk_key, (rows.shape[0], pool.shape[0]), jnp.float32)
        return select_matches(rows, coalition, pool, scores)

    picked: Array = jax.lax.map(chunk, (jax.random.split(key, chunks), padded.reshape(chunks, size, -1)))
    return picked.reshape(-1)[:count]


def select_matches(
    observations: RowObservations, coalition: Coalition, pool: RowObservations, scores: PoolScores
) -> PoolRows:
    """Top score per row, matching pool rows first; `scores` lie in [0, 1)."""
    matches = jnp.all((observations[:, None, :] == pool[None, :, :]) | ~coalition, axis=-1)
    return jnp.argmax(jnp.where(matches, scores + 1.0, scores), axis=-1).astype(jnp.int32)
