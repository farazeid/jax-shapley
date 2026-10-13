"""Feature coalitions: FastSVERL's enumeration order, Shapley-kernel sampler and observation masking.

A coalition is a boolean feature mask; True marks a known feature. Unknown
features are replaced by the source's MASK value directly in the int32
observation, before the model's float32 boundary. The source's uniform sampler
(`sample_rand`) is `jax.random.bernoulli(key, 0.5, (batch, features))`.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Int

type Coalitions = Bool[Array, "... features"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type Observations = Int[Array, "... features"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type Coalition = Bool[Array, "features"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.
type CoalitionTable = Bool[Array, "coalitions features"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type CoalitionBatch = Bool[Array, "batch features"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.

MASK = -2


def coalition_table(features: int) -> np.ndarray[tuple[int, int], np.dtype[np.bool_]]:
    """Every coalition in torch.cartesian_prod order (the first feature varies slowest), as a host constant."""
    if features < 1:
        raise ValueError("coalitions require at least one feature")
    return np.asarray(list(itertools.product((False, True), repeat=features)), np.bool_)


def kernel_coalitions(key: Array, batch: int, features: int) -> CoalitionBatch:
    """Shapley-kernel coalitions: a size k in 1..F-1 with probability proportional to 1 / (k (F - k)),
    then a uniformly random k-subset."""
    if features < 2:
        raise ValueError("Shapley-kernel coalitions require at least two features")
    sizes = np.arange(1, features)
    weights = 1 / (sizes * (features - sizes))
    probabilities = jnp.asarray(weights / weights.sum(), jnp.float32)
    size_key, order_key = jax.random.split(key)
    drawn = 1 + jax.random.categorical(size_key, jnp.log(probabilities), shape=(batch,))
    ordered = jnp.arange(features) < drawn[:, None]
    return jax.random.permutation(order_key, ordered, axis=1, independent=True)


def mask(observations: Observations, coalitions: Coalitions) -> Observations:
    """Replace unknown features by MASK, broadcasting coalitions over leading axes."""
    if observations.dtype != jnp.int32:
        raise TypeError("masked observations must be int32 environment observations")
    if coalitions.dtype != jnp.bool_:
        raise TypeError("coalitions must be boolean feature masks")
    return jnp.where(coalitions, observations, jnp.asarray(MASK, jnp.int32))
