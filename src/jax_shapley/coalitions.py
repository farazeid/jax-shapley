"""Feature coalitions: FastSVERL's samplers, enumeration order and observation masking.

A coalition is a boolean feature mask; True marks a known feature. Unknown
features are replaced by the source's MASK value directly in the int32
observation, before the model's float32 boundary.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Bool, Float, Int

type Coalitions = Bool[Array, "... features"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type Observations = Int[Array, "... features"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type Coalition = Bool[Array, "features"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.
type CoalitionTable = Bool[Array, "coalitions features"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type CoalitionBatch = Bool[Array, "batch features"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type SizeProbabilities = Float[Array, "sizes"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.

MASK = -2


def all_coalitions(features: int) -> CoalitionTable:
    """Every coalition in torch.cartesian_prod order: the first feature varies slowest."""
    return jnp.asarray(coalition_table(features))


def uniform_coalitions(key: Array, batch: int, features: int) -> CoalitionBatch:
    """Each feature is independently known with probability 1/2, as in the source's sample_rand."""
    _features_at_least(features, 1)
    return jax.random.bernoulli(key, 0.5, (batch, features))


def kernel_coalitions(key: Array, batch: int, features: int, *, paired: bool = False) -> CoalitionBatch:
    """Shapley-kernel coalitions: a size k in 1..F-1, then a uniformly random k-subset.

    Paired sampling replaces every odd row by the complement of the row before it.
    """
    _features_at_least(features, 2)
    size_key, order_key = jax.random.split(key)
    sizes = 1 + jax.random.categorical(size_key, jnp.log(shapley_kernel_probabilities(features)), shape=(batch,))
    ordered = jnp.arange(features) < sizes[:, None]
    coalitions = jax.random.permutation(order_key, ordered, axis=1, independent=True)
    if paired and batch > 1:
        coalitions = coalitions.at[1::2].set(~coalitions[0 : batch - 1 : 2])
    return coalitions


def shapley_kernel_probabilities(features: int) -> SizeProbabilities:
    """Probability of coalition size k = 1..F-1, proportional to 1 / (k (F - k))."""
    _features_at_least(features, 2)
    sizes = np.arange(1, features)
    weights = 1 / (sizes * (features - sizes))
    return jnp.asarray(weights / weights.sum(), jnp.float32)


def mask(observations: Observations, coalitions: Coalitions) -> Observations:
    """Replace unknown features by MASK, broadcasting coalitions over leading axes."""
    if observations.dtype != jnp.int32:
        raise TypeError("masked observations must be int32 environment observations")
    if coalitions.dtype != jnp.bool_:
        raise TypeError("coalitions must be boolean feature masks")
    return jnp.where(coalitions, observations, jnp.asarray(MASK, jnp.int32))


def coalition_table(features: int) -> np.ndarray[tuple[int, int], np.dtype[np.bool_]]:
    """Host constant for all_coalitions, usable while tracing static matrices."""
    _features_at_least(features, 1)
    return np.asarray(list(itertools.product((False, True), repeat=features)), np.bool_)


def _features_at_least(features: int, minimum: int) -> None:
    if features < minimum:
        raise ValueError(f"coalitions require at least {minimum} feature(s)")
