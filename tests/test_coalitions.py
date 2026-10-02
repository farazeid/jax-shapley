"""Coalition enumeration, samplers and masking against the pinned source and their definitions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_shapley.coalitions import (
    MASK,
    all_coalitions,
    kernel_coalitions,
    mask,
    shapley_kernel_probabilities,
    uniform_coalitions,
)
from tests.explanation_support import CASES, layout, load_fixture


@pytest.mark.parametrize("case", CASES)
def test_enumeration_and_kernel_match_source(case: str) -> None:
    fixture, shape = load_fixture(), layout(case)
    np.testing.assert_array_equal(all_coalitions(shape.features), fixture[f"{case}_coalitions"])
    np.testing.assert_allclose(
        shapley_kernel_probabilities(shape.features), fixture[f"{case}_char_behaviour_sampler_probs"], rtol=1e-6
    )


def test_kernel_sampler_follows_size_distribution_and_uniform_subsets() -> None:
    features, samples = 6, 60_000
    coalitions = np.asarray(kernel_coalitions(jax.random.key(3), samples, features))
    sizes = coalitions.sum(axis=1)
    assert sizes.min() >= 1 and sizes.max() <= features - 1
    expected = np.asarray(shapley_kernel_probabilities(features))
    observed = np.bincount(sizes, minlength=features)[1:] / samples
    tolerance = 4 * np.sqrt(expected * (1 - expected) / samples)  # Four binomial standard errors.
    assert np.all(np.abs(observed - expected) <= tolerance), (observed, expected)
    # Given its size, each feature is equally likely to be included.
    for size in range(1, features):
        rows = coalitions[sizes == size]
        np.testing.assert_allclose(rows.mean(axis=0), size / features, atol=0.03)


def test_paired_kernel_rows_are_complements() -> None:
    coalitions = np.asarray(kernel_coalitions(jax.random.key(1), 7, 5, paired=True))
    np.testing.assert_array_equal(coalitions[1::2], ~coalitions[0:6:2])
    assert np.all(coalitions[0::2].sum(axis=1) >= 1)


def test_uniform_coalitions_are_independent_fair_bits() -> None:
    coalitions = np.asarray(uniform_coalitions(jax.random.key(2), 40_000, 8))
    assert coalitions.dtype == np.bool_
    np.testing.assert_allclose(coalitions.mean(axis=0), 0.5, atol=0.01)
    assert coalitions.all(axis=1).any() and (~coalitions).all(axis=1).any()


def test_mask_replaces_unknown_features_and_broadcasts() -> None:
    observations = jnp.asarray([[0, -1, 2], [1, 2, -1]], jnp.int32)
    masked = mask(observations, jnp.asarray([True, False, True]))
    np.testing.assert_array_equal(masked, [[0, MASK, 2], [1, MASK, -1]])
    assert masked.dtype == jnp.int32
    with pytest.raises(TypeError):
        mask(observations.astype(jnp.float32), jnp.ones(3, jnp.bool_))
    with pytest.raises(TypeError):
        mask(observations, jnp.ones(3, jnp.int32))


def test_invalid_feature_counts_fail_before_tracing() -> None:
    with pytest.raises(ValueError):
        kernel_coalitions(jax.random.key(0), 2, 1)
    with pytest.raises(ValueError):
        all_coalitions(0)
