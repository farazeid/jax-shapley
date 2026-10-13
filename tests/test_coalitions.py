"""Coalition enumeration, the Shapley-kernel sampler and masking against the pinned source and their definitions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from new_jax_shapley.common.coalitions import MASK, coalition_table, kernel_coalitions, mask
from tests.fixture_support import CASES, layout, load_fixture


@pytest.mark.parametrize("case", CASES)
def test_enumeration_and_kernel_match_source(case: str) -> None:
    fixture, shape = load_fixture(), layout(case)
    np.testing.assert_array_equal(coalition_table(shape.features), fixture[f"{case}_coalitions"])
    sizes = np.arange(1, shape.features)
    weights = 1 / (sizes * (shape.features - sizes))
    np.testing.assert_allclose(weights / weights.sum(), fixture[f"{case}_char_behaviour_sampler_probs"], rtol=1e-6)


def test_kernel_sampler_follows_size_distribution_and_uniform_subsets() -> None:
    features, samples = 6, 60_000
    coalitions = np.asarray(kernel_coalitions(jax.random.key(3), samples, features))
    sizes = coalitions.sum(axis=1)
    assert sizes.min() >= 1 and sizes.max() <= features - 1
    weights = 1 / (np.arange(1, features) * (features - np.arange(1, features)))
    expected = weights / weights.sum()
    observed = np.bincount(sizes, minlength=features)[1:] / samples
    tolerance = 4 * np.sqrt(expected * (1 - expected) / samples)  # Four binomial standard errors.
    assert np.all(np.abs(observed - expected) <= tolerance), (observed, expected)
    # Given its size, each feature is equally likely to be included.
    for size in range(1, features):
        rows = coalitions[sizes == size]
        np.testing.assert_allclose(rows.mean(axis=0), size / features, atol=0.03)


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
        coalition_table(0)
