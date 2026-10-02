"""Offline integrity of the pinned FastSVERL behaviour/prediction explanation fixtures.

These tests check the captured reference against independent definitions only;
JAX parity tests consume the same fixture once the explanation modules exist.
"""

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "fastsverl_explanations.npz"
TRANSITIONS = Path(__file__).parent / "fixtures" / "fastsverl_transitions.json"
CASES = ("gridworld", "mastermind222", "hypercube")
KINDS = ("behaviour", "prediction")
# Exact tables are keyed by characteristic label; each pairs with the agent output it averages.
EXACT_TARGETS = {"behaviour": "greedy", "behaviour_explore": "explore", "prediction": "value"}
OFF_POLICY_VARIANTS = ("weighted", "unweighted", "weighted_clipped")
STEPS = 3


def test_fixture_provenance() -> None:
    metadata = json.loads(FIXTURE.with_suffix(".json").read_text())
    assert metadata["fastsverl_commit"] == "25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b"
    assert metadata["fixture_sha256"] == hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert metadata["transitions_sha256"] == json.loads(TRANSITIONS.read_text())["fixture_sha256"]
    assert metadata["torch_version"].startswith("2.6.0")
    assert metadata["mask"] == -2
    assert set(metadata["cases"]) == set(CASES)
    assert set(metadata["off_policy_variants"]) == set(OFF_POLICY_VARIANTS)


@pytest.mark.parametrize("case", CASES)
def test_case_layout_and_dtypes(case: str) -> None:
    fixture = _load()
    features, actions, batch = _dimensions(case)
    unique = fixture[f"{case}_unique"]
    states = unique.shape[0]
    assert unique.dtype == np.int32
    assert fixture[f"{case}_exact_buffer"].dtype == np.int32
    assert fixture[f"{case}_counts"].dtype == np.int32
    assert fixture[f"{case}_coalitions"].shape == (2**features, features)
    assert fixture[f"{case}_coalitions"].dtype == np.bool_
    for label, outputs in (("behaviour", actions), ("behaviour_explore", actions), ("prediction", 1)):
        assert fixture[f"{case}_exact_{label}"].shape == (2**features, states, outputs)
        assert fixture[f"{case}_exact_{label}_null"].shape == (outputs,)
    for kind, outputs in (("behaviour", actions), ("prediction", 1)):
        char = f"{case}_char_{kind}"
        shapley = f"{case}_shapley_{kind}"
        assert fixture[f"{char}_targets"].shape == (batch, outputs)
        assert fixture[f"{char}_grand"].shape == (states, outputs)
        for step in range(STEPS):
            assert fixture[f"{char}_coalitions_{step}"].shape == (batch, features)
            assert fixture[f"{char}_coalitions_{step}"].dtype == np.bool_
            assert fixture[f"{char}_predictions_{step}"].shape == (batch, outputs)
            assert fixture[f"{char}_loss_{step}"].shape == ()
            assert fixture[f"{shapley}_coalition_{step}"].shape == (features,)
            assert fixture[f"{shapley}_targets_{step}"].shape == (batch, outputs)
        for variant in OFF_POLICY_VARIANTS:
            assert fixture[f"{char}_off_{variant}_actions"].dtype == np.int32
            assert fixture[f"{char}_off_{variant}_weights"].shape == (batch,)
        for label in ("initial", "gradient", "final"):
            assert len(_parameters(fixture, f"{char}_{label}")) == 6
            assert len(_parameters(fixture, f"{shapley}_{label}")) == 6
        assert fixture[f"{shapley}_raw"].shape == (batch, features, outputs)
        assert fixture[f"{shapley}_sv_unique"].shape == (states, features, outputs)
        for label in ("with_exact_char", "without_exact_char"):
            assert fixture[f"{shapley}_exact_{label}"].shape == (states, features, outputs)


@pytest.mark.parametrize("case", CASES)
def test_exact_buffer_distribution_and_coalition_order(case: str) -> None:
    fixture = _load()
    features, _, _ = _dimensions(case)
    unique, counts = np.unique(fixture[f"{case}_exact_buffer"], axis=0, return_counts=True)
    np.testing.assert_array_equal(fixture[f"{case}_unique"], unique)
    np.testing.assert_array_equal(fixture[f"{case}_counts"], counts)
    np.testing.assert_array_equal(fixture[f"{case}_counts"], fixture[f"{case}_multiplicities"])
    assert len(set(counts.tolist())) > 1, "unequal multiplicities exercise the empirical distribution"
    # torch.cartesian_prod order: the first feature varies slowest.
    expected = np.asarray(list(itertools.product((False, True), repeat=features)))
    np.testing.assert_array_equal(fixture[f"{case}_coalitions"], expected)


@pytest.mark.parametrize("case", CASES)
def test_sampler_follows_shapley_kernel(case: str) -> None:
    fixture = _load()
    features, _, _ = _dimensions(case)
    sizes = np.arange(1, features)
    kernel = 1 / (sizes * (features - sizes))
    for kind in KINDS:
        name = f"{case}_char_{kind}"
        np.testing.assert_allclose(fixture[f"{name}_sampler_probs"], kernel / kernel.sum(), rtol=1e-6)
        np.testing.assert_array_equal(fixture[f"{name}_sampler_tril"], np.tril(np.ones((features - 1, features)), k=0))


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("label", tuple(EXACT_TARGETS))
def test_exact_characteristic_is_conditional_expectation(case: str, label: str) -> None:
    fixture = _load()
    unique = fixture[f"{case}_unique"]
    targets = fixture[f"{case}_agent_{EXACT_TARGETS[label]}"]
    distribution = fixture[f"{case}_counts"] / fixture[f"{case}_counts"].sum()
    table = fixture[f"{case}_exact_{label}"]
    for coalition, values in zip(fixture[f"{case}_coalitions"], table, strict=True):
        for state, value in zip(unique, values, strict=True):
            matches = (unique[:, coalition] == state[coalition]).all(axis=1)
            weights = distribution[matches] / distribution[matches].sum()
            np.testing.assert_allclose(value, weights @ targets[matches], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(table[-1], targets, atol=1e-6)
    np.testing.assert_allclose(fixture[f"{case}_exact_{label}_null"], table[0, 0], atol=1e-7)


@pytest.mark.parametrize("case", CASES)
def test_behaviour_outputs_are_distributions(case: str) -> None:
    fixture = _load()
    for key in (
        f"{case}_agent_greedy",
        f"{case}_agent_explore",
        f"{case}_exact_behaviour",
        f"{case}_exact_behaviour_explore",
        f"{case}_char_behaviour_grand",
        *(f"{case}_char_behaviour_predictions_{step}" for step in range(STEPS)),
    ):
        np.testing.assert_allclose(fixture[key].sum(axis=-1), 1, atol=1e-6, err_msg=key)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_exact_shapley_values_match_permutation_definition(case: str, kind: str) -> None:
    fixture = _load()
    expected = _shapley_values(fixture[f"{case}_coalitions"], fixture[f"{case}_exact_{kind}"])
    actual = fixture[f"{case}_shapley_{kind}_exact_with_exact_char"]
    np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_normalised_shapley_values_are_efficient(case: str, kind: str) -> None:
    fixture = _load()
    exact = fixture[f"{case}_exact_{kind}"]
    learnt = fixture[f"{case}_char_{kind}_grand"] - fixture[f"{case}_char_{kind}_null"]
    shapley = f"{case}_shapley_{kind}"
    np.testing.assert_allclose(fixture[f"{shapley}_exact_with_exact_char"].sum(axis=1), exact[-1] - exact[0], atol=1e-6)
    # Learnt normalisation uses the learnt characteristic at the grand and empty coalitions.
    np.testing.assert_allclose(fixture[f"{shapley}_exact_without_exact_char"].sum(axis=1), learnt, atol=1e-6)
    np.testing.assert_allclose(fixture[f"{shapley}_sv_unique"].sum(axis=1), learnt, atol=1e-6)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_off_policy_importance_weights(case: str, kind: str) -> None:
    fixture = _load()
    name = f"{case}_char_{kind}_off"
    ratios = np.exp(fixture[f"{name}_weighted_new_logprobs"] - fixture[f"{name}_weighted_old_logprobs"])
    weighted = fixture[f"{name}_weighted_weights"]
    np.testing.assert_allclose(weighted, ratios / ratios.sum(), rtol=1e-5)
    np.testing.assert_allclose(fixture[f"{name}_unweighted_weights"], ratios, rtol=1e-5)
    np.testing.assert_allclose(fixture[f"{name}_weighted_clipped_weights"], np.clip(weighted, 0.01, 1.99), rtol=1e-5)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_shapley_training_coalitions_are_proper_subsets(case: str, kind: str) -> None:
    fixture = _load()
    for step in range(STEPS):
        coalition = fixture[f"{case}_shapley_{kind}_coalition_{step}"]
        assert 0 < coalition.sum() < coalition.size


def _load() -> dict[str, np.ndarray]:
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        return {key: fixture[key] for key in fixture.files}


def _dimensions(case: str) -> tuple[int, int, int]:
    metadata = json.loads(FIXTURE.with_suffix(".json").read_text())
    layout = metadata["cases"][case]
    return int(layout["features"]), int(layout["actions"]), int(metadata["batch_size"])


def _parameters(fixture: dict[str, np.ndarray], prefix: str) -> list[np.ndarray]:
    return [
        fixture[f"{prefix}_{index}"]
        for index in itertools.takewhile(lambda i: f"{prefix}_{i}" in fixture, itertools.count())
    ]


def _shapley_values(coalitions: np.ndarray, table: np.ndarray) -> np.ndarray:
    """phi_i = sum over S not containing i of |S|!(F-|S|-1)!/F! * (v(S + i) - v(S))."""
    features = coalitions.shape[1]
    index = {tuple(row): position for position, row in enumerate(coalitions)}
    values = np.zeros((table.shape[1], features, table.shape[2]))
    for feature in range(features):
        for row in coalitions[~coalitions[:, feature]]:
            size = int(row.sum())
            weight = math.factorial(size) * math.factorial(features - size - 1) / math.factorial(features)
            joined = row.copy()
            joined[feature] = True
            values[:, feature] += weight * (table[index[tuple(joined)]] - table[index[tuple(row)]])
    return values
