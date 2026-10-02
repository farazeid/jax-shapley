"""Behaviour/prediction characteristic parity with the pinned FastSVERL source."""

from math import inf
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax import Array

from jax_shapley.characteristics import (
    CharacteristicConfig,
    CharacteristicModel,
    Kind,
    LossMetrics,
    OffPolicyConfig,
    characteristic_loss,
    characteristic_network,
    exact_characteristic,
    exact_characteristic_loss,
    explained_values,
    importance_weighted_loss,
    max_unique_states,
    off_policy_weights,
    policy_probabilities,
    update_characteristic,
)
from jax_shapley.coalitions import MASK, all_coalitions, mask
from jax_shapley.train_dqn import AdamConfig, action_log_probabilities, make_adam
from tests.explanation_support import (
    ATOL,
    CASES,
    KINDS,
    RTOL,
    STEPS,
    CaseLayout,
    FixtureArrays,
    assert_gradients,
    assert_parameters,
    assign,
    layout,
    load_agent,
    load_fixture,
)

OFF_POLICY = {
    "weighted": OffPolicyConfig(True, inf),
    "unweighted": OffPolicyConfig(False, inf),
    "weighted_clipped": OffPolicyConfig(True, 0.99),
}


@pytest.mark.parametrize("case", CASES)
def test_explained_agent_outputs(case: str) -> None:
    fixture, shape = load_fixture(), layout(case)
    agent = load_agent(fixture, shape)
    q_values = agent(jnp.asarray(fixture[f"{case}_unique"])).q_values
    epsilon = jnp.asarray(shape.epsilon, jnp.float32)
    for policy, key in (("greedy", "greedy"), ("explore", "explore")):
        config = CharacteristicConfig("behaviour", policy)
        np.testing.assert_allclose(
            explained_values(q_values, config, epsilon), fixture[f"{case}_agent_{key}"], rtol=1e-6, atol=1e-7
        )
    values = explained_values(q_values, CharacteristicConfig("prediction"), epsilon)
    np.testing.assert_allclose(values, fixture[f"{case}_agent_value"], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_on_policy_training_matches_source(case: str, kind: Kind) -> None:
    fixture, shape = load_fixture(), layout(case)
    name = f"{case}_char_{kind}"
    agent = load_agent(fixture, shape)
    model, optimizer = _characteristic(fixture, shape, kind, f"{name}_initial")
    batch = jnp.asarray(fixture[f"{case}_exact_buffer"][: shape.batch_size])
    targets = explained_values(agent(batch).q_values, CharacteristicConfig(kind), jnp.asarray(shape.epsilon))
    np.testing.assert_allclose(targets, fixture[f"{name}_targets"], rtol=1e-6, atol=1e-6)
    for step in range(STEPS):
        coalitions = jnp.asarray(fixture[f"{name}_coalitions_{step}"])
        predictions = model(mask(batch, coalitions)).values
        np.testing.assert_allclose(predictions, fixture[f"{name}_predictions_{step}"], rtol=RTOL, atol=ATOL)
        if step == 0:
            loss, gradients = _gradients(model, batch, targets, coalitions)
            np.testing.assert_allclose(loss, fixture[f"{name}_loss_0"], rtol=RTOL, atol=ATOL)
            assert_gradients(gradients, fixture, f"{name}_gradient")
        metrics = update_characteristic(model, optimizer, batch, targets, coalitions)
        np.testing.assert_allclose(metrics.loss, fixture[f"{name}_loss_{step}"], rtol=RTOL, atol=ATOL)
    assert_parameters(model.mlp, fixture, f"{name}_final")
    unique = jnp.asarray(fixture[f"{case}_unique"])
    np.testing.assert_allclose(model(unique).values, fixture[f"{name}_grand"], rtol=RTOL, atol=ATOL)
    null = model(jnp.full((1, shape.features), MASK, jnp.int32)).values[0]
    np.testing.assert_allclose(null, fixture[f"{name}_null"], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("variant", tuple(OFF_POLICY))
def test_off_policy_training_matches_source(case: str, kind: Kind, variant: str) -> None:
    fixture, shape = load_fixture(), layout(case)
    name = f"{case}_char_{kind}_off_{variant}"
    agent = load_agent(fixture, shape)
    model, optimizer = _characteristic(fixture, shape, kind, f"{case}_char_{kind}_initial")
    batch = jnp.asarray(fixture[f"{case}_exact_buffer"][: shape.batch_size])
    q_values = agent(batch).q_values
    targets = explained_values(q_values, CharacteristicConfig(kind), jnp.asarray(shape.epsilon))
    actions = jnp.asarray(fixture[f"{name}_actions"])
    # The source evaluates the explained (greedy) policy's log-probabilities for buffer actions.
    new = action_log_probabilities(policy_probabilities(q_values, "greedy", jnp.asarray(shape.epsilon)), actions)
    np.testing.assert_allclose(new, fixture[f"{name}_new_logprobs"], rtol=1e-6, atol=1e-6)
    weights = off_policy_weights(new, jnp.asarray(fixture[f"{name}_old_logprobs"]), OFF_POLICY[variant])
    np.testing.assert_allclose(weights, fixture[f"{name}_weights"], rtol=1e-5, atol=1e-12)
    for step in range(STEPS):
        coalitions = jnp.asarray(fixture[f"{case}_char_{kind}_coalitions_{step}"])
        result = cast(
            tuple[tuple[Array, LossMetrics], nnx.State[str | int, nnx.Param[Array]]],
            nnx.value_and_grad(importance_weighted_loss, has_aux=True)(model, batch, targets, coalitions, weights),
        )
        (loss, _), gradients = result
        np.testing.assert_allclose(loss, fixture[f"{name}_loss_{step}"], rtol=RTOL, atol=1e-9)
        if step == 0:
            assert_gradients(gradients, fixture, f"{name}_gradient")
        optimizer.update(model, gradients)
    assert_parameters(model.mlp, fixture, f"{name}_final")


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("label", ("behaviour", "behaviour_explore", "prediction"))
def test_exact_characteristic_matches_source(case: str, label: str) -> None:
    fixture, shape = load_fixture(), layout(case)
    config = CharacteristicConfig(
        "prediction" if label == "prediction" else "behaviour", "explore" if label.endswith("explore") else "greedy"
    )
    observations = jnp.asarray(fixture[f"{case}_exact_buffer"])
    targets = explained_values(load_agent(fixture, shape)(observations).q_values, config, jnp.asarray(shape.epsilon))
    # Padding beyond the true number of unique states must not change valid entries.
    exact = exact_characteristic(
        observations, targets, all_coalitions(shape.features), max_unique=observations.shape[0]
    )
    unique = fixture[f"{case}_unique"]
    states = unique.shape[0]
    np.testing.assert_array_equal(exact.states[:states], unique)
    assert int(exact.state_valid.sum()) == states
    np.testing.assert_allclose(exact.at_states[:, :states], fixture[f"{case}_exact_{label}"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(exact.null, fixture[f"{case}_exact_{label}_null"], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_exact_loss_matches_source(case: str, kind: Kind) -> None:
    fixture, shape = load_fixture(), layout(case)
    name = f"{case}_char_{kind}"
    model, _ = _characteristic(fixture, shape, kind, f"{name}_final")
    observations = jnp.asarray(fixture[f"{case}_exact_buffer"])
    agent = load_agent(fixture, shape)
    targets = explained_values(agent(observations).q_values, CharacteristicConfig(kind), jnp.asarray(shape.epsilon))
    exact = exact_characteristic(observations, targets, all_coalitions(shape.features), max_unique=32)
    np.testing.assert_allclose(exact_characteristic_loss(model, exact), fixture[f"{name}_exact_loss"], rtol=1e-4)


def test_exact_values_weight_rows_by_frequency() -> None:
    observations = jnp.asarray([[0, 0], [0, 1], [0, 1], [1, 1]], jnp.int32)
    targets = jnp.asarray([[0.0], [3.0], [3.0], [9.0]])
    exact = exact_characteristic(observations, targets, all_coalitions(2), max_unique=4)
    np.testing.assert_allclose(exact.null, [15.0 / 4])
    # Knowing only feature 0 = 0 averages rows [0, 0], [0, 1], [0, 1].
    np.testing.assert_allclose(exact.at_states[2, :3, 0], [2.0, 2.0, 9.0])
    np.testing.assert_array_equal(exact.valid[0], [True, False, False, False])
    np.testing.assert_array_equal(exact.state_valid, [True, True, True, False])


def test_static_unique_bound() -> None:
    assert max_unique_states(0, 5, 5, 100_000) == 7776
    assert max_unique_states(np.asarray([0, 0]), np.asarray([1, 3]), 2, 8000) == 8
    assert max_unique_states(-1, 2, 8, 8000) == 8000
    with pytest.raises(ValueError):
        max_unique_states(2, 1, 1, 10)


def test_configuration_validation_and_networks() -> None:
    behaviour = characteristic_network(8, 4, CharacteristicConfig("behaviour", hidden_widths=(3,)))
    assert behaviour.layers[-1].activation == "softmax" and behaviour.output_features == 4
    prediction = characteristic_network(8, 4, CharacteristicConfig("prediction"))
    assert prediction.layers[-1].activation == "identity" and prediction.output_features == 1
    with pytest.raises(ValueError):
        CharacteristicConfig("outcome")  # ty: ignore[invalid-argument-type] -- runtime validation of untyped input.
    with pytest.raises(ValueError):
        OffPolicyConfig(clip_coef=-1)


def _characteristic(
    fixture: FixtureArrays, shape: CaseLayout, kind: Kind, prefix: str
) -> tuple[CharacteristicModel, nnx.Optimizer[CharacteristicModel]]:
    config = CharacteristicConfig(kind, hidden_widths=(shape.width,) * 2)
    model = CharacteristicModel(characteristic_network(shape.features, shape.actions, config), rngs=nnx.Rngs(0))
    assign(model.mlp, fixture, prefix)
    return model, make_adam(model, shape.learning_rate, AdamConfig())


def _gradients(
    model: CharacteristicModel, batch: Array, targets: Array, coalitions: Array
) -> tuple[Array, nnx.State[str | int, nnx.Param[Array]]]:
    result = cast(
        tuple[tuple[Array, LossMetrics], nnx.State[str | int, nnx.Param[Array]]],
        nnx.value_and_grad(characteristic_loss, has_aux=True)(model, batch, targets, coalitions),
    )
    (loss, _), gradients = result
    return loss, gradients
