"""Behaviour/prediction Shapley model parity with the pinned FastSVERL source."""

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
    characteristic_network,
    exact_characteristic,
    explained_values,
)
from jax_shapley.coalitions import all_coalitions
from jax_shapley.shapley import (
    MaskedCharacteristic,
    ShapleyLayout,
    ShapleyModel,
    characteristic_table,
    exact_shapley,
    exact_shapley_loss,
    explain,
    normalise,
    shapley_loss,
    shapley_matrix,
    shapley_network,
    update_shapley,
)
from jax_shapley.train_dqn import AdamConfig, make_adam
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


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_training_matches_source(case: str, kind: Kind) -> None:
    fixture, shape = load_fixture(), layout(case)
    name = f"{case}_shapley_{kind}"
    characteristic = _characteristic(fixture, shape, kind)
    model = _shapley(fixture, shape, kind, f"{name}_initial")
    optimizer = make_adam(model, shape.learning_rate, AdamConfig())
    batch = jnp.asarray(fixture[f"{case}_exact_buffer"][: shape.batch_size])
    for step in range(STEPS):
        coalition = jnp.asarray(fixture[f"{name}_coalition_{step}"])
        targets = characteristic.values(batch, coalition) - characteristic.null(batch)
        np.testing.assert_allclose(targets, fixture[f"{name}_targets_{step}"], rtol=RTOL, atol=ATOL)
        if step == 0:
            result = cast(
                tuple[tuple[Array, LossMetrics], nnx.State[str | int, nnx.Param[Array]]],
                nnx.value_and_grad(shapley_loss, has_aux=True)(model, characteristic, batch, coalition),
            )
            (loss, _), gradients = result
            np.testing.assert_allclose(loss, fixture[f"{name}_loss_0"], rtol=RTOL, atol=ATOL)
            assert_gradients(gradients, fixture, f"{name}_gradient")
        metrics = update_shapley(model, optimizer, characteristic, batch, coalition)
        np.testing.assert_allclose(metrics.loss, fixture[f"{name}_loss_{step}"], rtol=RTOL, atol=ATOL)
    assert_parameters(model.mlp, fixture, f"{name}_final")
    raw = model(batch).contributions
    np.testing.assert_allclose(raw, fixture[f"{name}_raw"], rtol=RTOL, atol=ATOL)
    grand = characteristic.model(batch).values
    normalised = normalise(raw, grand, characteristic.null(batch))
    np.testing.assert_allclose(normalised, fixture[f"{name}_normalised"], rtol=RTOL, atol=ATOL)
    unique = jnp.asarray(fixture[f"{case}_unique"])
    np.testing.assert_allclose(
        explain(model, characteristic, unique), fixture[f"{name}_sv_unique"], rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_exact_values_and_losses_match_source(case: str, kind: Kind) -> None:
    fixture, shape = load_fixture(), layout(case)
    name = f"{case}_shapley_{kind}"
    characteristic = _characteristic(fixture, shape, kind)
    model = _shapley(fixture, shape, kind, f"{name}_final")
    observations = jnp.asarray(fixture[f"{case}_exact_buffer"])
    targets = explained_values(
        load_agent(fixture, shape)(observations).q_values, CharacteristicConfig(kind), jnp.asarray(shape.epsilon)
    )
    exact = exact_characteristic(observations, targets, all_coalitions(shape.features), max_unique=32)
    states = fixture[f"{case}_unique"].shape[0]
    learnt = characteristic_table(characteristic, exact.states, all_coalitions(shape.features))
    for label, values in (("with_exact_char", exact.at_states), ("without_exact_char", learnt)):
        solved = exact_shapley(values)
        np.testing.assert_allclose(solved[:states], fixture[f"{name}_exact_{label}"], rtol=1e-4, atol=1e-6)
        loss = exact_shapley_loss(model, characteristic, exact.states, exact.state_valid, solved)
        np.testing.assert_allclose(loss, fixture[f"{name}_exact_loss_{label}"], rtol=1e-4)


def test_least_squares_matrix_reproduces_permutation_shapley_values() -> None:
    # v(S) = |S|^2 over three features: every player contributes 3 by symmetry.
    coalitions = np.asarray(all_coalitions(3))
    values = jnp.asarray(coalitions.sum(axis=1, keepdims=True)[:, None, :] ** 2, jnp.float32)
    np.testing.assert_allclose(exact_shapley(values)[0, :, 0], [3.0, 3.0, 3.0], rtol=1e-6)
    np.testing.assert_allclose(shapley_matrix(2), [[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]], atol=1e-7)


def test_layout_validation() -> None:
    layout = ShapleyLayout(3, 2)
    model = ShapleyModel(shapley_network(layout, (4,)), layout, rngs=nnx.Rngs(0))
    assert model(jnp.zeros((5, 3), jnp.int32)).contributions.shape == (5, 3, 2)
    with pytest.raises(ValueError):
        ShapleyModel(shapley_network(ShapleyLayout(3, 1), (4,)), layout, rngs=nnx.Rngs(0))
    with pytest.raises(ValueError):
        ShapleyLayout(1, 1)


def _characteristic(fixture: FixtureArrays, shape: CaseLayout, kind: Kind) -> MaskedCharacteristic:
    config = CharacteristicConfig(kind, hidden_widths=(shape.width,) * 2)
    model = CharacteristicModel(characteristic_network(shape.features, shape.actions, config), rngs=nnx.Rngs(0))
    # Shapley capture follows characteristic training, so it explains the trained characteristic.
    assign(model.mlp, fixture, f"{shape.name}_char_{kind}_final")
    return MaskedCharacteristic(model)


def _shapley(fixture: FixtureArrays, shape: CaseLayout, kind: Kind, prefix: str) -> ShapleyModel:
    layout = ShapleyLayout(shape.features, shape.actions if kind == "behaviour" else 1)
    model = ShapleyModel(shapley_network(layout, (shape.width,) * 2), layout, rngs=nnx.Rngs(0))
    assign(model.mlp, fixture, prefix)
    return model
