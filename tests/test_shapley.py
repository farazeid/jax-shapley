"""Behaviour/prediction Shapley model parity with the pinned FastSVERL source.

The training loss draws its own rows and coalition; the fixture fixes both, so the
training test replaces the two draws with the fixture's batch order and coalition.
"""

from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax import Array

from new_jax_shapley.characteristic_models import behaviour, prediction
from new_jax_shapley.characteristic_models.behaviour import CharacteristicConfig, CharacteristicModel, Kind
from new_jax_shapley.common.coalitions import coalition_table
from new_jax_shapley.common.networks import LayerConfig, MLPConfig
from new_jax_shapley.common.supervised_learning import (
    StateBuffers,
    SupervisedConfig,
    TrainState,
    gradient_step,
    seed_slice,
)
from new_jax_shapley.shapley_models import behaviour as shapley
from new_jax_shapley.shapley_models.behaviour import (
    ExactShapley,
    ShapleyConfig,
    ShapleyContext,
    ShapleyModel,
    masked_null,
    masked_values,
)
from tests.fixture_support import (
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

TARGETS = {"behaviour": behaviour.targets, "prediction": prediction.targets}
HEADS = {"behaviour": behaviour.head, "prediction": prediction.head}


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_training_matches_source(case: str, kind: Kind, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture, shape = load_fixture(), layout(case)
    name = f"{case}_shapley_{kind}"
    characteristic = _characteristic(fixture, shape, kind)
    training = SupervisedConfig(batch_size=shape.batch_size, buffer_size=shape.batch_size * 5 // 4, num_envs=1)
    state = _shapley(shape, kind, training)
    model, optimizer = state.model, state.optimizer
    assign(model.mlp, fixture, f"{name}_initial")
    batch = jnp.asarray(fixture[f"{case}_exact_buffer"][: shape.batch_size])
    context = ShapleyContext(StateBuffers(batch, batch, batch), None)
    loss = partial(shapley.loss, masked_values, masked_null)
    for step in range(STEPS):
        coalition = jnp.asarray(fixture[f"{name}_coalition_{step}"])
        targets = masked_values(characteristic, batch, coalition) - masked_null(characteristic, batch)
        np.testing.assert_allclose(targets, fixture[f"{name}_targets_{step}"], rtol=RTOL, atol=ATOL)
        _fix_draws(monkeypatch, coalition)
        if step == 0:
            value, gradients = cast(
                tuple[Array, nnx.State[str | int, nnx.Param[Array]]],
                nnx.value_and_grad(loss)(
                    model, characteristic, context, training, shape.features, jax.random.key(0), "train"
                ),
            )
            np.testing.assert_allclose(value, fixture[f"{name}_loss_0"], rtol=RTOL, atol=ATOL)
            assert_gradients(gradients, fixture, f"{name}_gradient")
        value = gradient_step(
            loss, model, optimizer, characteristic, context, training, shape.features, jax.random.key(0)
        )
        np.testing.assert_allclose(value, fixture[f"{name}_loss_{step}"], rtol=RTOL, atol=ATOL)
    assert_parameters(model.mlp, fixture, f"{name}_final")
    np.testing.assert_allclose(model(batch).contributions, fixture[f"{name}_raw"], rtol=RTOL, atol=ATOL)
    normalised = shapley.explain(masked_values, masked_null, model, characteristic, batch)
    np.testing.assert_allclose(normalised, fixture[f"{name}_normalised"], rtol=RTOL, atol=ATOL)
    unique = jnp.asarray(fixture[f"{case}_unique"])
    np.testing.assert_allclose(
        shapley.explain(masked_values, masked_null, model, characteristic, unique),
        fixture[f"{name}_sv_unique"],
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_exact_values_and_losses_match_source(case: str, kind: Kind) -> None:
    fixture, shape = load_fixture(), layout(case)
    name = f"{case}_shapley_{kind}"
    characteristic = _characteristic(fixture, shape, kind)
    model = _shapley(shape, kind, SupervisedConfig()).model
    assign(model.mlp, fixture, f"{name}_final")
    observations = jnp.asarray(fixture[f"{case}_exact_buffer"])
    targets = TARGETS[kind](CharacteristicConfig(shape.env, kind=kind), load_agent(fixture, shape), observations)
    coalitions = jnp.asarray(coalition_table(shape.features))
    exact = behaviour.exact_characteristic(observations, targets, coalitions, max_unique=32)
    states = fixture[f"{case}_unique"].shape[0]
    learnt = masked_values(characteristic, exact.states[None], coalitions[:, None])
    for label, values in (("with_exact_char", exact.at_states), ("without_exact_char", learnt)):
        solved = shapley.exact_shapley(values)
        np.testing.assert_allclose(solved[:states], fixture[f"{name}_exact_{label}"], rtol=1e-4, atol=1e-6)
        context = ShapleyContext(
            StateBuffers(observations, observations, observations),
            ExactShapley(exact.states, exact.state_valid, solved),
        )
        loss = shapley.loss(
            masked_values,
            masked_null,
            model,
            characteristic,
            context,
            SupervisedConfig(),
            shape.features,
            jax.random.key(0),
            "exact",
        )
        np.testing.assert_allclose(loss, fixture[f"{name}_exact_loss_{label}"], rtol=1e-4)


def test_least_squares_reproduces_permutation_shapley_values() -> None:
    # v(S) = |S|^2 over three features: every player contributes 3 by symmetry.
    coalitions = coalition_table(3)
    values = jnp.asarray(coalitions.sum(axis=1, keepdims=True)[:, None, :] ** 2, jnp.float32)
    np.testing.assert_allclose(shapley.exact_shapley(values)[0, :, 0], [3.0, 3.0, 3.0], rtol=1e-6)


def test_layout_validation() -> None:
    hidden = LayerConfig(4, "relu")
    model = ShapleyModel(MLPConfig(3, (hidden, LayerConfig(6))), 3, rngs=nnx.Rngs(0))
    assert model(jnp.zeros((5, 3), jnp.int32)).contributions.shape == (5, 3, 2)
    with pytest.raises(ValueError):
        ShapleyModel(MLPConfig(3, (hidden, LayerConfig(4))), 3, rngs=nnx.Rngs(0))
    with pytest.raises(ValueError):
        ShapleyModel(MLPConfig(3, (hidden, LayerConfig(6, "relu"))), 3, rngs=nnx.Rngs(0))
    with pytest.raises(ValueError):
        ShapleyModel(MLPConfig(1, (hidden, LayerConfig(1))), 1, rngs=nnx.Rngs(0))


def _characteristic(fixture: FixtureArrays, shape: CaseLayout, kind: Kind) -> CharacteristicModel:
    config = CharacteristicConfig(shape.env, kind=kind, hidden_widths=(shape.width,) * 2)
    model = seed_slice(behaviour.initialize(config, HEADS[kind]), 0).model
    # Shapley capture follows characteristic training, so it explains the trained characteristic.
    assign(model.mlp, fixture, f"{shape.name}_char_{kind}_final")
    return model


def _shapley(shape: CaseLayout, kind: Kind, training: SupervisedConfig) -> TrainState[ShapleyModel]:
    config = ShapleyConfig(shape.env, hidden_widths=(shape.width,) * 2, train=training)
    explained = CharacteristicConfig(shape.env, kind=kind, train=training)
    return seed_slice(shapley.initialize(config, explained, HEADS[kind]), 0)


def _fix_draws(monkeypatch: pytest.MonkeyPatch, coalition: Array) -> None:
    """The loss's batch rows become the buffer in order, and its kernel coalition the fixture's."""

    def rows(key: Array, shape: tuple[int, ...], minval: int, maxval: int) -> Array:
        del key, minval, maxval
        return jnp.arange(shape[0], dtype=jnp.int32)

    def draw(key: Array, batch: int, features: int) -> Array:
        del key
        assert (batch, features) == (1, coalition.shape[0])
        return coalition[None]

    monkeypatch.setattr(jax.random, "randint", rows)
    monkeypatch.setattr(shapley, "kernel_coalitions", draw)
