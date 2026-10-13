"""Behaviour/prediction characteristic parity with the pinned FastSVERL source.

The training loss draws its own rows and coalitions; the fixture fixes both, so these tests
replace the two draws with the fixture's batch order and coalitions.
"""

from functools import partial
from math import inf
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax import Array

from jax_shapley.algorithms import dqn
from jax_shapley.characteristic_models import behaviour, prediction
from jax_shapley.characteristic_models.behaviour import (
    EXACT_CHUNK,
    CharacteristicConfig,
    CharacteristicContext,
    CharacteristicModel,
    Kind,
)
from jax_shapley.common.coalitions import MASK, coalition_table, mask
from jax_shapley.common.supervised_learning import (
    ReplayBehaviour,
    StateBuffers,
    SupervisedConfig,
    gradient_step,
    seed_slice,
)
from tests.fixture_support import (
    ATOL,
    CASES,
    KINDS,
    RTOL,
    STEPS,
    assert_gradients,
    assert_parameters,
    assign,
    layout,
    load_agent,
    load_fixture,
)

# Weighting and clip coefficient of each source off-policy variant.
OFF_POLICY = {"weighted": (True, inf), "unweighted": (False, inf), "weighted_clipped": (True, 0.99)}
TARGETS = {"behaviour": behaviour.targets, "prediction": prediction.targets}
HEADS = {"behaviour": behaviour.head, "prediction": prediction.head}


@pytest.mark.parametrize("case", CASES)
def test_explained_agent_outputs(case: str) -> None:
    fixture, shape = load_fixture(), layout(case)
    agent = load_agent(fixture, shape)
    unique = jnp.asarray(fixture[f"{case}_unique"])
    for policy in ("greedy", "explore"):
        values = behaviour.targets(CharacteristicConfig(shape.env, policy=policy), agent, unique)
        np.testing.assert_allclose(values, fixture[f"{case}_agent_{policy}"], rtol=1e-6, atol=1e-7)
    values = prediction.targets(CharacteristicConfig(shape.env, kind="prediction"), agent, unique)
    np.testing.assert_allclose(values, fixture[f"{case}_agent_value"], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("kind", KINDS)
def test_on_policy_training_matches_source(case: str, kind: Kind, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture, shape = load_fixture(), layout(case)
    name = f"{case}_char_{kind}"
    config = CharacteristicConfig(
        shape.env, kind=kind, hidden_widths=(shape.width,) * 2, train=_training(shape.batch_size)
    )
    state = seed_slice(behaviour.initialize(config, HEADS[kind]), 0)
    model, optimizer = state.model, state.optimizer
    assign(model.mlp, fixture, f"{name}_initial")
    batch = jnp.asarray(fixture[f"{case}_exact_buffer"][: shape.batch_size])
    targets = TARGETS[kind](config, load_agent(fixture, shape), batch)
    np.testing.assert_allclose(targets, fixture[f"{name}_targets"], rtol=1e-6, atol=1e-6)
    context = CharacteristicContext(StateBuffers(batch, batch, batch), targets, targets, None)
    loss = partial(behaviour.loss, config)
    for step in range(STEPS):
        coalitions = jnp.asarray(fixture[f"{name}_coalitions_{step}"])
        _fix_draws(monkeypatch, coalitions)
        predictions = model(mask(batch, coalitions)).values
        np.testing.assert_allclose(predictions, fixture[f"{name}_predictions_{step}"], rtol=RTOL, atol=ATOL)
        if step == 0:
            value, gradients = cast(
                tuple[Array, nnx.State[str | int, nnx.Param[Array]]],
                nnx.value_and_grad(loss)(
                    model, None, context, config.train, shape.features, jax.random.key(0), "train"
                ),
            )
            np.testing.assert_allclose(value, fixture[f"{name}_loss_0"], rtol=RTOL, atol=ATOL)
            assert_gradients(gradients, fixture, f"{name}_gradient")
        value = gradient_step(loss, model, optimizer, None, context, config.train, shape.features, jax.random.key(0))
        np.testing.assert_allclose(value, fixture[f"{name}_loss_{step}"], rtol=RTOL, atol=ATOL)
    assert_parameters(model.mlp, fixture, f"{name}_final")
    unique = jnp.asarray(fixture[f"{case}_unique"])
    np.testing.assert_allclose(model(unique).values, fixture[f"{name}_grand"], rtol=RTOL, atol=ATOL)
    null = model(jnp.full((1, shape.features), MASK, jnp.int32)).values[0]
    np.testing.assert_allclose(null, fixture[f"{name}_null"], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("variant", tuple(OFF_POLICY))
def test_off_policy_training_matches_source(
    case: str, kind: Kind, variant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, shape = load_fixture(), layout(case)
    name = f"{case}_char_{kind}_off_{variant}"
    weighting, clip_coef = OFF_POLICY[variant]
    config = CharacteristicConfig(
        shape.env,
        kind=kind,
        hidden_widths=(shape.width,) * 2,
        data="replay",
        off_policy=True,
        weighting=weighting,
        clip_coef=clip_coef,
        train=_training(shape.batch_size, exact_loss=False),
    )
    state = seed_slice(behaviour.initialize(config, HEADS[kind]), 0)
    model, optimizer = state.model, state.optimizer
    assign(model.mlp, fixture, f"{case}_char_{kind}_initial")
    agent = load_agent(fixture, shape)
    batch = jnp.asarray(fixture[f"{case}_exact_buffer"][: shape.batch_size])
    targets = TARGETS[kind](config, agent, batch)
    actions = jnp.asarray(fixture[f"{name}_actions"])
    # The source evaluates the explained (greedy) policy's log-probabilities for buffer actions.
    explained = dqn.log_probabilities(behaviour.targets(config, agent, batch), actions)
    np.testing.assert_allclose(explained, fixture[f"{name}_new_logprobs"], rtol=1e-6, atol=1e-6)
    logged = jnp.asarray(fixture[f"{name}_old_logprobs"])
    replay = ReplayBehaviour(actions, logged, actions, logged)
    context = CharacteristicContext(
        StateBuffers(batch, batch, batch, replay), targets, targets, None, explained, explained
    )
    loss = partial(behaviour.loss, config)
    for step in range(STEPS):
        _fix_draws(monkeypatch, jnp.asarray(fixture[f"{case}_char_{kind}_coalitions_{step}"]))
        value, gradients = cast(
            tuple[Array, nnx.State[str | int, nnx.Param[Array]]],
            nnx.value_and_grad(loss)(model, None, context, config.train, shape.features, jax.random.key(0), "train"),
        )
        np.testing.assert_allclose(value, fixture[f"{name}_loss_{step}"], rtol=RTOL, atol=1e-9)
        if step == 0:
            assert_gradients(gradients, fixture, f"{name}_gradient")
        optimizer.update(model, gradients)
    assert_parameters(model.mlp, fixture, f"{name}_final")


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("label", ("behaviour", "behaviour_explore", "prediction"))
def test_exact_characteristic_matches_source(case: str, label: str) -> None:
    fixture, shape = load_fixture(), layout(case)
    kind: Kind = "prediction" if label == "prediction" else "behaviour"
    config = CharacteristicConfig(shape.env, kind=kind, policy="explore" if label.endswith("explore") else "greedy")
    observations = jnp.asarray(fixture[f"{case}_exact_buffer"])
    targets = TARGETS[kind](config, load_agent(fixture, shape), observations)
    # Padding beyond the true number of unique states must not change valid entries.
    coalitions = jnp.asarray(coalition_table(shape.features))
    exact = behaviour.exact_characteristic(observations, targets, coalitions, max_unique=observations.shape[0])
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
    config = CharacteristicConfig(shape.env, kind=kind, hidden_widths=(shape.width,) * 2)
    model = seed_slice(behaviour.initialize(config, HEADS[kind]), 0).model
    assign(model.mlp, fixture, f"{name}_final")
    observations = jnp.asarray(fixture[f"{case}_exact_buffer"])
    targets = TARGETS[kind](config, load_agent(fixture, shape), observations)
    coalitions = jnp.asarray(coalition_table(shape.features))
    exact = behaviour.exact_characteristic(observations, targets, coalitions, max_unique=32)
    np.testing.assert_allclose(behaviour.exact_loss(model, exact), fixture[f"{name}_exact_loss"], rtol=1e-4)


def test_exact_values_weight_rows_by_frequency() -> None:
    observations = jnp.asarray([[0, 0], [0, 1], [0, 1], [1, 1]], jnp.int32)
    targets = jnp.asarray([[0.0], [3.0], [3.0], [9.0]])
    exact = behaviour.exact_characteristic(observations, targets, jnp.asarray(coalition_table(2)), max_unique=4)
    np.testing.assert_allclose(exact.null, [15.0 / 4])
    # Knowing only feature 0 = 0 averages rows [0, 0], [0, 1], [0, 1].
    np.testing.assert_allclose(exact.at_states[2, :3, 0], [2.0, 2.0, 9.0])
    np.testing.assert_array_equal(exact.state_valid, [True, True, True, False])


def test_chunked_exact_loss_equals_the_dense_masked_mean() -> None:
    observations = jax.random.randint(jax.random.key(0), (600, 5), 0, 3, jnp.int32)
    targets = jax.random.uniform(jax.random.key(1), (600, 1))
    coalitions = jnp.asarray(coalition_table(5))
    exact = behaviour.exact_characteristic(observations, targets, coalitions, max_unique=243)
    assert int(exact.count) > EXACT_CHUNK  # Valid entries span more than one chunk.
    config = CharacteristicConfig(layout("hypercube").env, kind="prediction", hidden_widths=(8,))
    model: CharacteristicModel = seed_slice(behaviour.initialize(config, prediction.head), 0).model
    # Independent dense reference: every coalition's distinct masked observations, each valued
    # at the mean target of the rows that share it, each counted once.
    rows, values = np.asarray(observations), np.asarray(targets)
    keys, means = [], []
    for coalition in coalition_table(5):
        masked = np.where(coalition, rows, MASK)
        unique, inverse = np.unique(masked, axis=0, return_inverse=True)
        keys.append(unique)
        means.append(np.stack([values[inverse.reshape(-1) == index].mean(0) for index in range(len(unique))]))
    reference_keys, reference_values = np.concatenate(keys), np.concatenate(means)
    assert int(exact.count) == len(reference_keys)
    errors = jnp.square(model(jnp.asarray(reference_keys, jnp.int32)).values - reference_values)
    dense = jnp.mean(errors)
    np.testing.assert_allclose(behaviour.exact_loss(model, exact), dense, rtol=1e-5)
    np.testing.assert_allclose(nnx.jit(behaviour.exact_loss)(model, exact), dense, rtol=1e-5)


def test_configuration_validation_and_networks() -> None:
    env = layout("mastermind222").env
    model = seed_slice(behaviour.initialize(CharacteristicConfig(env, hidden_widths=(3,)), behaviour.head), 0).model
    assert model.mlp.config.layers[-1].activation == "softmax" and model.mlp.config.output_features == 4
    config = CharacteristicConfig(env, kind="prediction")
    model = seed_slice(behaviour.initialize(config, prediction.head), 0).model
    assert model.mlp.config.layers[-1].activation == "identity" and model.mlp.config.output_features == 1
    with pytest.raises(ValueError):
        CharacteristicConfig(env, kind="outcome")  # ty: ignore[invalid-argument-type] -- runtime validation of untyped input.
    with pytest.raises(ValueError):
        CharacteristicConfig(env, clip_coef=-1)
    with pytest.raises(ValueError):
        CharacteristicConfig(env, off_policy=True)  # Off-policy weighting needs replay data.


def _training(batch_size: int, *, exact_loss: bool = True) -> SupervisedConfig:
    """One batch of `batch_size` train rows; the source's learning rate is the default."""
    return SupervisedConfig(batch_size=batch_size, buffer_size=batch_size * 5 // 4, num_envs=1, exact_loss=exact_loss)


def _fix_draws(monkeypatch: pytest.MonkeyPatch, coalitions: Array) -> None:
    """The loss's batch rows become the buffer in order, and its coalitions the fixture's."""

    def rows(key: Array, shape: tuple[int, ...], minval: int, maxval: int) -> Array:
        del key, minval, maxval
        return jnp.arange(shape[0], dtype=jnp.int32)

    def draw(key: Array, p: float, shape: tuple[int, ...]) -> Array:
        del key, p
        assert shape == coalitions.shape
        return coalitions

    monkeypatch.setattr(jax.random, "randint", rows)
    monkeypatch.setattr(jax.random, "bernoulli", draw)
