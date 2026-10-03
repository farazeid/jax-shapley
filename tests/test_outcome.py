"""Outcome characteristic and outcome Shapley parity with the pinned FastSVERL source."""

from collections.abc import Callable
from pathlib import Path
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax import Array

from jax_shapley.characteristics import (
    CharacteristicConfig,
    CharacteristicModel,
    LossMetrics,
    characteristic_network,
    exact_characteristic,
)
from jax_shapley.coalitions import all_coalitions
from jax_shapley.outcome import (
    AgentPolicy,
    ConditionedPolicy,
    ExactOutcome,
    OutcomeCharacteristic,
    OutcomeConfig,
    OutcomeNetworks,
    Transitions,
    Variant,
    exact_behaviour_table,
    exact_outcome,
    exact_outcome_loss,
    learnt_behaviour_table,
    off_policy_loss,
    on_policy_loss,
    outcome_network,
    outcome_value,
    soft_update,
    update_off_policy,
    update_on_policy,
)
from jax_shapley.shapley import (
    MaskedCharacteristic,
    ShapleyLayout,
    ShapleyModel,
    characteristic_table,
    exact_shapley,
    exact_shapley_loss,
    explain,
    shapley_loss,
    shapley_network,
    update_shapley,
)
from jax_shapley.tabular import state_indices, transition_model
from jax_shapley.train_dqn import AdamConfig, EnvironmentConfig, GridworldConfig, Mastermind222Config, make_adam
from tests.explanation_support import (
    ATOL,
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

FIXTURE = Path(__file__).parent / "fixtures" / "fastsverl_outcome.npz"
CONFIGS: dict[str, EnvironmentConfig] = {"gridworld": GridworldConfig(), "mastermind222": Mastermind222Config()}
VARIANTS: dict[str, Variant] = {"off": "off_policy", "on": "on_policy"}
LABELS = ("with_exact_char", "without_exact_char")
TAU = 0.01
# Upstream iterates in float64 to a 1e-10 change; the port solves the same fixed point in float32.
EXACT_RTOL, EXACT_ATOL = 1e-5, 1e-4


@pytest.mark.parametrize("case", tuple(CONFIGS))
@pytest.mark.parametrize("prefix", tuple(VARIANTS))
def test_updates_match_source(case: str, prefix: str) -> None:
    fixture, shape = _fixture(), layout(case)
    variant = VARIANTS[prefix]
    networks = _networks(fixture, shape, variant, f"{case}_{prefix}_initial")
    optimizer = make_adam(networks, shape.learning_rate, AdamConfig())
    config = OutcomeConfig(variant, tau=TAU, hidden_widths=(shape.width,) * 2)
    batch, explained = _batch(fixture, case)
    first = jnp.asarray(fixture[f"{case}_off_coalitions_0"])
    first_actions = jnp.asarray(fixture[f"{case}_off_next_actions_0"])
    if prefix == "off":
        _assert_gradients(
            lambda model: off_policy_loss(model, networks.target, batch, explained, first, first_actions, config.gamma),
            networks.model,
            fixture,
            f"{case}_off_gradient",
        )
    else:
        _assert_gradients(
            lambda model: on_policy_loss(model, networks.target, batch, explained, first, config.gamma),
            networks.model,
            fixture,
            f"{case}_on_gradient",
        )
    for step in range(STEPS):
        # Both variants use the same fixed coalitions per step.
        coalitions = jnp.asarray(fixture[f"{case}_off_coalitions_{step}"])
        if prefix == "off":
            next_actions = jnp.asarray(fixture[f"{case}_off_next_actions_{step}"])
            metrics = update_off_policy(networks, optimizer, batch, explained, coalitions, next_actions, config)
        else:
            metrics = update_on_policy(networks, optimizer, batch, explained, coalitions, config)
            soft_update(networks, config.tau)  # The collector follows each update with update_target.
        np.testing.assert_allclose(metrics.loss, fixture[f"{case}_{prefix}_loss_{step}"], rtol=RTOL, atol=ATOL)
    assert_parameters(networks.model.mlp, fixture, f"{case}_{prefix}_final")
    assert_parameters(networks.target.mlp, fixture, f"{case}_{prefix}_target")


@pytest.mark.parametrize("case", tuple(CONFIGS))
@pytest.mark.parametrize("prefix", tuple(VARIANTS))
def test_values_and_null_match_source(case: str, prefix: str) -> None:
    fixture, shape = _fixture(), layout(case)
    characteristic = _characteristic(fixture, shape, VARIANTS[prefix], f"{case}_{prefix}_final")
    unique = jnp.asarray(load_fixture()[f"{case}_unique"])
    probe = jnp.asarray(fixture[f"{case}_probe_coalitions"])
    np.testing.assert_allclose(
        characteristic.values(unique, probe), fixture[f"{case}_{prefix}_get_val"], rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(characteristic.null(unique), fixture[f"{case}_{prefix}_null"], rtol=RTOL, atol=ATOL)
    # get_val is the source's sum over the conditioned policy (off-policy) or V at e (on-policy).
    direct = outcome_value(characteristic.networks.model, characteristic.policy, unique, probe, characteristic.variant)
    np.testing.assert_allclose(direct, fixture[f"{case}_{prefix}_get_val"], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("case", tuple(CONFIGS))
def test_exact_values_and_losses_match_source(case: str) -> None:
    fixture, shape = _fixture(), layout(case)
    exact, columns = _exact(shape)
    for label in LABELS:
        np.testing.assert_allclose(
            exact[label].values[:, columns], fixture[f"{case}_exact_{label}"], rtol=EXACT_RTOL, atol=EXACT_ATOL
        )
        for prefix, variant in VARIANTS.items():
            characteristic = _characteristic(fixture, shape, variant, f"{case}_{prefix}_final")
            loss = exact_outcome_loss(characteristic, exact[label], all_coalitions(shape.features))
            np.testing.assert_allclose(
                loss, fixture[f"{case}_{prefix}_exact_loss_{label}"], rtol=EXACT_RTOL, atol=EXACT_ATOL
            )


@pytest.mark.parametrize("case", tuple(CONFIGS))
def test_outcome_shapley_matches_source(case: str) -> None:
    fixture, shape = _fixture(), layout(case)
    characteristic = _characteristic(fixture, shape, "off_policy", f"{case}_off_final")
    shapley_layout = ShapleyLayout(shape.features, 1)
    model = ShapleyModel(shapley_network(shapley_layout, (shape.width,) * 2), shapley_layout, rngs=nnx.Rngs(0))
    assign(model.mlp, fixture, f"{case}_shapley_initial")
    optimizer = make_adam(model, shape.learning_rate, AdamConfig())
    observations = _batch(fixture, case)[0].observations
    first = jnp.asarray(fixture[f"{case}_shapley_coalition_0"])
    _assert_gradients(
        lambda shapley: shapley_loss(shapley, characteristic, observations, first),
        model,
        fixture,
        f"{case}_shapley_gradient",
    )
    for step in range(STEPS):
        coalition = jnp.asarray(fixture[f"{case}_shapley_coalition_{step}"])
        metrics = update_shapley(model, optimizer, characteristic, observations, coalition)
        np.testing.assert_allclose(metrics.loss, fixture[f"{case}_shapley_loss_{step}"], rtol=RTOL, atol=ATOL)
    assert_parameters(model.mlp, fixture, f"{case}_shapley_final")
    states = jnp.asarray(load_fixture()[f"{case}_unique"])
    np.testing.assert_allclose(
        explain(model, characteristic, states), fixture[f"{case}_shapley_sv_unique"], rtol=RTOL, atol=ATOL
    )
    exact, columns = _exact(shape)
    coalitions = all_coalitions(shape.features)
    tables = {
        "with_exact_char": exact["with_exact_char"].values[:, columns, None],
        "without_exact_char": characteristic_table(characteristic, states, coalitions),
    }
    valid = jnp.ones(states.shape[0], jnp.bool_)
    for label, table in tables.items():
        solved = exact_shapley(table)
        tolerance = (EXACT_RTOL, EXACT_ATOL) if label == "with_exact_char" else (RTOL, ATOL)
        np.testing.assert_allclose(
            solved, fixture[f"{case}_shapley_exact_{label}"], rtol=tolerance[0], atol=tolerance[1]
        )
        loss = exact_shapley_loss(model, characteristic, states, valid, solved)
        np.testing.assert_allclose(
            loss, fixture[f"{case}_shapley_exact_loss_{label}"], rtol=tolerance[0], atol=tolerance[1]
        )


def test_config_and_network_shapes() -> None:
    off = outcome_network(3, 4, OutcomeConfig("off_policy", hidden_widths=(5,)))
    on = outcome_network(3, 4, OutcomeConfig("on_policy", hidden_widths=(5,)))
    assert off.input_features == on.input_features == 9
    assert off.output_features == 4 and on.output_features == 1
    for invalid in ({"tau": 0.0}, {"gamma": 1.5}, {"hidden_widths": (0,)}):
        with pytest.raises(ValueError):
            OutcomeConfig(**invalid)  # ty: ignore[invalid-argument-type] -- runtime validation of untyped input.
    with pytest.raises(ValueError):
        OutcomeConfig("outcome")  # ty: ignore[invalid-argument-type] -- runtime validation of untyped input.


def _assert_gradients[Model: nnx.Module](
    loss: Callable[[Model], tuple[Array, LossMetrics]], model: Model, fixture: FixtureArrays, prefix: str
) -> None:
    # NNX transformation has a dynamic return signature; assert its known loss/state boundary.
    result = cast(
        tuple[tuple[Array, LossMetrics], nnx.State[str | int, nnx.Param[Array]]],
        nnx.value_and_grad(loss, has_aux=True)(model),
    )
    assert_gradients(result[1], fixture, prefix)


def _exact(shape: CaseLayout) -> tuple[dict[str, ExactOutcome], Array]:
    """Production exact outcome values (at every modelled state), and the columns of upstream's unique states.

    The exact behaviour characteristic evaluated at modelled states must agree with
    exact_characteristic at the buffer's unique states.
    """
    explanations = load_fixture()
    policy = _policy(shape)
    tabular = transition_model(CONFIGS[shape.name])
    observations = jnp.asarray(explanations[f"{shape.name}_exact_buffer"])
    unique = jnp.asarray(explanations[f"{shape.name}_unique"])
    coalitions = all_coalitions(shape.features)
    targets = policy.agent(observations)
    behaviour = exact_characteristic(observations, targets, coalitions, max_unique=int(unique.shape[0]))
    np.testing.assert_array_equal(behaviour.states, unique)
    exact_table, present = exact_behaviour_table(observations, targets, tabular.states, coalitions)
    columns, modelled = state_indices(tabular, unique)
    assert bool(modelled.all()) and int(present.sum()) == unique.shape[0] and bool(present[columns].all())
    np.testing.assert_allclose(exact_table[:, columns], behaviour.at_states, rtol=1e-6, atol=1e-7)
    agent = policy.agent(tabular.states)
    conditioned = {
        "with_exact_char": exact_table,
        "without_exact_char": learnt_behaviour_table(policy.behaviour, tabular.states, coalitions),
    }
    exact = {
        label: exact_outcome(tabular, tabular.states, present, table, agent, 1.0)
        for label, table in conditioned.items()
    }
    return exact, columns


def _characteristic(fixture: FixtureArrays, shape: CaseLayout, variant: Variant, prefix: str) -> OutcomeCharacteristic:
    networks = _networks(fixture, shape, variant, prefix)
    return OutcomeCharacteristic(networks, _policy(shape), variant)


def _networks(fixture: FixtureArrays, shape: CaseLayout, variant: Variant, prefix: str) -> OutcomeNetworks:
    config = OutcomeConfig(variant, tau=TAU, hidden_widths=(shape.width,) * 2)
    networks = OutcomeNetworks(outcome_network(shape.features, shape.actions, config), rngs=nnx.Rngs(0))
    assign(networks.model.mlp, fixture, prefix)
    assign(networks.target.mlp, fixture, prefix)  # The source copies the model into its target critic.
    return networks


def _policy(shape: CaseLayout) -> ConditionedPolicy:
    explanations = load_fixture()
    config = CharacteristicConfig("behaviour", hidden_widths=(shape.width,) * 2)
    behaviour = CharacteristicModel(characteristic_network(shape.features, shape.actions, config), rngs=nnx.Rngs(0))
    assign(behaviour.mlp, explanations, f"{shape.name}_char_behaviour_final")
    agent = AgentPolicy(load_agent(explanations, shape), "explore", jnp.asarray(shape.epsilon, jnp.float32))
    return ConditionedPolicy(MaskedCharacteristic(behaviour), agent)


def _batch(fixture: FixtureArrays, case: str) -> tuple[Transitions, Array]:
    batch = Transitions(
        jnp.asarray(fixture[f"{case}_batch_obs"], jnp.int32),
        jnp.asarray(fixture[f"{case}_batch_action"], jnp.int32),
        jnp.asarray(fixture[f"{case}_batch_reward"], jnp.float32),
        jnp.asarray(fixture[f"{case}_batch_next_obs"], jnp.int32),
        jnp.asarray(fixture[f"{case}_batch_terminated"]) > 0,
    )
    return batch, jnp.asarray(fixture[f"{case}_batch_e_obs"], jnp.int32)


def _fixture() -> FixtureArrays:
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        return {key: fixture[key] for key in fixture.files}
