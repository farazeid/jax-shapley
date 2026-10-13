"""Outcome characteristic and outcome Shapley parity with the pinned FastSVERL source.

The off-policy and Shapley losses draw their own rows, explained states, coalitions and
next actions; the fixture records them, so these tests replace the draws with the fixture's.
"""

from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax import Array
from jax.errors import JaxRuntimeError

from jax_shapley.algorithms import dqn
from jax_shapley.algorithms.dqn import Env, TransitionBatch
from jax_shapley.characteristic_models import behaviour, outcome
from jax_shapley.characteristic_models.behaviour import CharacteristicConfig, CharacteristicModel
from jax_shapley.characteristic_models.outcome import (
    ExactOutcome,
    OutcomeConfig,
    OutcomeContext,
    OutcomeExplained,
    OutcomeFrozen,
    OutcomeNetworks,
    OutcomeTrainConfig,
    Variant,
)
from jax_shapley.common.coalitions import coalition_table
from jax_shapley.common.networks import soft_update
from jax_shapley.common.supervised_learning import StateBuffers, SupervisedConfig, gradient_step, seed_slice
from jax_shapley.common.tabular import state_indices, transition_model
from jax_shapley.envs import Gridworld, Mastermind
from jax_shapley.shapley_models import behaviour as shapley
from jax_shapley.shapley_models import outcome as outcome_shapley
from jax_shapley.shapley_models.behaviour import ExactShapley, ShapleyConfig, ShapleyContext
from tests.fixture_support import (
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
ENVIRONMENTS: dict[str, Env] = {"gridworld": Gridworld(), "mastermind222": Mastermind(2, 2, 2)}
VARIANTS: dict[str, Variant] = {"off": "off_policy", "on": "on_policy"}
LABELS = ("with_exact_char", "without_exact_char")
TAU = 0.01
# Upstream iterates in float64 to a 1e-10 change; the port solves the same fixed point in float32.
EXACT_RTOL, EXACT_ATOL = 1e-5, 1e-4


@pytest.mark.parametrize("case", tuple(ENVIRONMENTS))
@pytest.mark.parametrize("prefix", tuple(VARIANTS))
def test_updates_match_source(case: str, prefix: str, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture, shape = _fixture(), layout(case)
    variant = VARIANTS[prefix]
    batch, explained_rows = _batch(fixture, case)
    rows = batch.observations.shape[0]
    training = OutcomeTrainConfig(batch_size=rows, train_size=2 * rows, validation_size=rows)
    config = OutcomeConfig(shape.env, variant=variant, tau=TAU, hidden_widths=(shape.width,) * 2, train=training)
    explained = CharacteristicConfig(shape.env, policy="explore")
    state = seed_slice(outcome.initialize(config, explained), 0)
    networks, optimizer = state.model, state.optimizer
    assign(networks.model.mlp, fixture, f"{case}_{prefix}_initial")
    assign(networks.target.mlp, fixture, f"{case}_{prefix}_initial")  # The source copies the model into its target.
    frozen = _frozen(shape)
    # Batch rows come first in the train prefix, the explained states after them.
    stacked = TransitionBatch(*[jnp.concatenate((leaf, leaf)) for leaf in jax.tree.leaves(batch)])
    context = OutcomeContext(
        replace(stacked, observations=jnp.concatenate((batch.observations, explained_rows))), stacked, None
    )
    loss = partial(outcome.loss, config, explained)
    for step in range(STEPS):
        # Both variants use the same fixed coalitions per step.
        coalitions = jnp.asarray(fixture[f"{case}_off_coalitions_{step}"])
        if prefix == "off":
            _fix_draws(monkeypatch, rows, coalitions, jnp.asarray(fixture[f"{case}_off_next_actions_{step}"]))
            arguments = (networks, frozen, context, training, shape.features, jax.random.key(0))
            value, gradients = cast(
                tuple[Array, nnx.State[str | int, nnx.Param[Array]]],
                nnx.value_and_grad(loss)(*arguments, "train"),
            )
            if step == 0:
                assert_gradients(_model_gradients(gradients), fixture, f"{case}_off_gradient")
            value = outcome.update(
                config, explained, networks, optimizer, frozen, context, training, shape.features, jax.random.key(0)
            )
        else:
            value, gradients = cast(
                tuple[Array, nnx.State[str | int, nnx.Param[Array]]],
                nnx.value_and_grad(outcome.on_policy_loss)(networks, batch, explained_rows, coalitions, config.gamma),
            )
            if step == 0:
                assert_gradients(_model_gradients(gradients), fixture, f"{case}_on_gradient")
            optimizer.update(networks, gradients)
            soft_update(networks.model.mlp, networks.target.mlp, config.tau)  # The collector's update_target.
        np.testing.assert_allclose(value, fixture[f"{case}_{prefix}_loss_{step}"], rtol=RTOL, atol=ATOL)
    assert_parameters(networks.model.mlp, fixture, f"{case}_{prefix}_final")
    assert_parameters(networks.target.mlp, fixture, f"{case}_{prefix}_target")


@pytest.mark.parametrize("case", tuple(ENVIRONMENTS))
@pytest.mark.parametrize("prefix", tuple(VARIANTS))
def test_values_and_null_match_source(case: str, prefix: str) -> None:
    fixture, shape = _fixture(), layout(case)
    config, explained, characteristic = _characteristic(fixture, shape, VARIANTS[prefix], f"{case}_{prefix}_final")
    unique = jnp.asarray(load_fixture()[f"{case}_unique"])
    probe = jnp.asarray(fixture[f"{case}_probe_coalitions"])
    # get_val is the source's sum over the conditioned policy (off-policy) or V at e (on-policy).
    np.testing.assert_allclose(
        outcome.values(config, explained, characteristic, unique, probe),
        fixture[f"{case}_{prefix}_get_val"],
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        outcome.null(config, characteristic, unique), fixture[f"{case}_{prefix}_null"], rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("case", tuple(ENVIRONMENTS))
def test_exact_values_and_losses_match_source(case: str) -> None:
    fixture, shape = _fixture(), layout(case)
    exact, columns = _exact(shape)
    for label in LABELS:
        np.testing.assert_allclose(
            exact[label].values[:, columns], fixture[f"{case}_exact_{label}"], rtol=EXACT_RTOL, atol=EXACT_ATOL
        )
        for prefix, variant in VARIANTS.items():
            config, explained, characteristic = _characteristic(fixture, shape, variant, f"{case}_{prefix}_final")
            np.testing.assert_allclose(
                outcome.exact_values_loss(config, explained, characteristic, exact[label]),
                fixture[f"{case}_{prefix}_exact_loss_{label}"],
                rtol=EXACT_RTOL,
                atol=EXACT_ATOL,
            )


@pytest.mark.parametrize("case", tuple(ENVIRONMENTS))
def test_outcome_shapley_matches_source(case: str, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture, shape = _fixture(), layout(case)
    config, explained, characteristic = _characteristic(fixture, shape, "off_policy", f"{case}_off_final")
    observations = _batch(fixture, case)[0].observations
    rows = observations.shape[0]
    training = SupervisedConfig(batch_size=rows, buffer_size=rows * 5 // 4, num_envs=1)
    shapley_config = ShapleyConfig(shape.env, hidden_widths=(shape.width,) * 2, train=training)
    state = seed_slice(shapley.initialize(shapley_config, explained, outcome_shapley.head), 0)
    model, optimizer = state.model, state.optimizer
    assign(model.mlp, fixture, f"{case}_shapley_initial")
    values, null = partial(outcome.values, config, explained), partial(outcome.null, config)
    loss = partial(shapley.loss, values, null)
    context = ShapleyContext(StateBuffers(observations, observations, observations), None)
    for step in range(STEPS):
        coalition = jnp.asarray(fixture[f"{case}_shapley_coalition_{step}"])
        _fix_shapley_draws(monkeypatch, coalition)
        if step == 0:
            _, gradients = cast(
                tuple[Array, nnx.State[str | int, nnx.Param[Array]]],
                nnx.value_and_grad(loss)(
                    model, characteristic, context, training, shape.features, jax.random.key(0), "train"
                ),
            )
            assert_gradients(gradients, fixture, f"{case}_shapley_gradient")
        value = gradient_step(
            loss, model, optimizer, characteristic, context, training, shape.features, jax.random.key(0)
        )
        np.testing.assert_allclose(value, fixture[f"{case}_shapley_loss_{step}"], rtol=RTOL, atol=ATOL)
    assert_parameters(model.mlp, fixture, f"{case}_shapley_final")
    states = jnp.asarray(load_fixture()[f"{case}_unique"])
    np.testing.assert_allclose(
        shapley.explain(values, null, model, characteristic, states),
        fixture[f"{case}_shapley_sv_unique"],
        rtol=RTOL,
        atol=ATOL,
    )
    exact, columns = _exact(shape)
    coalitions = jnp.asarray(coalition_table(shape.features))
    tables = {
        "with_exact_char": exact["with_exact_char"].values[:, columns, None],
        "without_exact_char": values(characteristic, states[None], coalitions[:, None]),
    }
    for label, table in tables.items():
        solved = shapley.exact_shapley(table)
        tolerance = (EXACT_RTOL, EXACT_ATOL) if label == "with_exact_char" else (RTOL, ATOL)
        np.testing.assert_allclose(
            solved, fixture[f"{case}_shapley_exact_{label}"], rtol=tolerance[0], atol=tolerance[1]
        )
        solved_context = ShapleyContext(
            context.buffers, ExactShapley(states, jnp.ones(states.shape[0], jnp.bool_), solved)
        )
        exact_loss = loss(model, characteristic, solved_context, training, shape.features, jax.random.key(0), "exact")
        np.testing.assert_allclose(
            exact_loss, fixture[f"{case}_shapley_exact_loss_{label}"], rtol=tolerance[0], atol=tolerance[1]
        )


def test_config_and_network_shapes() -> None:
    env = Mastermind(2, 2, 2)
    explained = CharacteristicConfig(env)
    for variant, outputs in (("off_policy", 4), ("on_policy", 1)):
        config = OutcomeConfig(env, variant=variant, hidden_widths=(5,))
        networks = seed_slice(outcome.initialize(config, explained), 0).model
        assert networks.model.mlp.config.input_features == 3 * 8
        assert networks.model.mlp.config.output_features == outputs
    for invalid in ({"tau": 0.0}, {"gamma": 1.5}, {"hidden_widths": (0,)}):
        with pytest.raises(ValueError):
            OutcomeConfig(env, **invalid)  # ty: ignore[invalid-argument-type] -- runtime validation of untyped input.
    with pytest.raises(ValueError):
        OutcomeConfig(env, variant="outcome")  # ty: ignore[invalid-argument-type] -- runtime validation of untyped input.


def test_exact_outcome_fails_when_the_conditioned_policy_can_loop_forever() -> None:
    """A greedy agent that always takes one action never reaches Gridworld's goal: with gamma = 1
    upstream's value iteration hangs, and the port stops the run."""
    env = Gridworld()
    tabular = transition_model(env)
    model = dqn.initialize(dqn.DQNConfig(env=env, hidden_widths=()), jax.random.key(0)).model
    model.mlp.layers[0].kernel[...] = jnp.zeros((2, 4), jnp.float32)
    model.mlp.layers[0].bias[...] = jnp.asarray([1.0, 0.0, 0.0, 0.0])
    explained = CharacteristicConfig(env, hidden_widths=())
    characteristic: CharacteristicModel = seed_slice(behaviour.initialize(explained, behaviour.head), 0).model
    frozen = OutcomeFrozen(dqn.Agent(model, jnp.asarray(0.0, jnp.float32)), characteristic)
    with pytest.raises(ValueError, match="non-finite"):
        outcome.exact_outcome(tabular, tabular.states, frozen, explained, True, 1.0)
    compiled = nnx.jit(partial(outcome.exact_outcome, tabular, explained=explained, with_exact_char=False, gamma=1.0))
    with pytest.raises(JaxRuntimeError, match="non-finite"):
        jax.block_until_ready(compiled(tabular.states, frozen))


def _exact(shape: CaseLayout) -> tuple[dict[str, ExactOutcome], Array]:
    """Production exact outcome values (at every modelled state), and the columns of upstream's unique states."""
    explanations = load_fixture()
    tabular = transition_model(ENVIRONMENTS[shape.name])
    observations = jnp.asarray(explanations[f"{shape.name}_exact_buffer"])
    unique = jnp.asarray(explanations[f"{shape.name}_unique"])
    explained = CharacteristicConfig(shape.env, policy="explore")
    frozen = _frozen(shape)
    exact = {
        label: outcome.exact_outcome(tabular, observations, frozen, explained, label == "with_exact_char", 1.0)
        for label in LABELS
    }
    columns, modelled = state_indices(tabular, unique)
    present = exact["with_exact_char"].state_valid
    assert bool(modelled.all()) and int(present.sum()) == unique.shape[0] and bool(present[columns].all())
    return exact, columns


def _characteristic(
    fixture: FixtureArrays, shape: CaseLayout, variant: Variant, prefix: str
) -> tuple[OutcomeConfig, CharacteristicConfig, OutcomeExplained]:
    config = OutcomeConfig(shape.env, variant=variant, tau=TAU, hidden_widths=(shape.width,) * 2)
    explained = CharacteristicConfig(shape.env, policy="explore")
    networks: OutcomeNetworks = seed_slice(outcome.initialize(config, explained), 0).model
    assign(networks.model.mlp, fixture, prefix)
    assign(networks.target.mlp, fixture, prefix)
    frozen = _frozen(shape)
    return config, explained, OutcomeExplained(frozen.agent, frozen.behaviour, networks)


def _frozen(shape: CaseLayout) -> OutcomeFrozen:
    """The fixture's explore agent and its trained behaviour characteristic."""
    explanations = load_fixture()
    config = CharacteristicConfig(shape.env, hidden_widths=(shape.width,) * 2)
    model: CharacteristicModel = seed_slice(behaviour.initialize(config, behaviour.head), 0).model
    assign(model.mlp, explanations, f"{shape.name}_char_behaviour_final")
    return OutcomeFrozen(load_agent(explanations, shape), model)


def _model_gradients(gradients: nnx.State[str | int, nnx.Param[Array]]) -> nnx.State[str | int, nnx.Param[Array]]:
    return cast(nnx.State[str | int, nnx.Param[Array]], gradients["model"])


def _batch(fixture: FixtureArrays, case: str) -> tuple[TransitionBatch, Array]:
    batch = TransitionBatch(
        jnp.asarray(fixture[f"{case}_batch_obs"], jnp.int32),
        jnp.asarray(fixture[f"{case}_batch_action"], jnp.int32),
        jnp.asarray(fixture[f"{case}_batch_reward"], jnp.float32),
        jnp.asarray(fixture[f"{case}_batch_next_obs"], jnp.int32),
        jnp.asarray(fixture[f"{case}_batch_terminated"]) > 0,
        jnp.zeros(fixture[f"{case}_batch_obs"].shape[0], jnp.float32),
    )
    return batch, jnp.asarray(fixture[f"{case}_batch_e_obs"], jnp.int32)


def _fix_draws(monkeypatch: pytest.MonkeyPatch, rows: int, coalitions: Array, next_actions: Array) -> None:
    """The off-policy loss's draws: the batch rows in order, then the explained states after them,
    the fixture's coalitions, and its next actions."""
    indices: list[Array] = [jnp.arange(rows, dtype=jnp.int32), rows + jnp.arange(rows, dtype=jnp.int32)]

    def randint(key: Array, shape: tuple[int, ...], minval: int, maxval: int) -> Array:
        del key, shape, minval, maxval
        return indices.pop(0)

    def bernoulli(key: Array, p: float, shape: tuple[int, ...]) -> Array:
        del key, p
        assert shape == coalitions.shape
        return coalitions

    def categorical(key: Array, logits: Array, axis: int) -> Array:
        del key, axis
        assert logits.shape[0] == next_actions.shape[0]
        return next_actions

    # The loss is traced twice per step: once for gradients, once inside the update.
    indices.extend(indices.copy())
    monkeypatch.setattr(jax.random, "randint", randint)
    monkeypatch.setattr(jax.random, "bernoulli", bernoulli)
    monkeypatch.setattr(jax.random, "categorical", categorical)


def _fix_shapley_draws(monkeypatch: pytest.MonkeyPatch, coalition: Array) -> None:
    """The Shapley loss's draws: the batch rows in order and the fixture's kernel coalition."""

    def randint(key: Array, shape: tuple[int, ...], minval: int, maxval: int) -> Array:
        del key, minval, maxval
        return jnp.arange(shape[0], dtype=jnp.int32)

    def kernel_coalitions(key: Array, batch: int, features: int) -> Array:
        del key
        assert (batch, features) == (1, coalition.shape[0])
        return coalition[None]

    monkeypatch.setattr(jax.random, "randint", randint)
    monkeypatch.setattr(shapley, "kernel_coalitions", kernel_coalitions)


def _fixture() -> FixtureArrays:
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        return {key: fixture[key] for key in fixture.files}
