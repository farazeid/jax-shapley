"""Compiled explanation code equals eager evaluation at production batch sizes.

With jax 0.11.0, XLA CPU miscompiled a network applied to a broadcast-constant batch from
about 64 rows; the null values therefore use one all-masked row. These checks guard that.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jax_shapley.algorithms import dqn
from jax_shapley.algorithms.dqn import Env, TransitionBatch
from jax_shapley.characteristic_models import behaviour, outcome
from jax_shapley.characteristic_models.behaviour import CharacteristicConfig, CharacteristicContext, CharacteristicModel
from jax_shapley.characteristic_models.outcome import OutcomeConfig, OutcomeContext, OutcomeExplained, Variant
from jax_shapley.common.coalitions import coalition_table
from jax_shapley.common.supervised_learning import (
    StateBuffers,
    SupervisedConfig,
    seed_slice,
    stack_seeds,
)
from jax_shapley.envs import Gridworld, Hypercube, Mastermind
from jax_shapley.shapley_models import behaviour as shapley
from jax_shapley.shapley_models.behaviour import ExactShapley, ShapleyConfig, ShapleyContext, ShapleyModel

RTOL, ATOL = 1e-5, 1e-6


@pytest.mark.parametrize("rows", [63, 64, 128, 2000])
@pytest.mark.parametrize("env", [Gridworld(), Hypercube(2, 5)], ids=["gridworld", "hypercube"])
def test_shapley_targets_and_values_match_eager(rows: int, env: Env) -> None:
    characteristic, model, features = _models(env, width=32)
    observations = jax.random.randint(jax.random.key(1), (rows, features), 0, 2, jnp.int32)
    null = shapley.masked_null
    np.testing.assert_allclose(nnx.jit(null)(characteristic, observations), null(characteristic, observations))
    training = SupervisedConfig(batch_size=rows, buffer_size=rows * 5 // 4 + 5, num_envs=1)
    context = ShapleyContext(StateBuffers(observations, observations, observations), None)
    loss = partial(
        shapley.loss, shapley.masked_values, shapley.masked_null, config=training, features=features, partition="train"
    )
    key = jax.random.key(2)
    np.testing.assert_allclose(
        nnx.jit(loss)(model, characteristic, context, key=key),
        loss(model, characteristic, context, key=key),
        rtol=RTOL,
        atol=ATOL,
    )
    explain = partial(shapley.explain, shapley.masked_values, shapley.masked_null)
    np.testing.assert_allclose(
        nnx.jit(explain)(model, characteristic, observations),
        explain(model, characteristic, observations),
        rtol=RTOL,
        atol=ATOL,
    )


@pytest.mark.parametrize("rows", [64, 2000])
def test_characteristic_losses_and_tables_match_eager(rows: int) -> None:
    characteristic, model, features = _models(Hypercube(2, 5), width=32)
    observations = jax.random.randint(jax.random.key(3), (rows, features), 0, 3, jnp.int32)
    targets = jax.nn.softmax(jax.random.normal(jax.random.key(4), (rows, 10)))
    config = CharacteristicConfig(Hypercube(2, 5))
    training = SupervisedConfig(batch_size=rows, buffer_size=rows * 5 // 4 + 5, num_envs=1)
    every = jnp.asarray(coalition_table(features))
    exact = jax.jit(partial(behaviour.exact_characteristic, coalitions=every, max_unique=243))(observations, targets)
    context = CharacteristicContext(StateBuffers(observations, observations, observations), targets, targets, exact)
    key = jax.random.key(5)
    for partition in ("train", "validation", "exact"):
        loss = partial(behaviour.loss, config, config=training, features=features, partition=partition)
        np.testing.assert_allclose(
            nnx.jit(loss)(characteristic, None, context, key=key),
            loss(characteristic, None, context, key=key),
            rtol=RTOL,
            atol=ATOL,
        )
    table = nnx.jit(shapley.masked_values)(characteristic, exact.states[None], every[:, None])
    np.testing.assert_allclose(
        table, shapley.masked_values(characteristic, exact.states[None], every[:, None]), rtol=RTOL, atol=ATOL
    )
    solved = ShapleyContext(
        context.buffers, ExactShapley(exact.states, exact.state_valid, shapley.exact_shapley(table))
    )
    shapley_loss = partial(
        shapley.loss, shapley.masked_values, shapley.masked_null, config=training, features=features, partition="exact"
    )
    np.testing.assert_allclose(
        nnx.jit(shapley_loss)(model, characteristic, solved, key=key),
        shapley_loss(model, characteristic, solved, key=key),
        rtol=RTOL,
    )


@pytest.mark.parametrize("rows", [63, 64, 128, 2000])
@pytest.mark.parametrize("variant", ["off_policy", "on_policy"])
def test_outcome_values_null_and_losses_match_eager(rows: int, variant: Variant) -> None:
    env = Mastermind(2, 2, 2)
    characteristic, _, features = _models(env, width=32)
    agent = dqn.Agent(
        dqn.initialize(dqn.DQNConfig(env=env, hidden_widths=(32,)), jax.random.key(2)).model, jnp.asarray(0.2)
    )
    explained = CharacteristicConfig(env, policy="explore")
    config = OutcomeConfig(env, variant=variant, hidden_widths=(32, 32))
    networks = seed_slice(outcome.initialize(config, explained), 0).model
    explained_outcome = OutcomeExplained(agent, characteristic, networks)
    observations = jax.random.randint(jax.random.key(6), (rows, features), 0, 3, jnp.int32)
    coalitions = jax.random.bernoulli(jax.random.key(7), 0.5, (rows, features)).at[0].set(False)
    values = partial(outcome.values, config, explained)
    np.testing.assert_allclose(
        nnx.jit(values)(explained_outcome, observations, coalitions),
        values(explained_outcome, observations, coalitions),
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        nnx.jit(partial(outcome.null, config))(explained_outcome, observations),
        values(explained_outcome, observations, jnp.zeros_like(coalitions)),
        rtol=RTOL,
        atol=ATOL,
    )
    batch = TransitionBatch(
        observations,
        jnp.arange(rows, dtype=jnp.int32) % 4,
        jnp.full((rows,), -1.0, jnp.float32),
        jnp.roll(observations, 1, axis=0),
        jnp.arange(rows) % 5 == 0,
        jnp.zeros(rows, jnp.float32),
    )
    if variant == "on_policy":
        on_policy = partial(outcome.on_policy_loss, gamma=1.0)
        np.testing.assert_allclose(
            nnx.jit(on_policy)(networks, batch, observations, coalitions),
            on_policy(networks, batch, observations, coalitions),
            rtol=RTOL,
            atol=ATOL,
        )
        return
    training = outcome.OutcomeTrainConfig(batch_size=rows, train_size=rows, validation_size=rows)
    loss = partial(outcome.loss, config, explained, training=training, features=features, partition="train")
    frozen = outcome.OutcomeFrozen(agent, characteristic)
    context, key = OutcomeContext(batch, batch, None), jax.random.key(8)
    np.testing.assert_allclose(
        nnx.jit(loss)(networks, frozen, context, key=key),
        loss(networks, frozen, context, key=key),
        rtol=RTOL,
        atol=ATOL,
    )


def test_logged_exact_loss_matches_eager_evaluation_of_the_returned_model() -> None:
    """End to end: the best epoch's logged exact loss is the restored best model's eager exact loss."""
    env = Mastermind(2, 2, 2)
    training = SupervisedConfig(epochs=6, batch_size=128, buffer_size=640, num_envs=16)
    explained = CharacteristicConfig(env, hidden_widths=(16,), train=training)
    agent = dqn.Agent(
        dqn.initialize(dqn.DQNConfig(env=env, hidden_widths=(16,)), jax.random.key(0)).model, jnp.asarray(0.1)
    )
    characteristics = behaviour.initialize(explained, behaviour.head, jnp.stack([jax.random.key(2)]))
    characteristic = seed_slice(characteristics, 0)
    config = ShapleyConfig(env, hidden_widths=(16,), with_exact_char=False, train=training)
    _, inherited = nnx.jit(partial(behaviour.collect, explained))(
        jnp.stack([jax.random.key(1)]), stack_seeds([agent]), None
    )
    state, metrics = shapley.main(config, explained, stack_seeds([agent]), characteristics.model, inherited)
    assert metrics is not None
    result = seed_slice(state, 0)
    exact_rows = seed_slice(inherited, 0).exact
    every = jnp.asarray(coalition_table(8))
    exact = behaviour.exact_characteristic(
        exact_rows, jnp.zeros((exact_rows.shape[0], 4)), every, max_unique=exact_rows.shape[0]
    )
    learnt = shapley.masked_values(characteristic.model, exact.states[None], every[:, None])
    context = ShapleyContext(
        result.buffers, ExactShapley(exact.states, exact.state_valid, shapley.exact_shapley(learnt))
    )
    eager = shapley.loss(
        shapley.masked_values,
        shapley.masked_null,
        result.model,
        characteristic.model,
        context,
        training,
        8,
        jax.random.key(0),
        "exact",
    )
    epochs = np.asarray(metrics.epoch)[:, 0]
    best = int(result.best_epoch)
    logged = np.asarray(metrics.exact_loss)[:, 0][epochs == best]
    np.testing.assert_allclose(logged, [eager], rtol=1e-5)


def _models(env: Env, *, width: int) -> tuple[CharacteristicModel, ShapleyModel, int]:
    explained = CharacteristicConfig(env, hidden_widths=(width, width))
    characteristic = seed_slice(behaviour.initialize(explained, behaviour.head), 0).model
    config = ShapleyConfig(env, hidden_widths=(width, width))
    model = seed_slice(shapley.initialize(config, explained, behaviour.head), 0).model
    return characteristic, model, characteristic.mlp.config.input_features
