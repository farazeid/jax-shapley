"""Abstract compilation of coalition, characteristic and Shapley units at production shapes and dtypes."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jax_shapley.characteristics import (
    CharacteristicConfig,
    CharacteristicModel,
    Kind,
    OffPolicyConfig,
    Policy,
    characteristic_network,
    exact_characteristic,
    exact_characteristic_loss,
    explained_values,
    off_policy_weights,
    update_characteristic,
)
from jax_shapley.coalitions import all_coalitions, kernel_coalitions, mask, uniform_coalitions
from jax_shapley.shapley import (
    ShapleyLayout,
    ShapleyModel,
    characteristic_table,
    exact_shapley,
    exact_shapley_loss,
    shapley_network,
    update_shapley,
)
from jax_shapley.train_dqn import AdamConfig, make_adam

# (features, actions, width, exact rows, static unique bound) for Gridworld, Mastermind-222 and Hypercube 6x5.
SHAPES = ((2, 4, 64, 40, 8), (8, 4, 64, 40, 40), (5, 10, 128, 40, 40))
KEY = jax.ShapeDtypeStruct((), jax.random.key(0).dtype)


@pytest.mark.compile
@pytest.mark.parametrize("features", [2, 5, 8])
def test_samplers_and_mask_compile(features: int) -> None:
    for sampler in (
        partial(uniform_coalitions, batch=7, features=features),
        partial(kernel_coalitions, batch=7, features=features),
        partial(kernel_coalitions, batch=7, features=features, paired=True),
    ):
        compiled = jax.jit(sampler).trace(KEY).lower().compile()
        coalitions = compiled(jax.random.key(0))
        assert coalitions.shape == (7, features) and coalitions.dtype == jnp.bool_
    observations = jax.ShapeDtypeStruct((3, 7, features), jnp.int32)
    coalitions = jax.ShapeDtypeStruct((7, features), jnp.bool_)
    masked = jax.jit(mask).trace(observations, coalitions).lower().compile()
    result = masked(jnp.zeros(observations.shape, jnp.int32), jnp.ones(coalitions.shape, jnp.bool_))
    assert result.dtype == jnp.int32


@pytest.mark.compile
@pytest.mark.parametrize("shape", SHAPES, ids=["gridworld", "mastermind222", "hypercube"])
@pytest.mark.parametrize("kind,policy", [("behaviour", "greedy"), ("behaviour", "explore"), ("prediction", "greedy")])
def test_characteristic_and_shapley_units_compile(
    shape: tuple[int, int, int, int, int], kind: Kind, policy: Policy
) -> None:
    features, actions, width, rows, unique = shape
    config = CharacteristicConfig(kind, policy, (width, width))
    characteristic = CharacteristicModel(characteristic_network(features, actions, config), rngs=nnx.Rngs(0))
    optimizer = make_adam(characteristic, 2.5e-4, AdamConfig())
    batch = jax.ShapeDtypeStruct((128, features), jnp.int32)
    outputs = actions if kind == "behaviour" else 1
    targets = jax.ShapeDtypeStruct((128, outputs), jnp.float32)
    coalitions = jax.ShapeDtypeStruct((128, features), jnp.bool_)
    update = nnx.jit(update_characteristic).trace(characteristic, optimizer, batch, targets, coalitions)
    metrics = update.lower().compile()(
        characteristic,
        optimizer,
        jnp.zeros(batch.shape, jnp.int32),
        jnp.zeros(targets.shape, jnp.float32),
        jnp.ones(coalitions.shape, jnp.bool_),
    )
    assert jnp.isfinite(metrics.loss)
    q_values = jax.ShapeDtypeStruct((rows, actions), jnp.float32)
    epsilon = jax.ShapeDtypeStruct((), jnp.float32)
    explained = jax.jit(partial(explained_values, config=config)).trace(q_values, epsilon=epsilon).lower().compile()
    assert explained(jnp.zeros(q_values.shape), epsilon=jnp.float32(0.1)).shape == (rows, outputs)
    weights = jax.jit(partial(off_policy_weights, config=OffPolicyConfig(True, 0.99))).trace(
        jax.ShapeDtypeStruct((128,), jnp.float32), jax.ShapeDtypeStruct((128,), jnp.float32)
    )
    weights.lower().compile()

    observations = jax.ShapeDtypeStruct((rows, features), jnp.int32)
    exact_targets = jax.ShapeDtypeStruct((rows, outputs), jnp.float32)
    every = all_coalitions(features)
    exact_fn = jax.jit(partial(exact_characteristic, coalitions=every, max_unique=unique))
    exact = (
        exact_fn
        .trace(observations, exact_targets)
        .lower()
        .compile()(
            jnp.asarray(np.arange(rows * features).reshape(rows, features) % 3, jnp.int32),
            jnp.ones(exact_targets.shape, jnp.float32),
        )
    )
    assert exact.at_states.shape == (2**features, unique, outputs)
    loss = nnx.jit(exact_characteristic_loss).trace(characteristic, exact).lower().compile()(characteristic, exact)
    assert jnp.isfinite(loss)

    layout = ShapleyLayout(features, outputs)
    model = ShapleyModel(shapley_network(layout, (width, width)), layout, rngs=nnx.Rngs(1))
    shapley_optimizer = make_adam(model, 2.5e-4, AdamConfig())
    coalition = jax.ShapeDtypeStruct((features,), jnp.bool_)
    learn = nnx.jit(update_shapley).trace(model, shapley_optimizer, characteristic, batch, coalition)
    learnt = learn.lower().compile()(
        model, shapley_optimizer, characteristic, jnp.zeros(batch.shape, jnp.int32), every[1]
    )
    assert jnp.isfinite(learnt.loss)

    def exact_losses(
        model: ShapleyModel, characteristic: CharacteristicModel, exact_values: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        # Both ground truths: the exact characteristic, and the learnt one (with_exact_char=False).
        learnt = exact_shapley(characteristic_table(characteristic, exact.states, every))
        exact_loss = exact_shapley_loss(
            model, characteristic, exact.states, exact.state_valid, exact_shapley(exact_values)
        )
        return exact_loss, exact_shapley_loss(model, characteristic, exact.states, exact.state_valid, learnt)

    compiled = nnx.jit(exact_losses).trace(model, characteristic, exact.at_states).lower().compile()
    assert all(jnp.isfinite(value) for value in compiled(model, characteristic, exact.at_states))
