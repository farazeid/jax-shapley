"""The refactored networks (new_jax_shapley) initialise and evaluate bitwise like jax_shapley's.

Every network case (DQN per environment, PPO actor and critic with orthogonal
initialisation) is built from the same seed by both packages: parameters, outputs on
int32 and float32 observations must agree.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jax_shapley import networks as old
from new_jax_shapley.common.networks import OrthogonalInit
from tests.network_cases import NETWORK_CASES, NetworkCase, make_network


@pytest.mark.parametrize("case", NETWORK_CASES, ids=lambda case: case.name)
def test_networks_match_the_current_package(case: NetworkCase) -> None:
    # The cases build new_jax_shapley networks; the reference is jax_shapley's from the same config.
    layers = tuple(
        old.LayerConfig(
            layer.features,
            layer.activation,
            old.OrthogonalInit(layer.initialization.gain, layer.initialization.bias)
            if isinstance(layer.initialization, OrthogonalInit)
            else old.TorchLinearInit(),
        )
        for layer in case.config.layers
    )
    config = old.MLPConfig(case.config.input_features, layers)
    rngs = nnx.Rngs(0)
    reference = (
        old.PPOActor(config, rngs=rngs)
        if case.kind == "actor"
        else old.PPOCritic(config, rngs=rngs)
        if case.kind == "critic"
        else old.QNetwork(config, rngs=rngs)
    )
    network = make_network(case, seed=0)
    for left, right in zip(jax.tree.leaves(nnx.state(reference)), jax.tree.leaves(nnx.state(network)), strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    features = case.config.input_features
    integers = jax.random.randint(jax.random.key(1), (3, 5, features), -2, 4, jnp.int32)
    for observations in (integers, integers.astype(jnp.float32)):
        np.testing.assert_array_equal(np.asarray(reference(observations)[0]), np.asarray(network(observations)[0]))
