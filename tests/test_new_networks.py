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
from new_jax_shapley.algorithms.dqn import QNetwork
from new_jax_shapley.algorithms.ppo import PPOActor, PPOCritic
from new_jax_shapley.common.networks import LayerConfig, MLPConfig, OrthogonalInit, TorchLinearInit
from tests.network_cases import NETWORK_CASES, NetworkCase, make_network


@pytest.mark.parametrize("case", NETWORK_CASES, ids=lambda case: case.name)
def test_networks_match_the_current_package(case: NetworkCase) -> None:
    layers = tuple(
        LayerConfig(
            layer.features,
            layer.activation,
            OrthogonalInit(layer.initialization.gain, layer.initialization.bias)
            if isinstance(layer.initialization, old.OrthogonalInit)
            else TorchLinearInit(),
        )
        for layer in case.config.layers
    )
    config = MLPConfig(case.config.input_features, layers)
    rngs = nnx.Rngs(0)
    network = (
        PPOActor(config, rngs=rngs)
        if case.kind == "actor"
        else PPOCritic(config, rngs=rngs)
        if case.kind == "critic"
        else QNetwork(config, rngs=rngs)
    )
    reference = make_network(case, seed=0)
    for left, right in zip(jax.tree.leaves(nnx.state(reference)), jax.tree.leaves(nnx.state(network)), strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
    features = case.config.input_features
    integers = jax.random.randint(jax.random.key(1), (3, 5, features), -2, 4, jnp.int32)
    for observations in (integers, integers.astype(jnp.float32)):
        np.testing.assert_array_equal(np.asarray(reference(observations)[0]), np.asarray(network(observations)[0]))
