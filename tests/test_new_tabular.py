"""The refactored tabular MDPs (new_jax_shapley) equal jax_shapley's bitwise.

Transition tables, exact Q-values under a fixed stochastic policy, and observation
lookups agree for every environment family (Mastermind with hidden goals included).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_shapley import tabular as old
from jax_shapley.train_dqn import GridworldConfig, HypercubeConfig, Mastermind222Config, MastermindConfig
from new_jax_shapley.common import tabular
from new_jax_shapley.envs import Gridworld, Hypercube, Mastermind

CASES = {
    "gridworld": (Gridworld(), GridworldConfig()),
    "mastermind222": (Mastermind(2, 2, 2), Mastermind222Config()),
    "mastermind322": (Mastermind(3, 2, 2), MastermindConfig(3, 2, 2)),
    "hypercube": (Hypercube(3, 3), HypercubeConfig(3, 3)),
}


@pytest.mark.parametrize("name", list(CASES))
def test_tables_match_the_current_package(name: str) -> None:
    env, config = CASES[name]
    new, reference = tabular.transition_model(env), old.transition_model(config)
    for field in ("states", "rewards", "continuation"):
        np.testing.assert_array_equal(np.asarray(getattr(new, field)), np.asarray(getattr(reference, field)))
    logits = jax.random.normal(jax.random.key(0), new.rewards.shape)
    policy = jax.nn.softmax(logits, axis=-1)
    np.testing.assert_array_equal(
        np.asarray(tabular.policy_q_values(new, policy, 0.9)), np.asarray(old.policy_q_values(reference, policy, 0.9))
    )
    observations = jnp.concatenate([new.states[::2], jnp.full((3, new.states.shape[1]), 7, jnp.int32)])
    for left, right in zip(
        tabular.state_indices(new, observations), old.state_indices(reference, observations), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
