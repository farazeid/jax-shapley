"""Tabular transition models and exact policy evaluation against upstream get_P and value_iteration."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jax_shapley.networks import QNetwork, dqn_config
from jax_shapley.tabular import TabularModel, policy_q_values, state_indices, transition_model
from jax_shapley.train_dqn import (
    EnvironmentConfig,
    GridworldConfig,
    HypercubeConfig,
    Mastermind222Config,
    exploration_probabilities,
)
from tests.explanation_support import assign
from tests.explanation_support import load_fixture as load_explanations

FIXTURE = Path(__file__).parent / "fixtures" / "fastsverl_outcome.npz"
CONFIGS: dict[str, EnvironmentConfig] = {"gridworld": GridworldConfig(), "mastermind222": Mastermind222Config()}


@pytest.mark.parametrize("case", tuple(CONFIGS))
def test_transition_model_matches_upstream_get_p(case: str) -> None:
    fixture = _fixture()
    model = transition_model(CONFIGS[case])
    states = [tuple(row) for row in np.asarray(model.states).tolist()]
    upstream = [tuple(row) for row in fixture[f"{case}_p_states"].tolist()]
    assert sorted(states) == sorted(upstream)
    index = {state: row for row, state in enumerate(states)}
    probability, reward = fixture[f"{case}_p_probability"], fixture[f"{case}_p_reward"]
    following, done = fixture[f"{case}_p_next"], fixture[f"{case}_p_done"]
    for source, state in enumerate(upstream):
        row = index[state]
        expected_reward = (probability[source] * reward[source]).sum(axis=-1)
        np.testing.assert_allclose(model.rewards[row], expected_reward, rtol=1e-6, atol=1e-6)
        expected = np.zeros(model.continuation.shape[1:], np.float32)
        for action in range(expected.shape[0]):
            for outcome in range(probability.shape[-1]):
                if probability[source, action, outcome] > 0 and not done[source, action, outcome]:
                    target = index[tuple(following[source, action, outcome].tolist())]
                    expected[action, target] += probability[source, action, outcome]
        np.testing.assert_allclose(model.continuation[row], expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("case", tuple(CONFIGS))
@pytest.mark.parametrize("policy", ["uniform", "explore"])
def test_policy_q_values_match_upstream_value_iteration(case: str, policy: str) -> None:
    fixture = _fixture()
    model = transition_model(CONFIGS[case])
    actions = model.rewards.shape[1]
    if policy == "uniform":
        probabilities = jnp.full(model.rewards.shape, 1.0 / actions, jnp.float32)
    else:
        explanations = load_explanations()
        agent = QNetwork(dqn_config(model.states.shape[1], actions, hidden_widths=(64, 64)), rngs=nnx.Rngs(0))
        assign(agent.mlp, explanations, f"{case}_agent")
        probabilities = exploration_probabilities(agent(model.states).q_values, jnp.asarray(0.2, jnp.float32))
    q_values = policy_q_values(model, probabilities, 1.0)
    rows, valid = state_indices(model, jnp.asarray(fixture[f"{case}_q_{policy}_states"]))
    assert bool(valid.all())
    # Upstream iterates in float64 to a 1e-10 change; this solves the same fixed point in float32.
    np.testing.assert_allclose(q_values[rows], fixture[f"{case}_q_{policy}"], rtol=1e-5, atol=1e-4)


def test_hypercube_model_structure_and_index_lookup() -> None:
    model = transition_model(HypercubeConfig(3, 3))
    assert model.states.shape == (26, 3)  # Every cell except the goal.
    sums = np.asarray(model.continuation.sum(axis=-1))
    np.testing.assert_allclose(sums[sums > 0], 1.0)  # Only goal-reaching moves terminate.
    assert np.asarray(model.rewards).max() == pytest.approx(3 * 3 - 1)
    rows, valid = state_indices(model, jnp.asarray([[0, 0, 0], [2, 2, 2]], jnp.int32))
    np.testing.assert_array_equal(model.states[rows[0]], [0, 0, 0])
    np.testing.assert_array_equal(valid, [True, False])  # The goal is terminal, not a modelled state.


def test_vectorised_policy_evaluation_compiles_and_matches() -> None:
    model = transition_model(Mastermind222Config())
    policies = jax.nn.softmax(jax.random.normal(jax.random.key(0), (6, *model.rewards.shape)), axis=-1)

    def evaluate(model: TabularModel, policies: jax.Array) -> jax.Array:
        return jax.vmap(lambda policy: policy_q_values(model, policy, 1.0))(policies)

    compiled = jax.jit(evaluate).trace(model, policies).lower().compile()
    batched = compiled(model, policies)
    for index in range(policies.shape[0]):
        np.testing.assert_allclose(batched[index], policy_q_values(model, policies[index], 1.0), rtol=1e-5, atol=1e-5)


def _fixture() -> dict[str, np.ndarray]:
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        return {key: fixture[key] for key in fixture.files}
