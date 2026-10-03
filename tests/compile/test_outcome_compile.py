"""Abstract compilation of outcome characteristic units at production shapes and dtypes."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from jax_shapley.characteristics import (
    CharacteristicConfig,
    CharacteristicModel,
    LossMetrics,
    characteristic_network,
    exact_characteristic,
)
from jax_shapley.coalitions import all_coalitions
from jax_shapley.networks import QNetwork, dqn_config
from jax_shapley.outcome import (
    AgentPolicy,
    ConditionedPolicy,
    ExactOutcome,
    OutcomeCharacteristic,
    OutcomeConfig,
    OutcomeNetworks,
    Transitions,
    Variant,
    exact_outcome,
    exact_outcome_loss,
    learnt_behaviour_table,
    outcome_network,
    sample_next_actions,
    soft_update,
    update_off_policy,
    update_on_policy,
)
from jax_shapley.shapley import MaskedCharacteristic, characteristic_table, exact_shapley
from jax_shapley.tabular import transition_model
from jax_shapley.train_dqn import AdamConfig, EnvironmentConfig, GridworldConfig, Mastermind222Config, make_adam

# Audited source settings: (environment, tau) for Gridworld and Mastermind-222; batch 64, widths (64, 64).
CASES: dict[str, tuple[EnvironmentConfig, float]] = {
    "gridworld": (GridworldConfig(), 0.001),
    "mastermind222": (Mastermind222Config(), 0.01),
}
BATCH, WIDTH, UNIQUE = 64, 64, 40


@pytest.mark.compile
@pytest.mark.parametrize("case", tuple(CASES))
@pytest.mark.parametrize("variant", ["off_policy", "on_policy"])
def test_outcome_updates_compile(case: str, variant: Variant) -> None:
    environment, tau = CASES[case]
    tabular = transition_model(environment)
    features, actions = tabular.states.shape[1], tabular.rewards.shape[1]
    config = OutcomeConfig(variant, tau=tau, hidden_widths=(WIDTH, WIDTH))
    networks = OutcomeNetworks(outcome_network(features, actions, config), rngs=nnx.Rngs(0))
    optimizer = make_adam(networks, 2.5e-4, AdamConfig())
    behaviour, agent = _frozen(features, actions)
    concrete = Transitions(
        tabular.states[jnp.arange(BATCH) % tabular.states.shape[0]],
        jnp.arange(BATCH, dtype=jnp.int32) % actions,
        jnp.full((BATCH,), -1.0, jnp.float32),
        tabular.states[(jnp.arange(BATCH) + 1) % tabular.states.shape[0]],
        jnp.arange(BATCH) % 7 == 0,
    )
    batch = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), concrete)
    rows = jax.ShapeDtypeStruct((BATCH, features), jnp.int32)
    coalitions = jax.ShapeDtypeStruct((BATCH, features), jnp.bool_)

    def learn(
        networks: OutcomeNetworks,
        optimizer: nnx.Optimizer[OutcomeNetworks],
        behaviour: CharacteristicModel,
        agent: QNetwork,
        batch: Transitions,
        explained: jax.Array,
        coalitions: jax.Array,
        key: jax.Array,
    ) -> LossMetrics:
        if variant == "on_policy":
            metrics = update_on_policy(networks, optimizer, batch, explained, coalitions, config)
            soft_update(networks, config.tau)
            return metrics
        policy = _policy(behaviour, agent)
        next_actions = sample_next_actions(policy, key, batch.next_observations, explained, coalitions)
        return update_off_policy(networks, optimizer, batch, explained, coalitions, next_actions, config)

    key = jax.ShapeDtypeStruct((), jax.random.key(0).dtype)
    compiled = (
        nnx.jit(learn).trace(networks, optimizer, behaviour, agent, batch, rows, coalitions, key).lower().compile()
    )
    metrics = compiled(
        networks,
        optimizer,
        behaviour,
        agent,
        concrete,
        concrete.next_observations,
        jnp.arange(BATCH * features).reshape(BATCH, features) % 3 == 0,
        jax.random.key(1),
    )
    assert metrics.loss.shape == () and metrics.loss.dtype == jnp.float32 and jnp.isfinite(metrics.loss)


@pytest.mark.compile
@pytest.mark.parametrize("case", tuple(CASES))
@pytest.mark.parametrize("variant", ["off_policy", "on_policy"])
def test_exact_outcome_and_losses_compile(case: str, variant: Variant) -> None:
    environment, _ = CASES[case]
    tabular = transition_model(environment)
    features, actions = tabular.states.shape[1], tabular.rewards.shape[1]
    config = OutcomeConfig(variant, hidden_widths=(WIDTH, WIDTH))
    networks = OutcomeNetworks(outcome_network(features, actions, config), rngs=nnx.Rngs(0))
    behaviour, agent = _frozen(features, actions)
    every = all_coalitions(features)
    observations = tabular.states[jnp.arange(UNIQUE) % tabular.states.shape[0]]

    def exact_values(
        behaviour: CharacteristicModel, agent: QNetwork, observations: jax.Array
    ) -> dict[str, ExactOutcome]:
        policy = _policy(behaviour, agent)
        exact = exact_characteristic(observations, policy.agent(observations), every, max_unique=UNIQUE)
        conditioned = {
            "with_exact_char": exact.at_states,
            "without_exact_char": learnt_behaviour_table(policy.behaviour, exact.states, every),
        }
        table = policy.agent(tabular.states)
        return {
            label: exact_outcome(tabular, exact.states, exact.state_valid, values, table, config.gamma)
            for label, values in conditioned.items()
        }

    observation_spec = jax.ShapeDtypeStruct(observations.shape, jnp.int32)
    compiled = nnx.jit(exact_values).trace(behaviour, agent, observation_spec).lower().compile()
    exact = compiled(behaviour, agent, observations)
    unique = tabular.states.shape[0]
    for values in exact.values():
        assert values.values.shape == (2**features, UNIQUE) and values.values.dtype == jnp.float32
        assert int(values.state_valid.sum()) == unique  # Padding beyond the modelled states is invalid.
        assert bool(jnp.isfinite(values.values[:, :unique]).all())

    def losses(
        networks: OutcomeNetworks, behaviour: CharacteristicModel, agent: QNetwork, exact: ExactOutcome
    ) -> tuple[jax.Array, jax.Array]:
        characteristic = OutcomeCharacteristic(networks, _policy(behaviour, agent), variant)
        learnt = exact_shapley(characteristic_table(characteristic, exact.states, every))
        return exact_outcome_loss(characteristic, exact, every), learnt.sum()

    exact_spec = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), exact["with_exact_char"])
    compiled_losses = nnx.jit(losses).trace(networks, behaviour, agent, exact_spec).lower().compile()
    loss, total = compiled_losses(networks, behaviour, agent, exact["with_exact_char"])
    assert loss.shape == () and jnp.isfinite(loss) and jnp.isfinite(total)


def _frozen(features: int, actions: int) -> tuple[CharacteristicModel, QNetwork]:
    config = CharacteristicConfig("behaviour", "explore", (WIDTH, WIDTH))
    behaviour = CharacteristicModel(characteristic_network(features, actions, config), rngs=nnx.Rngs(1))
    agent = QNetwork(dqn_config(features, actions, hidden_widths=(WIDTH, WIDTH)), rngs=nnx.Rngs(2))
    return behaviour, agent


def _policy(behaviour: CharacteristicModel, agent: QNetwork) -> ConditionedPolicy:
    explore = AgentPolicy(agent, "explore", jnp.asarray(0.2, jnp.float32))
    return ConditionedPolicy(MaskedCharacteristic(behaviour), explore)
