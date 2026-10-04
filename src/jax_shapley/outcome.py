"""Outcome characteristics: the value of an agent conditioned to act on a partially known state.

The source's `PerformanceCharacteristic` asks what return the agent would obtain
if, in state `e`, it acted according to the behaviour characteristic given only
coalition `C` of `e`'s features, and otherwise followed its own policy. A
goal-conditioned DQN-style network learns this from `(s, e, C)`:

- off-policy (`PerformanceCharacteristicOffPolicy`): Q(s, a | e, C) from the
  agent's replay, with the next action sampled from the conditioned policy, and
  `v(e, C) = sum_a pi_hat(a | e; e, C) Q(e, a | e, C)`
- on-policy (`PerformanceCharacteristicOnPolicy`): V(s | e, C) from experience
  collected with the conditioned policy, and `v(e, C) = V(e | e, C)`

Exact values solve the tabular MDP under the conditioned policy for every
coalition and every unique exact-buffer state.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
from flax import nnx, struct
from jax import Array
from jaxtyping import Bool, Float, Int

from jax_shapley.characteristics import LossMetrics, Policy, Values, policy_probabilities
from jax_shapley.coalitions import Coalitions, CoalitionTable, Observations, mask
from jax_shapley.networks import MLP, LayerConfig, MLPConfig, QNetwork
from jax_shapley.shapley import CharacteristicFunction
from jax_shapley.tabular import ActionTable, TabularModel, policy_q_values, state_indices

type Variant = Literal["off_policy", "on_policy"]
type Batch = Int[Array, "batch features"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type BatchCoalitions = Bool[Array, "batch features"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type BatchScalars = Float[Array, "batch"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.
type BatchActions = Int[Array, "batch"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.
type BatchFlags = Bool[Array, "batch"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.
type ExactTable = Float[Array, "coalitions unique"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type UniqueStates = Int[Array, "unique features"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type UniqueFlags = Bool[Array, "unique"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.
type ConditionedTable = Float[Array, "coalitions unique actions"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.


@dataclass(frozen=True, slots=True)
class OutcomeConfig:
    """Which outcome characteristic, its soft target-update rate, discount and hidden ReLU widths."""

    variant: Variant = "off_policy"
    tau: float = 0.001
    gamma: float = 1.0
    hidden_widths: tuple[int, ...] = (64, 64)

    def __post_init__(self) -> None:
        if self.variant not in ("off_policy", "on_policy"):
            raise ValueError("variant must be off_policy or on_policy")
        if not 0 < self.tau <= 1 or not 0 <= self.gamma <= 1:
            raise ValueError("tau must be in (0, 1] and gamma in [0, 1]")
        if any(width < 1 for width in self.hidden_widths):
            raise ValueError("hidden widths must be positive")


@struct.dataclass
class Transitions:
    """Replay rows the off-policy characteristic learns from (the agent's replay prefixes)."""

    observations: Batch
    actions: BatchActions
    rewards: BatchScalars
    next_observations: Batch
    terminated: BatchFlags


@struct.dataclass
class ConditionedReplay:
    """On-policy experience and the (e, C) each decision was conditioned on (source V.buffer with e_obs, C)."""

    transitions: Transitions
    explained: Batch
    coalitions: BatchCoalitions
    pointer: Array
    size: Array


@struct.dataclass
class ExactOutcome:
    """Exact v(e, C) for every coalition and every unique exact-buffer state (padded, with validity)."""

    states: UniqueStates
    state_valid: UniqueFlags
    values: ExactTable


class OutcomeOutput(NamedTuple):
    values: Values


class OutcomeModel(nnx.Module):
    """Goal-conditioned network on concat(s, e, C): Q per action (off-policy) or one V (on-policy)."""

    def __init__(self, config: MLPConfig, *, rngs: nnx.Rngs) -> None:
        self.mlp = MLP(config, rngs=rngs)

    def __call__(self, observations: Observations, explained: Observations, coalitions: Coalitions) -> OutcomeOutput:
        shape = jnp.broadcast_shapes(observations.shape, explained.shape, coalitions.shape)
        inputs = jnp.concatenate(
            (
                jnp.broadcast_to(observations, shape).astype(jnp.float32),
                jnp.broadcast_to(explained, shape).astype(jnp.float32),
                jnp.broadcast_to(coalitions, shape).astype(jnp.float32),
            ),
            axis=-1,
        )
        return OutcomeOutput(self.mlp(inputs))


class OutcomeNetworks(nnx.Module):
    """The learnt network and its soft-updated target, trained and checkpointed together.

    The source copies the model into the target at construction. One Adam covers
    both: the target only enters losses through stop-gradient TD targets, so its
    gradients are exactly zero, Adam leaves it unchanged, and `soft_update` moves it.
    """

    def __init__(self, config: MLPConfig, *, rngs: nnx.Rngs) -> None:
        self.model = OutcomeModel(config, rngs=rngs)
        self.target = nnx.clone(self.model)


@dataclass(frozen=True, slots=True)
class AgentPolicy:
    """The explained agent's policy v_F: greedy (ties shared) or epsilon-greedy over its Q values."""

    network: QNetwork
    policy: Policy
    epsilon: Array

    def __call__(self, observations: Observations) -> Values:
        return policy_probabilities(self.network(observations).q_values, self.policy, self.epsilon)


@dataclass(frozen=True, slots=True)
class ConditionedPolicy:
    """pi_hat(a | s; e, C): the behaviour characteristic at e given C when s == e, else the agent.

    The behaviour characteristic is a learnt model or, for the source's sampling groups, a
    sampled characteristic (`jax_shapley.sampling`).
    """

    behaviour: CharacteristicFunction
    agent: AgentPolicy

    def __call__(self, observations: Observations, explained: Observations, coalitions: Coalitions) -> Values:
        at_explained = jnp.all(observations == explained, axis=-1, keepdims=True)
        return jnp.where(at_explained, self.behaviour.values(explained, coalitions), self.agent(observations))


@dataclass(frozen=True, slots=True)
class OutcomeCharacteristic:
    """The learnt outcome characteristic as a CharacteristicFunction for Shapley training."""

    networks: OutcomeNetworks
    policy: ConditionedPolicy
    variant: Variant

    def values(self, observations: Observations, coalitions: Coalitions) -> Values:
        shape = jnp.broadcast_shapes(observations.shape, coalitions.shape)
        observations, coalitions = jnp.broadcast_to(observations, shape), jnp.broadcast_to(coalitions, shape)
        return outcome_value(self.networks.model, self.policy, observations, coalitions, self.variant)

    def null(self, observations: Observations) -> Values:
        """v(s, C = 0); the behaviour part uses its single-row null (see MaskedCharacteristic.null)."""
        empty = jnp.zeros(observations.shape, jnp.bool_)
        values = self.networks.model(observations, observations, empty).values
        if self.variant == "on_policy":
            return values
        return jnp.sum(self.policy.behaviour.null(observations) * values, axis=-1, keepdims=True)


def outcome_network(features: int, actions: int, config: OutcomeConfig) -> MLPConfig:
    outputs = actions if config.variant == "off_policy" else 1
    hidden = tuple(LayerConfig(width, "relu") for width in config.hidden_widths)
    return MLPConfig(3 * features, (*hidden, LayerConfig(outputs)))


def outcome_value(
    model: OutcomeModel, policy: ConditionedPolicy, explained: Observations, coalitions: Coalitions, variant: Variant
) -> Values:
    """Source get_val: sum_a pi_hat(a | e; e, C) Q(e, a | e, C), or V(e | e, C)."""
    if variant == "on_policy":
        return model(explained, explained, coalitions).values
    probabilities = policy(explained, explained, coalitions)
    return jnp.sum(probabilities * model(explained, explained, coalitions).values, axis=-1, keepdims=True)


def update_off_policy(
    networks: OutcomeNetworks,
    optimizer: nnx.Optimizer[OutcomeNetworks],
    batch: Transitions,
    explained: Batch,
    coalitions: BatchCoalitions,
    next_actions: BatchActions,
    config: OutcomeConfig,
) -> LossMetrics:
    """Source PerformanceCharacteristicOffPolicy.update: one Adam step, then the soft target update."""

    def loss(networks: OutcomeNetworks) -> tuple[Array, LossMetrics]:
        return off_policy_loss(
            networks.model, networks.target, batch, explained, coalitions, next_actions, config.gamma
        )

    metrics = _step(networks, optimizer, loss)
    soft_update(networks, config.tau)
    return metrics


def update_on_policy(
    networks: OutcomeNetworks,
    optimizer: nnx.Optimizer[OutcomeNetworks],
    batch: Transitions,
    explained: Batch,
    coalitions: BatchCoalitions,
    config: OutcomeConfig,
) -> LossMetrics:
    """Source PerformanceCharacteristicOnPolicy.update: one Adam step (the training loop moves the target)."""

    def loss(networks: OutcomeNetworks) -> tuple[Array, LossMetrics]:
        return on_policy_loss(networks.model, networks.target, batch, explained, coalitions, config.gamma)

    return _step(networks, optimizer, loss)


def off_policy_loss(
    model: OutcomeModel,
    target: OutcomeModel,
    batch: Transitions,
    explained: Batch,
    coalitions: BatchCoalitions,
    next_actions: BatchActions,
    gamma: float,
) -> tuple[Array, LossMetrics]:
    """Q-learning towards r + gamma Q_target(s', a' ~ pi_hat(s'; e, C) | e, C) (1 - terminated)."""
    target_values = target(batch.next_observations, explained, coalitions).values
    following = jnp.take_along_axis(target_values, next_actions[:, None], axis=-1)[:, 0]
    targets = jax.lax.stop_gradient(batch.rewards + gamma * following * (~batch.terminated).astype(jnp.float32))
    values = model(batch.observations, explained, coalitions).values
    predictions = jnp.take_along_axis(values, batch.actions[:, None], axis=-1)[:, 0]
    loss = jnp.mean(jnp.square(predictions - targets))
    return loss, LossMetrics(loss, jnp.mean(targets), jnp.mean(predictions))


def on_policy_loss(
    model: OutcomeModel,
    target: OutcomeModel,
    batch: Transitions,
    explained: Batch,
    coalitions: BatchCoalitions,
    gamma: float,
) -> tuple[Array, LossMetrics]:
    """TD(0) towards r + gamma V_target(s' | e, C) (1 - terminated); actions are not used."""
    following = target(batch.next_observations, explained, coalitions).values[:, 0]
    targets = jax.lax.stop_gradient(batch.rewards + gamma * following * (~batch.terminated).astype(jnp.float32))
    predictions = model(batch.observations, explained, coalitions).values[:, 0]
    loss = jnp.mean(jnp.square(predictions - targets))
    return loss, LossMetrics(loss, jnp.mean(targets), jnp.mean(predictions))


def empty_conditioned_replay(capacity: int, features: int) -> ConditionedReplay:
    if capacity < 1 or features < 1:
        raise ValueError("replay capacity and features must be positive")
    rows = jnp.zeros((capacity, features), jnp.int32)
    transitions = Transitions(
        rows,
        jnp.zeros((capacity,), jnp.int32),
        jnp.zeros((capacity,), jnp.float32),
        rows,
        jnp.zeros((capacity,), jnp.bool_),
    )
    return ConditionedReplay(
        transitions,
        rows,
        jnp.zeros((capacity, features), jnp.bool_),
        jnp.asarray(0, jnp.int32),
        jnp.asarray(0, jnp.int32),
    )


def append_conditioned(
    replay: ConditionedReplay, transitions: Transitions, explained: Batch, coalitions: BatchCoalitions
) -> ConditionedReplay:
    """Insert one vector step's rows in environment order, as the source Buffer.add does."""
    capacity = replay.explained.shape[0]
    count = transitions.observations.shape[0]
    if count > capacity:
        raise ValueError("a vector step exceeds the replay capacity")
    rows = (replay.pointer + jnp.arange(count, dtype=jnp.int32)) % capacity

    def insert(storage: Array, values: Array) -> Array:
        return storage.at[rows].set(values)

    stored = Transitions(
        insert(replay.transitions.observations, transitions.observations),
        insert(replay.transitions.actions, transitions.actions),
        insert(replay.transitions.rewards, transitions.rewards),
        insert(replay.transitions.next_observations, transitions.next_observations),
        insert(replay.transitions.terminated, transitions.terminated),
    )
    return ConditionedReplay(
        stored,
        insert(replay.explained, explained),
        insert(replay.coalitions, coalitions),
        (replay.pointer + count) % capacity,
        jnp.minimum(replay.size + count, capacity).astype(jnp.int32),
    )


def sample_conditioned(
    replay: ConditionedReplay, key: Array, batch_size: int
) -> tuple[Transitions, Batch, BatchCoalitions]:
    """Uniform rows with replacement from the filled prefix (source Buffer.sample)."""
    rows = jax.random.randint(key, (batch_size,), 0, replay.size, dtype=jnp.int32)
    stored = replay.transitions
    batch = Transitions(
        stored.observations[rows],
        stored.actions[rows],
        stored.rewards[rows],
        stored.next_observations[rows],
        stored.terminated[rows],
    )
    return batch, replay.explained[rows], replay.coalitions[rows]


def sample_actions(
    policy: ConditionedPolicy, key: Array, observations: Batch, explained: Batch, coalitions: BatchCoalitions
) -> BatchActions:
    """a ~ pi_hat(. | s; e, C): source choose_action, and the off-policy target's next actions."""
    probabilities = policy(observations, explained, coalitions)
    return jax.random.categorical(key, jnp.log(probabilities), axis=-1).astype(jnp.int32)


def soft_update(networks: OutcomeNetworks, tau: float) -> None:
    """Source DQN_Super.update_target: target <- tau * model + (1 - tau) * target."""
    for source, target in zip(networks.model.mlp.layers, networks.target.mlp.layers, strict=True):
        target.kernel[...] = tau * source.kernel[...] + (1 - tau) * target.kernel[...]
        target.bias[...] = tau * source.bias[...] + (1 - tau) * target.bias[...]


def exact_outcome(
    model: TabularModel,
    states: UniqueStates,
    state_valid: UniqueFlags,
    conditioned: ConditionedTable,
    agent: ActionTable,
    gamma: float,
) -> ExactOutcome:
    """Solve the tabular MDP under pi_hat(.; e, C) for every (C, e), then average Q at e under pi_hat.

    `conditioned[c, u]` is the behaviour characteristic's action distribution at
    state u given coalition c (the exact behaviour
    characteristic's `at_states`, or `learnt_behaviour_table`), and `agent` the agent's policy at
    every modelled state. Unmodelled states (padding) are marked invalid.
    """
    rows, modelled = state_indices(model, states)

    def per_coalition(probabilities: Array) -> Array:
        def per_state(row: Array, chosen: Array) -> Array:
            policy = agent.at[row].set(chosen)
            return jnp.sum(chosen * policy_q_values(model, policy, gamma)[row])

        return jax.vmap(per_state)(rows, probabilities)

    values = jax.lax.map(per_coalition, conditioned)
    return ExactOutcome(states, state_valid & modelled, values)


def exact_behaviour_table(
    observations: Batch, targets: Values, states: UniqueStates, coalitions: CoalitionTable
) -> tuple[ConditionedTable, UniqueFlags]:
    """The exact behaviour characteristic at given states, and whether each state occurs in the buffer.

    For coalition C and state s this is the mean explained policy over buffer rows
    that agree with s on C: the source's conditional expectation under the buffer's
    empirical state distribution (characteristics.exact_characteristic), evaluated
    directly at the transition model's states instead of every padded unique row.
    """
    present = jnp.any(jnp.all(states[:, None, :] == observations[None, :, :], axis=-1), axis=-1)

    def per_coalition(coalition: Array) -> Values:
        known = mask(states, coalition)[:, None, :] == mask(observations, coalition)[None, :, :]
        matches = jnp.all(known, axis=-1).astype(jnp.float32)
        return (matches @ targets) / jnp.maximum(jnp.sum(matches, axis=-1, keepdims=True), 1.0)

    table: ConditionedTable = jax.lax.map(per_coalition, coalitions)
    return table, present


def learnt_behaviour_table(
    behaviour: CharacteristicFunction, states: UniqueStates, coalitions: CoalitionTable
) -> ConditionedTable:
    """The learnt behaviour characteristic at each (coalition, unique state) (source char.get_val)."""

    def per_coalition(coalition: Array) -> Values:
        return behaviour.values(states, jnp.broadcast_to(coalition, states.shape))

    table: ConditionedTable = jax.lax.map(per_coalition, coalitions)
    return table


def exact_outcome_loss(
    characteristic: CharacteristicFunction, exact: ExactOutcome, coalitions: CoalitionTable
) -> Array:
    """MSE over every (coalition, valid unique state) pair, equally weighted."""
    states = exact.states

    def per_coalition(coalition: Array) -> Array:
        return characteristic.values(states, jnp.broadcast_to(coalition, states.shape))[:, 0]

    approximations = jax.lax.map(per_coalition, coalitions)
    valid = exact.state_valid.astype(jnp.float32)[None, :]
    errors = jnp.square(approximations - exact.values) * valid
    return jnp.sum(errors) / (jnp.sum(valid) * coalitions.shape[0])


def _step(
    networks: OutcomeNetworks,
    optimizer: nnx.Optimizer[OutcomeNetworks],
    loss: Callable[[OutcomeNetworks], tuple[Array, LossMetrics]],
) -> LossMetrics:
    # NNX transformation has a dynamic return signature; assert its known loss/state boundary.
    result = cast(
        tuple[tuple[Array, LossMetrics], nnx.State[str | int, nnx.Param[Array]]],
        nnx.value_and_grad(loss, has_aux=True)(networks),
    )
    (_, metrics), gradients = result
    optimizer.update(networks, gradients)
    return metrics
