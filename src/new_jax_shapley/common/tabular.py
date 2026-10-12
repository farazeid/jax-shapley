"""Tabular observation-level models and exact policy evaluation.

FastSVERL's exact outcome values run value iteration over each environment's
transition table `P`, keyed by observation. `transition_model` builds the same
table from the JAX environments by enumerating every reachable hidden state with
`step_env`. Observations shared by several hidden states (Mastermind's unseen
goal) average their outcomes with equal weight per hidden state, which is how
the source's `get_P` counts them. Enumeration is host-side model construction;
the resulting arrays feed compiled code.
"""

from collections import defaultdict
from functools import partial
from typing import Protocol, Self, cast

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import Array
from jaxtyping import Bool, Float, Int

from new_jax_shapley.algorithms.dqn import Env
from new_jax_shapley.envs import Gridworld, GridworldState, HypercubeState, Mastermind, MastermindState
from new_jax_shapley.envs.wrappers import EnvironmentProtocol

type TableStates = Int[Array, "states features"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type ActionTable = Float[Array, "states actions"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type Continuation = Float[Array, "states actions states"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type RowObservations = Int[Array, "rows features"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type RowIndices = Int[Array, "rows"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.
type RowFlags = Bool[Array, "rows"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.

# Enumeration bound: every paper domain is far smaller (Hypercube 6x5 has 7775 non-goal states).
MAX_STATES = 100_000


class Timed(Protocol):
    """Flax struct environment states, whose step counter can be reset."""

    def replace(self, **updates: object) -> Self: ...


class ObservableEnvironment[State, Params](EnvironmentProtocol[State, Params], Protocol):
    """Environments that expose the observation of any hidden state."""

    def get_obs(self, state: State) -> Array: ...


@struct.dataclass
class TabularModel:
    """Observation-level MDP: expected reward and non-terminal next-state probabilities.

    `continuation[s, a, t]` is P(t | s, a) restricted to non-terminal transitions, so
    `Q = rewards + gamma * continuation @ V` is the source's
    `sum p (r + (1 - done) * gamma * V[next])`.
    """

    states: TableStates
    rewards: ActionTable
    continuation: Continuation


def transition_model(env: Env) -> TabularModel:
    """The source `P` for an environment, from every start state its reset can produce."""
    zero, false = jnp.asarray(0, jnp.int32), jnp.asarray(False)
    if isinstance(env, Gridworld):
        return enumerate_model(
            env, [GridworldState(time=zero, position=start, terminated=false) for start in Gridworld._starts]
        )
    if isinstance(env, Mastermind):
        board = jnp.full((env.num_guesses, env.code_size + 2), -1, jnp.int32)
        goals = [env.decode_action(jnp.asarray(goal, jnp.int32)) for goal in range(env.num_actions)]
        return enumerate_model(
            env,
            [MastermindState(time=zero, board=board, goal=goal, guess_index=zero, terminated=false) for goal in goals],
        )
    start = HypercubeState(time=zero, position=jnp.zeros(env.dimension, jnp.int32), terminated=false)
    return enumerate_model(env, [start])


def enumerate_model[State, Params](env: ObservableEnvironment[State, Params], roots: list[State]) -> TabularModel:
    """Breadth-first over hidden states, all actions per layer at once, then the observation-level table.

    The step counter never changes dynamics, so it is reset to keep the hidden state space
    finite; hidden states are identified by their leaves' bytes.
    """
    actions = env.num_actions

    def per_state(env: ObservableEnvironment[State, Params], state: State) -> tuple[Array, Array, State, Array, Array]:
        # Paper domains are deterministic given the hidden state, so any key will do.
        step = partial(env.step_env, jax.random.key(0), state, params=env.default_params)
        next_observations, next_states, rewards, done, _ = jax.vmap(step)(jnp.arange(actions, dtype=jnp.int32))
        return env.get_obs(state), next_observations, next_states, rewards, done

    expand = jax.jit(jax.vmap(partial(per_state, env)))
    zero = jnp.asarray(0, jnp.int32)
    untimed = [cast(State, cast(Timed, root).replace(time=zero)) for root in roots]  # Flax struct states.
    frontier = {b"".join(np.asarray(leaf).tobytes() for leaf in jax.tree.leaves(state)): state for state in untimed}
    # Each hidden state's observation, and per action its next observation, reward and episode end.
    observations: dict[bytes, tuple[int, ...]] = {}
    outcomes: dict[bytes, list[tuple[tuple[int, ...], float, bool]]] = {}
    while frontier:
        batch = jax.tree.map(lambda *leaves: jnp.stack(leaves), *frontier.values())
        observation, next_observation, expanded, rewards, done = expand(batch)
        observation, next_observation = np.asarray(observation), np.asarray(next_observation)
        rewards, done = np.asarray(rewards), np.asarray(done)
        expanded_leaves, structure = jax.tree.flatten(expanded)
        upcoming: dict[bytes, State] = {}
        for index, identity in enumerate(frontier):
            observations[identity] = tuple(observation[index].tolist())
            outcomes[identity] = [
                (
                    tuple(next_observation[index, action].tolist()),
                    float(rewards[index, action]),
                    bool(done[index, action]),
                )
                for action in range(actions)
            ]
            for action in np.flatnonzero(~done[index]).tolist():
                successor = jax.tree.unflatten(structure, [leaf[index, action] for leaf in expanded_leaves])
                successor = cast(State, cast(Timed, successor).replace(time=zero))  # Flax struct states.
                key = b"".join(np.asarray(leaf).tobytes() for leaf in jax.tree.leaves(successor))
                if key not in outcomes and key not in frontier:
                    upcoming[key] = successor
        if len(outcomes) > MAX_STATES:
            raise ValueError("environment exceeds the enumerable state bound")
        frontier = upcoming
    return tabulate(observations, outcomes, actions)


def tabulate(
    observations: dict[bytes, tuple[int, ...]],
    outcomes: dict[bytes, list[tuple[tuple[int, ...], float, bool]]],
    actions: int,
) -> TabularModel:
    """The observation-level MDP: hidden states sharing an observation weigh equally (the source's get_P)."""
    states = sorted(set(observations.values()))
    position = {state: row for row, state in enumerate(states)}
    members: dict[tuple[int, ...], list[bytes]] = defaultdict(list)
    for identity, observation in observations.items():
        members[observation].append(identity)
    rewards = np.zeros((len(states), actions), np.float64)
    continuation = np.zeros((len(states), actions, len(states)), np.float64)
    for observation, identities in members.items():
        row, weight = position[observation], 1.0 / len(identities)
        for identity in identities:
            for action, (next_observation, reward, done) in enumerate(outcomes[identity]):
                rewards[row, action] += weight * reward
                if not done:
                    continuation[row, action, position[next_observation]] += weight
    return TabularModel(
        jnp.asarray(np.asarray(states, np.int32)),
        jnp.asarray(rewards, jnp.float32),
        jnp.asarray(continuation, jnp.float32),
    )


def policy_q_values(model: TabularModel, policy: ActionTable, gamma: float) -> ActionTable:
    """Q(s, a) under `policy`, matching the source's returned `Q_table`.

    V is the exact solution of V = R_pi + gamma C_pi V: the fixed point the source's
    value iteration approaches.
    """
    rewards = jnp.sum(policy * model.rewards, axis=-1)
    continuation = jnp.einsum("sa,sat->st", policy, model.continuation)
    identity = jnp.eye(model.states.shape[0], dtype=jnp.float32)
    values: Array = jnp.linalg.solve(identity - gamma * continuation, rewards)
    return model.rewards + gamma * jnp.einsum("sat,t->sa", model.continuation, values)


def state_indices(model: TabularModel, observations: RowObservations) -> tuple[RowIndices, RowFlags]:
    """Row of each observation in `model.states`, and whether it is a modelled (non-terminal) state."""
    matches = jnp.all(observations[:, None, :] == model.states[None, :, :], axis=-1)
    return jnp.argmax(matches, axis=-1).astype(jnp.int32), jnp.any(matches, axis=-1)
