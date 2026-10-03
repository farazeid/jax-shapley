"""Tabular observation-level models and exact policy evaluation.

FastSVERL's exact outcome values run value iteration over each environment's
transition table `P`, keyed by observation. `transition_model` builds the same
table from the JAX environments by enumerating every reachable hidden state with
`step_env`. Observations shared by several hidden states (Mastermind's unseen
goal) average their outcomes with equal weight per hidden state, which is how
the source's `get_P` counts them. Enumeration is host-side model construction.
The resulting arrays feed compiled code.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import Array
from jaxtyping import Bool, Float, Int

from jax_shapley.envs import (
    Gridworld,
    GridworldState,
    HypercubeState,
    MastermindState,
)
from jax_shapley.train_dqn import EnvironmentConfig, GridworldConfig, Mastermind222Config, MastermindConfig
from jax_shapley.wrappers import EnvironmentProtocol

type TableStates = Int[Array, "states features"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type ActionTable = Float[Array, "states actions"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type Continuation = Float[Array, "states actions states"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type StateValues = Float[Array, "states"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.
type RowObservations = Int[Array, "rows features"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type RowIndices = Int[Array, "rows"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.
type RowFlags = Bool[Array, "rows"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.

# Enumeration bound: every paper domain is far smaller (Hypercube 6x5 has 7775 non-goal states).
MAX_STATES = 100_000


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


class ObservableEnvironment[State, Params](EnvironmentProtocol[State, Params], Protocol):
    """Environments that expose the observation of any hidden state."""

    def get_obs(self, state: State) -> Array: ...


class _Timed(Protocol):
    def replace(self, **updates: object) -> object: ...


@dataclass(frozen=True, slots=True)
class _Outcome:
    next_observation: tuple[int, ...]
    reward: float
    done: bool


def transition_model(config: EnvironmentConfig) -> TabularModel:
    """The source `P` for an environment, from every start state its reset can produce."""
    if isinstance(config, GridworldConfig):
        env = Gridworld()
        starts = [
            GridworldState(time=jnp.asarray(0, jnp.int32), position=start, terminated=jnp.asarray(False))
            for start in env._starts
        ]
        return enumerate_model(env, starts)
    if isinstance(config, Mastermind222Config | MastermindConfig):
        env = config.make()
        roots = [
            MastermindState(
                time=jnp.asarray(0, jnp.int32),
                board=jnp.full((env.num_guesses, env.code_size + 2), -1, jnp.int32),
                goal=env.decode_action(jnp.asarray(goal, jnp.int32)),
                guess_index=jnp.asarray(0, jnp.int32),
                terminated=jnp.asarray(False),
            )
            for goal in range(env.num_actions)
        ]
        return enumerate_model(env, roots)
    env = config.make()
    start = HypercubeState(
        time=jnp.asarray(0, jnp.int32),
        position=jnp.zeros(config.dimension, jnp.int32),
        terminated=jnp.asarray(False),
    )
    return enumerate_model(env, [start])


def enumerate_model[State, Params](env: ObservableEnvironment[State, Params], roots: list[State]) -> TabularModel:
    """Breadth-first over hidden states (ignoring the step counter), all actions per layer at once."""
    params = env.default_params
    actions = env.num_actions
    key = jax.random.key(0)  # Paper domains are deterministic given the hidden state.

    def expand(states: State) -> tuple[Array, Array, State, Array, Array]:
        def per_state(state: State) -> tuple[Array, Array, State, Array, Array]:
            observation = env.get_obs(state)
            next_observations, next_states, rewards, done, _ = jax.vmap(
                lambda action: env.step_env(key, state, action, params)
            )(jnp.arange(actions, dtype=jnp.int32))
            return observation, next_observations, next_states, rewards, done

        return jax.vmap(per_state)(states)

    compiled = jax.jit(expand)
    hidden: dict[bytes, list[_Outcome]] = {}
    observations: dict[bytes, tuple[int, ...]] = {}
    frontier = [_untimed(root) for root in roots]
    while frontier:
        batch = jax.tree.map(lambda *leaves: jnp.stack(leaves), *frontier)
        observation_array, next_array, expanded, reward_array, done_array = compiled(batch)
        observation, next_observation = np.asarray(observation_array), np.asarray(next_array)
        rewards, done = np.asarray(reward_array), np.asarray(done_array)
        upcoming: list[State] = []
        for index, state in enumerate(frontier):
            identity = _identity(state)
            if identity in hidden:
                continue
            observations[identity] = tuple(int(value) for value in observation[index])
            hidden[identity] = [
                _Outcome(
                    tuple(int(value) for value in next_observation[index, action]),
                    float(rewards[index, action]),
                    bool(done[index, action]),
                )
                for action in range(actions)
            ]
            for action in range(actions):
                if not done[index, action]:
                    successor = _untimed(
                        cast(State, jax.tree.map(lambda leaf, i=index, a=action: leaf[i, a], expanded))
                    )
                    if _identity(successor) not in hidden:
                        upcoming.append(successor)
            if len(hidden) > MAX_STATES:
                raise ValueError("environment exceeds the enumerable state bound")
        frontier = list({_identity(state): state for state in upcoming}.values())
    return _tabulate(observations, hidden, actions)


def policy_values(model: TabularModel, policy: ActionTable, gamma: float) -> StateValues:
    """Exact solution of V = R_pi + gamma C_pi V: the fixed point the source's value iteration approaches."""
    rewards = jnp.sum(policy * model.rewards, axis=-1)
    continuation = jnp.einsum("sa,sat->st", policy, model.continuation)
    identity = jnp.eye(model.states.shape[0], dtype=jnp.float32)
    solution = jnp.linalg.solve(identity - gamma * continuation, rewards)
    return cast(StateValues, solution)  # Typed linear-algebra boundary.


def policy_q_values(model: TabularModel, policy: ActionTable, gamma: float) -> ActionTable:
    """Q(s, a) under `policy`, matching the source's returned `Q_table`."""
    values = policy_values(model, policy, gamma)
    return model.rewards + gamma * jnp.einsum("sat,t->sa", model.continuation, values)


def state_indices(model: TabularModel, observations: RowObservations) -> tuple[RowIndices, RowFlags]:
    """Row of each observation in `model.states`, and whether it is a modelled (non-terminal) state."""
    matches = jnp.all(observations[:, None, :] == model.states[None, :, :], axis=-1)
    return jnp.argmax(matches, axis=-1).astype(jnp.int32), jnp.any(matches, axis=-1)


def _tabulate(
    observations: dict[bytes, tuple[int, ...]], hidden: dict[bytes, list[_Outcome]], actions: int
) -> TabularModel:
    states = sorted(set(observations.values()))
    index = {state: position for position, state in enumerate(states)}
    members: dict[tuple[int, ...], list[bytes]] = defaultdict(list)
    for identity, observation in observations.items():
        members[observation].append(identity)
    rewards = np.zeros((len(states), actions), np.float64)
    continuation = np.zeros((len(states), actions, len(states)), np.float64)
    for observation, identities in members.items():
        row, weight = index[observation], 1.0 / len(identities)
        for identity in identities:
            for action, outcome in enumerate(hidden[identity]):
                rewards[row, action] += weight * outcome.reward
                if not outcome.done:
                    continuation[row, action, index[outcome.next_observation]] += weight
    return TabularModel(
        jnp.asarray(np.asarray(states, np.int32)),
        jnp.asarray(rewards, jnp.float32),
        jnp.asarray(continuation, jnp.float32),
    )


def _untimed[State](state: State) -> State:
    """The step counter never changes dynamics; dropping it keeps the hidden state space finite."""
    return cast(State, cast(_Timed, state).replace(time=jnp.asarray(0, jnp.int32)))  # Flax struct states.


def _identity(state: object) -> bytes:
    return b"".join(np.asarray(leaf).tobytes() for leaf in jax.tree.leaves(state))
