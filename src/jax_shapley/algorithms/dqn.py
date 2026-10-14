"""Step 1: train the FastSVERL DQN agent (CleanRL DQN with soft target updates).

`main` trains one agent per seed, or loads them from `checkpoint`, and returns their final
states, every vector step's metrics (None when loaded), and the frozen `Agent`s the
explanation steps explain, all with a leading seed axis. Each seed is the same compiled
single-seed program, so a seed's agent does not depend on the others. With a `track`
backend each seed logs the source's scalars live into its own run, and the whole
seed-stacked final state is pushed as one Orbax checkpoint artifact. `train` is the
compiled part: `nnx.jit(partial(train, config, log, log_evaluation))(key, seed_index)`. Timestep and frequency
units count all environment transitions, as the source's `global_step` does. Replay
matches FastSVERL `Buffer` insertion and sampling, with fixed shapes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, fields
from functools import partial
from math import isfinite
from pathlib import Path
from time import time_ns
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
import trackio
import tyro
import wandb
from flax import nnx, struct
from jax import Array
from jaxtyping import Bool, Float, Int

from jax_shapley import backends
from jax_shapley.backends import Track
from jax_shapley.common import checkpoints
from jax_shapley.common.networks import MLP, AdamConfig, LayerConfig, MLPConfig, ModelInput, soft_update
from jax_shapley.common.supervised_learning import stack_seeds
from jax_shapley.envs import (
    Gridworld,
    GridworldParams,
    GridworldState,
    Hypercube,
    HypercubeParams,
    HypercubeState,
    Mastermind,
    MastermindParams,
    MastermindState,
)
from jax_shapley.envs.wrappers import EpisodeStatisticsState, EpisodeStatisticsWrapper

type ActionValues = Float[Array, "... actions"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type Observations = Int[Array, "batch features"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type Actions = Int[Array, "batch"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.
type Scalars = Float[Array, "batch"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.
type Terminations = Bool[Array, "batch"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.
type StepLog = Callable[[StepMetrics], None]
"""A live log of a block of vector steps of every seed: metrics with leading (steps, seeds) axes."""
type EvaluationLog = Callable[[Array, EvaluationMetrics], None]
type Env = Gridworld | Mastermind | Hypercube
type EnvState = GridworldState | MastermindState | HypercubeState
type EnvParams = GridworldParams | MastermindParams | HypercubeParams

# Config fields that only say where a stage is tracked; a checkpoint ignores them.
TRACKING_FIELDS = ("checkpoint", "name", "track", "project", "entity", "group")

# The source logs the DQN loss when the global step crosses a multiple of this.
LOG_STEPS = 100
# Vector steps per block: one live log each (host round trips on accelerators).
STEP_BLOCK = 100


@dataclass(frozen=True, slots=True)
class DQNConfig:
    env: Env
    seeds: tuple[int, ...] = (1,)
    """One agent per seed; explanation stages explain the first, or agent s explains seed s."""
    hidden_widths: tuple[int, ...] = (64, 64)
    num_envs: int = 1
    total_timesteps: int = 50_000
    learning_rate: float = 2.5e-4
    buffer_size: int = 50_000
    gamma: float = 1.0
    tau: float = 0.01
    target_network_frequency: int = 1
    batch_size: int = 64
    start_e: float = 1.0
    end_e: float = 0.01
    exploration_fraction: float = 0.5
    learning_starts: int = 10_000
    train_frequency: int = 10
    eval_episodes: int = 0
    """Greedy evaluation episodes after training, logged as `agent/eval_episodic_return`."""
    adam: AdamConfig = field(default_factory=AdamConfig)
    checkpoint: str | None = None
    """An agent artifact to load instead of training, e.g. `GROUP--algorithm--dqn:latest`."""
    name: str = "dqn"
    """The experiment name runs are named after: `NAME--algorithm--dqn--seed-N`."""
    track: Track | None = None
    """The tracking backend; None trains untracked (tests)."""
    project: str = "dev-jax"
    entity: str | None = "fastsverl"
    """wandb entity (team or user)."""
    group: str | None = None
    """The run group; None starts a new one, `NAME-TIMESTAMP`."""

    def __post_init__(self) -> None:
        if any(getattr(self, item.name) is tyro.MISSING for item in fields(self)):
            return  # A sweep config: validated once the command line supplies every field.
        for name, value, minimum in (
            ("num_envs", self.num_envs, 1),
            ("total_timesteps", self.total_timesteps, 1),
            ("buffer_size", self.buffer_size, 1),
            ("batch_size", self.batch_size, 1),
            ("train_frequency", self.train_frequency, 1),
            ("target_network_frequency", self.target_network_frequency, 1),
            ("learning_starts", self.learning_starts, 0),
            ("eval_episodes", self.eval_episodes, 0),
        ):
            # Counters, replay indices and steps are int32 inside the compiled run.
            if not minimum <= value < 2**31:
                raise ValueError(f"{name} must be at least {minimum} and fit in int32")
        if self.vector_steps < 1:
            raise ValueError("total_timesteps must cover at least one vector step (total_timesteps // num_envs)")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds) or min(self.hidden_widths, default=1) < 1:
            raise ValueError("seeds must be distinct and nonempty, and hidden widths positive")
        for name, value in (("gamma", self.gamma), ("tau", self.tau), ("start_e", self.start_e), ("end_e", self.end_e)):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name, value in (("learning_rate", self.learning_rate), ("exploration_fraction", self.exploration_fraction)):
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def vector_steps(self) -> int:
        return self.total_timesteps // self.num_envs


@struct.dataclass
class TransitionBatch:
    observations: Observations
    actions: Actions
    rewards: Scalars
    next_observations: Observations
    terminated: Terminations
    log_probabilities: Scalars


@struct.dataclass
class ReplayState:
    data: TransitionBatch
    pointer: Array
    size: Array


@struct.dataclass
class RunState:
    model: QNetwork
    target: QNetwork
    optimizer: nnx.Optimizer[QNetwork]
    replay: ReplayState
    observations: Array
    env_state: EpisodeStatisticsState[EnvState]
    vector_step: Array
    updates: Array
    target_updates: Array
    key: Array


@struct.dataclass
class StepMetrics:
    global_step: Array
    epsilon: Array
    loss: Array
    updated: Array
    target_updated: Array
    episode_return: Array
    episode_length: Array
    completed: Array


@struct.dataclass
class EvaluationMetrics:
    """One evaluation step: each environment's episode return and whether it completed, and how many
    evaluation episodes completed before this step."""

    episode_return: Array
    completed: Array
    first_episode: Array


@struct.dataclass
class Agent:
    """The trained agent the explanation steps explain: its Q-network and final epsilon."""

    model: QNetwork
    epsilon: Array


@struct.dataclass
class EnvironmentStep:
    observations: Array
    state: EpisodeStatisticsState[EnvState]
    raw_observations: Array
    rewards: Array
    terminated: Array
    truncated: Array
    episode_return: Array
    episode_length: Array
    completed: Array


class QOutput(NamedTuple):
    q_values: ActionValues


class QNetwork(nnx.Module):
    """DQN's Q-network: one unactivated output per action."""

    def __init__(self, config: MLPConfig, *, rngs: nnx.Rngs) -> None:
        if config.layers[-1].activation != "identity":
            raise ValueError("DQN output must be unactivated Q-values")
        self.mlp = MLP(config, rngs=rngs)

    def __call__(self, observations: ModelInput) -> QOutput:
        return QOutput(self.mlp(observations))


def main(config: DQNConfig) -> tuple[RunState, StepMetrics | None, Agent]:
    """Train (or load) one agent per seed, tracked by `config.track`.

    Everything returned has a leading seed axis; metrics then a vector-step axis.
    """
    keys = jnp.stack([jax.random.key(seed) for seed in config.seeds])
    if config.track is None:
        state, metrics = nnx.jit(partial(train, config, None, None))(keys)
        return state, metrics, trained_agent(config, state)
    group = config.group or f"{config.name}-{time_ns()}"
    artifact = f"{group}--algorithm--dqn"
    settings = {
        "stage": "algorithm",
        "env": config.env.name,
        "config": {name: value for name, value in asdict(config).items() if name not in TRACKING_FIELDS},
    }
    with backends.session(config.track, config.entity) as root:
        tracking = wandb if config.track == "wandb" else trackio
        runs = tuple(
            tracking.init(
                project=config.project,
                name=f"{config.name}--algorithm--dqn--seed-{seed}",
                group=group,
                config={**asdict(config), "seed": seed},
            )
            for seed in config.seeds
        )
        try:
            if config.checkpoint is not None:
                directory = runs[0].use_artifact(config.checkpoint).download(root=str(root / "artifacts" / artifact))
                template = stack_seeds([initialize(config, jax.random.key(0))] * len(config.seeds))
                state = checkpoints.load(Path(directory), template, settings)
                return state, None, trained_agent(config, state)

            program = nnx.jit(
                partial(train, config, partial(log_step, runs, config.num_envs), partial(log_evaluation, runs))
            )
            state, metrics = program(keys)
            directory = root / "checkpoints" / artifact
            checkpoints.save(directory, state, settings)
            backends.upload(runs[0], directory, artifact, "agent")
            print(f"Agent checkpoint: {artifact}:latest")
            return state, metrics, trained_agent(config, state)
        finally:
            for run in runs:
                run.finish()


def log_step(runs: Sequence[backends.Logger], num_envs: int, metrics: StepMetrics) -> None:
    """The source's DQN scalars for a block of vector steps, (steps, seeds), into each seed's run (a host
    callback): episode statistics where episodes finished (averaged when several finish on the same
    step: a step holds one value per tag), and the loss where the global step crosses a multiple of 100."""
    global_steps, completed = np.asarray(metrics.global_step), np.asarray(metrics.completed)
    returns, lengths = np.asarray(metrics.episode_return), np.asarray(metrics.episode_length)
    epsilons, updated, losses = np.asarray(metrics.epsilon), np.asarray(metrics.updated), np.asarray(metrics.loss)
    for index in np.ndindex(global_steps.shape):
        step = int(global_steps[index])
        row: dict[str, float] = {}
        if completed[index].any():
            row["agent/episodic_return"] = float(returns[index][completed[index]].mean())
            row["agent/episodic_length"] = float(lengths[index][completed[index]].mean())
            row["agent/epsilon"] = float(epsilons[index])
        if bool(updated[index]) and step % LOG_STEPS < num_envs:
            row["agent/DQN_loss"] = float(losses[index])
        if row:
            runs[index[1]].log(row, step=step)


def log_evaluation(runs: Sequence[backends.Logger], seed: Array, metrics: EvaluationMetrics) -> None:
    """Each completed evaluation episode's return, stepped by its episode number (a host callback)."""
    returns = np.asarray(metrics.episode_return)[np.asarray(metrics.completed)]
    for index, value in enumerate(returns):
        runs[int(seed)].log({"agent/eval_episodic_return": float(value)}, step=int(metrics.first_episode) + index)


def train(
    config: DQNConfig, log: StepLog | None, log_evaluation: EvaluationLog | None, keys: Array
) -> tuple[RunState, StepMetrics]:
    """Every seed's compiled run, vectorised over seeds: training, then `eval_episodes` greedy
    evaluation episodes. Returns seed-stacked states and (seeds, vector steps) metrics.

    Vector steps run in blocks of `STEP_BLOCK`; `log` receives each block's metrics,
    (steps, seeds), during the run (a host callback). `log_evaluation` receives every
    evaluation step that completed an episode. Evaluation advances each run's key, as the
    source's does.
    """
    env = EpisodeStatisticsWrapper(config.env)
    state = nnx.vmap(partial(initialize, config), in_axes=0, out_axes=0)(keys)
    every_seed = nnx.vmap(partial(step, config, env), in_axes=(0, None), out_axes=(0, 0))
    full, rest = divmod(config.vector_steps, STEP_BLOCK)
    pieces = []
    if full:
        indices = jnp.arange(full * STEP_BLOCK, dtype=jnp.int32).reshape(full, STEP_BLOCK)
        state, blocks = nnx.scan(partial(step_block, every_seed, log), in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))(
            state, indices
        )
        pieces.append(jax.tree.map(lambda leaf: leaf.reshape(full * STEP_BLOCK, *leaf.shape[2:]), blocks))
    if rest:
        state, tail = step_block(every_seed, log, state, jnp.arange(rest, dtype=jnp.int32))
        pieces.append(tail)
    # (steps, seeds) to (seeds, steps).
    columns = zip(*[jax.tree.leaves(piece) for piece in pieces], strict=True)
    metrics = StepMetrics(*[jnp.moveaxis(jnp.concatenate(column), 0, 1) for column in columns])
    if config.eval_episodes:
        seeds = jnp.arange(keys.shape[0], dtype=jnp.int32)
        state = nnx.vmap(partial(evaluate, config, log_evaluation), in_axes=(0, 0), out_axes=0)(state, seeds)
    return state, metrics


def step_block(
    every_seed: Callable[[RunState, Array], tuple[RunState, StepMetrics]],
    log: StepLog | None,
    state: RunState,
    indices: Array,
) -> tuple[RunState, StepMetrics]:
    """Consecutive vector steps of every seed, then one live log of them all."""
    state, metrics = nnx.scan(every_seed, in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))(state, indices)
    if log is not None:
        jax.debug.callback(log, metrics, ordered=True)
    return state, metrics


def step(
    config: DQNConfig,
    env: EpisodeStatisticsWrapper[EnvState, EnvParams],
    state: RunState,
    index: Array,
) -> tuple[RunState, StepMetrics]:
    """One vector step: act, store, then learn and soft-update the target when due (`index` is unused)."""
    del index
    global_step = state.vector_step * config.num_envs
    fraction = global_step.astype(jnp.float32) / (config.exploration_fraction * config.total_timesteps)
    epsilon = jnp.maximum(config.start_e + (config.end_e - config.start_e) * fraction, config.end_e)
    epsilon = epsilon.astype(jnp.float32)
    actions, log_probabilities, key = choose_action(state.model, state.observations, epsilon, state.key)
    keys: Array = jax.random.split(key, config.num_envs + 1)
    transition = jax.vmap(partial(environment_step, env), in_axes=(0, 0, 0, None))(
        keys[1:], state.env_state, actions, env.default_params
    )
    # Insert: upstream discards the prefix of an oversized insertion before advancing its pointer.
    batch = TransitionBatch(
        state.observations,
        actions,
        transition.rewards,
        transition.raw_observations,
        transition.terminated,
        log_probabilities,
    )
    capacity = state.replay.data.observations.shape[0]
    count = min(config.num_envs, capacity)
    indices = (state.replay.pointer + jnp.arange(count, dtype=jnp.int32)) % capacity
    stored = zip(jax.tree.leaves(state.replay.data), jax.tree.leaves(batch), strict=True)
    data = TransitionBatch(*[storage.at[indices].set(values[-count:]) for storage, values in stored])
    replay = ReplayState(
        data,
        (state.replay.pointer + count) % capacity,
        jnp.minimum(state.replay.size + count, capacity).astype(jnp.int32),
    )
    ready = global_step > config.learning_starts
    should_learn = ready & (global_step % config.train_frequency < config.num_envs)
    should_target = ready & (global_step % config.target_network_frequency < config.num_envs)
    # Sampling outside the condition is safe after insertion. Advance its stream
    # only on actual updates, so grouping steps into chunks cannot change samples.
    sample_keys: Array = jax.random.split(keys[0])
    rows = jax.random.randint(sample_keys[1], (config.batch_size,), 0, replay.size, dtype=jnp.int32)
    sampled = TransitionBatch(*[array[rows] for array in jax.tree.leaves(replay.data)])
    key = jnp.where(should_learn, sample_keys[0], keys[0])

    def learn(
        model: QNetwork,
        target: QNetwork,
        optimizer: nnx.Optimizer[QNetwork],
        batch: TransitionBatch,
        gamma: float,
    ) -> Array:
        # NNX transformation has a dynamic return signature; assert its loss/gradient boundary.
        loss, gradients = cast(
            tuple[Array, nnx.State[str | int, nnx.Param[Array]]],
            nnx.value_and_grad(dqn_loss)(model, target, batch, gamma),
        )
        optimizer.update(model, gradients)
        return loss

    def skip(
        model: QNetwork,
        target: QNetwork,
        optimizer: nnx.Optimizer[QNetwork],
        batch: TransitionBatch,
        gamma: float,
    ) -> Array:
        del model, target, optimizer, batch, gamma
        return jnp.asarray(0.0, jnp.float32)

    loss = nnx.cond(should_learn, learn, skip, state.model, state.target, state.optimizer, sampled, config.gamma)

    nnx.cond(
        should_target,
        lambda model, target, tau: soft_update(model.mlp, target.mlp, tau),
        lambda model, target, tau: None,
        state.model,
        state.target,
        config.tau,
    )
    next_state = RunState(
        state.model,
        state.target,
        state.optimizer,
        replay,
        transition.observations,
        transition.state,
        state.vector_step + 1,
        state.updates + should_learn.astype(jnp.int32),
        state.target_updates + should_target.astype(jnp.int32),
        key,
    )
    metrics = StepMetrics(
        global_step,
        epsilon,
        loss,
        should_learn,
        should_target,
        transition.episode_return,
        transition.episode_length,
        transition.completed,
    )
    return next_state, metrics


def evaluate(config: DQNConfig, log: EvaluationLog | None, state: RunState, seed: Array) -> RunState:
    """The source's greedy evaluation until `eval_episodes` episodes completed, from fresh resets
    keyed by the run's key; returns the state with that key advanced."""
    env = EpisodeStatisticsWrapper(config.env)
    type Evaluation = tuple[Array, EpisodeStatisticsState[EnvState], Array, Array]

    def step(
        config: DQNConfig,
        env: EpisodeStatisticsWrapper[EnvState, EnvParams],
        log: EvaluationLog | None,
        seed: Array,
        model: QNetwork,
        carry: Evaluation,
    ) -> Evaluation:
        """One greedy vector step; completed episodes count towards `eval_episodes`."""
        observations, env_state, count, key = carry
        actions, _, key = choose_action(model, observations, jnp.asarray(0.0, jnp.float32), key)
        keys: Array = jax.random.split(key, config.num_envs + 1)
        transition = jax.vmap(partial(environment_step, env), in_axes=(0, 0, 0, None))(
            keys[1:], env_state, actions, env.default_params
        )
        if log is not None:
            metrics = EvaluationMetrics(transition.episode_return, transition.completed, count)
            jax.lax.cond(
                jnp.any(transition.completed),
                partial(jax.debug.callback, log, ordered=True),
                lambda seed, metrics: None,
                seed,
                metrics,
            )
        return (
            transition.observations,
            transition.state,
            count + jnp.sum(transition.completed, dtype=jnp.int32),
            keys[0],
        )

    def unfinished(config: DQNConfig, carry: Evaluation) -> Array:
        return carry[2] < config.eval_episodes

    reset_keys: Array = jax.random.split(state.key, config.num_envs + 1)
    observations, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys[1:], env.default_params)
    initial: Evaluation = (observations, env_state, jnp.asarray(0, jnp.int32), reset_keys[0])
    _, _, _, key = jax.lax.while_loop(
        partial(unfinished, config), partial(step, config, env, log, seed, state.model), initial
    )
    evaluated = RunState(
        state.model,
        state.target,
        state.optimizer,
        state.replay,
        state.observations,
        state.env_state,
        state.vector_step,
        state.updates,
        state.target_updates,
        key,
    )
    return evaluated


def initialize(config: DQNConfig, key: Array) -> RunState:
    """A fresh run: networks, optimizer, empty replay and reset environments.

    Also the template a checkpoint is restored into (any key).
    """
    wrapped = EpisodeStatisticsWrapper(config.env)
    shape = wrapped.observation_space(wrapped.default_params).shape
    if len(shape) != 1:
        raise ValueError("DQN requires flat observations")
    hidden = tuple(LayerConfig(width, "relu") for width in config.hidden_widths)
    network = MLPConfig(shape[0], (*hidden, LayerConfig(wrapped.num_actions)))
    keys: Array = jax.random.split(key)
    model = QNetwork(network, rngs=nnx.Rngs(keys[1]))
    target = nnx.clone(model)
    adam = config.adam
    # Torch Adam: constant rate, epsilon outside the square root, no root epsilon.
    optimizer = nnx.Optimizer(
        model,
        optax.adam(config.learning_rate, b1=adam.beta1, b2=adam.beta2, eps=adam.epsilon, eps_root=0.0),
        wrt=nnx.Param,
    )
    env_keys: Array = jax.random.split(keys[0], config.num_envs + 1)
    observations, env_state = jax.vmap(wrapped.reset, in_axes=(0, None))(env_keys[1:], wrapped.default_params)
    zero = jnp.asarray(0, jnp.int32)
    capacity, features = config.buffer_size, network.input_features
    stored = jnp.zeros((capacity, features), dtype=jnp.int32)
    scalars = jnp.zeros(capacity, dtype=jnp.float32)
    replay = ReplayState(
        TransitionBatch(
            stored, jnp.zeros(capacity, jnp.int32), scalars, stored, jnp.zeros(capacity, jnp.bool_), scalars
        ),
        zero,
        zero,
    )
    return RunState(model, target, optimizer, replay, observations, env_state, zero, zero, zero, env_keys[0])


def trained_agent(config: DQNConfig, state: RunState) -> Agent:
    """The agent a finished (or restored) run explains; the source saves the epsilon of the last step taken."""
    last_step = jnp.maximum(state.vector_step - 1, 0) * config.num_envs
    fraction = last_step.astype(jnp.float32) / (config.exploration_fraction * config.total_timesteps)
    epsilon = jnp.maximum(config.start_e + (config.end_e - config.start_e) * fraction, config.end_e)
    return Agent(state.model, epsilon.astype(jnp.float32))


def choose_action(model: QNetwork, observations: ModelInput, epsilon: Array, key: Array) -> tuple[Array, Array, Array]:
    """Epsilon-greedy actions (greedy ties shared equally), the source's clamped log-probability of each,
    and the advanced key."""
    q_values = model(observations).q_values
    ties = q_values == jnp.max(q_values, axis=-1, keepdims=True)
    greedy = ties.astype(jnp.float32) / jnp.sum(ties, axis=-1, keepdims=True)
    probabilities = epsilon / q_values.shape[-1] + (1 - epsilon) * greedy
    keys: Array = jax.random.split(key)
    # Sampling uses true zero-probability logits; the clamp applies only to reported log probabilities.
    actions: Array = jax.random.categorical(keys[1], jnp.log(probabilities), axis=-1).astype(jnp.int32)
    return actions, log_probabilities(probabilities, actions), keys[0]


def log_probabilities(probabilities: Array, actions: Array) -> Array:
    """The log-probability of each action, as Torch Categorical reports it: probabilities renormalised
    and clamped to [eps, 1 - eps] before taking logs."""
    normalised = probabilities / jnp.sum(probabilities, axis=-1, keepdims=True)
    clamp = jnp.finfo(jnp.float32).eps
    logits = jnp.log(jnp.clip(normalised, clamp, 1 - clamp))
    return jnp.take_along_axis(logits, actions[..., None], axis=-1)[..., 0]


def environment_step(
    env: EpisodeStatisticsWrapper[EnvState, EnvParams],
    key: Array,
    state: EpisodeStatisticsState[EnvState],
    action: Array,
    params: EnvParams,
) -> EnvironmentStep:
    """Keep Gymnax auto-reset and expose its pre-reset observation with the same step key."""
    keys: Array = jax.random.split(key)
    raw_observations, _, _, _, _ = env.env.step_env(keys[0], state.env_state, action, params)
    observation, next_state, reward, _, info = env.step(key, state, action, params)
    episode = info["episode"]
    return EnvironmentStep(
        observation,
        next_state,
        raw_observations,
        reward,
        info["terminated"],
        info["truncated"],
        episode["return"],
        episode["length"],
        episode["completed"],
    )


def dqn_loss(model: QNetwork, target: QNetwork, batch: TransitionBatch, gamma: float) -> Array:
    """Mean squared TD error against the target network's greedy value."""
    if batch.observations.dtype != jnp.int32 or batch.rewards.dtype != jnp.float32 or batch.terminated.dtype != bool:
        raise TypeError("replay observations must be int32, rewards float32 and terminations bool")
    next_values = jnp.max(target(batch.next_observations).q_values, axis=-1)
    targets = jax.lax.stop_gradient(batch.rewards + gamma * next_values * (~batch.terminated).astype(jnp.float32))
    values = jnp.take_along_axis(model(batch.observations).q_values, batch.actions[:, None], axis=-1)[:, 0]
    return jnp.mean(jnp.square(targets - values))
