"""Step 1: train the FastSVERL DQN agent (CleanRL DQN with soft target updates).

`main` trains one seed's agent, or loads it from `checkpoint`, and returns its final
state, every vector step's metrics (None when loaded), and the frozen `Agent` the
explanation steps explain. With a `track` backend it logs the source's scalars live and
pushes the whole final state as an Orbax checkpoint artifact. `train` is the compiled
part: `nnx.jit(partial(train, config, log))(key)`. Timestep and frequency
units count all environment transitions, as the source's `global_step` does. Replay
matches FastSVERL `Buffer` insertion and sampling, with fixed shapes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
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
import wandb
from flax import nnx, struct
from jax import Array
from jaxtyping import Bool, Float, Int

from new_jax_shapley import backends
from new_jax_shapley.backends import Track
from new_jax_shapley.common import checkpoints
from new_jax_shapley.common.networks import MLP, AdamConfig, LayerConfig, MLPConfig, ModelInput
from new_jax_shapley.envs import (
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
from new_jax_shapley.envs.wrappers import EpisodeStatisticsState, EpisodeStatisticsWrapper

type ActionValues = Float[Array, "... actions"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type Observations = Int[Array, "batch features"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type Actions = Int[Array, "batch"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.
type Scalars = Float[Array, "batch"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.
type Terminations = Bool[Array, "batch"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.
type StepLog = Callable[[StepMetrics], None]
type Env = Gridworld | Mastermind | Hypercube
type EnvState = GridworldState | MastermindState | HypercubeState
type EnvParams = GridworldParams | MastermindParams | HypercubeParams

# Config fields that only say where a stage is tracked; a checkpoint ignores them.
TRACKING_FIELDS = ("checkpoint", "name", "track", "project", "entity", "group")

# The source logs the DQN loss when the global step crosses a multiple of this.
LOG_STEPS = 100


@dataclass(frozen=True, slots=True)
class DQNConfig:
    env: Env
    seed: int = 1
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
    adam: AdamConfig = field(default_factory=AdamConfig)
    checkpoint: str | None = None
    """An agent artifact to load instead of training, e.g. `GROUP--algorithm--dqn:latest`."""
    name: str = "dqn"
    """The experiment name runs are named after: `NAME--algorithm--dqn`."""
    track: Track | None = None
    """The tracking backend; None trains untracked (tests)."""
    project: str = "dev-jax"
    entity: str | None = "fastsverl"
    """wandb entity (team or user)."""
    group: str | None = None
    """The run group; None starts a new one, `NAME-TIMESTAMP`."""

    def __post_init__(self) -> None:
        for name, value in (
            ("num_envs", self.num_envs),
            ("total_timesteps", self.total_timesteps),
            ("buffer_size", self.buffer_size),
            ("batch_size", self.batch_size),
            ("train_frequency", self.train_frequency),
            ("target_network_frequency", self.target_network_frequency),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.vector_steps < 1:
            raise ValueError("total_timesteps must include at least one vector step")
        if self.learning_starts < 0:
            raise ValueError("learning_starts must be nonnegative")
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
    """Train (or load) one agent, tracked by `config.track`; metrics carry a leading vector-step axis."""
    key = jax.random.key(config.seed)
    if config.track is None:
        state, metrics = nnx.jit(partial(train, config, None))(key)
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
        run = tracking.init(
            project=config.project, name=f"{config.name}--algorithm--dqn", group=group, config=asdict(config)
        )
        try:
            if config.checkpoint is not None:
                directory = run.use_artifact(config.checkpoint).download(root=str(root / "artifacts" / artifact))
                state = checkpoints.load(Path(directory), initialize(config, jax.random.key(0)), settings)
                return state, None, trained_agent(config, state)

            def log(run: wandb.Run | trackio.Run, num_envs: int, metrics: StepMetrics) -> None:
                """The source's DQN scalars; episodes finishing on the same step are averaged."""
                step = int(metrics.global_step)
                completed = np.asarray(metrics.completed)
                row: dict[str, float] = {}
                if completed.any():
                    row["agent/episodic_return"] = float(np.asarray(metrics.episode_return)[completed].mean())
                    row["agent/episodic_length"] = float(np.asarray(metrics.episode_length)[completed].mean())
                    row["agent/epsilon"] = float(metrics.epsilon)
                if bool(metrics.updated) and step % LOG_STEPS < num_envs:
                    row["agent/DQN_loss"] = float(metrics.loss)
                if row:
                    run.log(row, step=step)

            state, metrics = nnx.jit(partial(train, config, partial(log, run, config.num_envs)))(key)
            directory = root / "checkpoints" / artifact
            checkpoints.save(directory, state, settings)
            run.log_artifact(str(directory), name=artifact, type="agent").wait()
            print(f"Agent checkpoint: {artifact}:latest")
            return state, metrics, trained_agent(config, state)
        finally:
            run.finish()


def train(config: DQNConfig, log: StepLog | None, key: Array) -> tuple[RunState, StepMetrics]:
    """The compiled training run. `log` receives a step's metrics during the run (a host
    callback), on steps where an episode finished or the source would log the DQN loss."""
    env = EpisodeStatisticsWrapper(config.env)
    state = initialize(config, key)

    def step(
        config: DQNConfig,
        env: EpisodeStatisticsWrapper[EnvState, EnvParams],
        log: StepLog | None,
        state: RunState,
        index: Array,
    ) -> tuple[RunState, StepMetrics]:
        """Act, store, then learn and soft-update the target when due."""
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

        def soft_update(model: QNetwork, target: QNetwork, tau: float) -> None:
            for source, destination in zip(model.mlp.layers, target.mlp.layers, strict=True):
                destination.kernel[...] = tau * source.kernel[...] + (1 - tau) * destination.kernel[...]
                destination.bias[...] = tau * source.bias[...] + (1 - tau) * destination.bias[...]

        nnx.cond(should_target, soft_update, lambda model, target, tau: None, state.model, state.target, config.tau)
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
        if log is not None:
            loss_due = should_learn & (global_step % LOG_STEPS < config.num_envs)
            jax.lax.cond(
                jnp.any(transition.completed) | loss_due,
                partial(jax.debug.callback, log, ordered=True),
                lambda metrics: None,
                metrics,
            )
        return next_state, metrics

    return nnx.scan(partial(step, config, env, log), in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))(
        state, jnp.arange(config.vector_steps, dtype=jnp.int32)
    )


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
    # Torch Categorical renormalizes probabilities and clamps before converting to logits.
    normalised = probabilities / jnp.sum(probabilities, axis=-1, keepdims=True)
    clamp = jnp.finfo(jnp.float32).eps
    logits = jnp.log(jnp.clip(normalised, clamp, 1 - clamp))
    return actions, jnp.take_along_axis(logits, actions[..., None], axis=-1)[..., 0], keys[0]


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
