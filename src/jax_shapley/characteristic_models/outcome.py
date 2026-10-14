"""Step 3 (outcome): the value of an agent conditioned to act on a partially known state.

The source's `PerformanceCharacteristic` asks what return the agent would obtain if, in
state `e`, it acted according to the behaviour characteristic given only coalition `C`
of `e`'s features, and otherwise followed its own policy. A goal-conditioned DQN-style
network learns this from `(s, e, C)`. Off-policy (`PerformanceCharacteristicOffPolicy`):
Q(s, a | e, C) from prefixes of the agent's replay, with the next action sampled from the
conditioned policy, and `v(e, C) = sum_a pi_hat(a | e; e, C) Q(e, a | e, C)`.

Exact values solve the tabular MDP under the conditioned policy for every coalition and
every unique state of the behaviour characteristic's exact buffer. `with_exact_char`
conditions them on the exact behaviour characteristic, otherwise on the learnt one;
training always uses the learnt one. The behaviour characteristic stage's `targets` (the
explained policy) and shared epoch loop are imported, not repeated.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from functools import partial
from math import isfinite
from pathlib import Path
from time import time_ns
from typing import Literal, NamedTuple, cast

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
from jax_shapley.algorithms.dqn import (
    LOG_STEPS,
    STEP_BLOCK,
    TRACKING_FIELDS,
    Agent,
    Env,
    EnvParams,
    EnvState,
    TransitionBatch,
    environment_step,
)
from jax_shapley.backends import Track
from jax_shapley.characteristic_models.behaviour import (
    CharacteristicConfig,
    CharacteristicModel,
    SampledPool,
    Values,
    sampled_values,
    targets,
)
from jax_shapley.common import checkpoints
from jax_shapley.common.coalitions import MASK, Coalitions, Observations, coalition_table, mask
from jax_shapley.common.networks import MLP, AdamConfig, LayerConfig, MLPConfig, soft_update
from jax_shapley.common.supervised_learning import (
    EpochMetrics,
    Partition,
    StateBuffers,
    TrainState,
    log_epochs,
    stack_seeds,
    train,
)
from jax_shapley.common.tabular import ActionTable, TabularModel, policy_q_values, state_indices, transition_model
from jax_shapley.envs.wrappers import EpisodeStatisticsState, EpisodeStatisticsWrapper

type Variant = Literal["off_policy", "on_policy"]
type ExactTable = Float[Array, "coalitions unique"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type UniqueStates = Int[Array, "unique features"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type UniqueFlags = Bool[Array, "unique"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.
type ConditionedTable = Float[Array, "coalitions unique actions"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.


@dataclass(frozen=True, slots=True)
class OutcomeTrainConfig:
    """Source outcome settings, shared like the source's CharacteristicArgs.

    Off-policy runs use train_models over prefixes of the agent's replay: the first
    `train_size` and `validation_size` transitions, so they overlap. On-policy runs use
    train_V_on_policy: `total_timesteps` environment steps across `num_envs` environments,
    with a replay as large as the run.
    """

    epochs: int = 128
    patience: int | None = None
    learning_rate: float = 2.5e-4
    batch_size: int = 64
    train_size: int = 50_000
    validation_size: int = 80
    exact_loss: bool = True
    exact_condition: float | None = None
    track_batch: bool = True
    total_timesteps: int = 100_000
    num_envs: int = 8
    train_frequency: int = 1
    target_network_frequency: int = 1
    adam: AdamConfig = field(default_factory=AdamConfig)

    def __post_init__(self) -> None:
        if any(getattr(self, item.name) is tyro.MISSING for item in fields(self)):
            return  # A sweep config: validated once the command line supplies every field.
        for name, value in (
            ("epochs", self.epochs),
            ("batch_size", self.batch_size),
            ("train_size", self.train_size),
            ("validation_size", self.validation_size),
            ("patience", self.epochs if self.patience is None else self.patience),
            ("batches", self.batches),
            ("num_envs", self.num_envs),
            ("train_frequency", self.train_frequency),
            ("target_network_frequency", self.target_network_frequency),
            ("vector steps (total_timesteps // num_envs)", self.vector_steps),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if not isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.exact_condition is not None and (not self.exact_loss or not self.exact_condition > 0):
            raise ValueError("exact_condition must be positive and requires exact_loss")

    @property
    def batches(self) -> int:
        return self.train_size // self.batch_size

    @property
    def patience_epochs(self) -> int:
        return self.epochs if self.patience is None else self.patience

    @property
    def vector_steps(self) -> int:
        """On-policy loop length: the source's range(total_timesteps // num_envs)."""
        return self.total_timesteps // max(self.num_envs, 1)


@dataclass(frozen=True, slots=True)
class OutcomeConfig:
    """Which outcome characteristic, its soft target-update rate, discount, network and settings.

    Every seed explains the agent behind the explained behaviour characteristic seed, and
    trains in one compiled program.
    """

    env: Env
    variant: Variant = "off_policy"
    with_exact_char: bool = True
    tau: float = 0.001
    gamma: float = 1.0
    hidden_widths: tuple[int, ...] = (64, 64)
    seeds: tuple[int, ...] = (1,)
    train: OutcomeTrainConfig = field(default_factory=OutcomeTrainConfig)
    checkpoint: str | None = None
    """An outcome artifact to load instead of training, e.g. `GROUP--outcome--off-policy--with-exact:latest`."""
    name: str = "outcome"
    """The experiment name runs are named after: `NAME--outcome--VARIANT--{with,without}-exact--seed-N`."""
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
        if self.variant not in ("off_policy", "on_policy"):
            raise ValueError("variant must be off_policy or on_policy")
        if not 0 < self.tau <= 1 or not 0 <= self.gamma <= 1:
            raise ValueError("tau must be in (0, 1] and gamma in [0, 1]")
        if any(width < 1 for width in self.hidden_widths):
            raise ValueError("hidden widths must be positive")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct and nonempty")


@struct.dataclass
class OutcomeFrozen:
    """The frozen models an outcome characteristic conditions on: the agent and its behaviour characteristic."""

    agent: Agent
    behaviour: CharacteristicModel


@struct.dataclass
class ExactOutcome:
    """Exact v(e, C) for every coalition and every unique exact-buffer state (padded, with validity)."""

    states: UniqueStates
    state_valid: UniqueFlags
    values: ExactTable


@struct.dataclass
class OutcomeContext:
    """Per-seed constants an outcome learner reads: replay prefixes and the exact ground truth."""

    train: TransitionBatch
    validation: TransitionBatch
    exact: ExactOutcome | None


@struct.dataclass
class ConditionedReplay:
    """On-policy experience and the (e, C) each decision was conditioned on (source V.buffer with e_obs, C)."""

    transitions: TransitionBatch
    explained: Array
    coalitions: Array
    pointer: Array
    size: Array


@struct.dataclass
class OnPolicyState:
    """Source train_V_on_policy loop state for one seed; `step` counts completed vector steps.

    The final model is the last one (the source saves no best model on-policy).
    """

    networks: OutcomeNetworks
    optimizer: nnx.Optimizer[OutcomeNetworks]
    replay: ConditionedReplay
    observations: Array
    env_state: EpisodeStatisticsState[EnvState]
    explained: Array
    coalitions: Array
    step: Array
    stopped: Array
    key: Array


@struct.dataclass
class OnPolicyContext:
    """Per-seed constants: the behaviour characteristic's train buffer (the e_obs pool) and exact values,
    and that buffer with v_F (`sampled`) when the behaviour characteristic is sampled."""

    pool: Array
    exact: ExactOutcome | None
    sampled: SampledPool | None = None


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

    The source copies the model into the target at construction. One Adam covers both: the
    target only enters losses through stop-gradient TD targets, so its gradients are exactly
    zero, Adam leaves it unchanged, and the soft update after each step moves it.
    """

    def __init__(self, config: MLPConfig, *, rngs: nnx.Rngs) -> None:
        self.model = OutcomeModel(config, rngs=rngs)
        self.target = nnx.clone(self.model)


@struct.dataclass
class OutcomeExplained:
    """A trained outcome characteristic as Shapley models explain it: its networks, and the agent
    and behaviour characteristic its conditioned policy uses."""

    agent: Agent
    behaviour: CharacteristicModel
    networks: OutcomeNetworks


def main(
    config: OutcomeConfig,
    explained: CharacteristicConfig,
    agents: Agent,
    behaviours: CharacteristicModel,
    behaviour_buffers: StateBuffers,
    replays: TransitionBatch,
) -> tuple[TrainState[OutcomeNetworks] | OnPolicyState, EpochMetrics | None]:
    """Train (or load) every seed's outcome characteristic, tracked by `config.track`.

    The agents, behaviour characteristics (and their buffers, whose exact buffer the exact
    values explain) and the agents' replays are seed-stacked, one per outcome seed.
    `explained` is the behaviour characteristic's config. Off-policy runs learn from the
    replays' prefixes; on-policy runs act and draw e from the behaviour characteristic's
    train buffer. With a backend each seed logs the source's
    `model/PerformanceCharacteristic{Off,On}Policy_*` scalars into its own run, and the whole
    seed-stacked final state is pushed as one Orbax checkpoint artifact.
    """
    if explained.kind != "behaviour":
        raise ValueError("outcome characteristics condition on a behaviour characteristic")
    if explained.estimator == "sampled" and (config.variant != "on_policy" or not config.with_exact_char):
        # The source's sampling groups: on-policy outcome runs, with exact behaviour values in the ground truth.
        raise ValueError("sampled behaviour characteristics condition on-policy outcome runs with exact values")
    seeds = len(config.seeds)
    if agents.epsilon.shape[0] != seeds or behaviour_buffers.exact.shape[0] != seeds:
        raise ValueError("outcome inputs need one agent and behaviour characteristic per seed")
    prefix = max(config.train.train_size, config.train.validation_size)
    if config.variant == "off_policy" and replays.observations.shape[1] < prefix:
        raise ValueError("the agents' replays hold fewer transitions than the outcome prefixes")
    keys = jnp.stack([jax.random.key(seed) for seed in config.seeds])
    # Exact values solve the environment's tabular MDP (host-side enumeration).
    tabular = transition_model(config.env) if config.train.exact_loss else None
    off_policy = config.variant == "off_policy"
    if config.track is None:
        if off_policy:
            return nnx.jit(partial(fit, config, explained, tabular, None))(
                keys, agents, behaviours, behaviour_buffers, replays
            )
        return nnx.jit(partial(fit_on_policy, config, explained, tabular, None))(
            keys, agents, behaviours, behaviour_buffers
        )
    group = config.group or f"{config.name}-{time_ns()}"
    variant = config.variant.replace("_", "-")
    exactness = "with-exact" if config.with_exact_char else "without-exact"
    artifact = f"{group}--outcome--{variant}--{exactness}"
    settings = {
        "stage": "outcome",
        "env": config.env.name,
        "explained": {name: value for name, value in asdict(explained).items() if name not in TRACKING_FIELDS},
        "config": {name: value for name, value in asdict(config).items() if name not in TRACKING_FIELDS},
    }
    with backends.session(config.track, config.entity) as root:
        tracking = wandb if config.track == "wandb" else trackio
        runs = tuple(
            tracking.init(
                project=config.project,
                name=f"{config.name}--outcome--{variant}--{exactness}--seed-{seed}",
                group=group,
                config={**asdict(config), "seed": seed},
            )
            for seed in config.seeds
        )
        try:
            if config.checkpoint is not None:
                directory = runs[0].use_artifact(config.checkpoint).download(root=str(root / "artifacts" / artifact))
                template = initialize(config, explained) if off_policy else initialize_on_policy(config, explained)
                return checkpoints.load(Path(directory), template, settings), None
            model = f"PerformanceCharacteristic{'OffPolicy' if off_policy else 'OnPolicy'}"
            epoch_log = partial(log_epochs, runs, model, config.train.exact_loss)
            if off_policy:
                program = nnx.jit(partial(fit, config, explained, tabular, epoch_log))
                state, metrics = program(keys, agents, behaviours, behaviour_buffers, replays)
            else:
                program = nnx.jit(partial(fit_on_policy, config, explained, tabular, epoch_log))
                state, metrics = program(keys, agents, behaviours, behaviour_buffers)
            directory = root / "checkpoints" / artifact
            checkpoints.save(directory, state, settings)
            backends.upload(runs[0], directory, artifact, "outcome")
            print(f"Outcome checkpoint: {artifact}:latest")
            return state, metrics
        finally:
            for run in runs:
                run.finish()


def fit(
    config: OutcomeConfig,
    explained: CharacteristicConfig,
    tabular: TabularModel | None,
    log: Callable[[EpochMetrics], None] | None,
    keys: Array,
    agents: Agent,
    behaviours: CharacteristicModel,
    behaviour_buffers: StateBuffers,
    replays: TransitionBatch,
) -> tuple[TrainState[OutcomeNetworks], EpochMetrics]:
    """Learn every seed's off-policy outcome characteristic from its agent's replay prefixes.

    The buffers kept with the state are the prefixes' observations and the behaviour
    characteristic's exact buffer, whose unique states the exact values explain.
    """
    training = config.train
    prefixes = (
        jax.tree.map(lambda rows: rows[:, : training.train_size], replays),
        jax.tree.map(lambda rows: rows[:, : training.validation_size], replays),
    )
    train_rows, validation_rows = cast(tuple[TransitionBatch, TransitionBatch], prefixes)  # tree.map keeps structs.
    buffers = StateBuffers(train_rows.observations, validation_rows.observations, behaviour_buffers.exact)
    state = initialize(config, explained, keys, buffers)
    features = buffers.train.shape[-1]

    def context(
        config: OutcomeConfig,
        explained: CharacteristicConfig,
        tabular: TabularModel | None,
        train_rows: TransitionBatch,
        validation_rows: TransitionBatch,
        exact_rows: Array,
        frozen: OutcomeFrozen,
    ) -> OutcomeContext:
        if tabular is None:
            return OutcomeContext(train_rows, validation_rows, None)
        exact = exact_outcome(tabular, exact_rows, frozen, explained, config.with_exact_char, config.gamma)
        return OutcomeContext(train_rows, validation_rows, exact)

    frozen = OutcomeFrozen(agents, behaviours)
    contexts = nnx.vmap(partial(context, config, explained, tabular), in_axes=(0, 0, 0, 0), out_axes=0)(
        train_rows, validation_rows, behaviour_buffers.exact, frozen
    )
    return train(
        state,
        contexts,
        frozen,
        partial(loss, config, explained),
        training,
        features,
        log,
        partial(update, config, explained),
    )


def initialize(
    config: OutcomeConfig,
    explained: CharacteristicConfig,
    keys: Array | None = None,
    buffers: StateBuffers | None = None,
) -> TrainState[OutcomeNetworks]:
    """Every seed's fresh training state, seed-stacked.

    Without keys and buffers it is the template a checkpoint is restored into: the
    configured seeds, and zero buffers with the prefixes' sizes and the behaviour
    characteristic's exact size.
    """
    wrapped = EpisodeStatisticsWrapper(config.env)
    shape = wrapped.observation_space(wrapped.default_params).shape
    if len(shape) != 1:
        raise ValueError("explanations require flat observations")
    features = int(shape[0])
    outputs = wrapped.num_actions if config.variant == "off_policy" else 1
    hidden = tuple(LayerConfig(width, "relu") for width in config.hidden_widths)
    network = MLPConfig(3 * features, (*hidden, LayerConfig(outputs)))
    training = config.train
    if keys is None:
        keys = jnp.stack([jax.random.key(seed) for seed in config.seeds])
    if buffers is None:
        empty = StateBuffers(
            jnp.zeros((training.train_size, features), jnp.int32),
            jnp.zeros((training.validation_size, features), jnp.int32),
            jnp.zeros((explained.train.exact_size, features), jnp.int32),
        )
        buffers = stack_seeds([empty] * len(config.seeds))

    def seed(
        network: MLPConfig, training: OutcomeTrainConfig, key: Array, buffers: StateBuffers
    ) -> TrainState[OutcomeNetworks]:
        model_key, key = jax.random.split(key)
        networks = OutcomeNetworks(network, rngs=nnx.Rngs(model_key))
        adam = training.adam
        # Torch Adam: constant rate, epsilon outside the square root, no root epsilon.
        optimizer = nnx.Optimizer(
            networks,
            optax.adam(training.learning_rate, b1=adam.beta1, b2=adam.beta2, eps=adam.epsilon, eps_root=0.0),
            wrt=nnx.Param,
        )
        return TrainState(
            networks,
            nnx.clone(networks),
            optimizer,
            buffers,
            jnp.asarray(0, jnp.int32),
            jnp.asarray(jnp.inf, jnp.float32),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(False),
            key,
        )

    return nnx.vmap(partial(seed, network, training), in_axes=(0, 0), out_axes=0)(keys, buffers)


def fit_on_policy(
    config: OutcomeConfig,
    explained: CharacteristicConfig,
    tabular: TabularModel | None,
    log: Callable[[EpochMetrics], None] | None,
    keys: Array,
    agents: Agent,
    behaviours: CharacteristicModel,
    behaviour_buffers: StateBuffers,
) -> tuple[OnPolicyState, EpochMetrics]:
    """Source train_V_on_policy for every seed: act with pi_hat, store (e, C) with each step, learn V.

    e_obs rows come from the behaviour characteristic's train buffer and one coalition is
    shared by all environments; both are resampled every decision. A seed stops once its
    exact loss falls below `exact_condition`; metrics have one row per vector step, active
    where the source logs (learning steps crossing a multiple of 100 environment steps).
    """
    env = EpisodeStatisticsWrapper(config.env)
    state = initialize_on_policy(config, explained, keys, behaviour_buffers.train)

    def context(
        config: OutcomeConfig,
        explained: CharacteristicConfig,
        tabular: TabularModel | None,
        buffers: StateBuffers,
        frozen: OutcomeFrozen,
    ) -> OnPolicyContext:
        sampled = None
        if explained.estimator == "sampled":
            pooled = targets(explained, frozen.agent, buffers.train)
            sampled = SampledPool(buffers.train, pooled, jnp.mean(pooled, axis=0))
        if tabular is None:
            return OnPolicyContext(buffers.train, None, sampled)
        exact = exact_outcome(tabular, buffers.exact, frozen, explained, config.with_exact_char, config.gamma)
        return OnPolicyContext(buffers.train, exact, sampled)

    frozen = OutcomeFrozen(agents, behaviours)
    contexts = nnx.vmap(partial(context, config, explained, tabular), in_axes=(0, 0), out_axes=0)(
        behaviour_buffers, frozen
    )

    def seed_step(
        config: OutcomeConfig,
        explained: CharacteristicConfig,
        env: EpisodeStatisticsWrapper[EnvState, EnvParams],
        state: OnPolicyState,
        context: OnPolicyContext,
        frozen: OutcomeFrozen,
    ) -> tuple[OnPolicyState, EpochMetrics]:
        """One source train_V_on_policy iteration for one seed, or a no-op once it has stopped."""
        training = config.train
        running = (state.step < training.vector_steps) & ~state.stopped
        return nnx.cond(
            running,
            partial(on_policy_step, config, explained, env),
            partial(skip_on_policy_step),
            state,
            context,
            frozen,
        )

    # Frozen models ride in the scan carry: NNX transforms only accept graph nodes created at
    # their own trace level as arguments.
    type Carry = tuple[OnPolicyState, OutcomeFrozen]

    def step(
        config: OutcomeConfig,
        explained: CharacteristicConfig,
        env: EpisodeStatisticsWrapper[EnvState, EnvParams],
        contexts: OnPolicyContext,
        carry: Carry,
        index: Array,
    ) -> tuple[Carry, EpochMetrics]:
        del index
        state, frozen = carry
        every_seed = nnx.vmap(partial(seed_step, config, explained, env), in_axes=(0, 0, 0), out_axes=(0, 0))
        # Padding steps past the run, once no seed acts, are skipped outright: under the seed vmap a
        # per-seed skip computes both branches.
        acting = jnp.any((state.step < config.train.vector_steps) & ~state.stopped)
        skip = nnx.vmap(skip_on_policy_step, in_axes=(0, 0, 0), out_axes=(0, 0))
        state, metrics = nnx.cond(acting, every_seed, skip, state, contexts, frozen)
        return (state, frozen), metrics

    def block(
        config: OutcomeConfig,
        explained: CharacteristicConfig,
        env: EpisodeStatisticsWrapper[EnvState, EnvParams],
        contexts: OnPolicyContext,
        log: Callable[[EpochMetrics], None] | None,
        carry: Carry,
        indices: Array,
    ) -> tuple[Carry, EpochMetrics]:
        """`STEP_BLOCK` vector steps of every seed (steps past the run, or after a seed stopped, do
        nothing), then one live log of them all."""
        carry, metrics = nnx.scan(
            partial(step, config, explained, env, contexts), in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0)
        )(carry, indices)
        if log is not None:
            jax.debug.callback(log, metrics, ordered=True)
        return carry, metrics

    blocks = -(-config.train.vector_steps // STEP_BLOCK)
    indices = jnp.arange(blocks * STEP_BLOCK, dtype=jnp.int32).reshape(blocks, STEP_BLOCK)
    (trained, _), blocked = nnx.scan(
        partial(block, config, explained, env, contexts, log), in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0)
    )((state, frozen), indices)
    metrics = EpochMetrics(*[
        leaf.reshape(blocks * STEP_BLOCK, *leaf.shape[2:])[: config.train.vector_steps]
        for leaf in jax.tree.leaves(blocked)
    ])
    return trained, metrics


def on_policy_step(
    config: OutcomeConfig,
    explained: CharacteristicConfig,
    env: EpisodeStatisticsWrapper[EnvState, EnvParams],
    state: OnPolicyState,
    context: OnPolicyContext,
    frozen: OutcomeFrozen,
) -> tuple[OnPolicyState, EpochMetrics]:
    """Act with pi_hat in every environment, store the step, redraw (e, C), learn V by TD(0) when due,
    measure the exact loss when the source logs, and soft-update the target."""
    training = config.train
    global_step = state.step * training.num_envs
    keys: Array = jax.random.split(state.key, 5)
    if context.sampled is None:
        behaviour = frozen.behaviour(mask(state.explained, state.coalitions)).values
    else:
        # A sampled behaviour draws its values afresh at every decision, as the source's get_val does.
        behaviour = sampled_values(context.sampled, jax.random.fold_in(keys[0], 1), state.explained, state.coalitions)
    probabilities = conditioned_policy(explained, frozen.agent, behaviour, state.observations, state.explained)
    actions = jax.random.categorical(keys[0], jnp.log(probabilities), axis=-1).astype(jnp.int32)
    step_keys: Array = jax.random.split(keys[1], training.num_envs)
    transition = jax.vmap(partial(environment_step, env), in_axes=(0, 0, 0, None))(
        step_keys, state.env_state, actions, env.default_params
    )
    # Insert in environment order, as the source Buffer.add does. Terminal rows store the
    # pre-reset observation; the (1 - terminated) mask makes the source's stored reset
    # observation equivalent.
    replay = state.replay
    capacity = replay.explained.shape[0]
    rows = (replay.pointer + jnp.arange(training.num_envs, dtype=jnp.int32)) % capacity
    inserted = TransitionBatch(
        state.observations,
        actions,
        transition.rewards,
        transition.raw_observations.astype(jnp.int32),
        transition.terminated,
        jnp.zeros(training.num_envs, jnp.float32),
    )
    stored = zip(jax.tree.leaves(replay.transitions), jax.tree.leaves(inserted), strict=True)
    replay = ConditionedReplay(
        TransitionBatch(*[storage.at[rows].set(values) for storage, values in stored]),
        replay.explained.at[rows].set(state.explained),
        replay.coalitions.at[rows].set(state.coalitions),
        (replay.pointer + training.num_envs) % capacity,
        jnp.minimum(replay.size + training.num_envs, capacity).astype(jnp.int32),
    )
    # The next decision's e_obs rows (with replacement) and one coalition for every environment.
    draw_keys: Array = jax.random.split(keys[2])
    pool = context.pool
    explained_rows = pool[jax.random.randint(draw_keys[0], (training.num_envs,), 0, pool.shape[0])]
    coalition = jax.random.bernoulli(draw_keys[1], 0.5, (1, pool.shape[-1]))
    coalitions = jnp.broadcast_to(coalition, (training.num_envs, pool.shape[-1]))
    learn = global_step % training.train_frequency < training.num_envs
    sampled = jax.random.randint(keys[3], (training.batch_size,), 0, replay.size, dtype=jnp.int32)
    batch = TransitionBatch(*[leaf[sampled] for leaf in jax.tree.leaves(replay.transitions)])

    def learn_step(
        gamma: float,
        networks: OutcomeNetworks,
        optimizer: nnx.Optimizer[OutcomeNetworks],
        batch: TransitionBatch,
        explained: Array,
        coalitions: Array,
    ) -> Array:
        # NNX transformation has a dynamic return signature; assert its loss/gradient boundary.
        value, gradients = cast(
            tuple[Array, nnx.State[str | int, nnx.Param[Array]]],
            nnx.value_and_grad(on_policy_loss)(networks, batch, explained, coalitions, gamma),
        )
        optimizer.update(networks, gradients)
        return value

    def skip_learn(
        gamma: float,
        networks: OutcomeNetworks,
        optimizer: nnx.Optimizer[OutcomeNetworks],
        batch: TransitionBatch,
        explained: Array,
        coalitions: Array,
    ) -> Array:
        del gamma, networks, optimizer, batch, explained, coalitions
        return jnp.asarray(jnp.nan, jnp.float32)

    loss_value = nnx.cond(
        learn,
        partial(learn_step, config.gamma),
        partial(skip_learn, config.gamma),
        state.networks,
        state.optimizer,
        batch,
        replay.explained[sampled],
        replay.coalitions[sampled],
    )
    logged = learn & (global_step % LOG_STEPS < training.num_envs)
    exact = jnp.asarray(jnp.nan, jnp.float32)
    if context.exact is not None:

        def exact_loss(
            config: OutcomeConfig,
            explained: CharacteristicConfig,
            exact: ExactOutcome,
            frozen: OutcomeFrozen,
            networks: OutcomeNetworks,
        ) -> Array:
            outcome = OutcomeExplained(frozen.agent, frozen.behaviour, networks)
            return exact_values_loss(config, explained, outcome, exact)

        exact = nnx.cond(
            logged,
            partial(exact_loss, config, explained, context.exact),
            lambda frozen, networks: jnp.asarray(jnp.nan, jnp.float32),
            frozen,
            state.networks,
        )
    stopped = jnp.asarray(False)
    if training.exact_condition is not None:
        stopped = logged & (exact < training.exact_condition)
    # The source breaks out of the loop before this step's target update.
    target = (global_step % training.target_network_frequency < training.num_envs) & ~stopped

    nnx.cond(
        target,
        lambda networks, tau: soft_update(networks.model.mlp, networks.target.mlp, tau),
        lambda networks, tau: None,
        state.networks,
        config.tau,
    )
    step = state.step + 1
    next_state = OnPolicyState(
        state.networks,
        state.optimizer,
        replay,
        transition.observations.astype(jnp.int32),
        transition.state,
        explained_rows,
        coalitions,
        step,
        stopped,
        keys[4],
    )
    nothing = jnp.asarray(jnp.nan, jnp.float32)
    return next_state, EpochMetrics(step, global_step, loss_value, nothing, exact, step, logged)


def skip_on_policy_step(
    state: OnPolicyState, context: OnPolicyContext, frozen: OutcomeFrozen
) -> tuple[OnPolicyState, EpochMetrics]:
    """A stopped seed's step: nothing changes, nothing is logged."""
    del context, frozen
    zero, nothing = jnp.asarray(0, jnp.int32), jnp.asarray(0.0, jnp.float32)
    return state, EpochMetrics(zero, zero, nothing, nothing, nothing, zero, jnp.asarray(False))


def initialize_on_policy(
    config: OutcomeConfig, explained: CharacteristicConfig, keys: Array | None = None, pools: Array | None = None
) -> OnPolicyState:
    """Every seed's fresh on-policy state, seed-stacked: networks, optimizer, an empty replay as large
    as the run, reset environments and a first (e, C) drawn from the pool.

    Without keys and pools it is the template a checkpoint is restored into.
    """
    env = EpisodeStatisticsWrapper(config.env)
    shape = env.observation_space(env.default_params).shape
    if len(shape) != 1:
        raise ValueError("explanations require flat observations")
    features = int(shape[0])
    hidden = tuple(LayerConfig(width, "relu") for width in config.hidden_widths)
    network = MLPConfig(3 * features, (*hidden, LayerConfig(1)))
    training = config.train
    if keys is None:
        keys = jnp.stack([jax.random.key(seed) for seed in config.seeds])
    if pools is None:
        pools = jnp.zeros((len(config.seeds), explained.train.train_size, features), jnp.int32)

    def seed(
        env: EpisodeStatisticsWrapper[EnvState, EnvParams],
        network: MLPConfig,
        training: OutcomeTrainConfig,
        key: Array,
        pool: Array,
    ) -> OnPolicyState:
        keys: Array = jax.random.split(key, 4)
        networks = OutcomeNetworks(network, rngs=nnx.Rngs(keys[0]))
        adam = training.adam
        # Torch Adam: constant rate, epsilon outside the square root, no root epsilon.
        optimizer = nnx.Optimizer(
            networks,
            optax.adam(training.learning_rate, b1=adam.beta1, b2=adam.beta2, eps=adam.epsilon, eps_root=0.0),
            wrt=nnx.Param,
        )
        reset_keys: Array = jax.random.split(keys[1], training.num_envs)
        observations, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_keys, env.default_params)
        draw_keys: Array = jax.random.split(keys[2])
        explained_rows = pool[jax.random.randint(draw_keys[0], (training.num_envs,), 0, pool.shape[0])]
        coalition = jax.random.bernoulli(draw_keys[1], 0.5, (1, pool.shape[-1]))
        capacity, width = training.total_timesteps, pool.shape[-1]
        stored = jnp.zeros((capacity, width), jnp.int32)
        replay = ConditionedReplay(
            TransitionBatch(
                stored,
                jnp.zeros(capacity, jnp.int32),
                jnp.zeros(capacity, jnp.float32),
                stored,
                jnp.zeros(capacity, jnp.bool_),
                jnp.zeros(capacity, jnp.float32),
            ),
            stored,
            jnp.zeros((capacity, width), jnp.bool_),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(0, jnp.int32),
        )
        return OnPolicyState(
            networks,
            optimizer,
            replay,
            observations.astype(jnp.int32),
            env_state,
            explained_rows,
            jnp.broadcast_to(coalition, (training.num_envs, width)),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(False),
            keys[3],
        )

    return nnx.vmap(partial(seed, env, network, training), in_axes=(0, 0), out_axes=0)(keys, pools)


def on_policy_loss(
    networks: OutcomeNetworks, batch: TransitionBatch, explained: Array, coalitions: Array, gamma: float
) -> Array:
    """TD(0) towards r + gamma V_target(s' | e, C) (1 - terminated); actions are not used."""
    following = networks.target(batch.next_observations, explained, coalitions).values[:, 0]
    alive = (~batch.terminated).astype(jnp.float32)
    targets = jax.lax.stop_gradient(batch.rewards + gamma * following * alive)
    predictions = networks.model(batch.observations, explained, coalitions).values[:, 0]
    return jnp.mean(jnp.square(predictions - targets))


def exact_values_loss(
    config: OutcomeConfig, explained: CharacteristicConfig, outcome: OutcomeExplained, exact: ExactOutcome
) -> Array:
    """MSE between the learnt and exact outcome values over every (coalition, valid unique state) pair."""
    states = exact.states

    def per_coalition(
        config: OutcomeConfig,
        explained: CharacteristicConfig,
        outcome: OutcomeExplained,
        states: Array,
        coalition: Array,
    ) -> Array:
        return values(config, explained, outcome, states, jnp.broadcast_to(coalition, states.shape))[:, 0]

    coalitions = jnp.asarray(coalition_table(states.shape[-1]))
    approximations = jax.lax.map(partial(per_coalition, config, explained, outcome, states), coalitions)
    valid = exact.state_valid.astype(jnp.float32)[None, :]
    errors = jnp.square(approximations - exact.values) * valid
    return jnp.sum(errors) / (jnp.sum(valid) * coalitions.shape[0])


def loss(
    config: OutcomeConfig,
    explained: CharacteristicConfig,
    networks: OutcomeNetworks,
    frozen: OutcomeFrozen,
    context: OutcomeContext,
    training: OutcomeTrainConfig,
    features: int,
    key: Array,
    partition: Partition,
) -> Array:
    """A sampled train or validation batch's Q-learning loss, or the exact loss.

    Batches (the source's get_data) draw transition rows and, independently, the explained
    states e from the same replay prefix, uniform coalitions per row, and a key the
    conditioned policy samples next actions with: Q-learning towards
    r + gamma Q_target(s', a' ~ pi_hat(s'; e, C) | e, C) (1 - terminated). The exact loss is
    the MSE over every (coalition, valid unique state) pair, equally weighted.
    """
    if partition == "exact":
        exact = context.exact
        assert exact is not None
        return exact_values_loss(config, explained, OutcomeExplained(frozen.agent, frozen.behaviour, networks), exact)
    rows_from, size = (
        (context.train, training.batch_size) if partition == "train" else (context.validation, training.validation_size)
    )
    buffer = training.train_size if partition == "train" else training.validation_size
    keys: Array = jax.random.split(key, 4)
    rows = jax.random.randint(keys[0], (size,), 0, buffer)
    explained_rows = rows_from.observations[jax.random.randint(keys[1], (size,), 0, buffer)]
    coalitions = jax.random.bernoulli(keys[2], 0.5, (size, features))
    observations, next_observations = rows_from.observations[rows], rows_from.next_observations[rows]
    behaviour = frozen.behaviour(mask(explained_rows, coalitions)).values
    probabilities = conditioned_policy(explained, frozen.agent, behaviour, next_observations, explained_rows)
    next_actions = jax.random.categorical(keys[3], jnp.log(probabilities), axis=-1).astype(jnp.int32)
    target_values = networks.target(next_observations, explained_rows, coalitions).values
    following = jnp.take_along_axis(target_values, next_actions[:, None], axis=-1)[:, 0]
    alive = (~rows_from.terminated[rows]).astype(jnp.float32)
    td_targets = jax.lax.stop_gradient(rows_from.rewards[rows] + config.gamma * following * alive)
    predictions = networks.model(observations, explained_rows, coalitions).values
    chosen = jnp.take_along_axis(predictions, rows_from.actions[rows][:, None], axis=-1)[:, 0]
    return jnp.mean(jnp.square(chosen - td_targets))


def update(
    config: OutcomeConfig,
    explained: CharacteristicConfig,
    networks: OutcomeNetworks,
    optimizer: nnx.Optimizer[OutcomeNetworks],
    frozen: OutcomeFrozen,
    context: OutcomeContext,
    training: OutcomeTrainConfig,
    features: int,
    key: Array,
) -> Array:
    """Source PerformanceCharacteristicOffPolicy.update: one Adam step, then the soft target update
    target <- tau * model + (1 - tau) * target."""
    # NNX transformation has a dynamic return signature; assert its loss/gradient boundary.
    value, gradients = cast(
        tuple[Array, nnx.State[str | int, nnx.Param[Array]]],
        nnx.value_and_grad(partial(loss, config, explained))(
            networks, frozen, context, training, features, key, "train"
        ),
    )
    optimizer.update(networks, gradients)
    soft_update(networks.model.mlp, networks.target.mlp, config.tau)
    return value


def conditioned_policy(
    explained: CharacteristicConfig,
    agent: Agent,
    behaviour: Values,
    observations: Observations,
    explained_observations: Observations,
) -> Values:
    """pi_hat(a | s; e, C): the behaviour characteristic's values at e given C (`behaviour`, a learnt
    model's or a sampled one's) when s == e, else the agent's explained (greedy or explore) policy."""
    at_explained = jnp.all(observations == explained_observations, axis=-1, keepdims=True)
    return jnp.where(at_explained, behaviour, targets(explained, agent, observations))


def values(
    config: OutcomeConfig,
    explained: CharacteristicConfig,
    outcome: OutcomeExplained,
    observations: Observations,
    coalitions: Coalitions,
) -> Values:
    """The learnt outcome characteristic v(e, C) (source get_val): sum_a pi_hat(a | e; e, C) Q(e, a | e, C)
    off-policy, or V(e | e, C) on-policy; coalitions broadcast against the states e."""
    shape = jnp.broadcast_shapes(observations.shape, coalitions.shape)
    observations, coalitions = jnp.broadcast_to(observations, shape), jnp.broadcast_to(coalitions, shape)
    predicted = outcome.networks.model(observations, observations, coalitions).values
    if config.variant == "on_policy":
        return predicted
    behaviour = outcome.behaviour(mask(observations, coalitions)).values
    probabilities = conditioned_policy(explained, outcome.agent, behaviour, observations, observations)
    return jnp.sum(probabilities * predicted, axis=-1, keepdims=True)


def null(config: OutcomeConfig, outcome: OutcomeExplained, observations: Observations) -> Values:
    """v(s, C = 0). Off-policy, the behaviour characteristic's empty coalition is one all-MASK row
    broadcast over the rows (evaluating a broadcast-constant batch miscompiles on XLA CPU)."""
    empty = jnp.zeros(observations.shape, jnp.bool_)
    predicted = outcome.networks.model(observations, observations, empty).values
    if config.variant == "on_policy":
        return predicted
    behaviour = outcome.behaviour(jnp.full((1, observations.shape[-1]), MASK, jnp.int32)).values[0]
    return jnp.sum(behaviour * predicted, axis=-1, keepdims=True)


def exact_outcome(
    tabular: TabularModel,
    observations: Array,
    frozen: OutcomeFrozen,
    explained: CharacteristicConfig,
    with_exact_char: bool,
    gamma: float,
) -> ExactOutcome:
    """Source PerformanceCharacteristic.get_exact over the exact buffer's unique states.

    The behaviour characteristic at the tabular states, for every coalition, is the exact one
    (with_exact_char: the mean explained policy over buffer rows agreeing with the state on C,
    the source's conditional expectation under the buffer's empirical state distribution) or
    the learnt one. Then the tabular MDP is solved under pi_hat(.; e, C) for every (C, e), and
    Q at e averaged under pi_hat. States that are not in the buffer, or not modelled, are invalid.
    """
    states = tabular.states
    coalitions = jnp.asarray(coalition_table(states.shape[-1]))
    agent_targets = targets(explained, frozen.agent, observations)
    present = jnp.any(jnp.all(states[:, None, :] == observations[None, :, :], axis=-1), axis=-1)

    def exact_behaviour(states: Array, observations: Array, agent_targets: Array, coalition: Array) -> Values:
        known = mask(states, coalition)[:, None, :] == mask(observations, coalition)[None, :, :]
        matches = jnp.all(known, axis=-1).astype(jnp.float32)
        return (matches @ agent_targets) / jnp.maximum(jnp.sum(matches, axis=-1, keepdims=True), 1.0)

    def learnt_behaviour(behaviour: CharacteristicModel, states: Array, coalition: Array) -> Values:
        return behaviour(mask(states, jnp.broadcast_to(coalition, states.shape))).values

    conditioned: ConditionedTable = (
        jax.lax.map(partial(exact_behaviour, states, observations, agent_targets), coalitions)
        if with_exact_char
        else jax.lax.map(partial(learnt_behaviour, frozen.behaviour, states), coalitions)
    )
    agent = targets(explained, frozen.agent, states)
    rows, modelled = state_indices(tabular, states)

    def per_state(tabular: TabularModel, agent: ActionTable, gamma: float, row: Array, chosen: Array) -> Array:
        policy = agent.at[row].set(chosen)
        return jnp.sum(chosen * policy_q_values(tabular, policy, gamma)[row])

    def per_coalition(tabular: TabularModel, agent: ActionTable, gamma: float, rows: Array, chosen: Array) -> Array:
        return jax.vmap(partial(per_state, tabular, agent, gamma))(rows, chosen)

    exact_values = jax.lax.map(partial(per_coalition, tabular, agent, gamma, rows), conditioned)
    valid = present & modelled
    # Stop the run here, before any training, rather than learn against non-finite targets.
    jax.debug.callback(require_finite, exact_values, valid, ordered=True)
    return ExactOutcome(states, valid, exact_values)


def require_finite(values: np.ndarray, valid: np.ndarray) -> None:
    """Raise when any exact outcome value at a valid state is non-finite (a host callback).

    With gamma = 1 the values diverge when the conditioned policy can loop forever without
    reaching a terminal state; upstream's value iteration then never terminates.
    """
    finite = np.isfinite(values) | ~valid[..., None, :]
    if not finite.all():
        raise ValueError(
            f"{int((~finite).sum())} exact outcome values are non-finite: with gamma = 1 the conditioned "
            "policy can loop forever without reaching a terminal state (upstream's value iteration hangs here). "
            "Train the agent longer or explain a policy that always terminates."
        )
