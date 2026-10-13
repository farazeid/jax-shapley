"""Step 3: behaviour characteristic models, v(s, C) learnt from masked states; the shared stage.

A characteristic model maps a state whose unknown features are MASKed to the explained
output `v_F(s)`: here the agent's action probabilities. `stage`, `collect`, `fit`,
`initialize`, `loss` and `exact_characteristic` are shared with the other kinds, which
supply their own `targets` (the explained output) and `head` (the network's last layer).
Least squares over random coalitions makes it learn the mean over states sharing the
known features, which is the characteristic value. Exact values over the exact buffer
measure its error. The reference is FastSVERL's `ValuePolicyCharacteristicModel`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from functools import partial
from math import inf, isnan
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
from jax_shapley.algorithms import dqn
from jax_shapley.algorithms.dqn import TRACKING_FIELDS, Agent, Env, EnvParams, EnvState, TransitionBatch
from jax_shapley.backends import Track
from jax_shapley.common import checkpoints
from jax_shapley.common.coalitions import Coalitions, CoalitionTable, Observations, coalition_table, mask
from jax_shapley.common.networks import MLP, LayerConfig, MLPConfig
from jax_shapley.common.rollouts import collect_buffers, collect_observations
from jax_shapley.common.supervised_learning import (
    EpochMetrics,
    Partition,
    ReplayBehaviour,
    StateBuffers,
    SupervisedConfig,
    TrainState,
    log_epochs,
    seed_slice,
    stack_seeds,
    train,
)
from jax_shapley.envs.wrappers import EpisodeStatisticsWrapper

type Kind = Literal["behaviour", "prediction"]
type Data = Literal["rollouts", "replay"]
type Estimator = Literal["model", "sampled"]
type Targets = Callable[[CharacteristicConfig, Agent, Array], Array]
"""A kind's explained output `v_F` of the agent at observations."""
type Head = Callable[[int], LayerConfig]
"""A kind's network output layer, given the number of actions."""
type Policy = Literal["greedy", "explore"]
type Values = Float[Array, "... outputs"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type Output = Float[Array, "outputs"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.
type RowObservations = Int[Array, "rows features"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type RowValues = Float[Array, "rows outputs"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type UniqueValues = Float[Array, "coalitions unique outputs"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type States = Int[Array, "unique features"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type StateFlags = Bool[Array, "unique"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.

# Rows per exact-loss evaluation chunk; padding beyond the valid entries is never evaluated.
EXACT_CHUNK = 512
# Rows a sampled characteristic matches against its pool at once; bounds the (rows, pool, features) comparison.
SAMPLE_CHUNK = 256


@dataclass(frozen=True, slots=True)
class CharacteristicConfig:
    """Which agent output is explained, the network, its data, and the source's train_models settings.

    Every seed trains in one compiled program. `data="replay"` learns from the agent's
    training transitions (upstream `offpolicy/run_policy_prediction.py`): the first
    `train_size` rows of a shuffled replay train and the next `validation_size` validate;
    the exact buffer is fresh rollouts of the explained policy, so `train.exact_buffer_size`
    is required. `off_policy` weights each row by the explained-to-behaviour policy ratio,
    normalised over the batch when `weighting` and clipped to `[1 - clip_coef, 1 + clip_coef]`.
    """

    env: Env
    kind: Kind = "behaviour"
    policy: Policy = "greedy"
    hidden_widths: tuple[int, ...] = (64, 64)
    data: Data = "rollouts"
    estimator: Estimator = "model"
    """`sampled` trains nothing: values are single draws from the train buffer (upstream
    `ValuePolicyCharacteristicSample`); Shapley models then reuse its buffers."""
    off_policy: bool = False
    weighting: bool = True
    clip_coef: float = inf
    seeds: tuple[int, ...] = (1,)
    train: SupervisedConfig = field(default_factory=SupervisedConfig)
    checkpoint: str | None = None
    """A characteristic artifact to load instead of training, e.g. `GROUP--characteristic--behaviour:latest`."""
    name: str = "characteristic"
    """The experiment name runs are named after: `NAME--characteristic--KIND--seed-N`."""
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
        if self.kind not in ("behaviour", "prediction"):
            raise ValueError("characteristic kind must be behaviour or prediction")
        if self.policy not in ("greedy", "explore"):
            raise ValueError("explained policy must be greedy or explore")
        if self.data not in ("rollouts", "replay") or (self.off_policy and self.data != "replay"):
            raise ValueError("data must be rollouts or replay; off-policy weighting needs replay data")
        if self.estimator not in ("model", "sampled") or (
            self.estimator == "sampled"
            and (self.data != "rollouts" or self.train.exact_loss or self.train.exact_buffer_size is not None)
        ):
            raise ValueError("sampled characteristics draw from rollouts, with exact values on their pool and no loss")
        if self.data == "replay" and self.train.exact_loss and self.train.exact_buffer_size is None:
            raise ValueError("replay characteristics need a separate exact buffer (train.exact_buffer_size)")
        if isnan(self.clip_coef) or self.clip_coef < 0:
            raise ValueError("clip_coef must be nonnegative")
        if any(width < 1 for width in self.hidden_widths):
            raise ValueError("hidden widths must be positive")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct and nonempty")


@struct.dataclass
class SampledPool:
    """The rows a sampled characteristic draws from, with v_F of each, and v(empty): v_F's pool mean."""

    observations: RowObservations
    values: RowValues
    null: Output


@struct.dataclass
class ExactCharacteristic:
    """Exact characteristic values under a buffer's empirical state distribution.

    `states` are the buffer's unique observations (sorted, padded to a static size;
    `state_valid` marks real ones) and `at_states[c, u]` is coalition c's exact value
    at state u. `compact_keys`/`compact_values` list every coalition's distinct masked
    observation and its value once (`count` of them, valid entries first, padded to
    whole chunks), which the per-epoch exact loss evaluates. `null` is the empty
    coalition's value.
    """

    states: States
    state_valid: StateFlags
    at_states: UniqueValues
    compact_keys: RowObservations
    compact_values: RowValues
    count: Array
    null: Output


@struct.dataclass
class CharacteristicContext:
    """Per-seed constants a characteristic learner reads: buffers, targets and exact values, and
    for replay rows the explained policy's log-probability of each replay action."""

    buffers: StateBuffers
    train_targets: Array
    validation_targets: Array
    exact: ExactCharacteristic | None
    train_log_probabilities: Array | None = None
    validation_log_probabilities: Array | None = None


class CharacteristicOutput(NamedTuple):
    values: Values


class CharacteristicModel(nnx.Module):
    """Characteristic network over masked int32 observations."""

    def __init__(self, config: MLPConfig, *, rngs: nnx.Rngs) -> None:
        self.mlp = MLP(config, rngs=rngs)

    def __call__(self, masked_observations: Observations) -> CharacteristicOutput:
        return CharacteristicOutput(self.mlp(masked_observations))


def main(
    config: CharacteristicConfig, agents: Agent, replays: TransitionBatch | None = None
) -> tuple[TrainState[CharacteristicModel], EpochMetrics | None]:
    """Behaviour characteristic models explaining the agents' greedy or explore policy."""
    if config.kind != "behaviour":
        raise ValueError("the behaviour stage needs kind='behaviour'")
    return stage(config, agents, replays, targets, head, "PolicyCharacteristicModel")


def stage(
    config: CharacteristicConfig,
    agents: Agent,
    replays: TransitionBatch | None,
    targets: Targets,
    head: Head,
    model: str,
) -> tuple[TrainState[CharacteristicModel], EpochMetrics | None]:
    """Train (or load) every seed of a kind, tracked by `config.track`; `model` is the source's
    model class name that prefixes the logged tags.

    `agents` are seed-stacked: one agent every seed explains, or one per seed (seed s
    explains agent s). Replay data needs `replays`, each agent's filled replay rows.

    Collects each seed's buffers, sizes the exact tables from them, then trains every
    seed in one compiled program. With a backend each seed logs the source's
    `model/{Policy,Value}CharacteristicModel_*` scalars into its own run, and the whole
    seed-stacked final state is pushed as one Orbax checkpoint artifact. The returned
    state and metrics carry a leading seed axis (metrics: (epochs, seeds); None when loaded).
    """
    keys = jnp.stack([jax.random.key(seed) for seed in config.seeds])
    if agents.epsilon.shape[0] == 1:
        agents = stack_seeds([seed_slice(agents, 0)] * len(config.seeds))
    elif agents.epsilon.shape[0] != len(config.seeds):
        raise ValueError("characteristic seeds need one agent, or one agent per seed")
    if config.data == "replay":
        if replays is None or replays.observations.shape[-2] < config.train.train_size + config.train.validation_size:
            raise ValueError("replay characteristics need the agents' replays, holding train and validation rows")
        if replays.observations.shape[0] == 1:
            replays = stack_seeds([seed_slice(replays, 0)] * len(config.seeds))
    else:
        replays = None
    if config.track is None:
        keys, buffers = nnx.jit(partial(collect, config))(keys, agents, replays)
        # Exact tables are sized by the most unique observations in any seed's exact buffer (a static shape).
        max_unique = max(int(np.unique(rows, axis=0).shape[0]) for rows in np.asarray(buffers.exact))
        return nnx.jit(partial(fit, config, targets, head, max_unique, None))(keys, agents, buffers)
    group = config.group or f"{config.name}-{time_ns()}"
    artifact = f"{group}--characteristic--{config.kind}"
    settings = {
        "stage": "characteristic",
        "env": config.env.name,
        "config": {name: value for name, value in asdict(config).items() if name not in TRACKING_FIELDS},
    }
    with backends.session(config.track, config.entity) as root:
        tracking = wandb if config.track == "wandb" else trackio
        runs = tuple(
            tracking.init(
                project=config.project,
                name=f"{config.name}--characteristic--{config.kind}--seed-{seed}",
                group=group,
                config={**asdict(config), "seed": seed},
            )
            for seed in config.seeds
        )
        try:
            if config.checkpoint is not None:
                directory = runs[0].use_artifact(config.checkpoint).download(root=str(root / "artifacts" / artifact))
                return checkpoints.load(Path(directory), initialize(config, head), settings), None

            keys, buffers = nnx.jit(partial(collect, config))(keys, agents, replays)
            # Exact tables are sized by the most unique observations in any seed's exact buffer (a static shape).
            max_unique = max(int(np.unique(rows, axis=0).shape[0]) for rows in np.asarray(buffers.exact))
            epoch_log = partial(log_epochs, runs, model, config.train.exact_loss)
            program = nnx.jit(partial(fit, config, targets, head, max_unique, epoch_log))
            state, metrics = program(keys, agents, buffers)
            directory = root / "checkpoints" / artifact
            checkpoints.save(directory, state, settings)
            runs[0].log_artifact(str(directory), name=artifact, type="characteristic").wait()
            print(f"Characteristic checkpoint: {artifact}:latest")
            return state, metrics
        finally:
            for run in runs:
                run.finish()


def collect(
    config: CharacteristicConfig, keys: Array, agents: Agent, replays: TransitionBatch | None
) -> tuple[Array, StateBuffers]:
    """Each seed's buffers, and its key advanced past them: rollouts of the explained policy, or
    the agent's shuffled replay rows with a rollout exact buffer."""

    def seed(
        config: CharacteristicConfig,
        env: EpisodeStatisticsWrapper[EnvState, EnvParams],
        key: Array,
        agent: Agent,
        replay: TransitionBatch | None,
    ) -> tuple[Array, StateBuffers]:
        epsilon = agent.epsilon if config.policy == "explore" else jnp.asarray(0.0, jnp.float32)
        training = config.train
        if replay is None:
            return collect_buffers(
                env,
                agent.model,
                epsilon,
                key,
                train_size=training.train_size,
                validation_size=training.validation_size,
                exact_size=training.exact_buffer_size,
                num_envs=training.num_envs,
            )
        keys: Array = jax.random.split(key, 3)
        order = jax.random.permutation(keys[1], replay.observations.shape[0])
        train_rows = order[: training.train_size]
        validation_rows = order[training.train_size : training.train_size + training.validation_size]
        exact = collect_observations(env, agent.model, epsilon, keys[2], training.exact_size, training.num_envs)
        behaviour = ReplayBehaviour(
            replay.actions[train_rows],
            replay.log_probabilities[train_rows],
            replay.actions[validation_rows],
            replay.log_probabilities[validation_rows],
        )
        observations = replay.observations
        return keys[0], StateBuffers(observations[train_rows], observations[validation_rows], exact, behaviour)

    every_seed = nnx.vmap(
        partial(seed, config, EpisodeStatisticsWrapper(config.env)),
        in_axes=(0, 0, None if replays is None else 0),
        out_axes=(0, 0),
    )
    return every_seed(keys, agents, replays)


def fit(
    config: CharacteristicConfig,
    targets: Targets,
    head: Head,
    max_unique: int,
    log: Callable[[EpochMetrics], None] | None,
    keys: Array,
    agents: Agent,
    buffers: StateBuffers,
) -> tuple[TrainState[CharacteristicModel], EpochMetrics | None]:
    """Train every seed from its buffers (a sampled characteristic trains nothing); exact tables
    hold `max_unique` states.

    Each seed's context holds the frozen agent's explained outputs on every buffer and,
    with the exact loss, exact characteristic values on the exact buffer.
    """
    state = initialize(config, head, keys, buffers)
    if config.estimator == "sampled":
        # The source's sampled characteristic trains nothing; its pool is the train buffer.
        return state, None
    features = state.buffers.train.shape[-1]

    def context(
        config: CharacteristicConfig, targets: Targets, max_unique: int, buffers: StateBuffers, agent: Agent
    ) -> CharacteristicContext:
        train_targets = targets(config, agent, buffers.train)
        validation_targets = targets(config, agent, buffers.validation)
        exact = None
        if config.train.exact_loss:
            # Every coalition: 2^features rows, so only built when exact values are wanted.
            coalitions = jnp.asarray(coalition_table(buffers.exact.shape[-1]))
            exact = exact_characteristic(
                buffers.exact, targets(config, agent, buffers.exact), coalitions, max_unique=max_unique
            )
        behaviour = buffers.behaviour
        if behaviour is None:
            return CharacteristicContext(buffers, train_targets, validation_targets, exact)
        # The explained policy's log-probability of each replay action (upstream choose_action):
        # Torch Categorical renormalises and clamps before taking logs.
        log_probabilities = []
        for observations, actions in (
            (buffers.train, behaviour.train_actions),
            (buffers.validation, behaviour.validation_actions),
        ):
            q_values = agent.model(observations).q_values
            ties = q_values == jnp.max(q_values, axis=-1, keepdims=True)
            probabilities = ties.astype(jnp.float32) / jnp.sum(ties, axis=-1, keepdims=True)
            if config.policy == "explore":
                probabilities = agent.epsilon / q_values.shape[-1] + (1 - agent.epsilon) * probabilities
            log_probabilities.append(dqn.log_probabilities(probabilities, actions))
        return CharacteristicContext(
            buffers, train_targets, validation_targets, exact, log_probabilities[0], log_probabilities[1]
        )

    contexts = nnx.vmap(partial(context, config, targets, max_unique), in_axes=(0, 0), out_axes=0)(
        state.buffers, agents
    )
    return train(state, contexts, None, partial(loss, config), config.train, features, log)


def initialize(
    config: CharacteristicConfig, head: Head, keys: Array | None = None, buffers: StateBuffers | None = None
) -> TrainState[CharacteristicModel]:
    """Every seed's fresh training state, seed-stacked.

    Without keys and buffers it is the template a checkpoint is restored into: the
    configured seeds and zero buffers of the configured sizes.
    """
    wrapped = EpisodeStatisticsWrapper(config.env)
    shape = wrapped.observation_space(wrapped.default_params).shape
    if len(shape) != 1:
        raise ValueError("explanations require flat observations")
    features = int(shape[0])
    # Experiment-script layout: ReLU hidden layers, then the kind's head.
    hidden = tuple(LayerConfig(width, "relu") for width in config.hidden_widths)
    network = MLPConfig(features, (*hidden, head(wrapped.num_actions)))
    training = config.train
    if keys is None:
        keys = jnp.stack([jax.random.key(seed) for seed in config.seeds])
    if buffers is None:
        behaviour = (
            ReplayBehaviour(
                jnp.zeros(training.train_size, jnp.int32),
                jnp.zeros(training.train_size, jnp.float32),
                jnp.zeros(training.validation_size, jnp.int32),
                jnp.zeros(training.validation_size, jnp.float32),
            )
            if config.data == "replay"
            else None
        )
        empty = StateBuffers(
            jnp.zeros((training.train_size, features), jnp.int32),
            jnp.zeros((training.validation_size, features), jnp.int32),
            jnp.zeros((training.exact_size, features), jnp.int32),
            behaviour,
        )
        buffers = stack_seeds([empty] * len(config.seeds))

    def seed(
        network: MLPConfig, training: SupervisedConfig, key: Array, buffers: StateBuffers
    ) -> TrainState[CharacteristicModel]:
        model_key, key = jax.random.split(key)
        model = CharacteristicModel(network, rngs=nnx.Rngs(model_key))
        adam = training.adam
        # Torch Adam: constant rate, epsilon outside the square root, no root epsilon.
        optimizer = nnx.Optimizer(
            model,
            optax.adam(training.learning_rate, b1=adam.beta1, b2=adam.beta2, eps=adam.epsilon, eps_root=0.0),
            wrt=nnx.Param,
        )
        return TrainState(
            model,
            nnx.clone(model),
            optimizer,
            buffers,
            jnp.asarray(0, jnp.int32),
            jnp.asarray(jnp.inf, jnp.float32),
            jnp.asarray(0, jnp.int32),
            jnp.asarray(False),
            key,
        )

    return nnx.vmap(partial(seed, network, training), in_axes=(0, 0), out_axes=0)(keys, buffers)


def targets(config: CharacteristicConfig, agent: Agent, observations: Array) -> Values:
    """`v_F`: the agent's greedy policy (ties shared equally), or its epsilon-greedy explore policy."""
    q_values = agent.model(observations).q_values
    ties = q_values == jnp.max(q_values, axis=-1, keepdims=True)
    greedy = ties.astype(jnp.float32) / jnp.sum(ties, axis=-1, keepdims=True)
    if config.policy == "greedy":
        return greedy
    return agent.epsilon / q_values.shape[-1] + (1 - agent.epsilon) * greedy


def head(actions: int) -> LayerConfig:
    """Action probabilities: the experiment scripts end behaviour models in Softmax(-1)."""
    return LayerConfig(actions, "softmax")


def loss(
    characteristic: CharacteristicConfig,
    model: CharacteristicModel,
    frozen: None,
    context: CharacteristicContext,
    config: SupervisedConfig,
    features: int,
    key: Array,
    partition: Partition,
) -> Array:
    """MSE of a sampled train or validation batch, or the exact loss.

    Batches draw rows with replacement and uniform coalitions per row (the source's
    sample_rand); validation resamples the whole validation size. Off-policy replay rows
    weigh each squared error by the explained-to-behaviour policy ratio (the source's
    ImportanceWeightedMSELoss), normalised over the draw and clipped.
    """
    del frozen
    if partition == "exact":
        assert context.exact is not None
        return exact_loss(model, context.exact)
    index_key, coalition_key = jax.random.split(key)
    behaviour = context.buffers.behaviour
    if partition == "train":
        rows = jax.random.randint(index_key, (config.batch_size,), 0, config.train_size)
        observations, targets = context.buffers.train[rows], context.train_targets[rows]
        explained, logged = (
            context.train_log_probabilities,
            None if behaviour is None else behaviour.train_log_probabilities,
        )
    else:
        rows = jax.random.randint(index_key, (config.validation_size,), 0, config.validation_size)
        observations, targets = context.buffers.validation[rows], context.validation_targets[rows]
        explained = context.validation_log_probabilities
        logged = None if behaviour is None else behaviour.validation_log_probabilities
    coalitions = jax.random.bernoulli(coalition_key, 0.5, (rows.shape[0], features))
    predictions = model(mask(observations, coalitions)).values
    errors = jnp.square(predictions - jax.lax.stop_gradient(targets))
    if not characteristic.off_policy:
        return jnp.mean(errors)
    assert explained is not None and logged is not None
    weights = jnp.exp(explained[rows] - logged[rows])
    if characteristic.weighting:
        weights = weights / jnp.sum(weights)
    weights = jnp.clip(weights, 1 - characteristic.clip_coef, 1 + characteristic.clip_coef)
    return jnp.mean(jax.lax.stop_gradient(weights)[:, None] * errors)


def exact_loss(model: CharacteristicModel, exact: ExactCharacteristic) -> Array:
    """MSE over every valid unique masked observation of every coalition, equally weighted.

    Only the chunks holding valid entries are evaluated: their number varies at run time,
    while the padded arrays keep static shapes.
    """
    outputs = exact.compact_values.shape[-1]
    chunks = (exact.count + EXACT_CHUNK - 1) // EXACT_CHUNK

    def body(exact: ExactCharacteristic, model: CharacteristicModel, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        index, total = carry
        start = index * EXACT_CHUNK
        keys = jax.lax.dynamic_slice_in_dim(exact.compact_keys, start, EXACT_CHUNK)
        values = jax.lax.dynamic_slice_in_dim(exact.compact_values, start, EXACT_CHUNK)
        valid = (start + jnp.arange(EXACT_CHUNK)) < exact.count
        errors = jnp.square(model(keys).values - values)
        return index + 1, total + jnp.sum(jnp.where(valid[:, None], errors, 0.0))

    def running(chunks: Array, carry: tuple[Array, Array]) -> Array:
        return carry[0] < chunks

    _, total = jax.lax.while_loop(
        partial(running, chunks),
        partial(body, exact, model),
        (jnp.asarray(0, jnp.int32), jnp.asarray(0.0, jnp.float32)),
    )
    return total / (exact.count.astype(jnp.float32) * outputs)


def exact_characteristic(
    observations: RowObservations, targets: RowValues, coalitions: CoalitionTable, *, max_unique: int
) -> ExactCharacteristic:
    """Conditional expectation of `v_F` given each coalition's known features.

    Each coalition's value is the mean target over buffer rows that agree on the known
    features: the source's Bayes rule over the empirical distribution. Rows are first
    deduplicated into unique states with summed targets, so every coalition processes
    `max_unique` states rather than every row. `max_unique` must bound the number of
    unique observations; padded entries are marked invalid.
    """
    if observations.ndim != 2 or targets.ndim != 2 or targets.shape[0] != observations.shape[0]:
        raise ValueError("exact characteristics need (rows, features) observations and (rows, outputs) targets")
    if max_unique < 1:
        raise ValueError("max_unique must be positive")
    # jnp.unique's multi-output overloads are untyped; the requested outputs are arrays.
    states, state_inverse, state_counts = cast(
        tuple[Array, Array, Array],
        jnp.unique(observations, axis=0, size=max_unique, fill_value=0, return_inverse=True, return_counts=True),
    )
    weights = state_counts.astype(jnp.float32)  # Padded states weigh nothing.
    # Each state's summed targets over its rows: the exact value is a row mean.
    weighted_targets = jax.ops.segment_sum(targets, state_inverse.reshape(-1), num_segments=max_unique)

    def evaluate(
        states: Array, weighted_targets: Array, weights: Array, max_unique: int, coalition: Array
    ) -> tuple[Array, Array, Array, Array]:
        # jnp.unique's multi-output overloads are untyped; the requested outputs are arrays.
        keys, inverse = cast(
            tuple[Array, Array],
            jnp.unique(mask(states, coalition), axis=0, size=max_unique, fill_value=0, return_inverse=True),
        )
        segments = inverse.reshape(-1)
        totals = jax.ops.segment_sum(weighted_targets, segments, num_segments=max_unique)
        mass = jax.ops.segment_sum(weights, segments, num_segments=max_unique)
        values = totals / jnp.maximum(mass, 1.0)[:, None]
        return keys, values, mass > 0, values[segments]

    keys, values, valid, at_states = jax.lax.map(
        partial(evaluate, states, weighted_targets, weights, max_unique), coalitions
    )
    flat_valid = valid.reshape(-1)
    entries = flat_valid.shape[0]
    padded = -(-entries // EXACT_CHUNK) * EXACT_CHUNK
    # Stable order: valid entries first, in coalition-major order, then padding.
    order = jnp.argsort(~flat_valid, stable=True)
    compact_keys = jnp.zeros((padded, keys.shape[-1]), jnp.int32).at[:entries].set(keys.reshape(entries, -1)[order])
    compact_values = (
        jnp.zeros((padded, values.shape[-1]), jnp.float32).at[:entries].set(values.reshape(entries, -1)[order])
    )
    count = jnp.sum(flat_valid, dtype=jnp.int32)
    # Coalition 0 masks every feature, so every state shares its single key.
    return ExactCharacteristic(states, state_counts > 0, at_states, compact_keys, compact_values, count, values[0, 0])


def sampled_values(pool: SampledPool, key: Array, observations: Observations, coalitions: Coalitions) -> Values:
    """A sampled characteristic (upstream ValuePolicyCharacteristicSample, one draw): for every row,
    v_F of a pool row agreeing with it on the coalition, uniform among the matching rows.

    Like the source, which reads `Cs[0]`, every row uses the first coalition. The source scores
    each (row, pool row) pair with `torch.rand` where they match and MASK where they do not, and
    takes the top score (a uniform pool row when nothing matches, as `topk` ties then pick from
    the freshly permuted pool). Here one uniform score per pair is raised by 1 for matches.
    """
    features = observations.shape[-1]
    rows = observations.reshape(-1, features)
    coalition = coalitions.reshape(-1, features)[0]
    count = rows.shape[0]
    size = min(SAMPLE_CHUNK, count)
    chunks = -(-count // size)
    padded = jnp.pad(rows, ((0, chunks * size - count), (0, 0)))

    def chunk(coalition: Array, pool: Array, inputs: tuple[Array, Array]) -> Array:
        chunk_key, rows = inputs
        scores = jax.random.uniform(chunk_key, (rows.shape[0], pool.shape[0]), jnp.float32)
        matches = jnp.all((rows[:, None, :] == pool[None, :, :]) | ~coalition, axis=-1)
        return jnp.argmax(jnp.where(matches, scores + 1.0, scores), axis=-1).astype(jnp.int32)

    picked: Array = jax.lax.map(
        partial(chunk, coalition, pool.observations), (jax.random.split(key, chunks), padded.reshape(chunks, size, -1))
    )
    indices = picked.reshape(-1)[:count]
    return pool.values[indices].reshape(*observations.shape[:-1], pool.values.shape[-1])
