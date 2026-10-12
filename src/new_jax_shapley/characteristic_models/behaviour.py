"""Step 3: characteristic models, v(s, C) learnt from masked states (behaviour; shared with prediction).

A characteristic model maps a state whose unknown features are MASKed to the explained
output `v_F(s)`: the agent's action probabilities (behaviour) or value (prediction).
Least squares over random coalitions makes it learn the mean over states sharing the
known features, which is the characteristic value. Exact values over the exact buffer
measure its error. The reference is FastSVERL's `ValuePolicyCharacteristicModel`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from time import time_ns
from typing import Literal, NamedTuple, cast

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
from new_jax_shapley.algorithms.dqn import TRACKING_FIELDS, Agent, Env, EnvParams, EnvState
from new_jax_shapley.backends import Track
from new_jax_shapley.common import checkpoints
from new_jax_shapley.common.coalitions import CoalitionTable, Observations, coalition_table, mask
from new_jax_shapley.common.networks import MLP, LayerConfig, MLPConfig
from new_jax_shapley.common.rollouts import collect_buffers
from new_jax_shapley.common.supervised_learning import (
    EpochMetrics,
    Partition,
    StateBuffers,
    SupervisedConfig,
    TrainState,
    stack_seeds,
    train,
)
from new_jax_shapley.envs.wrappers import EpisodeStatisticsWrapper

type Kind = Literal["behaviour", "prediction"]
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


@dataclass(frozen=True, slots=True)
class CharacteristicConfig:
    """Which agent output is explained, the network, and the source's train_models settings.

    Every seed explains the same agent and trains in one compiled program.
    """

    env: Env
    kind: Kind = "behaviour"
    policy: Policy = "greedy"
    hidden_widths: tuple[int, ...] = (64, 64)
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
        if self.kind not in ("behaviour", "prediction"):
            raise ValueError("characteristic kind must be behaviour or prediction")
        if self.policy not in ("greedy", "explore"):
            raise ValueError("explained policy must be greedy or explore")
        if any(width < 1 for width in self.hidden_widths):
            raise ValueError("hidden widths must be positive")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct and nonempty")


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
    """Per-seed constants a characteristic learner reads: buffers, targets and exact values."""

    buffers: StateBuffers
    train_targets: Array
    validation_targets: Array
    exact: ExactCharacteristic | None


class CharacteristicOutput(NamedTuple):
    values: Values


class CharacteristicModel(nnx.Module):
    """Characteristic network over masked int32 observations."""

    def __init__(self, config: MLPConfig, *, rngs: nnx.Rngs) -> None:
        self.mlp = MLP(config, rngs=rngs)

    def __call__(self, masked_observations: Observations) -> CharacteristicOutput:
        return CharacteristicOutput(self.mlp(masked_observations))


def main(config: CharacteristicConfig, agent: Agent) -> tuple[TrainState[CharacteristicModel], EpochMetrics | None]:
    """Train (or load) every seed explaining `agent`, tracked by `config.track`.

    Collects each seed's buffers, sizes the exact tables from them, then trains every
    seed in one compiled program. With a backend each seed logs the source's
    `model/{Policy,Value}CharacteristicModel_*` scalars into its own run, and the whole
    seed-stacked final state is pushed as one Orbax checkpoint artifact. The returned
    state and metrics carry a leading seed axis (metrics: (epochs, seeds); None when loaded).
    """
    keys = jnp.stack([jax.random.key(seed) for seed in config.seeds])
    agents = stack_seeds([agent] * len(config.seeds))
    if config.track is None:
        keys, buffers = nnx.jit(partial(collect, config))(keys, agents)
        # Exact tables are sized by the most unique observations in any seed's exact buffer (a static shape).
        max_unique = max(int(np.unique(rows, axis=0).shape[0]) for rows in np.asarray(buffers.exact))
        return nnx.jit(partial(fit, config, max_unique, None))(keys, agents, buffers)
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
                return checkpoints.load(Path(directory), initialize(config), settings), None

            def log(
                runs: tuple[wandb.Run | trackio.Run, ...], model: str, exact_loss: bool, metrics: EpochMetrics
            ) -> None:
                """The source's train_models scalars, `model/{model}_{epoch,val,exact}_loss`, into each seed's run."""
                active, steps = np.asarray(metrics.active), np.asarray(metrics.step)
                train_loss, validation = np.asarray(metrics.train_loss), np.asarray(metrics.validation_loss)
                exact = np.asarray(metrics.exact_loss)
                for index, run in enumerate(runs):
                    if not active[index]:
                        continue
                    row = {
                        f"model/{model}_epoch_loss": float(train_loss[index]),
                        f"model/{model}_val_loss": float(validation[index]),
                    }
                    if exact_loss:
                        row[f"model/{model}_exact_loss"] = float(exact[index])
                    run.log(row, step=int(steps[index]))

            model = f"{'Policy' if config.kind == 'behaviour' else 'Value'}CharacteristicModel"
            keys, buffers = nnx.jit(partial(collect, config))(keys, agents)
            # Exact tables are sized by the most unique observations in any seed's exact buffer (a static shape).
            max_unique = max(int(np.unique(rows, axis=0).shape[0]) for rows in np.asarray(buffers.exact))
            epoch_log = partial(log, runs, model, config.train.exact_loss)
            state, metrics = nnx.jit(partial(fit, config, max_unique, epoch_log))(keys, agents, buffers)
            directory = root / "checkpoints" / artifact
            checkpoints.save(directory, state, settings)
            runs[0].log_artifact(str(directory), name=artifact, type="characteristic").wait()
            print(f"Characteristic checkpoint: {artifact}:latest")
            return state, metrics
        finally:
            for run in runs:
                run.finish()


def collect(config: CharacteristicConfig, keys: Array, agents: Agent) -> tuple[Array, StateBuffers]:
    """Rollouts of the explained policy: each seed's buffers, and its key advanced past them."""

    def seed(
        config: CharacteristicConfig, env: EpisodeStatisticsWrapper[EnvState, EnvParams], key: Array, agent: Agent
    ) -> tuple[Array, StateBuffers]:
        epsilon = agent.epsilon if config.policy == "explore" else jnp.asarray(0.0, jnp.float32)
        training = config.train
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

    every_seed = nnx.vmap(partial(seed, config, EpisodeStatisticsWrapper(config.env)), in_axes=(0, 0), out_axes=(0, 0))
    return every_seed(keys, agents)


def fit(
    config: CharacteristicConfig,
    max_unique: int,
    log: Callable[[EpochMetrics], None] | None,
    keys: Array,
    agents: Agent,
    buffers: StateBuffers,
) -> tuple[TrainState[CharacteristicModel], EpochMetrics]:
    """Train every seed from its buffers; exact tables hold `max_unique` states.

    Each seed's context holds the frozen agent's explained outputs on every buffer and,
    with the exact loss, exact characteristic values on the exact buffer.
    """
    state = initialize(config, keys, buffers)
    features = state.buffers.train.shape[-1]

    def context(
        config: CharacteristicConfig, max_unique: int, buffers: StateBuffers, agent: Agent
    ) -> CharacteristicContext:
        # `v_F`: the agent's greedy/explore policy for behaviour, or its value for prediction.
        targets = []
        for observations in (buffers.train, buffers.validation, buffers.exact):
            q_values = agent.model(observations).q_values
            if config.kind == "prediction":
                targets.append(jnp.max(q_values, axis=-1, keepdims=True))
                continue
            ties = q_values == jnp.max(q_values, axis=-1, keepdims=True)
            greedy = ties.astype(jnp.float32) / jnp.sum(ties, axis=-1, keepdims=True)
            explore = agent.epsilon / q_values.shape[-1] + (1 - agent.epsilon) * greedy
            targets.append(greedy if config.policy == "greedy" else explore)
        coalitions = jnp.asarray(coalition_table(buffers.exact.shape[-1]))
        exact = (
            exact_characteristic(buffers.exact, targets[2], coalitions, max_unique=max_unique)
            if config.train.exact_loss
            else None
        )
        return CharacteristicContext(buffers, targets[0], targets[1], exact)

    contexts = nnx.vmap(partial(context, config, max_unique), in_axes=(0, 0), out_axes=0)(state.buffers, agents)
    return train(state, contexts, None, loss, config.train, features, log)


def initialize(
    config: CharacteristicConfig, keys: Array | None = None, buffers: StateBuffers | None = None
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
    # Experiment-script layout: ReLU hidden layers; behaviour ends in Softmax(-1).
    # Prediction keeps the source's trailing singleton value axis.
    output = LayerConfig(wrapped.num_actions, "softmax") if config.kind == "behaviour" else LayerConfig(1)
    network = MLPConfig(features, (*(LayerConfig(width, "relu") for width in config.hidden_widths), output))
    training = config.train
    if keys is None:
        keys = jnp.stack([jax.random.key(seed) for seed in config.seeds])
    if buffers is None:
        empty = StateBuffers(
            jnp.zeros((training.train_size, features), jnp.int32),
            jnp.zeros((training.validation_size, features), jnp.int32),
            jnp.zeros((training.exact_size, features), jnp.int32),
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


def loss(
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
    sample_rand); validation resamples the whole validation size. The exact loss is
    the MSE over every valid unique masked observation of every coalition, equally
    weighted; only the chunks holding valid entries are evaluated, as their number
    varies at run time while the padded arrays keep static shapes.
    """
    del frozen
    if partition == "exact":
        exact = context.exact
        assert exact is not None
        outputs = exact.compact_values.shape[-1]
        chunks = (exact.count + EXACT_CHUNK - 1) // EXACT_CHUNK

        def body(
            exact: ExactCharacteristic, model: CharacteristicModel, carry: tuple[Array, Array]
        ) -> tuple[Array, Array]:
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
    index_key, coalition_key = jax.random.split(key)
    if partition == "train":
        rows = jax.random.randint(index_key, (config.batch_size,), 0, config.train_size)
        observations, targets = context.buffers.train[rows], context.train_targets[rows]
    else:
        rows = jax.random.randint(index_key, (config.validation_size,), 0, config.validation_size)
        observations, targets = context.buffers.validation[rows], context.validation_targets[rows]
    coalitions = jax.random.bernoulli(coalition_key, 0.5, (rows.shape[0], features))
    predictions = model(mask(observations, coalitions)).values
    return jnp.mean(jnp.square(predictions - jax.lax.stop_gradient(targets)))


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
