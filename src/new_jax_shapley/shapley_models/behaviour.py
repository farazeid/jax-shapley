"""Step 4: behaviour Shapley models, phi(s) learnt against a frozen characteristic; the shared stage.

A Shapley model maps a full state to per-feature contributions of shape
(..., features, outputs). Following FastSHAP, it is trained by weighted least squares:
for a Shapley-kernel coalition C, sum_{i in C} phi_i(s) ~ v(s, C) - v(s, empty). An
additive correction then makes contributions sum to v(s, F) - v(s, empty). The
reference is FastSVERL's `Shapley` and its Policy, Value and Performance subclasses,
which differ only in the characteristic they explain: `stage`, `fit`, `initialize`, `loss`,
`explain` and `exact_shapley` are shared with the other kinds, which supply their
characteristic kind's `targets` and `head`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from functools import partial
from math import comb
from pathlib import Path
from time import time_ns
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import trackio
import tyro
import wandb
from einops import rearrange
from flax import nnx, struct
from jax import Array
from jaxtyping import Float

from new_jax_shapley import backends
from new_jax_shapley.algorithms.dqn import TRACKING_FIELDS, Agent, Env, EnvParams, EnvState
from new_jax_shapley.backends import Track
from new_jax_shapley.characteristic_models.behaviour import (
    CharacteristicConfig,
    CharacteristicModel,
    Head,
    Targets,
    UniqueValues,
    exact_characteristic,
    head,
    targets,
)
from new_jax_shapley.common import checkpoints
from new_jax_shapley.common.coalitions import MASK, Observations, coalition_table, kernel_coalitions, mask
from new_jax_shapley.common.networks import MLP, LayerConfig, MLPConfig
from new_jax_shapley.common.rollouts import collect_buffers
from new_jax_shapley.common.supervised_learning import (
    EpochMetrics,
    Partition,
    StateBuffers,
    SupervisedConfig,
    TrainState,
    log_epochs,
    stack_seeds,
    train,
)
from new_jax_shapley.envs.wrappers import EpisodeStatisticsWrapper

type BufferSource = Literal["fresh", "characteristic"]
type Contributions = Float[Array, "... features outputs"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type StateContributions = Float[Array, "unique features outputs"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.


@dataclass(frozen=True, slots=True)
class ShapleyConfig:
    """The Shapley model, its buffers and the source's train_models settings.

    `with_exact_char` selects the exact-loss ground truth: Shapley values of the exact
    characteristic (measuring propagated characteristic error, the paper's "approx
    characteristic" curves) or of the learnt one. Training always uses the learnt one.
    `buffers` collects fresh rollouts of the explained policy, or reuses the
    characteristic's. Exact values always use the characteristic's exact buffer.
    """

    env: Env
    hidden_widths: tuple[int, ...] = (64, 64)
    with_exact_char: bool = True
    buffers: BufferSource = "fresh"
    seeds: tuple[int, ...] = (1,)
    train: SupervisedConfig = field(default_factory=SupervisedConfig)
    checkpoint: str | None = None
    """A Shapley artifact to load instead of training, e.g. `GROUP--shapley--behaviour--with-exact:latest`."""
    name: str = "shapley"
    """The experiment name runs are named after: `NAME--shapley--KIND--{with,without}-exact--seed-N`."""
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
        if any(width < 1 for width in self.hidden_widths):
            raise ValueError("hidden widths must be positive")
        if self.buffers not in ("fresh", "characteristic"):
            raise ValueError("buffers must be fresh or characteristic")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct and nonempty")


@struct.dataclass
class ExactShapley:
    """Exact normalised Shapley values at the exact buffer's unique states (padded, with validity)."""

    states: Array
    state_valid: Array
    values: Array


@struct.dataclass
class ShapleyContext:
    """Per-seed constants a Shapley learner reads: buffers and the exact-loss ground truth."""

    buffers: StateBuffers
    exact: ExactShapley | None


class ShapleyOutput(NamedTuple):
    contributions: Contributions


class ShapleyModel(nnx.Module):
    """ReLU MLP whose flat output is laid out as (features, outputs), like the source's view()."""

    def __init__(self, config: MLPConfig, features: int, *, rngs: nnx.Rngs) -> None:
        if features < 2 or config.input_features != features or config.output_features % features:
            raise ValueError("Shapley networks map two or more features to a contribution per feature and output")
        if config.layers[-1].activation != "identity":
            raise ValueError("Shapley outputs are unactivated contributions")
        self.features = features
        self.mlp = MLP(config, rngs=rngs)

    def __call__(self, observations: Observations) -> ShapleyOutput:
        flat = self.mlp(observations)
        return ShapleyOutput(rearrange(flat, "... (f a) -> ... f a", f=self.features))


def main(
    config: ShapleyConfig,
    explained: CharacteristicConfig,
    agents: Agent,
    characteristics: CharacteristicModel,
    characteristic_buffers: StateBuffers,
) -> tuple[TrainState[ShapleyModel], EpochMetrics | None]:
    """Behaviour Shapley models explaining frozen behaviour characteristics."""
    if explained.kind != "behaviour":
        raise ValueError("the behaviour stage explains a behaviour characteristic")
    return stage(config, explained, agents, characteristics, characteristic_buffers, targets, head, "PolicyShapley")


def stage(
    config: ShapleyConfig,
    explained: CharacteristicConfig,
    agents: Agent,
    characteristics: CharacteristicModel,
    characteristic_buffers: StateBuffers,
    targets: Targets,
    head: Head,
    model: str,
) -> tuple[TrainState[ShapleyModel], EpochMetrics | None]:
    """Train (or load) every seed explaining frozen characteristics, tracked by `config.track`.

    The agents, characteristic models and their buffers are seed-stacked, one per Shapley
    seed: the same one repeated (the paper protocol), or Shapley seed s explaining
    characteristic seed s and its agent. `explained` is the characteristic's config;
    `targets` and `head` are its kind's; `model` is the source's model class name that
    prefixes the logged tags. With a backend each seed logs the source's
    `model/{Policy,Value}Shapley_*` scalars into its own run, and the whole seed-stacked
    final state is pushed as one Orbax checkpoint artifact. The returned state and metrics
    carry a leading seed axis (metrics: (epochs, seeds); None when loaded).
    """
    seeds = len(config.seeds)
    keys = jnp.stack([jax.random.key(seed) for seed in config.seeds])
    if characteristic_buffers.exact.shape[0] != seeds or agents.epsilon.shape[0] != seeds:
        raise ValueError("Shapley inputs need one agent and characteristic per seed")
    inherited = characteristic_buffers
    # Exact tables are sized by the most unique observations in any seed's exact buffer (a static shape).
    max_unique = max(int(np.unique(rows, axis=0).shape[0]) for rows in np.asarray(inherited.exact))
    if config.track is None:
        program = nnx.jit(partial(fit, config, explained, targets, head, max_unique, None))
        return program(keys, agents, characteristics, inherited)
    group = config.group or f"{config.name}-{time_ns()}"
    exactness = "with-exact" if config.with_exact_char else "without-exact"
    artifact = f"{group}--shapley--{explained.kind}--{exactness}"
    settings = {
        "stage": "shapley",
        "env": config.env.name,
        "explained": {name: value for name, value in asdict(explained).items() if name not in TRACKING_FIELDS},
        "config": {name: value for name, value in asdict(config).items() if name not in TRACKING_FIELDS},
    }
    with backends.session(config.track, config.entity) as root:
        tracking = wandb if config.track == "wandb" else trackio
        runs = tuple(
            tracking.init(
                project=config.project,
                name=f"{config.name}--shapley--{explained.kind}--{exactness}--seed-{seed}",
                group=group,
                config={**asdict(config), "seed": seed},
            )
            for seed in config.seeds
        )
        try:
            if config.checkpoint is not None:
                directory = runs[0].use_artifact(config.checkpoint).download(root=str(root / "artifacts" / artifact))
                return checkpoints.load(Path(directory), initialize(config, explained, head), settings), None

            epoch_log = partial(log_epochs, runs, model, config.train.exact_loss)
            program = nnx.jit(partial(fit, config, explained, targets, head, max_unique, epoch_log))
            state, metrics = program(keys, agents, characteristics, inherited)
            directory = root / "checkpoints" / artifact
            checkpoints.save(directory, state, settings)
            runs[0].log_artifact(str(directory), name=artifact, type="shapley").wait()
            print(f"Shapley checkpoint: {artifact}:latest")
            return state, metrics
        finally:
            for run in runs:
                run.finish()


def fit(
    config: ShapleyConfig,
    explained: CharacteristicConfig,
    targets: Targets,
    head: Head,
    max_unique: int,
    log: Callable[[EpochMetrics], None] | None,
    keys: Array,
    agents: Agent,
    characteristics: CharacteristicModel,
    inherited: StateBuffers,
) -> tuple[TrainState[ShapleyModel], EpochMetrics]:
    """Train every seed; exact tables hold `max_unique` states.

    Buffers are fresh rollouts of the explained policy, or the characteristic's; exact
    values always use the characteristic's exact buffer. With the exact loss each seed's
    context holds exact Shapley values on it, of the exact characteristic
    (`with_exact_char`) or of the learnt one.
    """

    def rollouts(
        config: ShapleyConfig,
        explained: CharacteristicConfig,
        env: EpisodeStatisticsWrapper[EnvState, EnvParams],
        key: Array,
        agent: Agent,
        inherited: StateBuffers,
    ) -> tuple[Array, StateBuffers]:
        if config.buffers == "characteristic":
            return key, inherited
        epsilon = agent.epsilon if explained.policy == "explore" else jnp.asarray(0.0, jnp.float32)
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
            exact=inherited.exact,
        )

    collect = nnx.vmap(
        partial(rollouts, config, explained, EpisodeStatisticsWrapper(config.env)), in_axes=(0, 0, 0), out_axes=(0, 0)
    )
    keys, buffers = collect(keys, agents, inherited)
    state = initialize(config, explained, head, keys, buffers)
    features = state.buffers.train.shape[-1]

    def context(
        config: ShapleyConfig,
        explained: CharacteristicConfig,
        targets: Targets,
        max_unique: int,
        buffers: StateBuffers,
        agent: Agent,
        characteristic: CharacteristicModel,
    ) -> ShapleyContext:
        if not config.train.exact_loss:
            return ShapleyContext(buffers, None)
        coalitions = jnp.asarray(coalition_table(buffers.exact.shape[-1]))
        exact = exact_characteristic(
            buffers.exact, targets(explained, agent, buffers.exact), coalitions, max_unique=max_unique
        )
        # Training always uses the learnt characteristic; with_exact_char only changes the ground truth.
        values = (
            exact.at_states
            if config.with_exact_char
            else characteristic(mask(exact.states[None], coalitions[:, None])).values
        )
        return ShapleyContext(buffers, ExactShapley(exact.states, exact.state_valid, exact_shapley(values)))

    contexts = nnx.vmap(partial(context, config, explained, targets, max_unique), in_axes=(0, 0, 0), out_axes=0)(
        state.buffers, agents, characteristics
    )
    return train(state, contexts, characteristics, loss, config.train, features, log)


def initialize(
    config: ShapleyConfig,
    explained: CharacteristicConfig,
    head: Head,
    keys: Array | None = None,
    buffers: StateBuffers | None = None,
) -> TrainState[ShapleyModel]:
    """Every seed's fresh training state, seed-stacked.

    Without keys and buffers it is the template a checkpoint is restored into: the
    configured seeds, and zero buffers with the Shapley run's train and validation
    sizes and the characteristic's exact size.
    """
    wrapped = EpisodeStatisticsWrapper(config.env)
    shape = wrapped.observation_space(wrapped.default_params).shape
    if len(shape) != 1:
        raise ValueError("explanations require flat observations")
    features = int(shape[0])
    # One contribution per feature and explained output (the characteristic head's width).
    outputs = head(wrapped.num_actions).features
    hidden = tuple(LayerConfig(width, "relu") for width in config.hidden_widths)
    network = MLPConfig(features, (*hidden, LayerConfig(features * outputs)))
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
        network: MLPConfig, features: int, training: SupervisedConfig, key: Array, buffers: StateBuffers
    ) -> TrainState[ShapleyModel]:
        model_key, key = jax.random.split(key)
        model = ShapleyModel(network, features, rngs=nnx.Rngs(model_key))
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

    return nnx.vmap(partial(seed, network, features, training), in_axes=(0, 0), out_axes=0)(keys, buffers)


def loss(
    model: ShapleyModel,
    characteristic: CharacteristicModel,
    context: ShapleyContext,
    config: SupervisedConfig,
    features: int,
    key: Array,
    partition: Partition,
) -> Array:
    """A sampled train or validation batch's loss, or the exact loss.

    Batches use one Shapley-kernel coalition for the whole batch, as in the source:
    sum_{i in C} phi_i ~ v(C) - v(empty). The exact loss is the MSE between learnt
    normalised and exact values over the valid unique states.
    """
    if partition == "exact":
        exact = context.exact
        assert exact is not None
        errors = jnp.square(explain(model, characteristic, exact.states) - exact.values)
        valid = exact.state_valid.astype(jnp.float32)
        return jnp.sum(errors * valid[:, None, None]) / (jnp.sum(valid) * errors.shape[-2] * errors.shape[-1])
    index_key, coalition_key = jax.random.split(key)
    if partition == "train":
        rows = jax.random.randint(index_key, (config.batch_size,), 0, config.train_size)
        observations = context.buffers.train[rows]
    else:
        rows = jax.random.randint(index_key, (config.validation_size,), 0, config.validation_size)
        observations = context.buffers.validation[rows]
    coalition = kernel_coalitions(coalition_key, 1, features)[0]
    # v(empty): one all-MASK row broadcast over the rows. It is identical for every row, and
    # evaluating a broadcast-constant batch miscompiles on XLA CPU (jax 0.11.0) from 64 rows.
    null = characteristic(jnp.full((1, features), MASK, jnp.int32)).values[0]
    targets = characteristic(mask(observations, coalition)).values - null
    predictions = jnp.sum(coalition[:, None].astype(jnp.float32) * model(observations).contributions, axis=-2)
    return jnp.mean(jnp.square(predictions - jax.lax.stop_gradient(targets)))


def explain(model: ShapleyModel, characteristic: CharacteristicModel, observations: Observations) -> Contributions:
    """Normalised Shapley values: v(F) - v(empty) - sum(phi) spread equally over features (additive
    efficient normalisation), using the learnt characteristic's grand and empty coalitions."""
    features = observations.shape[-1]
    grand = characteristic(mask(observations, jnp.ones(features, jnp.bool_))).values
    # v(empty): one all-MASK row, broadcast (see `loss`).
    null = characteristic(jnp.full((1, features), MASK, jnp.int32)).values[0]
    contributions = model(observations).contributions
    return contributions + ((grand - null - jnp.sum(contributions, axis=-2)) / features)[..., None, :]


def exact_shapley(values: UniqueValues) -> StateContributions:
    """Source exact values: weighted least squares over all coalitions, then normalisation.

    `values` must enumerate coalitions in `coalition_table` order, so index 0 is the empty
    coalition and index -1 the grand one. The solve is (X^T W X)^-1 X^T W with the Shapley
    kernel weights; empty and grand coalitions get zero weight.
    """
    features = round(np.log2(values.shape[0]))
    if 2**features != values.shape[0]:
        raise ValueError("exact Shapley values need every coalition")
    coalitions = coalition_table(features).astype(np.float64)
    sizes = coalitions.sum(axis=1).astype(int)
    weights = np.asarray([
        (features - 1) / (comb(features, size) * size * (features - size)) if 0 < size < features else 0.0
        for size in sizes
    ])
    weighted = coalitions.T * weights
    matrix = jnp.asarray(np.linalg.solve(weighted @ coalitions, weighted), jnp.float32)
    solved = jnp.einsum("fc,cso->sfo", matrix, values)
    grand, null = values[-1], values[0]
    normalised: StateContributions = solved + ((grand - null - jnp.sum(solved, axis=-2)) / features)[..., None, :]
    return normalised
