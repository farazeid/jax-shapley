"""Behaviour and prediction characteristics: models, losses and exact values.

A characteristic predicts the explained agent output `v_F(s)` from a masked
state. Behaviour explains the agent's policy (greedy or epsilon-greedy);
prediction explains its value `max_a Q(s, a)`. The reference is FastSVERL's
`ValuePolicyCharacteristicModel` and its exact-value computation.
"""

from dataclasses import dataclass
from math import inf, isnan, prod
from typing import Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx, struct
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Bool, Float, Int

from jax_shapley.coalitions import Coalitions, CoalitionTable, Observations, mask
from jax_shapley.networks import MLP, ActionValues, LayerConfig, MLPConfig
from jax_shapley.train_dqn import exploration_probabilities, greedy_probabilities

type Kind = Literal["behaviour", "prediction"]

# Rows per exact-loss evaluation chunk; padding beyond the valid entries is never evaluated.
EXACT_CHUNK = 512
type Policy = Literal["greedy", "explore"]
type Values = Float[Array, "... outputs"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type Output = Float[Array, "outputs"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.
type RowWeights = Float[Array, "batch"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.
type RowObservations = Int[Array, "rows features"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type RowValues = Float[Array, "rows outputs"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type UniqueKeys = Int[Array, "coalitions unique features"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type UniqueValues = Float[Array, "coalitions unique outputs"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type UniqueFlags = Bool[Array, "coalitions unique"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type States = Int[Array, "unique features"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type StateFlags = Bool[Array, "unique"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.


@dataclass(frozen=True, slots=True)
class CharacteristicConfig:
    """Which agent output is explained, and the characteristic's hidden ReLU widths."""

    kind: Kind
    policy: Policy = "greedy"
    hidden_widths: tuple[int, ...] = (64, 64)

    def __post_init__(self) -> None:
        if self.kind not in ("behaviour", "prediction"):
            raise ValueError("characteristic kind must be behaviour or prediction")
        if self.policy not in ("greedy", "explore"):
            raise ValueError("explained policy must be greedy or explore")
        if any(width < 1 for width in self.hidden_widths):
            raise ValueError("hidden widths must be positive")


@dataclass(frozen=True, slots=True)
class OffPolicyConfig:
    """Importance weighting for off-policy characteristic training (source defaults)."""

    weighting: bool = True
    clip_coef: float = inf

    def __post_init__(self) -> None:
        if isnan(self.clip_coef) or self.clip_coef < 0:
            raise ValueError("clip_coef must be nonnegative")


@struct.dataclass
class ExactCharacteristic:
    """Exact characteristic values under a buffer's empirical state distribution.

    `states` are the buffer's unique observations (sorted, padded to a static size;
    `state_valid` marks real ones) and `at_states[c, u]` is coalition c's exact value
    at state u. `compact_keys`/`compact_values` list every coalition's distinct masked
    observation and its value once (`count` of them, valid entries first, padded to
    whole chunks), which the per-epoch exact loss evaluates. `null` is the empty
    coalition's value and `distinct[c]` the number of distinct masked observations of
    coalition c.
    """

    states: States
    state_valid: StateFlags
    at_states: UniqueValues
    compact_keys: RowObservations
    compact_values: RowValues
    count: Array
    null: Output
    distinct: Array


class CharacteristicOutput(NamedTuple):
    values: Values


class LossMetrics(NamedTuple):
    loss: Array
    target_mean: Array
    prediction_mean: Array


class CharacteristicModel(nnx.Module):
    """Characteristic network over masked int32 observations."""

    def __init__(self, config: MLPConfig, *, rngs: nnx.Rngs) -> None:
        self.mlp = MLP(config, rngs=rngs)

    def __call__(self, masked_observations: Observations) -> CharacteristicOutput:
        return CharacteristicOutput(self.mlp(masked_observations))


def characteristic_network(features: int, actions: int, config: CharacteristicConfig) -> MLPConfig:
    """Experiment-script layout: ReLU hidden layers; behaviour ends in Softmax(-1)."""
    output = (
        LayerConfig(actions, "softmax") if config.kind == "behaviour" else LayerConfig(1)
    )  # Prediction keeps the source's trailing singleton value axis.
    return MLPConfig(features, (*tuple(LayerConfig(width, "relu") for width in config.hidden_widths), output))


def explained_values(q_values: ActionValues, config: CharacteristicConfig, epsilon: Array) -> Values:
    """`v_F`: the agent's greedy/explore policy for behaviour, or its value for prediction."""
    if config.kind == "prediction":
        return jnp.max(q_values, axis=-1, keepdims=True)
    return policy_probabilities(q_values, config.policy, epsilon)


def policy_probabilities(q_values: ActionValues, policy: Policy, epsilon: Array) -> Values:
    if policy == "greedy":
        return greedy_probabilities(q_values)
    return exploration_probabilities(q_values, epsilon)


def characteristic_loss(
    model: CharacteristicModel, observations: Observations, targets: Values, coalitions: Coalitions
) -> tuple[Array, LossMetrics]:
    """On-policy MSE over every output of every row."""
    predictions = model(mask(observations, coalitions)).values
    targets = jax.lax.stop_gradient(targets)
    loss = jnp.mean(jnp.square(predictions - targets))
    return loss, LossMetrics(loss, jnp.mean(targets), jnp.mean(predictions))


def importance_weighted_loss(
    model: CharacteristicModel,
    observations: Observations,
    targets: Values,
    coalitions: Coalitions,
    weights: RowWeights,
) -> tuple[Array, LossMetrics]:
    """Source ImportanceWeightedMSELoss: per-row weights, mean over rows and outputs."""
    predictions = model(mask(observations, coalitions)).values
    targets = jax.lax.stop_gradient(targets)
    loss = jnp.mean(jax.lax.stop_gradient(weights)[:, None] * jnp.square(predictions - targets))
    return loss, LossMetrics(loss, jnp.mean(targets), jnp.mean(predictions))


def off_policy_weights(
    new_log_probabilities: RowWeights, old_log_probabilities: RowWeights, config: OffPolicyConfig
) -> RowWeights:
    """Ratio, optionally normalised to sum to one over the batch, then clipped around one."""
    ratio = jnp.exp(new_log_probabilities - old_log_probabilities)
    if config.weighting:
        ratio = ratio / jnp.sum(ratio)
    return jnp.clip(ratio, 1 - config.clip_coef, 1 + config.clip_coef)


def update_characteristic(
    model: CharacteristicModel,
    optimizer: nnx.Optimizer[CharacteristicModel],
    observations: Observations,
    targets: Values,
    coalitions: Coalitions,
) -> LossMetrics:
    # NNX transformation has a dynamic return signature; assert its known loss/state boundary.
    result = cast(
        tuple[tuple[Array, LossMetrics], nnx.State[str | int, nnx.Param[Array]]],
        nnx.value_and_grad(characteristic_loss, has_aux=True)(model, observations, targets, coalitions),
    )
    (_, metrics), gradients = result
    optimizer.update(model, gradients)
    return metrics


def exact_characteristic(
    observations: RowObservations,
    targets: RowValues,
    coalitions: CoalitionTable,
    *,
    max_unique: int,
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

    def evaluate(coalition: Array) -> tuple[Array, Array, Array, Array]:
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

    keys, values, valid, at_states = jax.lax.map(evaluate, coalitions)
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
    return ExactCharacteristic(
        states,
        state_counts > 0,
        at_states,
        compact_keys,
        compact_values,
        count,
        values[0, 0],
        jnp.sum(valid, axis=-1, dtype=jnp.int32),
    )


def exact_characteristic_loss(model: CharacteristicModel, exact: ExactCharacteristic) -> Array:
    """MSE over every valid unique masked observation of every coalition, equally weighted.

    Only the chunks holding valid entries are evaluated: their number varies at run
    time, while the padded arrays keep static shapes.
    """
    outputs = exact.compact_values.shape[-1]
    chunks = (exact.count + EXACT_CHUNK - 1) // EXACT_CHUNK

    def body(state: tuple[Array, Array]) -> tuple[Array, Array]:
        index, total = state
        start = index * EXACT_CHUNK
        keys = jax.lax.dynamic_slice_in_dim(exact.compact_keys, start, EXACT_CHUNK)
        values = jax.lax.dynamic_slice_in_dim(exact.compact_values, start, EXACT_CHUNK)
        valid = (start + jnp.arange(EXACT_CHUNK)) < exact.count
        errors = jnp.square(model(keys).values - values)
        return index + 1, total + jnp.sum(jnp.where(valid[:, None], errors, 0.0))

    _, total = jax.lax.while_loop(
        lambda state: state[0] < chunks, body, (jnp.asarray(0, jnp.int32), jnp.asarray(0.0, jnp.float32))
    )
    return total / (exact.count.astype(jnp.float32) * outputs)


def max_unique_states(low: ArrayLike, high: ArrayLike, features: int, rows: int) -> int:
    """Static bound on distinct observations: the bounded observation space, or the buffer rows."""
    lows = np.broadcast_to(np.asarray(low), (features,)).astype(np.int64)
    highs = np.broadcast_to(np.asarray(high), (features,)).astype(np.int64)
    if np.any(highs < lows):
        raise ValueError("observation bounds must satisfy low <= high")
    return int(min(rows, prod(int(size) for size in highs - lows + 1)))
