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

    `keys`/`values`/`valid` hold each coalition's unique masked observations, padded
    to a static size. `states` are the buffer's unique full observations (sorted,
    padded) and `at_states` gives every coalition's value at each of them.
    """

    keys: UniqueKeys
    values: UniqueValues
    valid: UniqueFlags
    states: States
    state_valid: StateFlags
    at_states: UniqueValues

    @property
    def null(self) -> Output:
        """The empty coalition's value; coalition 0 masks every feature."""
        return self.values[0, 0]


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

    Every buffer row has equal weight, which equals the source's Bayes rule over the
    empirical distribution of unique states. `max_unique` must bound the number of
    unique observations; padded entries are marked invalid.
    """
    if observations.ndim != 2 or targets.ndim != 2 or targets.shape[0] != observations.shape[0]:
        raise ValueError("exact characteristics need (rows, features) observations and (rows, outputs) targets")
    if max_unique < 1:
        raise ValueError("max_unique must be positive")
    states, state_rows, state_counts = jnp.unique(
        observations, axis=0, size=max_unique, fill_value=0, return_index=True, return_counts=True
    )

    def evaluate(coalition: Array) -> tuple[Array, Array, Array, Array]:
        # jnp.unique's multi-output overloads are untyped; the requested outputs are arrays.
        keys, inverse, counts = cast(
            tuple[Array, Array, Array],
            jnp.unique(
                mask(observations, coalition),
                axis=0,
                size=max_unique,
                fill_value=0,
                return_inverse=True,
                return_counts=True,
            ),
        )
        segments = inverse.reshape(-1)
        sums = jax.ops.segment_sum(targets, segments, num_segments=max_unique)
        values = sums / jnp.maximum(counts, 1).astype(jnp.float32)[:, None]
        return keys, values, counts > 0, values[segments[state_rows]]

    keys, values, valid, at_states = jax.lax.map(evaluate, coalitions)
    return ExactCharacteristic(keys, values, valid, states, state_counts > 0, at_states)


def exact_characteristic_loss(model: CharacteristicModel, exact: ExactCharacteristic) -> Array:
    """MSE over every valid unique masked observation of every coalition, equally weighted."""
    errors = jnp.square(model(exact.keys).values - exact.values)
    valid = exact.valid.astype(jnp.float32)
    return jnp.sum(errors * valid[..., None]) / (jnp.sum(valid) * exact.values.shape[-1])


def max_unique_states(low: ArrayLike, high: ArrayLike, features: int, rows: int) -> int:
    """Static bound on distinct observations: the bounded observation space, or the buffer rows."""
    lows = np.broadcast_to(np.asarray(low), (features,)).astype(np.int64)
    highs = np.broadcast_to(np.asarray(high), (features,)).astype(np.int64)
    if np.any(highs < lows):
        raise ValueError("observation bounds must satisfy low <= high")
    return int(min(rows, prod(int(size) for size in highs - lows + 1)))
