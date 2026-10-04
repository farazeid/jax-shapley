"""Shapley models for behaviour, prediction and outcome explanations.

A Shapley model maps a full observation to per-feature contributions `phi` of
shape (..., features, outputs), trained against a frozen characteristic function
`v(s, C)`. The reference is FastSVERL's `Shapley` and its Policy, Value and
Performance subclasses, which differ only in the characteristic they explain.
"""

from dataclasses import dataclass
from math import comb
from typing import NamedTuple, Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange
from flax import nnx
from jax import Array
from jaxtyping import Float

from jax_shapley.characteristics import CharacteristicModel, LossMetrics, StateFlags, States, UniqueValues, Values
from jax_shapley.coalitions import MASK, Coalition, Coalitions, CoalitionTable, Observations, coalition_table, mask
from jax_shapley.networks import MLP, LayerConfig, MLPConfig

type Contributions = Float[Array, "... features outputs"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type StateContributions = Float[Array, "unique features outputs"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type ShapleyMatrix = Float[Array, "features coalitions"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.


@dataclass(frozen=True, slots=True)
class ShapleyLayout:
    """Static output layout: one contribution per feature and explained output."""

    features: int
    outputs: int

    def __post_init__(self) -> None:
        if self.features < 2 or self.outputs < 1:
            raise ValueError("Shapley models need at least two features and one output")


@dataclass(frozen=True, slots=True)
class MaskedCharacteristic:
    """Behaviour/prediction characteristic: the model applied to masked observations."""

    model: CharacteristicModel

    def values(self, observations: Observations, coalitions: Coalitions) -> Values:
        return self.model(mask(observations, coalitions)).values

    def null(self, observations: Observations) -> Values:
        """v(empty) for every row: one all-MASK row, broadcast over the leading axes.

        The value is identical for every row. Evaluating a broadcast-constant batch also
        miscompiles on XLA CPU (jax 0.11.0) from 64 rows; tests/test_jit_equivalence.py
        guards the compiled result against eager evaluation.
        """
        null = self.model(jnp.full((1, observations.shape[-1]), MASK, jnp.int32)).values[0]
        return jnp.broadcast_to(null, (*observations.shape[:-1], null.shape[-1]))


class CharacteristicFunction(Protocol):
    """A frozen characteristic v(s, C); coalitions broadcast against observations."""

    def values(self, observations: Observations, coalitions: Coalitions) -> Values: ...

    def null(self, observations: Observations) -> Values: ...


class ShapleyOutput(NamedTuple):
    contributions: Contributions


class ShapleyModel(nnx.Module):
    """ReLU MLP whose flat output is laid out as (features, outputs), like the source's view()."""

    def __init__(self, config: MLPConfig, layout: ShapleyLayout, *, rngs: nnx.Rngs) -> None:
        if config.input_features != layout.features or config.output_features != layout.features * layout.outputs:
            raise ValueError("Shapley network dimensions must match its layout")
        if config.layers[-1].activation != "identity":
            raise ValueError("Shapley outputs are unactivated contributions")
        self.layout = layout
        self.mlp = MLP(config, rngs=rngs)

    def __call__(self, observations: Observations) -> ShapleyOutput:
        flat = self.mlp(observations)
        return ShapleyOutput(rearrange(flat, "... (f a) -> ... f a", f=self.layout.features))


def shapley_network(layout: ShapleyLayout, hidden_widths: tuple[int, ...]) -> MLPConfig:
    return MLPConfig(
        layout.features,
        (*tuple(LayerConfig(width, "relu") for width in hidden_widths), LayerConfig(layout.features * layout.outputs)),
    )


def shapley_loss(
    model: ShapleyModel,
    characteristic: CharacteristicFunction,
    observations: Observations,
    coalition: Coalition,
) -> tuple[Array, LossMetrics]:
    """One coalition for the whole batch, as in the source: sum_{i in C} phi_i ~ v(C) - v(empty)."""
    targets = jax.lax.stop_gradient(characteristic.values(observations, coalition) - characteristic.null(observations))
    predictions = jnp.sum(coalition[:, None].astype(jnp.float32) * model(observations).contributions, axis=-2)
    loss = jnp.mean(jnp.square(predictions - targets))
    return loss, LossMetrics(loss, jnp.mean(targets), jnp.mean(predictions))


def update_shapley(
    model: ShapleyModel,
    optimizer: nnx.Optimizer[ShapleyModel],
    characteristic: CharacteristicFunction,
    observations: Observations,
    coalition: Coalition,
) -> LossMetrics:
    def loss(model: ShapleyModel) -> tuple[Array, LossMetrics]:
        # The frozen characteristic is closed over, not a differentiated argument: inside compiled
        # loops it belongs to an outer trace, and NNX transforms only split their own arguments.
        return shapley_loss(model, characteristic, observations, coalition)

    # NNX transformation has a dynamic return signature; assert its known loss/state boundary.
    result = cast(
        tuple[tuple[Array, LossMetrics], nnx.State[str | int, nnx.Param[Array]]],
        nnx.value_and_grad(loss, has_aux=True)(model),
    )
    (_, metrics), gradients = result
    optimizer.update(model, gradients)
    return metrics


def explain(model: ShapleyModel, characteristic: CharacteristicFunction, observations: Observations) -> Contributions:
    """Normalised Shapley values, using the learnt characteristic's grand and empty coalitions."""
    grand = characteristic.values(observations, jnp.ones(observations.shape[-1], jnp.bool_))
    return normalise(model(observations).contributions, grand, characteristic.null(observations))


def normalise(contributions: Contributions, grand: Values, null: Values) -> Contributions:
    """Additive efficient normalisation: spread v(F) - v(empty) - sum(phi) equally over features."""
    features = contributions.shape[-2]
    correction = (grand - null - jnp.sum(contributions, axis=-2)) / features
    return contributions + correction[..., None, :]


def exact_shapley(values: UniqueValues) -> StateContributions:
    """Source exact values: weighted least squares over all coalitions, then normalisation.

    `values` must enumerate coalitions in `all_coalitions` order, so index 0 is the
    empty coalition and index -1 the grand coalition.
    """
    features = round(np.log2(values.shape[0]))
    if 2**features != values.shape[0]:
        raise ValueError("exact Shapley values need every coalition")
    solved = jnp.einsum("fc,cso->sfo", shapley_matrix(features), values)
    return normalise(solved, values[-1], values[0])


def characteristic_table(
    characteristic: CharacteristicFunction,
    states: States,
    coalitions: CoalitionTable,
) -> UniqueValues:
    """The learnt characteristic at every coalition of every state."""
    return characteristic.values(states[None], coalitions[:, None])


def exact_shapley_loss(
    model: ShapleyModel,
    characteristic: CharacteristicFunction,
    states: States,
    state_valid: StateFlags,
    exact: StateContributions,
) -> Array:
    """MSE between learnt normalised and exact values over the valid unique states."""
    return contribution_loss(explain(model, characteristic, states), state_valid, exact)


def contribution_loss(contributions: StateContributions, state_valid: StateFlags, exact: StateContributions) -> Array:
    """MSE between normalised contributions and exact values over the valid unique states."""
    errors = jnp.square(contributions - exact)
    valid = state_valid.astype(jnp.float32)
    return jnp.sum(errors * valid[:, None, None]) / (jnp.sum(valid) * exact.shape[-2] * exact.shape[-1])


def shapley_matrix(features: int) -> ShapleyMatrix:
    """(X^T W X)^-1 X^T W with the Shapley kernel weights; empty and grand coalitions get zero weight."""
    coalitions = coalition_table(features).astype(np.float64)
    sizes = coalitions.sum(axis=1).astype(int)
    weights = np.asarray([
        (features - 1) / (comb(features, size) * size * (features - size)) if 0 < size < features else 0.0
        for size in sizes
    ])
    weighted = coalitions.T * weights
    return jnp.asarray(np.linalg.solve(weighted @ coalitions, weighted), jnp.float32)
