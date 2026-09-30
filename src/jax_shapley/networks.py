"""Typed NNX agent networks matching FastSVERL's SimpleNN semantics.

Leading observation axes are preserved. Environment int32 observations become
float32 at this boundary; all parameters and numerical outputs use float32.
Initialization matches PyTorch distributions, not its seed-specific RNG stream.
"""

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array
from jaxtyping import Float, Int

type Features = Float[Array, "... features"]  # ruff: ignore[forward-annotation-syntax-error]
type ModelInput = Int[Array, "... features"] | Features  # ruff: ignore[forward-annotation-syntax-error]
type ActionValues = Float[Array, "... actions"]  # ruff: ignore[forward-annotation-syntax-error]
type StateValues = Float[Array, "... 1"]  # ruff: ignore[forward-annotation-syntax-error]
type Activation = Literal["identity", "relu", "tanh", "softmax"]


@dataclass(frozen=True, slots=True)
class TorchLinearInit:
    """PyTorch Linear default: kernel and bias uniform in ±1/sqrt(fan_in)."""


@dataclass(frozen=True, slots=True)
class OrthogonalInit:
    """FastSVERL's explicit layer_init override: orthogonal kernel, constant bias."""

    gain: float = 1.0
    bias: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.gain) or not isfinite(self.bias):
            raise ValueError("orthogonal initialization gain and bias must be finite")


type LinearInit = TorchLinearInit | OrthogonalInit


@dataclass(frozen=True, slots=True)
class LayerConfig:
    """One affine layer followed by an explicitly selected activation."""

    features: int
    activation: Activation = "identity"
    initialization: LinearInit = TorchLinearInit()

    def __post_init__(self) -> None:
        if self.features < 1:
            raise ValueError("layer features must be positive")
        if self.activation not in ("identity", "relu", "tanh", "softmax"):
            raise ValueError("unsupported activation")


@dataclass(frozen=True, slots=True)
class MLPConfig:
    """Static architecture, validated before model construction or tracing."""

    input_features: int
    layers: tuple[LayerConfig, ...]

    def __post_init__(self) -> None:
        if self.input_features < 1:
            raise ValueError("input features must be positive")
        if not self.layers:
            raise ValueError("an MLP needs at least one layer")

    @property
    def output_features(self) -> int:
        return self.layers[-1].features


def dqn_config(input_features: int, num_actions: int, *, hidden_widths: tuple[int, ...] = (64, 64)) -> MLPConfig:
    """Source Gridworld/Mastermind-222 architecture; use (128, 128) for Hypercube.

    Larger Mastermind experiments also use (128, 128). Widths are explicit so a
    caller cannot silently infer a paper configuration from observation count.
    """
    return MLPConfig(
        input_features, (*tuple(LayerConfig(width, "relu") for width in hidden_widths), LayerConfig(num_actions))
    )


def _activate(inputs: Features, activation: Activation) -> Features:
    if activation == "relu":
        return jax.nn.relu(inputs)
    if activation == "tanh":
        return jnp.tanh(inputs)
    if activation == "softmax":
        return jax.nn.softmax(inputs, axis=-1)
    return inputs


class Linear(nnx.Module):
    """NNX affine layer with explicit, upstream-compatible initialization."""

    def __init__(self, input_features: int, config: LayerConfig, *, rngs: nnx.Rngs) -> None:
        kernel_key, bias_key = rngs.params(), rngs.params()
        shape = (input_features, config.features)
        if isinstance(config.initialization, OrthogonalInit):
            # Torch initializes (out, in); retain that orientation before transposing.
            kernel = jax.nn.initializers.orthogonal(config.initialization.gain)(
                kernel_key, (config.features, input_features), jnp.float32
            ).T
            bias = jnp.full((config.features,), config.initialization.bias, dtype=jnp.float32)
        else:
            bound = 1.0 / sqrt(input_features)
            kernel = jax.random.uniform(kernel_key, shape, jnp.float32, minval=-bound, maxval=bound)
            bias = jax.random.uniform(bias_key, (config.features,), jnp.float32, minval=-bound, maxval=bound)
        self.kernel: nnx.Param[Array] = nnx.Param(kernel)
        self.bias: nnx.Param[Array] = nnx.Param(bias)

    def __call__(self, inputs: Features) -> Features:
        return inputs @ self.kernel[...] + self.bias[...]


class MLP(nnx.Module):
    """Per-layer configurable MLP; preserves arbitrary leading batch axes."""

    def __init__(self, config: MLPConfig, *, rngs: nnx.Rngs) -> None:
        self.config = config
        widths = (config.input_features, *(layer.features for layer in config.layers))
        self.layers: nnx.List[Linear] = nnx.List(
            Linear(width, layer, rngs=rngs) for width, layer in zip(widths[:-1], config.layers, strict=True)
        )

    def __call__(self, observations: ModelInput) -> Features:
        # Shapes/dtypes are static under tracing; fail before any expensive run.
        if observations.ndim < 1 or observations.shape[-1] != self.config.input_features:
            raise ValueError(f"expected final observation axis of size {self.config.input_features}")
        if observations.dtype not in (jnp.dtype("int32"), jnp.dtype("float32")):
            raise TypeError("model observations must use int32 or float32")
        outputs = observations.astype(jnp.float32)
        for layer, config in zip(self.layers, self.config.layers, strict=True):
            outputs = _activate(layer(outputs), config.activation)
        return outputs


class QOutput(NamedTuple):
    """DQN values for every action; no policy or value reduction is implied."""

    q_values: ActionValues


class PolicyOutput(NamedTuple):
    """PPO action probabilities, matching Categorical(probs=...) upstream."""

    probabilities: ActionValues


class ValueOutput(NamedTuple):
    """PPO critic value with the upstream trailing singleton axis."""

    value: StateValues


class QNetwork(nnx.Module):
    """DQN's independent Q-network."""

    def __init__(self, config: MLPConfig, *, rngs: nnx.Rngs) -> None:
        if config.layers[-1].activation != "identity":
            raise ValueError("DQN output must be unactivated Q-values")
        self.mlp = MLP(config, rngs=rngs)

    def __call__(self, observations: ModelInput) -> QOutput:
        return QOutput(self.mlp(observations))


class PPOActor(nnx.Module):
    """Independent PPO actor. Supply the architecture explicitly, including Softmax."""

    def __init__(self, config: MLPConfig, *, rngs: nnx.Rngs) -> None:
        if config.layers[-1].activation != "softmax":
            raise ValueError("PPO actor output must use softmax probabilities")
        self.mlp = MLP(config, rngs=rngs)

    def __call__(self, observations: ModelInput) -> PolicyOutput:
        return PolicyOutput(self.mlp(observations))


class PPOCritic(nnx.Module):
    """Independent PPO critic; configurations are not inferred from Baxtub."""

    def __init__(self, config: MLPConfig, *, rngs: nnx.Rngs) -> None:
        if config.output_features != 1 or config.layers[-1].activation != "identity":
            raise ValueError("PPO critic requires one unactivated output")
        self.mlp = MLP(config, rngs=rngs)

    def __call__(self, observations: ModelInput) -> ValueOutput:
        return ValueOutput(self.mlp(observations))
