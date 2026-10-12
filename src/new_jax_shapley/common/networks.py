"""MLPs matching FastSVERL's SimpleNN.

Every model trains with Torch-compatible Adam: `optax.adam(eps_root=0.0)` at a constant
rate, epsilon outside the square root. Leading observation axes are preserved. Environment int32 observations become
float32 at this boundary; all parameters and outputs use float32. Initialization
matches PyTorch's Linear default distribution, not its seed-specific RNG stream.
"""

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Literal

import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array
from jaxtyping import Float, Int

type Features = Float[Array, "... features"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type ModelInput = Int[Array, "... features"] | Features  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
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


@dataclass(frozen=True, slots=True)
class AdamConfig:
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        for name, value in (("beta1", self.beta1), ("beta2", self.beta2)):
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if not isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be positive")


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
            outputs = outputs @ layer.kernel[...] + layer.bias[...]
            if config.activation == "relu":
                outputs = jax.nn.relu(outputs)
            elif config.activation == "tanh":
                outputs = jnp.tanh(outputs)
            elif config.activation == "softmax":
                outputs = jax.nn.softmax(outputs, axis=-1)
        return outputs


class Linear(nnx.Module):
    """Affine layer with upstream-compatible initialization: PyTorch Linear's default, or orthogonal."""

    def __init__(self, input_features: int, config: LayerConfig, *, rngs: nnx.Rngs) -> None:
        kernel_key, bias_key = rngs.params(), rngs.params()
        if isinstance(config.initialization, OrthogonalInit):
            # Torch initializes (out, in); retain that orientation before transposing.
            kernel = jax.nn.initializers.orthogonal(config.initialization.gain)(
                kernel_key, (config.features, input_features), jnp.float32
            ).T
            bias = jnp.full((config.features,), config.initialization.bias, dtype=jnp.float32)
        else:
            bound = 1.0 / sqrt(input_features)
            kernel = jax.random.uniform(
                kernel_key, (input_features, config.features), jnp.float32, minval=-bound, maxval=bound
            )
            bias = jax.random.uniform(bias_key, (config.features,), jnp.float32, minval=-bound, maxval=bound)
        self.kernel: nnx.Param[Array] = nnx.Param(kernel)
        self.bias: nnx.Param[Array] = nnx.Param(bias)
