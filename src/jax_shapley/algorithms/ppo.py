"""FastSVERL's PPO actor and critic networks (no PPO training: the paper reports DQN agents only).

Architectures are supplied explicitly, including the actor's Softmax and FastSVERL's
orthogonal `layer_init` overrides (`OrthogonalInit`).
"""

from typing import NamedTuple

from flax import nnx
from jax import Array
from jaxtyping import Float

from jax_shapley.common.networks import MLP, MLPConfig, ModelInput

type ActionProbabilities = Float[Array, "... actions"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type StateValues = Float[Array, "... 1"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.


class PolicyOutput(NamedTuple):
    """Action probabilities, matching Categorical(probs=...) upstream."""

    probabilities: ActionProbabilities


class ValueOutput(NamedTuple):
    """The critic's value with the upstream trailing singleton axis."""

    value: StateValues


class PPOActor(nnx.Module):
    """Independent PPO actor ending in Softmax probabilities."""

    def __init__(self, config: MLPConfig, *, rngs: nnx.Rngs) -> None:
        if config.layers[-1].activation != "softmax":
            raise ValueError("PPO actor output must use softmax probabilities")
        self.mlp = MLP(config, rngs=rngs)

    def __call__(self, observations: ModelInput) -> PolicyOutput:
        return PolicyOutput(self.mlp(observations))


class PPOCritic(nnx.Module):
    """Independent PPO critic: one unactivated output."""

    def __init__(self, config: MLPConfig, *, rngs: nnx.Rngs) -> None:
        if config.output_features != 1 or config.layers[-1].activation != "identity":
            raise ValueError("PPO critic requires one unactivated output")
        self.mlp = MLP(config, rngs=rngs)

    def __call__(self, observations: ModelInput) -> ValueOutput:
        return ValueOutput(self.mlp(observations))
