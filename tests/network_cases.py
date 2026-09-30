"""Typed network cases shared by numerical and compilation tests."""

from dataclasses import dataclass
from typing import Literal

from flax import nnx

from jax_shapley.networks import LayerConfig, MLPConfig, OrthogonalInit, PPOActor, PPOCritic, QNetwork, dqn_config

type AgentNetwork = QNetwork | PPOActor | PPOCritic


@dataclass(frozen=True, slots=True)
class NetworkCase:
    name: str
    kind: Literal["dqn", "actor", "critic"]
    config: MLPConfig


def make_network(case: NetworkCase, seed: int = 0) -> AgentNetwork:
    rngs = nnx.Rngs(seed)
    if case.kind == "actor":
        return PPOActor(case.config, rngs=rngs)
    if case.kind == "critic":
        return PPOCritic(case.config, rngs=rngs)
    return QNetwork(case.config, rngs=rngs)


NETWORK_CASES = (
    NetworkCase("dqn_gridworld", "dqn", dqn_config(2, 4)),
    NetworkCase("dqn_mastermind222", "dqn", dqn_config(8, 4)),
    NetworkCase("dqn_hypercube", "dqn", dqn_config(5, 10, hidden_widths=(128, 128))),
    NetworkCase("dqn_mastermind333", "dqn", dqn_config(15, 27, hidden_widths=(128, 128))),
    # Upstream provides configurable PPO, not paper-specific PPO launch presets.
    # These exercise its supported Tanh, Softmax and explicit layer_init paths.
    NetworkCase(
        "ppo_actor",
        "actor",
        MLPConfig(
            8, (LayerConfig(16, "tanh", OrthogonalInit(2**0.5)), LayerConfig(4, "softmax", OrthogonalInit(0.01)))
        ),
    ),
    NetworkCase(
        "ppo_critic",
        "critic",
        MLPConfig(
            8, (LayerConfig(16, "tanh", OrthogonalInit(2**0.5)), LayerConfig(1, "identity", OrthogonalInit(1.0)))
        ),
    ),
)
