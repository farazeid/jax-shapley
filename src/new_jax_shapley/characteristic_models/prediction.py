"""Step 3: prediction characteristic models, v(s, C) of the agent's value, learnt from masked states.

Everything but the explained output is the behaviour stage's (`characteristic_models.behaviour`):
the target is the agent's value max_a Q(s, a), and the network ends in one unactivated
output, keeping the source's trailing singleton value axis.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from new_jax_shapley.algorithms.dqn import Agent, TransitionBatch
from new_jax_shapley.characteristic_models.behaviour import (
    CharacteristicConfig,
    CharacteristicModel,
    Values,
    stage,
)
from new_jax_shapley.common.networks import LayerConfig
from new_jax_shapley.common.supervised_learning import EpochMetrics, TrainState


def main(
    config: CharacteristicConfig, agents: Agent, replays: TransitionBatch | None = None
) -> tuple[TrainState[CharacteristicModel], EpochMetrics | None]:
    """Prediction characteristic models explaining the agents' value."""
    if config.kind != "prediction":
        raise ValueError("the prediction stage needs kind='prediction'")
    return stage(config, agents, replays, targets, head, "ValueCharacteristicModel")


def targets(config: CharacteristicConfig, agent: Agent, observations: Array) -> Values:
    """`v_F`: the agent's value, max_a Q(s, a)."""
    del config
    return jnp.max(agent.model(observations).q_values, axis=-1, keepdims=True)


def head(actions: int) -> LayerConfig:
    """One unactivated value output."""
    del actions
    return LayerConfig(1)
