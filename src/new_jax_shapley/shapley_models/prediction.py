"""Step 4: prediction Shapley models, phi(s) explaining a frozen prediction characteristic.

Everything but the explained characteristic is the behaviour stage's
(`shapley_models.behaviour`): one contribution per feature, of the agent's value.
"""

from __future__ import annotations

from new_jax_shapley.algorithms.dqn import Agent
from new_jax_shapley.characteristic_models import prediction
from new_jax_shapley.characteristic_models.behaviour import CharacteristicConfig, CharacteristicModel
from new_jax_shapley.common.supervised_learning import EpochMetrics, StateBuffers, TrainState
from new_jax_shapley.shapley_models.behaviour import ShapleyConfig, ShapleyModel, stage


def main(
    config: ShapleyConfig,
    explained: CharacteristicConfig,
    agent: Agent,
    characteristic: CharacteristicModel,
    characteristic_buffers: StateBuffers,
) -> tuple[TrainState[ShapleyModel], EpochMetrics | None]:
    """Prediction Shapley models explaining one frozen prediction characteristic."""
    if explained.kind != "prediction":
        raise ValueError("the prediction stage explains a prediction characteristic")
    return stage(
        config,
        explained,
        agent,
        characteristic,
        characteristic_buffers,
        prediction.targets,
        prediction.head,
        "ValueShapley",
    )
