"""Step 4: prediction Shapley models, phi(s) explaining a frozen prediction characteristic.

Everything but the explained characteristic is the behaviour stage's
(`shapley_models.behaviour`): one contribution per feature, of the agent's value.
"""

from __future__ import annotations

from dataclasses import asdict
from functools import partial

from new_jax_shapley.algorithms.dqn import TRACKING_FIELDS, Agent
from new_jax_shapley.characteristic_models import prediction
from new_jax_shapley.characteristic_models.behaviour import CharacteristicConfig, CharacteristicModel
from new_jax_shapley.common.supervised_learning import EpochMetrics, StateBuffers, TrainState
from new_jax_shapley.shapley_models.behaviour import ShapleyConfig, ShapleyModel, fit, stage


def main(
    config: ShapleyConfig,
    explained: CharacteristicConfig,
    agents: Agent,
    characteristics: CharacteristicModel,
    characteristic_buffers: StateBuffers,
) -> tuple[TrainState[ShapleyModel], EpochMetrics | None]:
    """Prediction Shapley models explaining frozen prediction characteristics."""
    if explained.kind != "prediction":
        raise ValueError("the prediction stage explains a prediction characteristic")
    settings = {name: value for name, value in asdict(explained).items() if name not in TRACKING_FIELDS}
    return stage(
        config,
        explained,
        agents,
        characteristics,
        characteristic_buffers,
        partial(fit, explained, prediction.targets, prediction.head),
        prediction.head,
        settings,
        "ValueShapley",
    )
