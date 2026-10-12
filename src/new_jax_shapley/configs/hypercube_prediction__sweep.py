"""Prediction explanations of per-seed DQN agents in Hypercubes (paper Figures 2 and 5): a sweep.

Pinned from FastSVERL 25b2cbe `hypercube/hypercube_value.py`: for one cube, 20 agents
(agent s trained with seed s), characteristic seed s explaining agent s, and Shapley seed
s explaining characteristic seed s. Characteristic models stop once their exact loss falls
below 0.01, Shapley models below 0.01; the bars report how many updates that took.

Swept (MISSING here, supplied by `experiments/hypercube_prediction__sweep.sh` for each of the
paper's 20 cubes): each stage's `env.side-length` and `env.dimension`, and the agent's
`total-timesteps`, `buffer-size` (= timesteps) and `learning-starts` (= timesteps / 50).
"""

from dataclasses import replace

import tyro

from new_jax_shapley.algorithms.dqn import DQNConfig
from new_jax_shapley.characteristic_models.behaviour import CharacteristicConfig
from new_jax_shapley.common.supervised_learning import SupervisedConfig
from new_jax_shapley.envs import Hypercube
from new_jax_shapley.experiment import ExperimentConfig
from new_jax_shapley.shapley_models.behaviour import ShapleyConfig

ENV = Hypercube(side_length=tyro.MISSING, dimension=tyro.MISSING)
SEEDS = tuple(range(1, 21))
# Source train_models settings shared by the characteristic and Shapley models.
EXPLANATION_TRAINING = SupervisedConfig(
    epochs=10_000,
    learning_rate=2.5e-4,
    batch_size=128,
    buffer_size=10_000,
    val_fraction=0.016,
    num_envs=16,
    exact_loss=True,
    track_batch=True,
)

CONFIG = ExperimentConfig(
    name="hypercube_prediction__sweep",
    algorithm=DQNConfig(
        env=ENV,
        seeds=SEEDS,
        hidden_widths=(128, 128),
        num_envs=16,
        total_timesteps=tyro.MISSING,
        learning_rate=2.5e-4,
        buffer_size=tyro.MISSING,
        gamma=1.0,
        tau=0.01,
        target_network_frequency=1,
        batch_size=64,
        start_e=1.0,
        end_e=0.05,
        exploration_fraction=0.5,
        learning_starts=tyro.MISSING,
        train_frequency=10,
    ),
    characteristic=CharacteristicConfig(
        env=ENV,
        kind="prediction",
        hidden_widths=(128, 128),
        seeds=SEEDS,
        train=replace(EXPLANATION_TRAINING, exact_buffer_size=100_000, exact_condition=0.01),
    ),
    shapley=(
        ShapleyConfig(
            env=ENV,
            hidden_widths=(128, 128),
            with_exact_char=False,
            buffers="characteristic",
            seeds=SEEDS,
            train=replace(EXPLANATION_TRAINING, exact_condition=0.01),
        ),
    ),
    explained_seed=None,
)
