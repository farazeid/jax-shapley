"""Prediction explanations against a sampled characteristic in Gridworld (paper Figure 11).

Pinned from FastSVERL 25b2cbe `sampling/{characteristics,shapleys}`: one agent; 20 sampled
characteristic seeds (no model: single draws from their train buffers); Shapley seed s
explaining sampled characteristic seed s on its buffers, against its exact values
(w_exact_sampling). The learnt-characteristic groups are `gridworld_sampling_prediction`.
"""

from jax_shapley.algorithms.dqn import DQNConfig
from jax_shapley.characteristic_models.behaviour import CharacteristicConfig
from jax_shapley.common.supervised_learning import SupervisedConfig
from jax_shapley.envs import Gridworld
from jax_shapley.experiment import ExperimentConfig
from jax_shapley.shapley_models.behaviour import ShapleyConfig

ENV = Gridworld()
SEEDS = tuple(range(1, 21))

CONFIG = ExperimentConfig(
    name="gridworld_sampled_prediction",
    algorithm=DQNConfig(
        env=ENV,
        seeds=(1,),
        hidden_widths=(64, 64),
        num_envs=1,
        total_timesteps=50_000,
        learning_rate=2.5e-4,
        buffer_size=50_000,
        gamma=1.0,
        tau=0.01,
        target_network_frequency=1,
        batch_size=64,
        start_e=1.0,
        end_e=0.01,
        exploration_fraction=0.5,
        learning_starts=10_000,
        train_frequency=10,
    ),
    characteristic=CharacteristicConfig(
        env=ENV,
        kind="prediction",
        hidden_widths=(64, 64),
        estimator="sampled",
        seeds=SEEDS,
        train=SupervisedConfig(
            epochs=30,
            learning_rate=2.5e-4,
            batch_size=128,
            buffer_size=10_000,
            val_fraction=0.2,
            num_envs=16,
            exact_loss=False,
        ),
    ),
    shapley=(
        ShapleyConfig(
            env=ENV,
            hidden_widths=(64, 64),
            with_exact_char=True,
            buffers="characteristic",
            seeds=SEEDS,
            train=SupervisedConfig(
                epochs=20,
                learning_rate=2.5e-4,
                batch_size=128,
                buffer_size=10_000,
                val_fraction=0.2,
                num_envs=16,
                exact_loss=True,
            ),
        ),
    ),
    explained_seed=None,
)
