"""Prediction explanations against a learnt characteristic in Gridworld, the sampling baseline (paper Figure 11).

Pinned from FastSVERL 25b2cbe `sampling/{characteristics,shapleys}`: one agent; 20 learnt
characteristic seeds; Shapley seed s explaining characteristic seed s on fresh buffers,
against the exact characteristic's Shapley values (w_exact_model) and the learnt one's
(wo_exact_model). The sampled group is `gridworld_sampled_prediction`.
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
    name="gridworld_sampling_prediction",
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
        seeds=SEEDS,
        train=SupervisedConfig(
            epochs=30,
            learning_rate=2.5e-4,
            batch_size=128,
            buffer_size=10_000,
            val_fraction=0.2,
            num_envs=16,
            exact_loss=True,
        ),
    ),
    shapley=tuple(
        ShapleyConfig(
            env=ENV,
            hidden_widths=(64, 64),
            with_exact_char=exact,
            buffers="fresh",
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
        )
        for exact in (True, False)
    ),
    explained_seed=None,
)
