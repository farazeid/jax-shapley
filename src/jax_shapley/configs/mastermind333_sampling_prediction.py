"""Prediction explanations against a learnt characteristic in Mastermind-333, the sampling baseline (paper Figure 11).

Pinned from FastSVERL 25b2cbe `sampling/{characteristics,shapleys}`: one agent; 20 learnt
characteristic seeds; Shapley seed s explaining characteristic seed s on fresh buffers,
against the exact characteristic's Shapley values (w_exact_model) and the learnt one's
(wo_exact_model). The sampled group is `mastermind333_sampled_prediction`.
"""

from jax_shapley.algorithms.dqn import DQNConfig
from jax_shapley.characteristic_models.behaviour import CharacteristicConfig
from jax_shapley.common.supervised_learning import SupervisedConfig
from jax_shapley.envs import Mastermind
from jax_shapley.experiment import ExperimentConfig
from jax_shapley.shapley_models.behaviour import ShapleyConfig

ENV = Mastermind(code_size=3, num_guesses=3, num_pegs=3)
SEEDS = tuple(range(1, 21))

CONFIG = ExperimentConfig(
    name="mastermind333_sampling_prediction",
    algorithm=DQNConfig(
        env=ENV,
        seeds=(1,),
        hidden_widths=(128, 128),
        num_envs=16,
        total_timesteps=1_000_000,
        learning_rate=2.5e-4,
        buffer_size=1_000_000,
        gamma=1.0,
        tau=0.01,
        target_network_frequency=1,
        batch_size=64,
        start_e=1.0,
        end_e=0.01,
        exploration_fraction=0.25,
        learning_starts=10_000,
        train_frequency=10,
    ),
    characteristic=CharacteristicConfig(
        env=ENV,
        kind="prediction",
        hidden_widths=(128, 128),
        seeds=SEEDS,
        train=SupervisedConfig(
            epochs=2000,
            patience=10_000,
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
            hidden_widths=(128, 128),
            with_exact_char=exact,
            buffers="fresh",
            seeds=SEEDS,
            train=SupervisedConfig(
                epochs=500,
                patience=10_000,
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
