"""Behaviour explanations against a sampled characteristic in Mastermind-222 (paper Figure 3c and 11).

Pinned from FastSVERL 25b2cbe `sampling/{characteristics,shapleys}`: one agent; 20 sampled
characteristic seeds (no model: single draws from their train buffers); Shapley seed s
explaining sampled characteristic seed s on its buffers, against its exact values
(w_exact_sampling). The learnt-characteristic groups are `mastermind222_sampling_behaviour`.
"""

from new_jax_shapley.algorithms.dqn import DQNConfig
from new_jax_shapley.characteristic_models.behaviour import CharacteristicConfig
from new_jax_shapley.common.supervised_learning import SupervisedConfig
from new_jax_shapley.envs import Mastermind
from new_jax_shapley.experiment import ExperimentConfig
from new_jax_shapley.shapley_models.behaviour import ShapleyConfig

ENV = Mastermind(code_size=2, num_guesses=2, num_pegs=2)
SEEDS = tuple(range(1, 21))

CONFIG = ExperimentConfig(
    name="mastermind222_sampled_behaviour",
    algorithm=DQNConfig(
        env=ENV,
        seeds=(1,),
        hidden_widths=(64, 64),
        num_envs=16,
        total_timesteps=100_000,
        learning_rate=2.5e-4,
        buffer_size=100_000,
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
        kind="behaviour",
        policy="greedy",
        hidden_widths=(64, 64),
        estimator="sampled",
        seeds=SEEDS,
        train=SupervisedConfig(
            epochs=50,
            patience=10_000,
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
                epochs=50,
                patience=10_000,
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
