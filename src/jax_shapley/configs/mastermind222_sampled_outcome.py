"""On-policy outcome characteristics in Mastermind-222, the sampling comparison (paper Figure 11).

Pinned from FastSVERL 25b2cbe `sampling/characteristics/dqn_performance_*`: one agent; 20 behaviour
characteristic seeds; outcome seed s conditioned on behaviour seed s, acting for 1,000,000
steps over 16 environments: the outcome conditions on a sampled behaviour characteristic, its exact values on the exact one (w_exact_sampling). No Shapley models.
"""

from jax_shapley.algorithms.dqn import DQNConfig
from jax_shapley.characteristic_models.behaviour import CharacteristicConfig
from jax_shapley.characteristic_models.outcome import OutcomeConfig, OutcomeTrainConfig
from jax_shapley.common.supervised_learning import SupervisedConfig
from jax_shapley.envs import Mastermind
from jax_shapley.experiment import ExperimentConfig

ENV = Mastermind(code_size=2, num_guesses=2, num_pegs=2)
SEEDS = tuple(range(1, 21))

CONFIG = ExperimentConfig(
    name="mastermind222_sampled_outcome",
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
    outcome=OutcomeConfig(
        env=ENV,
        variant="on_policy",
        with_exact_char=True,
        tau=0.01,
        gamma=1.0,
        hidden_widths=(64, 64),
        seeds=SEEDS,
        train=OutcomeTrainConfig(
            learning_rate=2.5e-4,
            batch_size=64,
            exact_loss=True,
            total_timesteps=1_000_000,
            num_envs=16,
            train_frequency=1,
            target_network_frequency=1,
        ),
    ),
    shapley=(),
    explained_seed=None,
)
