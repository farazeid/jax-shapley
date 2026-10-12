"""On-policy outcome characteristics in Gridworld, the sampling comparison (paper Figure 11).

Pinned from FastSVERL 25b2cbe `sampling/characteristics/dqn_performance_*`: one agent; 20 behaviour
characteristic seeds; outcome seed s conditioned on behaviour seed s, acting for 100,000
steps over 16 environments: the outcome conditions on a sampled behaviour characteristic, its exact values on the exact one (w_exact_sampling). No Shapley models.
"""

from new_jax_shapley.algorithms.dqn import DQNConfig
from new_jax_shapley.characteristic_models.behaviour import CharacteristicConfig
from new_jax_shapley.characteristic_models.outcome import OutcomeConfig, OutcomeTrainConfig
from new_jax_shapley.common.supervised_learning import SupervisedConfig
from new_jax_shapley.envs import Gridworld
from new_jax_shapley.experiment import ExperimentConfig

ENV = Gridworld()
SEEDS = tuple(range(1, 21))

CONFIG = ExperimentConfig(
    name="gridworld_sampled_outcome",
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
        kind="behaviour",
        policy="greedy",
        hidden_widths=(64, 64),
        estimator="sampled",
        seeds=SEEDS,
        train=SupervisedConfig(
            epochs=50,
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
        tau=0.001,
        gamma=1.0,
        hidden_widths=(64, 64),
        seeds=SEEDS,
        train=OutcomeTrainConfig(
            learning_rate=2.5e-4,
            batch_size=64,
            exact_loss=True,
            total_timesteps=100_000,
            num_envs=16,
            train_frequency=1,
            target_network_frequency=1,
        ),
    ),
    shapley=(),
    explained_seed=None,
)
