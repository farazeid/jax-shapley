"""Prediction explanations of a DQN agent in Mastermind-333 (paper Figure 4).

Pinned from FastSVERL 25b2cbe: `experiments/fastsverl/agents/dqn_mastermind.py` (agent, 3-3-3,
1,000,000 steps), `characteristics/dqn_value_mastermind.py` and `shapleys/dqn_value_mastermind.py`.
One agent (seed 1); 20 characteristic seeds; 20 Shapley seeds explaining characteristic
seed 1, against the exact characteristic's Shapley values and against the learnt one's.
"""

from new_jax_shapley.algorithms.dqn import DQNConfig
from new_jax_shapley.characteristic_models.behaviour import CharacteristicConfig
from new_jax_shapley.common.supervised_learning import SupervisedConfig
from new_jax_shapley.envs import Mastermind
from new_jax_shapley.experiment import ExperimentConfig
from new_jax_shapley.shapley_models.behaviour import ShapleyConfig

ENV = Mastermind(code_size=3, num_guesses=3, num_pegs=3)
SEEDS = tuple(range(1, 21))

CONFIG = ExperimentConfig(
    name="mastermind333_prediction",
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
    explained_seed=1,
)
