"""Behaviour explanations of a DQN agent in Mastermind-222 (paper Figures 1a and 4).

Pinned from FastSVERL 25b2cbe: `experiments/fastsverl/agents/dqn_mastermind.py` (agent),
`characteristics/dqn_policy_mastermind.py` and `shapleys/dqn_policy_mastermind.py`.
One agent (seed 1); 20 characteristic seeds explaining it; 20 Shapley seeds explaining
characteristic seed 1, with the exact loss measured against the exact characteristic's
Shapley values ("approx characteristic") and against the learnt one's ("exact characteristic").
"""

from new_jax_shapley.algorithms.dqn import DQNConfig
from new_jax_shapley.characteristic_models.behaviour import CharacteristicConfig
from new_jax_shapley.common.supervised_learning import SupervisedConfig
from new_jax_shapley.envs import Mastermind
from new_jax_shapley.experiment import ExperimentConfig
from new_jax_shapley.shapley_models.behaviour import ShapleyConfig

ENV = Mastermind(code_size=2, num_guesses=2, num_pegs=2)
SEEDS = tuple(range(1, 21))
# Source train_models settings shared by the characteristic and Shapley models.
EXPLANATION_TRAINING = SupervisedConfig(
    epochs=50,
    patience=10_000,
    learning_rate=2.5e-4,
    batch_size=128,
    buffer_size=10_000,
    val_fraction=0.2,
    num_envs=16,
    exact_loss=True,
)

CONFIG = ExperimentConfig(
    name="mastermind222_behaviour",
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
        env=ENV, kind="behaviour", policy="greedy", hidden_widths=(64, 64), seeds=SEEDS, train=EXPLANATION_TRAINING
    ),
    shapley=(
        ShapleyConfig(
            env=ENV,
            hidden_widths=(64, 64),
            with_exact_char=True,
            buffers="fresh",
            seeds=SEEDS,
            train=EXPLANATION_TRAINING,
        ),
        ShapleyConfig(
            env=ENV,
            hidden_widths=(64, 64),
            with_exact_char=False,
            buffers="fresh",
            seeds=SEEDS,
            train=EXPLANATION_TRAINING,
        ),
    ),
    explained_seed=1,
)
