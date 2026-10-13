"""Behaviour explanations of DQN agents in large Mastermind games (paper Table 1 and Figure 6): a sweep.

Pinned from FastSVERL 25b2cbe `experiments/fastsverl/*/dqn_*_mastermind.py` at 4 pegs over 3
colours: one agent (10,000,000 steps); 10 characteristic seeds and 10 Shapley seeds explaining
characteristic seed 1 against the learnt characteristic, with no exact loss (the games are too
large). The tables report updates and final training losses.

Swept (MISSING here, supplied by `experiments/mastermind_large_behaviour__sweep.sh` for Mastermind-443, -453 and -463):
every stage's `env.num-guesses`.
"""

import tyro

from jax_shapley.algorithms.dqn import DQNConfig
from jax_shapley.characteristic_models.behaviour import CharacteristicConfig
from jax_shapley.common.supervised_learning import SupervisedConfig
from jax_shapley.envs import Mastermind
from jax_shapley.experiment import ExperimentConfig
from jax_shapley.shapley_models.behaviour import ShapleyConfig

ENV = Mastermind(code_size=4, num_guesses=tyro.MISSING, num_pegs=3)
SEEDS = tuple(range(1, 11))
TRAINING = SupervisedConfig(
    epochs=50_000,
    patience=5_000,
    learning_rate=2.5e-4,
    batch_size=128,
    buffer_size=10_000,
    val_fraction=0.2,
    num_envs=16,
    exact_loss=False,
)

CONFIG = ExperimentConfig(
    name="mastermind_large_behaviour__sweep",
    algorithm=DQNConfig(
        env=ENV,
        seeds=(1,),
        hidden_widths=(128, 128),
        num_envs=16,
        total_timesteps=10_000_000,
        learning_rate=2.5e-4,
        buffer_size=10_000_000,
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
        env=ENV, kind="behaviour", policy="greedy", hidden_widths=(128, 128), seeds=SEEDS, train=TRAINING
    ),
    shapley=(
        ShapleyConfig(
            env=ENV, hidden_widths=(128, 128), with_exact_char=False, buffers="fresh", seeds=SEEDS, train=TRAINING
        ),
    ),
    explained_seed=1,
)
