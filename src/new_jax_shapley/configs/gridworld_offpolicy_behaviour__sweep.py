"""Behaviour off-policy characteristics in Gridworld (paper Figures 8 and 9): a sweep.

Pinned from FastSVERL 25b2cbe `offpolicy/*/dqn_policy_gwb.py`: 20 agents (agent s trained
with seed s, 15_000 steps filling its replay), and characteristic seed s explaining
agent s's greedy policy, from rollouts or from its replay, with or without
importance weights. Characteristics only: the paper's off-policy figures plot their exact loss.

Swept (MISSING here, supplied by `experiments/gridworld_offpolicy_behaviour__sweep.sh` for each group): the
characteristic's `data`, `off-policy`, `weighting` and `clip-coef`.
"""

import tyro

from new_jax_shapley.algorithms.dqn import DQNConfig
from new_jax_shapley.characteristic_models.behaviour import CharacteristicConfig
from new_jax_shapley.common.supervised_learning import SupervisedConfig
from new_jax_shapley.envs import Gridworld
from new_jax_shapley.experiment import ExperimentConfig

ENV = Gridworld()
SEEDS = tuple(range(1, 21))

CONFIG = ExperimentConfig(
    name="gridworld_offpolicy_behaviour__sweep",
    algorithm=DQNConfig(
        env=ENV,
        seeds=SEEDS,
        hidden_widths=(64, 64),
        num_envs=8,
        total_timesteps=15_000,
        learning_rate=2.5e-4,
        buffer_size=15_000,
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
        data=tyro.MISSING,
        off_policy=tyro.MISSING,
        weighting=tyro.MISSING,
        clip_coef=tyro.MISSING,
        seeds=SEEDS,
        train=SupervisedConfig(
            epochs=50,
            learning_rate=2.5e-4,
            batch_size=128,
            buffer_size=15_000,
            val_fraction=0.2,
            num_envs=8,
            exact_loss=True,
            exact_buffer_size=12_000,
        ),
    ),
    shapley=(),
)
