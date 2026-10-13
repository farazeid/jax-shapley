"""Explanation models trained alongside DQN agents in Gridworld (paper Figures 3b and 10): a sweep.

Pinned from FastSVERL 25b2cbe `experiments/parallel` (`dqn_{policy,value,performance}_*`): 20 agents
(100,000 steps over 16 environments, learning from the start) with their characteristic,
Shapley and (outcome runs) outcome models updated `update_ratio` times per agent update, and
exact losses measured every `exact_frequency` environment steps. Both exact-characteristic
groups are on (w_exact and w_policy). Run with `python -m new_jax_shapley.continual`.

Swept (MISSING here, supplied by `experiments/continual_gridworld__sweep.sh`): `explain.kind`, `explain.update-ratio`
(1, 2, 10, 50) and `explain.exact-frequency` (1,000 environment steps; 5,000 for outcome).
"""

import tyro

from new_jax_shapley.algorithms.dqn import DQNConfig
from new_jax_shapley.envs import Gridworld
from new_jax_shapley.experiment import ContinualConfig, ContinualExplainConfig

CONFIG = ContinualConfig(
    name="continual_gridworld__sweep",
    algorithm=DQNConfig(
        env=Gridworld(),
        seeds=tuple(range(1, 21)),
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
        exploration_fraction=0.5,
        learning_starts=0,
        train_frequency=10,
    ),
    explain=ContinualExplainConfig(
        kind=tyro.MISSING,
        update_ratio=tyro.MISSING,
        with_exact_char=True,
        outcome_with_exact_char=True,
        exact_frequency=tyro.MISSING,
        exact_buffer_size=10_000,
        hidden_widths=(64, 64),
        learning_rate=2.5e-4,
        tau=0.01,
        gamma=1.0,
    ),
)
