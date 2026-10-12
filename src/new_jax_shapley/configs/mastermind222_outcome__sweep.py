"""Outcome explanations of a DQN agent in Mastermind-222 (paper Figures 1c and 4): a sweep.

Pinned from FastSVERL 25b2cbe: `experiments/fastsverl/agents/dqn_mastermind.py` (agent),
`characteristics/dqn_policy_mastermind.py` (the behaviour characteristic sweep),
`characteristics/dqn_performance_mastermind.py` and `shapleys/dqn_performance_mastermind.py`.
20 outcome seeds condition on behaviour characteristic seed 1 and learn off-policy from the
agent's replay or on-policy by acting (200,000 steps over 16 environments); 20 Shapley
seeds explain outcome seed 1 against the exact outcome values and against the learnt ones.
The paper's Shapley groups explain the off-policy outcome: w_policy (exact behaviour, exact
outcome values), wo_policy (learnt behaviour, exact outcome values) and wo_perf (learnt
behaviour, learnt outcome values).

Swept (MISSING here, supplied by `experiments/mastermind222_outcome__sweep.sh`): the outcome
stage's `variant` and `with-exact-char`, which conditions the exact outcome values on the
exact or the learnt behaviour characteristic.
"""

import tyro

from new_jax_shapley.algorithms.dqn import DQNConfig
from new_jax_shapley.characteristic_models.behaviour import CharacteristicConfig
from new_jax_shapley.characteristic_models.outcome import OutcomeConfig, OutcomeTrainConfig
from new_jax_shapley.common.supervised_learning import SupervisedConfig
from new_jax_shapley.envs import Mastermind
from new_jax_shapley.experiment import ExperimentConfig
from new_jax_shapley.shapley_models.behaviour import ShapleyConfig

ENV = Mastermind(code_size=2, num_guesses=2, num_pegs=2)
SEEDS = tuple(range(1, 21))


def explanation_training(epochs: int) -> SupervisedConfig:
    """Source train_models settings of the behaviour characteristic and Shapley models."""
    return SupervisedConfig(
        epochs=epochs,
        patience=10_000,
        learning_rate=2.5e-4,
        batch_size=128,
        buffer_size=10_000,
        val_fraction=0.2,
        num_envs=16,
        exact_loss=True,
    )


CONFIG = ExperimentConfig(
    name="mastermind222_outcome__sweep",
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
        env=ENV, kind="behaviour", policy="greedy", hidden_widths=(64, 64), seeds=SEEDS, train=explanation_training(50)
    ),
    # Source outcome settings: off-policy, 200,000 steps' worth of batches of 64 from the first
    # 100,000 replay transitions, so epochs = 200,000 * 64 // 100,000.
    outcome=OutcomeConfig(
        env=ENV,
        variant=tyro.MISSING,
        with_exact_char=tyro.MISSING,
        tau=0.01,
        gamma=1.0,
        hidden_widths=(64, 64),
        seeds=SEEDS,
        train=OutcomeTrainConfig(
            epochs=128,
            learning_rate=2.5e-4,
            batch_size=64,
            train_size=100_000,
            validation_size=160,
            exact_loss=True,
            track_batch=True,
            total_timesteps=200_000,
            num_envs=16,
        ),
    ),
    shapley=tuple(
        ShapleyConfig(
            env=ENV,
            hidden_widths=(64, 64),
            with_exact_char=exact,
            buffers="fresh",
            seeds=SEEDS,
            train=explanation_training(100),
        )
        for exact in (True, False)
    ),
    explained_seed=1,
)
