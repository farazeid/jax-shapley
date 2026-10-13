"""Continual learning (new_jax_shapley) reproduces the current jax_shapley continual CLI bitwise.

At small scale on Gridworld, for every explanation kind at update ratio 2: every logged
and exact step's agent and explanation losses, and every completed episode, must equal the
current `train_continual` run; and the agents must train as plain DQN runs with the same seeds.
"""

import csv
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jax_shapley import train_continual
from new_jax_shapley import continual
from new_jax_shapley.algorithms import dqn
from new_jax_shapley.common.tabular import transition_model
from new_jax_shapley.envs import Gridworld
from new_jax_shapley.experiment import ContinualConfig, ContinualExplainConfig, ContinualKind

SEEDS = (1, 2)
WIDTHS = (16, 16)
KINDS: tuple[ContinualKind, ...] = ("behaviour", "prediction", "outcome")
COLUMNS = (
    ("dqn_loss", "loss"),
    ("characteristic_loss", "characteristic_loss"),
    ("outcome_loss", "outcome_loss"),
    ("shapley_loss", "shapley_loss"),
    ("characteristic_exact_loss", "characteristic_exact"),
    ("outcome_exact_loss", "outcome_exact"),
    ("shapley_exact_loss", "shapley_exact"),
)


@pytest.mark.parametrize("kind", KINDS)
def test_continual_matches_the_current_cli(kind: ContinualKind, tmp_path: Path) -> None:
    config = ContinualConfig(
        name="test_continual",
        algorithm=dqn.DQNConfig(
            env=Gridworld(),
            seeds=SEEDS,
            hidden_widths=WIDTHS,
            num_envs=4,
            total_timesteps=640,
            buffer_size=640,
            learning_starts=0,
            batch_size=16,
            exploration_fraction=0.5,
        ),
        explain=ContinualExplainConfig(
            kind=kind, update_ratio=2, exact_frequency=80, exact_buffer_size=160, hidden_widths=WIDTHS
        ),
    )
    train_continual.main([
        "gridworld",
        "--seeds",
        *map(str, SEEDS),
        "--hidden-widths",
        *map(str, WIDTHS),
        "--train.num-envs",
        "4",
        "--train.total-timesteps",
        "640",
        "--train.buffer-size",
        "640",
        "--train.learning-starts",
        "0",
        "--train.batch-size",
        "16",
        "--train.exploration-fraction",
        "0.5",
        "--train.eval-episodes",
        "0",
        "--explain.kind",
        kind,
        "--explain.update-ratio",
        "2",
        "--explain.exact-frequency",
        "80",
        "--explain.exact-buffer-size",
        "160",
        "--explain.hidden-widths",
        *map(str, WIDTHS),
        "--output",
        str(tmp_path),
    ])
    keys = jnp.stack([jax.random.key(seed) for seed in SEEDS])
    program = nnx.jit(partial(continual.fit, config, transition_model(config.algorithm.env), None))
    state, metrics = program(keys)
    (sweep,) = [path for path in tmp_path.iterdir() if path.is_dir()]
    agent = metrics.agent
    shown = np.asarray(metrics.logged) | np.asarray(metrics.exact_step)
    for index, seed in enumerate(SEEDS):
        rows = list(csv.DictReader((sweep / f"seed-{seed}" / "metrics.csv").open()))
        steps = np.flatnonzero(shown[:, index])
        assert [int(row["step"]) for row in rows] == np.asarray(agent.global_step)[steps, index].tolist()
        assert [float(row["epsilon"]) for row in rows] == np.asarray(agent.epsilon)[steps, index].tolist()
        for column, name in COLUMNS:
            values = np.asarray(getattr(agent if name == "loss" else metrics, name), np.float64)[steps, index]
            logged = np.asarray(metrics.logged)[steps, index] | column.endswith("exact_loss")
            new = [
                float(value) if keep and not np.isnan(value) else None
                for value, keep in zip(values, logged, strict=True)
            ]
            assert [float(row[column]) if row[column] else None for row in rows] == new, column
        episodes = list(csv.DictReader((sweep / f"seed-{seed}" / "episodes.csv").open()))
        completed = np.asarray(agent.completed)[:, index]
        returns = np.asarray(agent.episode_return)[:, index][completed].tolist()
        assert [float(row["episode_return"]) for row in episodes] == returns
    # The agents train as plain DQN runs: the same episodes, and DQN losses equal to float rounding
    # (the continual run vmaps its seeds' agents; plain runs compile each seed alone).
    _, plain, _ = dqn.main(config.algorithm)
    assert plain is not None
    for index in range(len(SEEDS)):
        completed = np.asarray(plain.completed)[index]
        np.testing.assert_array_equal(np.asarray(agent.completed)[:, index], completed)
        np.testing.assert_array_equal(
            np.asarray(agent.episode_return)[:, index][completed], np.asarray(plain.episode_return)[index][completed]
        )
        updated = np.asarray(plain.updated)[index]
        np.testing.assert_allclose(
            np.asarray(agent.loss)[:, index][updated], np.asarray(plain.loss)[index][updated], rtol=1e-6
        )
    del state
