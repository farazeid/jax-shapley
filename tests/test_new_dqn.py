"""The refactored DQN stage (new_jax_shapley) reproduces the current jax_shapley DQN CLI bitwise.

Every seed's agent weights, training episode returns and greedy evaluation returns must
equal one `jax_shapley.train_dqn` run per seed, on each environment family. Evaluation runs
on Mastermind only: Gridworld and Hypercube have no time limit, so a barely trained greedy
agent can loop forever there (as it can in the source).
"""

import csv
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jax_shapley import train_dqn
from jax_shapley.run_io import load_agent, load_settings
from new_jax_shapley.algorithms import dqn
from new_jax_shapley.common.supervised_learning import seed_slice
from new_jax_shapley.envs import Gridworld, Hypercube, Mastermind

SEEDS = (1, 2)
WIDTHS = (16, 16)
CASES = {
    "gridworld": (Gridworld(), ["gridworld"], 0),
    "mastermind222": (Mastermind(2, 2, 2), ["mastermind222"], 5),
    "hypercube": (Hypercube(2, 2), ["hypercube", "--env.side-length", "2", "--env.dimension", "2"], 0),
}


@pytest.mark.parametrize("name", list(CASES))
def test_agents_match_the_current_dqn(name: str, tmp_path: Path) -> None:
    env, subcommand, eval_episodes = CASES[name]
    config = dqn.DQNConfig(
        env=env,
        seeds=SEEDS,
        hidden_widths=WIDTHS,
        num_envs=4,
        total_timesteps=1024,
        buffer_size=1024,
        learning_starts=128,
        batch_size=16,
        exploration_fraction=0.25,
        eval_episodes=eval_episodes,
    )
    state, metrics, agents = dqn.main(config)
    assert metrics is not None
    for index, seed in enumerate(SEEDS):
        output = tmp_path / f"seed-{seed}"
        train_dqn.main([
            *subcommand,
            "--seed",
            str(seed),
            "--hidden-widths",
            *map(str, WIDTHS),
            "--train.num-envs",
            str(config.num_envs),
            "--train.total-timesteps",
            str(config.total_timesteps),
            "--train.buffer-size",
            str(config.buffer_size),
            "--train.learning-starts",
            str(config.learning_starts),
            "--train.batch-size",
            str(config.batch_size),
            "--train.exploration-fraction",
            str(config.exploration_fraction),
            "--train.eval-episodes",
            str(eval_episodes),
            "--output",
            str(output),
        ])
        (run,) = sorted(path for path in output.iterdir() if path.is_dir())
        loaded = load_agent(run, load_settings(run / "config.json").network, load_replay=False)
        old = jax.tree.leaves(nnx.state(loaded.model))
        new = jax.tree.leaves(nnx.state(seed_slice(state, index).model))
        for left, right in zip(old, new, strict=True):
            np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
        np.testing.assert_array_equal(np.asarray(seed_slice(agents, index).epsilon), np.asarray(loaded.epsilon))
        rows = list(csv.DictReader((run / "metrics.csv").open()))
        seed_metrics = seed_slice(metrics, index)
        completed = np.asarray(seed_metrics.completed)
        assert [float(row["episode_return"]) for row in rows if row["kind"] == "episode"] == np.asarray(
            seed_metrics.episode_return
        )[completed].tolist()
        if not eval_episodes:
            continue
        evaluations: list[dqn.EvaluationMetrics] = []
        program = nnx.jit(partial(dqn.train, config, None, partial(_record, evaluations)))
        program(jax.random.key(seed), jnp.asarray(index, jnp.int32))
        jax.effects_barrier()
        new_evaluation = [
            (int(item.first_episode) + offset, value)
            for item in evaluations
            for offset, value in enumerate(np.asarray(item.episode_return)[np.asarray(item.completed)].tolist())
        ]
        old_evaluation = [
            (int(row["step"]), float(row["episode_return"])) for row in rows if row["kind"] == "evaluation"
        ]
        assert len(old_evaluation) >= eval_episodes
        assert new_evaluation == old_evaluation


def _record(items: list[dqn.EvaluationMetrics], seed: jax.Array, metrics: dqn.EvaluationMetrics) -> None:
    del seed
    items.append(metrics)
