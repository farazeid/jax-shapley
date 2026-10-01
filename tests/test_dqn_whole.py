"""Whole-run and pure helper numerical equivalence, including PRNG and optimizer."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jax_shapley.algorithms.dqn import evaluate_dqn, initialize_run, run_chunk, whole_run
from jax_shapley.config import DQNTrainConfig
from jax_shapley.envs import Mastermind222
from jax_shapley.networks import dqn_config
from jax_shapley.wrappers import EpisodeStatisticsWrapper
from tests.dqn_support import RecordingSink


@pytest.mark.parametrize("seed,block_size", [(1, 1), (7, 3), (1, 7)])
def test_whole_run_matches_chunk_helpers(seed: int, block_size: int) -> None:
    config = DQNTrainConfig(
        num_envs=2,
        total_timesteps=32,
        buffer_size=7,
        batch_size=3,
        learning_starts=0,
        train_frequency=3,
        target_network_frequency=5,
        eval_episodes=2,
    )
    network = dqn_config(8, 4, hidden_widths=(4,))
    env = EpisodeStatisticsWrapper(Mastermind222())
    sink = RecordingSink()
    run = nnx.jit(
        partial(
            whole_run,
            env=env,
            params=env.default_params,
            network=network,
            config=config,
            chunk_steps=block_size,
            checkpoint_mode="run",
            sink=sink,
        )
    )
    compiled = run.trace(jax.ShapeDtypeStruct((), jax.random.key(seed).dtype)).lower().compile()
    assert not sink.training_steps and not sink.checkpoints
    result = compiled(jax.random.key(seed))
    jax.block_until_ready(result.run.vector_step)
    jax.effects_barrier()
    reference = initialize_run(env, env.default_params, network, config, jax.random.key(seed))
    reference, _ = nnx.jit(
        partial(run_chunk, env=env, params=env.default_params, config=config, chunk_steps=config.vector_steps)
    )(reference, jnp.asarray(config.vector_steps, jnp.int32))
    evaluation = nnx.jit(
        partial(
            evaluate_dqn, env=env, params=env.default_params, num_envs=config.num_envs, episodes=config.eval_episodes
        )
    )(reference.model, reference.key)
    for left, right in zip(
        jax.tree.leaves(nnx.state((reference.model, reference.target, reference.optimizer))),
        jax.tree.leaves(nnx.state((result.run.model, result.run.target, result.run.optimizer))),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)
    for left, right in zip(
        jax.tree.leaves((
            reference.replay,
            reference.observations,
            reference.env_state,
            reference.vector_step,
            reference.updates,
            reference.target_updates,
        )),
        jax.tree.leaves((
            result.run.replay,
            result.run.observations,
            result.run.env_state,
            result.run.vector_step,
            result.run.updates,
            result.run.target_updates,
        )),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(jax.random.key_data(evaluation.key), jax.random.key_data(result.run.key))
    np.testing.assert_array_equal(sink.evaluation_returns, evaluation.returns[: int(evaluation.count)])
    assert sink.training_steps == list(range(0, 32, 2))
    assert sink.largest_batch <= block_size
    assert sink.checkpoints[-2:] == [(16, 1), (16, 2)]
    assert int(result.pending_count) == 0
