"""Whole-run and pure helper numerical equivalence, including PRNG and optimizer."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jax_shapley.envs import Mastermind222
from jax_shapley.networks import dqn_config
from jax_shapley.train_dqn import DQNTrainConfig, evaluate_dqn, initialize_run, run, run_chunk
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
    runner = nnx.jit(
        partial(
            run,
            environment_factory=Mastermind222,
            hidden_widths=(4,),
            config=config,
            chunk_steps=block_size,
            sink=sink,
        )
    )
    compiled = runner.trace(jax.ShapeDtypeStruct((), jax.random.key(seed).dtype), None, None, None).lower().compile()
    assert not sink.training_steps and not sink.checkpoints
    result = compiled(jax.random.key(seed), None, None, None)
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


@pytest.mark.parametrize("missing", ["phase", "evaluation_episodes"])
def test_restored_run_requires_saved_execution_counters(missing: str) -> None:
    config = DQNTrainConfig(total_timesteps=2, buffer_size=2, batch_size=1)
    env = EpisodeStatisticsWrapper(Mastermind222())
    key = jax.random.key(1)
    state = initialize_run(env, env.default_params, dqn_config(8, 4, hidden_widths=(4,)), config, key)
    sink = RecordingSink()
    runner = nnx.jit(
        partial(
            run,
            environment_factory=Mastermind222,
            hidden_widths=(4,),
            config=config,
            chunk_steps=2,
            sink=sink,
        )
    )
    abstract = nnx.eval_shape(lambda value: value, state)
    scalar = jax.ShapeDtypeStruct((), jnp.int32)
    with pytest.raises(ValueError, match="restored state requires saved phase and evaluation counters"):
        runner.trace(
            jax.ShapeDtypeStruct((), key.dtype),
            abstract,
            None if missing == "phase" else scalar,
            None if missing == "evaluation_episodes" else scalar,
        )
    assert not sink.training_steps and not sink.checkpoints and not sink.evaluation_returns
