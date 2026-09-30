"""Compile real DQN graph mutations and typed state with representative shapes."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jax_shapley.algorithms.dqn import evaluate_dqn, initialize_run, run_chunk, update, update_target
from jax_shapley.config import DQNTrainConfig
from jax_shapley.envs import Gridworld, Hypercube, Mastermind222
from jax_shapley.networks import dqn_config
from jax_shapley.replay import append, empty_replay, sample
from jax_shapley.wrappers import EnvironmentProtocol, EpisodeStatisticsWrapper
from tests.test_replay import batch


def _compile_training[State, Params](
    environment: EnvironmentProtocol[State, Params], features: int, actions: int, num_envs: int, chunk_steps: int
) -> None:
    config = DQNTrainConfig(
        num_envs=num_envs,
        total_timesteps=64,
        buffer_size=7,
        batch_size=3,
        learning_starts=0,
        train_frequency=3,
        target_network_frequency=5,
    )
    env = EpisodeStatisticsWrapper(environment)
    state = initialize_run(
        env, env.default_params, dqn_config(features, actions, hidden_widths=(4,)), config, jax.random.key(0)
    )
    spec = jax.ShapeDtypeStruct((), jnp.int32)
    run = nnx.jit(partial(run_chunk, env=env, params=env.default_params, config=config, chunk_steps=chunk_steps))
    compiled = run.trace(state, spec).lower().compile()
    state, metrics = compiled(state, jnp.asarray(chunk_steps - 1, jnp.int32))
    assert int(state.vector_step) == chunk_steps - 1
    assert metrics.loss.shape == (chunk_steps,)
    assert metrics.completed.shape == (chunk_steps, num_envs)
    assert metrics.loss.dtype == jnp.float32
    assert metrics.completed.dtype == jnp.bool_
    assert state.observations.dtype == jnp.int32
    assert state.vector_step.dtype == jnp.int32
    assert jnp.all(jnp.isfinite(metrics.loss))
    if isinstance(environment, Mastermind222):
        evaluate = nnx.jit(partial(evaluate_dqn, env=env, params=env.default_params, num_envs=num_envs, episodes=2))
        result = (
            evaluate
            .trace(state.model, jax.ShapeDtypeStruct((), jax.random.key(0).dtype))
            .lower()
            .compile()(state.model, state.key)
        )
        assert int(result.count) >= 2
        assert result.returns.dtype == jnp.float32


@pytest.mark.compile
@pytest.mark.parametrize("num_envs,chunk_steps", [(1, 3), (3, 4)])
def test_dqn_environment_chunks_compile(num_envs: int, chunk_steps: int) -> None:
    _compile_training(Gridworld(), 2, 4, num_envs, chunk_steps)
    _compile_training(Mastermind222(), 8, 4, num_envs, chunk_steps)
    _compile_training(Hypercube(6, 5), 5, 10, num_envs, chunk_steps)


@pytest.mark.compile
def test_replay_gradient_optimizer_and_target_compile() -> None:
    replay = empty_replay(5, 2)
    data = batch(0, 3)
    compiled_append = (
        jax
        .jit(append)
        .trace(
            jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), replay),
            jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), data),
        )
        .lower()
        .compile()
    )
    replay = compiled_append(replay, data)
    sampled = (
        jax
        .jit(partial(sample, batch_size=7))
        .trace(replay, key=jax.ShapeDtypeStruct((), jax.random.key(0).dtype))
        .lower()
        .compile()
    )
    # The static batch_size parameter must be bound by keyword around sample.
    result, _ = sampled(replay, key=jax.random.key(0))
    config = DQNTrainConfig(batch_size=7)
    env = EpisodeStatisticsWrapper(Gridworld())
    state = initialize_run(env, env.default_params, dqn_config(2, 4, hidden_widths=(4,)), config, jax.random.key(0))
    learn = nnx.jit(partial(update, gamma=1.0))
    loss = (
        learn
        .trace(
            state.model,
            state.target,
            state.optimizer,
            jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), result),
        )
        .lower()
        .compile()(state.model, state.target, state.optimizer, result)
    )
    assert jnp.isfinite(loss.loss)
    copy = nnx.jit(partial(update_target, tau=1.0))
    copy.trace(state.model, state.target).lower().compile()(state.model, state.target)
    np.testing.assert_array_equal(state.model.mlp.layers[0].kernel[...], state.target.mlp.layers[0].kernel[...])


@pytest.mark.gpu
@pytest.mark.compile
def test_dqn_gpu_compile() -> None:
    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        devices = []
    if not devices:
        pytest.skip("DQN GPU compilation requires an available JAX GPU backend")
    with jax.default_device(devices[0]):
        _compile_training(Gridworld(), 2, 4, 3, 4)
        _compile_training(Mastermind222(), 8, 4, 3, 4)
        _compile_training(Hypercube(6, 5), 5, 10, 3, 4)
