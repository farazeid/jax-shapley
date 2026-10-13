"""Buffer collection, the shared epoch loop's stopping rules, and Shapley buffer reuse."""

from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jax_shapley.algorithms import dqn
from jax_shapley.characteristic_models import behaviour, prediction
from jax_shapley.characteristic_models.behaviour import CharacteristicConfig
from jax_shapley.common.rollouts import collect_buffers, collect_observations
from jax_shapley.common.supervised_learning import EpochMetrics, SupervisedConfig, TrainState, seed_slice, stack_seeds
from jax_shapley.envs import Gridworld, Mastermind
from jax_shapley.envs.wrappers import EpisodeStatisticsWrapper
from jax_shapley.shapley_models import prediction as shapley_prediction
from jax_shapley.shapley_models.behaviour import ShapleyConfig

ENV = Mastermind(2, 2, 2)
SMALL = SupervisedConfig(epochs=3, batch_size=8, buffer_size=40, num_envs=4)


def test_observations_are_recorded_before_each_step_in_step_major_order() -> None:
    config = dqn.DQNConfig(env=Gridworld(), hidden_widths=(4,))
    env = EpisodeStatisticsWrapper(config.env)
    params = env.default_params
    model = dqn.initialize(config, jax.random.key(0)).model
    epsilon, key, num_envs, rows = jnp.asarray(0.3, jnp.float32), jax.random.key(5), 3, 12
    recorded = collect_observations(env, model, epsilon, key, rows, num_envs)
    assert recorded.shape == (rows, 2) and recorded.dtype == jnp.int32
    # Eager replica of the source fill_buffer loop with the same key discipline.
    keys = jax.random.split(key, num_envs + 1)
    observations, state = jax.vmap(env.reset, in_axes=(0, None))(keys[1:], params)
    key = keys[0]
    expected = []
    for _ in range(rows // num_envs):
        expected.append(np.asarray(observations))
        actions, _, key = dqn.choose_action(model, observations, epsilon, key)
        step_keys = jax.random.split(key, num_envs + 1)
        transition = jax.vmap(partial(dqn.environment_step, env), in_axes=(0, 0, 0, None))(
            step_keys[1:], state, actions, params
        )
        observations, state, key = transition.observations, transition.state, step_keys[0]
    np.testing.assert_array_equal(recorded, np.concatenate(expected))


def test_buffers_use_the_train_buffer_for_exact_values_unless_configured() -> None:
    config = dqn.DQNConfig(env=ENV, hidden_widths=(4,))
    env = EpisodeStatisticsWrapper(config.env)
    agent = _agent()
    collect = partial(
        collect_buffers,
        train_size=SMALL.train_size,
        validation_size=SMALL.validation_size,
        num_envs=SMALL.num_envs,
    )
    zero = jnp.asarray(0.0, jnp.float32)
    _, shared = collect(env, agent.model, zero, jax.random.key(0), exact_size=None)
    assert shared.exact is shared.train
    assert shared.train.shape == (SMALL.train_size, 8) and shared.validation.shape == (SMALL.validation_size, 8)
    _, separate = collect(env, agent.model, agent.epsilon, jax.random.key(0), exact_size=24)
    assert separate.exact.shape == (24, 8)
    inherited = jnp.zeros((4, 8), jnp.int32)
    _, reused = collect(env, agent.model, zero, jax.random.key(0), exact_size=24, exact=inherited)
    assert reused.exact is inherited


def test_patience_stops_and_restores_the_best_validation_model() -> None:
    state, metrics = _characteristic(replace(SMALL, epochs=40, patience=2, learning_rate=0.05))
    epochs, best, validation = _rows(metrics, "epoch"), _rows(metrics, "best_epoch"), _rows(metrics, "validation_loss")
    assert epochs.tolist() == list(range(1, len(epochs) + 1)) and int(state.epoch) == epochs[-1] < 40
    assert epochs[-1] - best[-1] >= 2 and (epochs[:-1] - best[:-1] < 2).all()
    assert validation[epochs == best[-1]][0] == validation.min()
    assert int(state.best_epoch) == best[-1] and bool(state.stopped)
    for left, right in zip(
        jax.tree.leaves(nnx.state(state.model)), jax.tree.leaves(nnx.state(state.best)), strict=True
    ):
        np.testing.assert_array_equal(left, right)


def test_exact_condition_and_track_batch_logging() -> None:
    config = replace(SMALL, exact_condition=1e3, track_batch=True)
    state, metrics = _characteristic(config)
    assert int(state.epoch) == 1 and bool(state.stopped)
    assert _rows(metrics, "step").tolist() == [2 * config.batches]  # Source step: (epoch + 1) * batches.
    _, metrics = _characteristic(replace(SMALL, epochs=3, exact_loss=False))
    assert _rows(metrics, "step").tolist() == [1, 2, 3] and np.isnan(_rows(metrics, "exact_loss")).all()


def test_shapley_run_reuses_characteristic_buffers_and_trains() -> None:
    explained = CharacteristicConfig(ENV, kind="prediction", hidden_widths=(4,), train=SMALL)
    agents = stack_seeds([_agent()])
    characteristics, _ = prediction.main(explained, agents)
    config = ShapleyConfig(ENV, hidden_widths=(4,), buffers="characteristic", train=SMALL)
    state, metrics = shapley_prediction.main(config, explained, agents, characteristics.model, characteristics.buffers)
    assert metrics is not None
    for name in ("train", "validation", "exact"):
        np.testing.assert_array_equal(getattr(state.buffers, name), getattr(characteristics.buffers, name))
    assert int(seed_slice(state, 0).epoch) == SMALL.epochs and np.isfinite(_rows(metrics, "exact_loss")).all()
    # Fresh buffers with the characteristic's seed would repeat its rollouts, as the source's do.
    fresh, _ = shapley_prediction.main(
        replace(config, buffers="fresh", seeds=(2,)), explained, agents, characteristics.model, characteristics.buffers
    )
    assert not np.array_equal(fresh.buffers.train, characteristics.buffers.train)
    np.testing.assert_array_equal(fresh.buffers.exact, characteristics.buffers.exact)  # Exact values stay shared.


def test_configuration_arithmetic_and_validation() -> None:
    hypercube = SupervisedConfig(epochs=10_000, val_fraction=16 * 10 / 10_000, exact_buffer_size=100_000)
    assert (hypercube.train_size, hypercube.validation_size, hypercube.batches) == (9840, 160, 76)
    assert hypercube.exact_size == 100_000
    default = SupervisedConfig()
    assert (default.train_size, default.validation_size, default.exact_size, default.patience_epochs) == (
        8000,
        2000,
        8000,
        50,
    )
    for invalid in (
        {"buffer_size": 30},
        {"exact_buffer_size": 30},
        {"val_fraction": 1.0},
        {"exact_condition": 0.1, "exact_loss": False},
        {"batch_size": 9000},
        {"patience": 0},
    ):
        with pytest.raises(ValueError):
            SupervisedConfig(**invalid)  # ty: ignore[invalid-argument-type] -- deliberately invalid settings.


def _characteristic(config: SupervisedConfig) -> tuple[TrainState[behaviour.CharacteristicModel], EpochMetrics]:
    state, metrics = behaviour.main(
        CharacteristicConfig(ENV, hidden_widths=(4,), train=config), stack_seeds([_agent()])
    )
    assert metrics is not None
    return seed_slice(state, 0), metrics


def _rows(metrics: EpochMetrics, name: str) -> np.ndarray:
    """Seed 0's logged epochs of one metric."""
    return np.asarray(getattr(metrics, name))[np.asarray(metrics.active)[:, 0], 0]


def _agent() -> dqn.Agent:
    model = dqn.initialize(dqn.DQNConfig(env=ENV, hidden_widths=(4,)), jax.random.key(0)).model
    return dqn.Agent(model, jnp.asarray(0.2, jnp.float32))
