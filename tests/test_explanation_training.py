"""Compiled explanation loop semantics: buffers, early stopping, exact stopping and logging steps."""

from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jax_shapley.characteristics import CharacteristicConfig, CharacteristicModel, characteristic_network
from jax_shapley.envs import Gridworld, Mastermind222
from jax_shapley.networks import QNetwork, dqn_config
from jax_shapley.train_dqn import choose_action, environment_step
from jax_shapley.train_explanations import (
    CharacteristicRunConfig,
    ExplanationPhase,
    ExplanationState,
    ExplanationTrainConfig,
    FrozenAgent,
    ShapleyRunConfig,
    collect_buffers,
    collect_observations,
    run_characteristic,
    run_shapley,
    seed_slice,
    stack_seeds,
)
from jax_shapley.wrappers import EpisodeStatisticsWrapper
from tests.explanation_support import RecordingSink

SMALL = ExplanationTrainConfig(epochs=8, batch_size=8, buffer_size=40, num_envs=4)


def test_observations_are_recorded_before_each_step_in_step_major_order() -> None:
    env = EpisodeStatisticsWrapper(Gridworld())
    params = env.default_params
    model = QNetwork(dqn_config(2, 4, hidden_widths=(4,)), rngs=nnx.Rngs(0))
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
        actions, key = choose_action(model, observations, epsilon, key)
        step_keys = jax.random.split(key, num_envs + 1)
        transition = jax.vmap(partial(environment_step, env), in_axes=(0, 0, 0, None))(
            step_keys[1:], state, actions.actions, params
        )
        observations, state, key = transition.observations, transition.state, step_keys[0]
    np.testing.assert_array_equal(recorded, np.concatenate(expected))


def test_buffers_use_the_train_buffer_for_exact_values_unless_configured() -> None:
    env = EpisodeStatisticsWrapper(Mastermind222())
    agent = _agent()
    _, shared = collect_buffers(env, agent, "greedy", SMALL, jax.random.key(0), None)
    assert shared.exact is shared.train
    assert shared.train.shape == (SMALL.train_size, 8) and shared.validation.shape == (SMALL.validation_size, 8)
    separate = replace(SMALL, exact_buffer_size=24)
    _, buffers = collect_buffers(env, agent, "explore", separate, jax.random.key(0), None)
    assert buffers.exact.shape == (24, 8)
    inherited = jnp.zeros((4, 8), jnp.int32)
    _, reused = collect_buffers(env, agent, "greedy", SMALL, jax.random.key(0), inherited)
    assert reused.exact is inherited


def test_patience_stops_and_restores_the_best_validation_model() -> None:
    config = replace(SMALL, epochs=40, patience=2, learning_rate=0.05)
    sink, result = _characteristic(config)
    epochs = [row[0] for row in sink.rows]
    assert epochs == list(range(1, len(epochs) + 1)) and int(result.epoch) == epochs[-1] < 40
    final, best = sink.rows[-1][0], sink.rows[-1][5]
    assert final - best >= 2 and all(row[0] - row[5] < 2 for row in sink.rows[:-1])
    validation = {row[0]: row[3] for row in sink.rows}
    assert validation[best] == min(validation.values())
    assert int(result.best_epoch) == best and bool(result.stopped)
    assert int(result.phase) == ExplanationPhase.COMPLETE
    for left, right in zip(
        jax.tree.leaves(nnx.state(result.model)), jax.tree.leaves(nnx.state(result.best)), strict=True
    ):
        np.testing.assert_array_equal(left, right)
    assert sink.checkpoints[-1] == (final, ExplanationPhase.COMPLETE)


def test_exact_condition_and_track_batch_logging() -> None:
    config = replace(SMALL, exact_condition=1e3, track_batch=True)
    sink, result = _characteristic(config)
    assert int(result.epoch) == 1 and bool(result.stopped)
    assert sink.rows[0][1] == 2 * config.batches  # Source step: (epoch + 1) * batches.
    sink, _ = _characteristic(replace(SMALL, epochs=3, exact_loss=False))
    assert [row[1] for row in sink.rows] == [1, 2, 3] and all(np.isnan(row[4]) for row in sink.rows)


def test_checkpoints_every_tenth_of_the_epochs_then_on_completion() -> None:
    sink, _ = _characteristic(replace(SMALL, epochs=20), chunk_epochs=3)
    assert sink.checkpoints == [(3, 0), (6, 0), (9, 0), (12, 0), (15, 0), (18, 0), (20, ExplanationPhase.COMPLETE)]
    assert [row[0] for row in sink.rows] == list(range(1, 21))


def test_shapley_run_reuses_characteristic_buffers_and_trains() -> None:
    env_factory = Mastermind222
    agent = _agent()
    characteristic = CharacteristicModel(
        characteristic_network(8, 4, CharacteristicConfig("prediction", hidden_widths=(4,))), rngs=nnx.Rngs(1)
    )
    _, inherited = collect_buffers(
        EpisodeStatisticsWrapper(env_factory()), agent, "greedy", SMALL, jax.random.key(3), None
    )
    sink = RecordingSink()
    config = ShapleyRunConfig(
        agent=_placeholder(), characteristic=_placeholder(), buffers="characteristic", hidden_widths=(4,), train=SMALL
    )
    runner = nnx.jit(
        partial(
            run_shapley,
            environment_factory=env_factory,
            config=config,
            explained=CharacteristicConfig("prediction"),
            max_unique=SMALL.exact_size,
            sink=sink,
        )
    )
    batched = runner(_keys(0), stack_seeds([agent]), stack_seeds([characteristic]), stack_seeds([inherited]), None)
    result = cast(ExplanationState[CharacteristicModel], seed_slice(batched, 0))  # One-seed sweep.
    np.testing.assert_array_equal(result.buffers.train, inherited.train)
    assert int(result.epoch) == SMALL.epochs and all(np.isfinite(row[4]) for row in sink.rows)


def test_configuration_arithmetic_and_validation() -> None:
    hypercube = ExplanationTrainConfig(epochs=10_000, val_fraction=16 * 10 / 10_000, exact_buffer_size=100_000)
    assert (hypercube.train_size, hypercube.validation_size, hypercube.batches) == (9840, 160, 76)
    assert hypercube.exact_size == 100_000 and hypercube.checkpoint_epochs == 1000
    default = ExplanationTrainConfig()
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
            ExplanationTrainConfig(**invalid)  # ty: ignore[invalid-argument-type] -- deliberately invalid settings.
    with pytest.raises(ValueError):
        CharacteristicRunConfig(agent=_placeholder(), chunk_epochs=0)


def _characteristic(
    config: ExplanationTrainConfig, *, chunk_epochs: int = 1
) -> tuple[RecordingSink, ExplanationState[CharacteristicModel]]:
    sink = RecordingSink()
    run = CharacteristicRunConfig(agent=_placeholder(), hidden_widths=(4,), train=config, chunk_epochs=chunk_epochs)
    runner = nnx.jit(
        partial(
            run_characteristic,
            environment_factory=Mastermind222,
            config=run,
            max_unique=config.exact_size,
            sink=sink,
        )
    )
    result = runner(_keys(0), stack_seeds([_agent()]), None, None, None)
    return sink, seed_slice(result, 0)


def _keys(*seeds: int) -> jax.Array:
    return jnp.stack([jax.random.key(seed) for seed in seeds])


def _agent() -> FrozenAgent:
    return FrozenAgent(QNetwork(dqn_config(8, 4, hidden_widths=(4,)), rngs=nnx.Rngs(0)), jnp.asarray(0.2, jnp.float32))


def _placeholder() -> Path:
    """Run configs require artifact paths; these direct runs pass the models explicitly instead."""
    return Path("unused")
