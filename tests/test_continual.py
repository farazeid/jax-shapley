"""Continual learning: the agent trains as a plain DQN run, explanations follow the source's cadence."""

from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from new_jax_shapley import continual
from new_jax_shapley.algorithms import dqn
from new_jax_shapley.common.tabular import transition_model
from new_jax_shapley.envs import Gridworld
from new_jax_shapley.experiment import ContinualConfig, ContinualExplainConfig, ContinualKind

NUM_ENVS, STEPS, EXACT_FREQUENCY = 4, 640, 80
AGENT = dqn.DQNConfig(
    env=Gridworld(),
    seeds=(1, 2),
    hidden_widths=(16, 16),
    num_envs=NUM_ENVS,
    total_timesteps=STEPS,
    buffer_size=STEPS,
    learning_starts=0,
    batch_size=16,
    exploration_fraction=0.5,
)
EXPLAIN = ContinualExplainConfig(
    update_ratio=2, exact_frequency=EXACT_FREQUENCY, exact_buffer_size=160, hidden_widths=(16, 16)
)


@pytest.mark.parametrize("kind", ["behaviour", "prediction", "outcome"])
def test_agent_trains_as_a_plain_dqn_run(kind: ContinualKind) -> None:
    agent = _run(replace(EXPLAIN, kind=kind)).agent
    _, plain, _ = dqn.main(AGENT)
    assert plain is not None
    # The same episodes, and DQN losses equal to float rounding (the continual run vmaps its
    # seeds' agents; plain runs compile each seed alone).
    for index in range(len(AGENT.seeds)):
        completed = np.asarray(plain.completed)[index]
        np.testing.assert_array_equal(np.asarray(agent.completed)[:, index], completed)
        np.testing.assert_array_equal(
            np.asarray(agent.episode_return)[:, index][completed], np.asarray(plain.episode_return)[index][completed]
        )
        updated = np.asarray(plain.updated)[index]
        np.testing.assert_allclose(
            np.asarray(agent.loss)[:, index][updated], np.asarray(plain.loss)[index][updated], rtol=1e-6
        )


def test_logging_and_exact_cadence_follow_the_source() -> None:
    metrics = _run(replace(EXPLAIN, kind="outcome"))
    steps = np.asarray(metrics.agent.global_step)
    learning = steps > 0
    # Losses log every 100 environment steps, exact losses every `exact_frequency`.
    np.testing.assert_array_equal(np.asarray(metrics.logged), learning & (steps % 100 < NUM_ENVS))
    exact = np.asarray(metrics.exact_step)
    np.testing.assert_array_equal(exact, learning & (steps % EXACT_FREQUENCY < NUM_ENVS))
    for name in ("characteristic_exact", "outcome_exact", "shapley_exact"):
        values = np.asarray(getattr(metrics, name))
        assert np.isfinite(values[exact]).all() and np.isnan(values[~exact]).all()
    logged = np.asarray(metrics.logged)
    for name in ("characteristic_loss", "outcome_loss", "shapley_loss"):
        assert np.isfinite(np.asarray(getattr(metrics, name))[logged]).all()
    # Seeds learn independently.
    losses = np.asarray(metrics.characteristic_loss)[learning[:, 0]]
    assert not np.array_equal(losses[:, 0], losses[:, 1])


def test_update_ratio_changes_only_the_explanations() -> None:
    one = _run(replace(EXPLAIN, update_ratio=1))
    many = _run(replace(EXPLAIN, update_ratio=5))
    for name in ("episode_return", "completed", "loss"):
        np.testing.assert_array_equal(np.asarray(getattr(one.agent, name)), np.asarray(getattr(many.agent, name)))
    assert not np.array_equal(np.asarray(one.shapley_loss), np.asarray(many.shapley_loss))


def test_continual_validation() -> None:
    with pytest.raises(ValueError):
        ContinualExplainConfig(update_ratio=0)
    with pytest.raises(ValueError):
        ContinualConfig(
            name="invalid", algorithm=replace(AGENT, num_envs=3), explain=replace(EXPLAIN, exact_buffer_size=10)
        )
    with pytest.raises(ValueError):
        replace(AGENT, seeds=(1, 1))


def _run(explain: ContinualExplainConfig) -> continual.ContinualMetrics:
    config = ContinualConfig(name="test_continual", algorithm=AGENT, explain=explain)
    keys = jnp.stack([jax.random.key(seed) for seed in AGENT.seeds])
    _, metrics = nnx.jit(partial(continual.fit, config, transition_model(AGENT.env), None))(keys)
    return metrics
