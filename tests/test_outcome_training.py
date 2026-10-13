"""Outcome stages: replay prefixes, logged exact losses, the on-policy loop and outcome Shapley models."""

from dataclasses import replace
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_shapley.algorithms import dqn
from jax_shapley.characteristic_models import behaviour, outcome
from jax_shapley.characteristic_models.behaviour import CharacteristicConfig, CharacteristicModel
from jax_shapley.characteristic_models.outcome import (
    OnPolicyState,
    OutcomeConfig,
    OutcomeExplained,
    OutcomeFrozen,
    OutcomeNetworks,
    OutcomeTrainConfig,
)
from jax_shapley.common.coalitions import coalition_table
from jax_shapley.common.supervised_learning import (
    EpochMetrics,
    SupervisedConfig,
    TrainState,
    seed_slice,
    stack_seeds,
)
from jax_shapley.common.tabular import transition_model
from jax_shapley.envs import Gridworld
from jax_shapley.shapley_models import behaviour as shapley
from jax_shapley.shapley_models import outcome as outcome_shapley
from jax_shapley.shapley_models.behaviour import ExactShapley, ShapleyConfig, ShapleyContext

ENV = Gridworld()
EXPLAINED = CharacteristicConfig(
    ENV,
    policy="explore",
    hidden_widths=(8,),
    train=SupervisedConfig(epochs=3, buffer_size=40, num_envs=4, batch_size=8),
)
OFF_POLICY = OutcomeConfig(
    ENV,
    tau=0.01,
    hidden_widths=(8,),
    train=OutcomeTrainConfig(epochs=4, train_size=32, validation_size=8, batch_size=8),
)
ON_POLICY = replace(
    OFF_POLICY, variant="on_policy", train=OutcomeTrainConfig(total_timesteps=400, num_envs=8, batch_size=16)
)


class Explained(NamedTuple):
    """A small Gridworld agent, its replay, and an explore-policy behaviour characteristic of it (one seed)."""

    agents: dqn.Agent
    replays: dqn.TransitionBatch
    behaviours: TrainState[CharacteristicModel]


def test_outcome_run_logs_the_exact_loss_of_its_restored_model(explained: Explained) -> None:
    config = replace(OFF_POLICY, with_exact_char=False)
    state, metrics = _off_policy(config, explained)
    assert _rows(metrics, "epoch").tolist() == [1, 2, 3, 4]
    assert _rows(metrics, "step").tolist() == [8, 12, 16, 20]  # track_batch: (epoch + 1) * batches.
    # Recompute the ground truth eagerly from the inputs and score the restored best model.
    trained = seed_slice(state, 0)
    agent, characteristic = seed_slice(explained.agents, 0), seed_slice(explained.behaviours, 0)
    exact = outcome.exact_outcome(
        transition_model(ENV),
        characteristic.buffers.exact,
        OutcomeFrozen(agent, characteristic.model),
        EXPLAINED,
        config.with_exact_char,
        config.gamma,
    )
    eager = outcome.exact_values_loss(
        config, EXPLAINED, OutcomeExplained(agent, characteristic.model, trained.model), exact
    )
    best = _rows(metrics, "epoch") == int(trained.best_epoch)
    np.testing.assert_allclose(_rows(metrics, "exact_loss")[best], [eager], rtol=1e-5)


def test_outcome_buffers_are_the_agent_replay_prefixes(explained: Explained) -> None:
    state, _ = _off_policy(OFF_POLICY, explained)
    buffers, replay = seed_slice(state, 0).buffers, seed_slice(explained.replays, 0)
    # The source's overlapping prefixes: train is replay[:train_size], validation replay[:validation_size].
    np.testing.assert_array_equal(buffers.train, replay.observations[:32])
    np.testing.assert_array_equal(buffers.validation, replay.observations[:8])
    np.testing.assert_array_equal(buffers.exact, seed_slice(explained.behaviours, 0).buffers.exact)


def test_outcome_seeds_equal_single_seed_runs(explained: Explained) -> None:
    pair = Explained(
        stack_seeds([seed_slice(explained.agents, 0)] * 2),
        stack_seeds([seed_slice(explained.replays, 0)] * 2),
        stack_seeds([seed_slice(explained.behaviours, 0)] * 2),
    )
    _, both = _off_policy(replace(OFF_POLICY, seeds=(1, 2)), pair)
    _, single = _off_policy(replace(OFF_POLICY, seeds=(2,)), explained)
    for name in ("epoch", "step", "train_loss", "validation_loss", "exact_loss", "best_epoch"):
        active = np.asarray(both.active)[:, 1]
        np.testing.assert_allclose(
            np.asarray(getattr(both, name))[active, 1], _rows(single, name), rtol=1e-5, atol=1e-6
        )


@pytest.mark.parametrize("with_exact_char", [True, False])
def test_shapley_explains_an_outcome_run(explained: Explained, with_exact_char: bool) -> None:
    state, _ = _off_policy(OFF_POLICY, explained)
    outcomes = OutcomeExplained(explained.agents, explained.behaviours.model, state.model)
    training = SupervisedConfig(epochs=3, buffer_size=40, num_envs=4, batch_size=8)
    config = ShapleyConfig(ENV, hidden_widths=(8,), with_exact_char=with_exact_char, train=training)
    result, metrics = outcome_shapley.main(
        config, EXPLAINED, OFF_POLICY, explained.agents, outcomes, explained.behaviours.buffers
    )
    assert metrics is not None
    # Score the restored best Shapley model eagerly against the ground truth the run should use.
    trained, explained_outcome = seed_slice(result, 0), seed_slice(outcomes, 0)
    exact = outcome.exact_outcome(
        transition_model(ENV),
        seed_slice(explained.behaviours, 0).buffers.exact,
        OutcomeFrozen(explained_outcome.agent, explained_outcome.behaviour),
        EXPLAINED,
        OFF_POLICY.with_exact_char,
        OFF_POLICY.gamma,
    )
    values = partial(outcome.values, OFF_POLICY, EXPLAINED)
    every = jnp.asarray(coalition_table(2))
    table = (
        exact.values[..., None] if with_exact_char else values(explained_outcome, exact.states[None], every[:, None])
    )
    context = ShapleyContext(
        trained.buffers, ExactShapley(exact.states, exact.state_valid, shapley.exact_shapley(table))
    )
    eager = shapley.loss(
        values,
        partial(outcome.null, OFF_POLICY),
        trained.model,
        explained_outcome,
        context,
        training,
        2,
        jax.random.key(0),
        "exact",
    )
    best = _rows(metrics, "epoch") == int(trained.best_epoch)
    np.testing.assert_allclose(_rows(metrics, "exact_loss")[best], [eager], rtol=1e-5)
    with pytest.raises(ValueError, match="fresh buffers"):
        outcome_shapley.main(
            replace(config, buffers="characteristic"),
            EXPLAINED,
            OFF_POLICY,
            explained.agents,
            outcomes,
            explained.behaviours.buffers,
        )


def test_on_policy_run_follows_train_v_on_policy(explained: Explained) -> None:
    state, metrics = outcome.main(
        ON_POLICY,
        EXPLAINED,
        explained.agents,
        explained.behaviours.model,
        explained.behaviours.buffers,
        explained.replays,
    )
    assert isinstance(state, OnPolicyState) and metrics is not None
    # Source logging: global_step % 100 < num_envs among multiples of 8.
    assert _rows(metrics, "step").tolist() == [0, 104, 200, 304]
    assert np.isnan(_rows(metrics, "validation_loss")).all() and (_rows(metrics, "exact_loss") > 0).all()
    run = seed_slice(state, 0)
    replay = run.replay
    assert int(run.step) == int(run.optimizer.step[...]) == 50 and int(replay.size) == 400
    coalitions = np.asarray(replay.coalitions).reshape(50, 8, 2)
    assert (coalitions == coalitions[:, :1]).all()  # One coalition shared by every environment per decision.
    pool = {tuple(row) for row in np.asarray(seed_slice(explained.behaviours, 0).buffers.train).tolist()}
    assert {tuple(row) for row in np.asarray(replay.explained).tolist()} <= pool
    observations = np.asarray(replay.transitions.observations).reshape(50, 8, 2)
    following = np.asarray(replay.transitions.next_observations).reshape(50, 8, 2)
    terminated = np.asarray(replay.transitions.terminated).reshape(50, 8)
    continuing = ~terminated[:-1]
    # Without termination the next decision starts from the stored next observation (no truncation here).
    np.testing.assert_array_equal(observations[1:][continuing], following[:-1][continuing])
    assert terminated.any()


def test_outcome_validation(explained: Explained) -> None:
    for invalid in ({"variant": "outcome"}, {"tau": 0.0}, {"gamma": 1.5}):
        with pytest.raises(ValueError):
            replace(OFF_POLICY, **invalid)
    with pytest.raises(ValueError):
        OutcomeTrainConfig(num_envs=0)
    with pytest.raises(ValueError, match="fewer transitions"):
        _off_policy(replace(OFF_POLICY, train=replace(OFF_POLICY.train, train_size=128)), explained)
    with pytest.raises(ValueError, match="behaviour characteristic"):
        outcome.main(
            OFF_POLICY,
            replace(EXPLAINED, kind="prediction"),
            explained.agents,
            explained.behaviours.model,
            explained.behaviours.buffers,
            explained.replays,
        )


@pytest.fixture(scope="module")
def explained() -> Explained:
    agent_config = dqn.DQNConfig(
        env=ENV,
        hidden_widths=(8,),
        num_envs=2,
        total_timesteps=64,
        buffer_size=64,
        batch_size=4,
        learning_starts=0,
        train_frequency=1,
    )
    state, _, agents = dqn.main(agent_config)
    behaviours, _ = behaviour.main(EXPLAINED, agents)
    return Explained(agents, state.replay.data, behaviours)


def _off_policy(config: OutcomeConfig, explained: Explained) -> tuple[TrainState[OutcomeNetworks], EpochMetrics]:
    state, metrics = outcome.main(
        config, EXPLAINED, explained.agents, explained.behaviours.model, explained.behaviours.buffers, explained.replays
    )
    assert not isinstance(state, OnPolicyState) and metrics is not None
    return state, metrics


def _rows(metrics: EpochMetrics, name: str) -> np.ndarray:
    """Seed 0's logged rows of one metric."""
    return np.asarray(getattr(metrics, name))[np.asarray(metrics.active)[:, 0], 0]
