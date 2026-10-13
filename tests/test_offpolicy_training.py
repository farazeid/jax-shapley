"""Off-policy characteristics learn from the agent's replay rows, with their behaviour policy."""

from dataclasses import replace

import numpy as np
import pytest

from jax_shapley.algorithms import dqn
from jax_shapley.characteristic_models import behaviour
from jax_shapley.characteristic_models.behaviour import CharacteristicConfig
from jax_shapley.common.supervised_learning import SupervisedConfig, seed_slice
from jax_shapley.envs import Gridworld

ENV = Gridworld()
AGENT = dqn.DQNConfig(
    env=ENV,
    hidden_widths=(8,),
    num_envs=2,
    total_timesteps=64,
    buffer_size=64,
    batch_size=4,
    learning_starts=0,
    train_frequency=1,
)
REPLAY = CharacteristicConfig(
    env=ENV,
    hidden_widths=(8,),
    data="replay",
    train=SupervisedConfig(epochs=4, buffer_size=40, num_envs=4, batch_size=8, exact_buffer_size=16),
)


def test_replay_buffers_are_shuffled_replay_rows_with_their_behaviour() -> None:
    state, _, agents = dqn.main(AGENT)
    characteristic, _ = behaviour.main(replace(REPLAY, off_policy=True), agents, state.replay.data)
    buffers = seed_slice(characteristic, 0).buffers
    replay = seed_slice(state.replay.data, 0)
    assert buffers.behaviour is not None
    # The same (observation, action) recurs with different behaviour log-probabilities as epsilon decays.
    rows = {
        (tuple(observation), action, round(log_probability, 5))
        for observation, action, log_probability in zip(
            np.asarray(replay.observations).tolist(),
            np.asarray(replay.actions).tolist(),
            np.asarray(replay.log_probabilities).tolist(),
            strict=True,
        )
    }
    for observations, actions, log_probabilities in (
        (buffers.train, buffers.behaviour.train_actions, buffers.behaviour.train_log_probabilities),
        (buffers.validation, buffers.behaviour.validation_actions, buffers.behaviour.validation_log_probabilities),
    ):
        for observation, action, log_probability in zip(
            np.asarray(observations).tolist(),
            np.asarray(actions).tolist(),
            np.asarray(log_probabilities).tolist(),
            strict=True,
        ):
            assert (tuple(observation), action, round(log_probability, 5)) in rows
    # Train and validation rows are disjoint draws from one permutation of all 64 rows.
    assert buffers.train.shape == (32, 2) and buffers.validation.shape == (8, 2)
    assert buffers.exact.shape == (16, 2)  # Fresh rollouts of the explained policy.


@pytest.mark.parametrize(("off_policy", "weighting"), [(False, True), (True, True), (True, False)])
def test_weighting_variants_train(off_policy: bool, weighting: bool) -> None:
    state, _, agents = dqn.main(AGENT)
    config = replace(REPLAY, off_policy=off_policy, weighting=weighting)
    _, metrics = behaviour.main(config, agents, state.replay.data)
    assert metrics is not None
    active = np.asarray(metrics.active)
    assert active.sum() == 4
    for name in ("train_loss", "validation_loss", "exact_loss"):
        assert np.isfinite(np.asarray(getattr(metrics, name))[active]).all()


def test_replay_validation() -> None:
    with pytest.raises(ValueError, match="replay data"):
        replace(REPLAY, data="rollouts", off_policy=True)
    with pytest.raises(ValueError, match="exact buffer"):
        replace(REPLAY, train=replace(REPLAY.train, exact_buffer_size=None))
    with pytest.raises(ValueError, match="clip_coef"):
        replace(REPLAY, clip_coef=-1)
    state, _, agents = dqn.main(AGENT)
    with pytest.raises(ValueError, match="replays"):
        behaviour.main(replace(REPLAY, train=replace(REPLAY.train, buffer_size=80)), agents, state.replay.data)
    with pytest.raises(ValueError, match="replays"):
        behaviour.main(REPLAY, agents)
