"""Pipeline parity with upstream fill_buffer, train_models and the trained agent's epsilon.

The fixture records every random draw the unchanged upstream loop made. The training loop
derives each batch's key from the state's key in a fixed split order, so the tests replay
that order to learn which key each recorded draw belongs to, and replace the loop's draws
by looking their key up. Every logged value and the restored best weights must then match.
"""

import json
from dataclasses import replace
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax import Array

from jax_shapley.algorithms import dqn
from jax_shapley.characteristic_models import behaviour, prediction
from jax_shapley.characteristic_models.behaviour import CharacteristicConfig, CharacteristicContext, Kind
from jax_shapley.common.coalitions import coalition_table
from jax_shapley.common.rollouts import collect_observations
from jax_shapley.common.supervised_learning import (
    EpochMetrics,
    ReplayBehaviour,
    StateBuffers,
    SupervisedConfig,
    seed_slice,
    stack_seeds,
    train,
)
from jax_shapley.envs import Hypercube, Mastermind
from jax_shapley.envs.wrappers import EpisodeStatisticsWrapper
from jax_shapley.shapley_models import behaviour as shapley
from jax_shapley.shapley_models.behaviour import ExactShapley, ShapleyConfig, ShapleyContext
from tests.fixture_support import assert_parameters, assign

FIXTURE = Path(__file__).parent / "fixtures" / "fastsverl_pipeline.npz"
ENV = Mastermind(2, 2, 2)
FEATURES = 8
# Upstream Adam over up to 28 updates; float differences compound slightly beyond the 3-step unit parity.
RTOL, ATOL = 1e-4, 1e-6


def test_buffer_collection_matches_upstream_fill_buffer() -> None:
    fixture = _fixture()
    config = dqn.DQNConfig(env=Hypercube(3, 5), hidden_widths=(64, 64))
    model = dqn.initialize(config, jax.random.key(0)).model
    assign(model.mlp, fixture, "buffer_agent")
    expected = fixture["buffer_observations"]
    # The goal-seeking greedy policy has no ties, so sampling is deterministic and keys are irrelevant.
    # Bound like the stages bind it: the wrapper of the union-typed config environment.
    collect = partial(collect_observations, EpisodeStatisticsWrapper(config.env))
    observed = collect(model, jnp.asarray(0.0), jax.random.key(0), expected.shape[0], 16)
    assert int((expected == 0).all(axis=1).sum()) > 16, "the capture must include auto-resets"
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("kind", ["behaviour", "prediction"])
def test_characteristic_loop_matches_upstream_train_models(kind: Kind, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture()
    prefix = f"loop_{kind}"
    stopping = kind == "prediction"
    exact_condition = float(json.loads(FIXTURE.with_suffix(".json").read_text())["exact_condition"])
    training = _training(
        epochs=12 if stopping else 8,
        patience=12 if stopping else 2,
        exact_condition=exact_condition if stopping else None,
        track_batch=stopping,
    )
    config = CharacteristicConfig(ENV, kind=kind, train=training)
    targets, head = (
        (behaviour.targets, behaviour.head) if kind == "behaviour" else (prediction.targets, prediction.head)
    )
    buffers = _buffers(fixture)
    agent = _agent(fixture)
    exact = behaviour.exact_characteristic(
        buffers.exact,
        targets(config, agent, buffers.exact),
        jnp.asarray(coalition_table(FEATURES)),
        max_unique=32,
    )
    context = CharacteristicContext(
        buffers, targets(config, agent, buffers.train), targets(config, agent, buffers.validation), exact
    )
    state = seed_slice(behaviour.initialize(config, head), 0)
    assign(state.model.mlp, fixture, f"{prefix}_initial")
    state = stack_seeds([replace(state, buffers=buffers)])
    _recorded(monkeypatch, fixture, prefix, state.key, training.batches)
    program = partial(train, loss=partial(behaviour.loss, config), config=training, features=FEATURES, log=None)
    result, metrics = nnx.jit(program)(state, stack_seeds([context]), None)
    _assert_rows(metrics, fixture, prefix)
    assert_parameters(seed_slice(result, 0).model.mlp, fixture, f"{prefix}_final", rtol=2e-4, atol=1e-5)


def test_offpolicy_characteristic_loop_matches_upstream_train_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Paper Figure 3a's loop: normalised importance weights per batch and over the validation draw."""
    fixture = _fixture()
    prefix = "loop_offpolicy"
    # Replay characteristics take exact values on a separate buffer: here the train rows.
    training = replace(_training(epochs=8, patience=2, exact_condition=None, track_batch=False), exact_buffer_size=32)
    config = CharacteristicConfig(ENV, data="replay", off_policy=True, train=training)
    actions = jnp.asarray(fixture[f"{prefix}_actions"])
    logged = jnp.asarray(fixture[f"{prefix}_log_probabilities"])
    rows = training.train_size
    base = _buffers(fixture)
    replay = ReplayBehaviour(actions[:rows], logged[:rows], actions[rows:], logged[rows:])
    buffers = StateBuffers(base.train, base.validation, base.exact, replay)
    agent = _agent(fixture)
    exact = behaviour.exact_characteristic(
        buffers.exact,
        behaviour.targets(config, agent, buffers.exact),
        jnp.asarray(coalition_table(FEATURES)),
        max_unique=32,
    )
    train_targets = behaviour.targets(config, agent, buffers.train)
    validation_targets = behaviour.targets(config, agent, buffers.validation)
    context = CharacteristicContext(
        buffers,
        train_targets,
        validation_targets,
        exact,
        dqn.log_probabilities(train_targets, replay.train_actions),
        dqn.log_probabilities(validation_targets, replay.validation_actions),
    )
    state = seed_slice(behaviour.initialize(config, behaviour.head), 0)
    assign(state.model.mlp, fixture, f"{prefix}_initial")
    state = stack_seeds([replace(state, buffers=buffers)])
    _recorded(monkeypatch, fixture, prefix, state.key, training.batches)
    program = partial(train, loss=partial(behaviour.loss, config), config=training, features=FEATURES, log=None)
    result, metrics = nnx.jit(program)(state, stack_seeds([context]), None)
    _assert_rows(metrics, fixture, prefix)
    assert_parameters(seed_slice(result, 0).model.mlp, fixture, f"{prefix}_final", rtol=2e-4, atol=1e-5)


def test_shapley_loop_matches_upstream_train_models(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture()
    training = _training(epochs=8, patience=2, exact_condition=None, track_batch=False)
    explained = CharacteristicConfig(ENV, train=training)
    characteristic = seed_slice(behaviour.initialize(explained, behaviour.head), 0).model
    assign(characteristic.mlp, fixture, "loop_behaviour_final")
    buffers = _buffers(fixture)
    coalitions = jnp.asarray(coalition_table(FEATURES))
    exact = behaviour.exact_characteristic(
        buffers.exact, behaviour.targets(explained, _agent(fixture), buffers.exact), coalitions, max_unique=32
    )
    context = ShapleyContext(
        buffers, ExactShapley(exact.states, exact.state_valid, shapley.exact_shapley(exact.at_states))
    )
    config = ShapleyConfig(ENV, train=training)
    state = seed_slice(shapley.initialize(config, explained, behaviour.head), 0)
    assign(state.model.mlp, fixture, "loop_shapley_initial")
    state = stack_seeds([replace(state, buffers=buffers)])
    _recorded(monkeypatch, fixture, "loop_shapley", state.key, training.batches)
    loss = partial(shapley.loss, shapley.masked_values, shapley.masked_null)
    program = partial(train, loss=loss, config=training, features=FEATURES, log=None)
    result, metrics = nnx.jit(program)(state, stack_seeds([context]), stack_seeds([characteristic]))
    _assert_rows(metrics, fixture, "loop_shapley")
    assert_parameters(seed_slice(result, 0).model.mlp, fixture, "loop_shapley_final", rtol=2e-4, atol=1e-5)


def test_trained_epsilon_follows_the_upstream_save_rule() -> None:
    """Upstream pickles epsilon after linear_schedule((steps - 1) * num_envs) (training.py:213, 289)."""
    config = dqn.DQNConfig(
        env=ENV,
        hidden_widths=(4,),
        num_envs=2,
        total_timesteps=40,
        buffer_size=8,
        batch_size=2,
        learning_starts=0,
        exploration_fraction=1.5,
    )
    epsilon = float(dqn.main(config)[2].epsilon[0])
    last_step = (config.total_timesteps // config.num_envs - 1) * config.num_envs
    schedule = config.start_e + (config.end_e - config.start_e) * last_step / (1.5 * config.total_timesteps)
    assert 0.01 < epsilon < 1.0  # Mid-schedule, so the step choice matters.
    np.testing.assert_allclose(epsilon, max(schedule, config.end_e), rtol=0, atol=1e-7)


def _recorded(
    monkeypatch: pytest.MonkeyPatch, fixture: dict[str, np.ndarray], prefix: str, key: Array, batches: int
) -> None:
    """Replace the loop's row and coalition draws by upstream's, found by the key that draws them.

    Every epoch splits the state key into the next key, a batch key and a validation key;
    each batch's key is a split of the batch key; a loss splits its key into a row key and
    a coalition key.
    """
    train_rows = fixture[f"{prefix}_train_rows"]
    epochs = train_rows.shape[0]
    rows = np.concatenate((train_rows.reshape(-1, train_rows.shape[-1]), fixture[f"{prefix}_validation_rows"]))
    train_coalitions, validation_coalitions = (
        fixture[f"{prefix}_train_coalitions"],
        fixture[f"{prefix}_validation_coalitions"],
    )
    coalitions = np.concatenate((train_coalitions.reshape(-1, *train_coalitions.shape[2:]), validation_coalitions))
    loss_keys: list[Array] = []
    validation_keys: list[Array] = []
    key = key[0]
    for _ in range(epochs):
        key, batch_key, validation_key = jax.random.split(key, 3)
        loss_keys.extend(jax.random.split(batch_key, batches))
        validation_keys.append(validation_key)
    split = [jax.random.split(loss_key) for loss_key in [*loss_keys, *validation_keys]]
    row_keys = jnp.stack([jax.random.key_data(pair[0]) for pair in split])
    coalition_keys = jnp.stack([jax.random.key_data(pair[1]) for pair in split])

    def lookup(table: Array, key: Array) -> Array:
        return jnp.argmax(jnp.all(table == jax.random.key_data(key), axis=-1))

    def randint(key: Array, shape: tuple[int, ...], minval: int, maxval: int) -> Array:
        del shape, minval, maxval
        return jnp.asarray(rows, jnp.int32)[lookup(row_keys, key)]

    def bernoulli(key: Array, p: float, shape: tuple[int, ...]) -> Array:
        del p, shape
        return jnp.asarray(coalitions)[lookup(coalition_keys, key)]

    def kernel_coalitions(key: Array, batch: int, features: int) -> Array:
        del batch, features
        return jnp.asarray(coalitions)[lookup(coalition_keys, key)]

    monkeypatch.setattr(jax.random, "randint", randint)
    monkeypatch.setattr(jax.random, "bernoulli", bernoulli)
    monkeypatch.setattr(shapley, "kernel_coalitions", kernel_coalitions)


def _assert_rows(metrics: EpochMetrics, fixture: dict[str, np.ndarray], prefix: str) -> None:
    active = np.asarray(metrics.active)[:, 0]
    steps = np.asarray(metrics.step)[active, 0]
    assert steps.tolist() == fixture[f"{prefix}_epoch_loss_steps"].tolist(), "epochs run and logged steps must match"
    for name, recorded in (("train_loss", "epoch_loss"), ("validation_loss", "val_loss"), ("exact_loss", "exact_loss")):
        values = np.asarray(getattr(metrics, name))[active, 0]
        np.testing.assert_allclose(values, fixture[f"{prefix}_{recorded}"], rtol=RTOL, atol=ATOL)


def _training(*, epochs: int, patience: int, exact_condition: float | None, track_batch: bool) -> SupervisedConfig:
    # 40 buffered rows split 32/8, batches of 8; num_envs only constrains collection.
    return SupervisedConfig(
        epochs=epochs,
        patience=patience,
        batch_size=8,
        buffer_size=40,
        num_envs=8,
        exact_condition=exact_condition,
        track_batch=track_batch,
    )


def _buffers(fixture: dict[str, np.ndarray]) -> StateBuffers:
    train_rows = jnp.asarray(fixture["loop_train"])
    return StateBuffers(train_rows, jnp.asarray(fixture["loop_validation"]), train_rows)


def _agent(fixture: dict[str, np.ndarray]) -> dqn.Agent:
    model = dqn.initialize(dqn.DQNConfig(env=ENV, hidden_widths=(64, 64)), jax.random.key(0)).model
    assign(model.mlp, fixture, "loop_agent")
    return dqn.Agent(model, jnp.asarray(0.2, jnp.float32))


def _fixture() -> dict[str, np.ndarray]:
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        return {key: fixture[key] for key in fixture.files}
