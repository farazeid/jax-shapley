"""Pipeline parity with upstream fill_buffer, train_models and the reloaded agent epsilon.

The fixture records every random draw the unchanged upstream loop made; the JAX
loop replays them through its injectable draws, so every logged value and the
restored best weights must match.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jax_shapley import train_dqn
from jax_shapley.characteristics import (
    CharacteristicConfig,
    CharacteristicModel,
    OffPolicyConfig,
    characteristic_network,
)
from jax_shapley.envs import Hypercube
from jax_shapley.networks import QNetwork, dqn_config
from jax_shapley.run_io import load_agent, load_settings
from jax_shapley.shapley import MaskedCharacteristic, ShapleyLayout, ShapleyModel, shapley_network
from jax_shapley.train_dqn import epsilon_at
from jax_shapley.train_explanations import (
    Behaviour,
    Buffers,
    CharacteristicContext,
    Draw,
    ExplanationState,
    ExplanationTrainConfig,
    FrozenAgent,
    Learner,
    ShapleyContext,
    ShapleyRunConfig,
    characteristic_context,
    characteristic_learner,
    collect_observations,
    initialize_state,
    seed_slice,
    shapley_context,
    shapley_learner,
    stack_seeds,
    train,
)
from jax_shapley.wrappers import EpisodeStatisticsWrapper
from tests.explanation_support import RecordingSink, assert_parameters, assign

FIXTURE = Path(__file__).parent / "fixtures" / "fastsverl_pipeline.npz"
# Upstream Adam over up to 28 updates; float differences compound slightly beyond the 3-step unit parity.
RTOL, ATOL = 1e-4, 1e-6


def test_buffer_collection_matches_upstream_fill_buffer() -> None:
    fixture = _fixture()
    env = EpisodeStatisticsWrapper(Hypercube(3, 5))
    agent = QNetwork(dqn_config(5, 10, hidden_widths=(64, 64)), rngs=nnx.Rngs(0))
    assign(agent.mlp, fixture, "buffer_agent")
    expected = fixture["buffer_observations"]
    # The goal-seeking greedy policy has no ties, so sampling is deterministic and keys are irrelevant.
    observed = collect_observations(env, agent, jnp.asarray(0.0), jax.random.key(0), expected.shape[0], 16)
    assert int((expected == 0).all(axis=1).sum()) > 16, "the capture must include auto-resets"
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("case", ["behaviour", "prediction"])
def test_characteristic_loop_matches_upstream_train_models(case: str) -> None:
    fixture, metadata = _fixture(), _metadata()
    prefix = f"loop_{case}"
    stopping = case == "prediction"
    training = _training(
        epochs=12 if stopping else 8,
        patience=12 if stopping else 2,
        exact_condition=float(cast(float, metadata["exact_condition"])) if stopping else None,
        track_batch=stopping,
    )
    explained = CharacteristicConfig("behaviour" if case == "behaviour" else "prediction", hidden_widths=(64, 64))
    model = CharacteristicModel(characteristic_network(8, 4, explained), rngs=nnx.Rngs(0))
    assign(model.mlp, fixture, f"{prefix}_initial")
    buffers = _buffers(fixture)
    context = characteristic_context(buffers, _agent(fixture), explained, training, 8, 32)

    def learner(context: CharacteristicContext, frozen: None) -> Learner[CharacteristicModel, Draw]:
        del frozen
        return _recorded(characteristic_learner(context, training, 8), fixture, prefix)

    sink, result = _replay(model, buffers, training, context, None, learner)
    _assert_rows(sink, fixture, prefix)
    assert_parameters(result.model.mlp, fixture, f"{prefix}_final", rtol=2e-4, atol=1e-5)


def test_offpolicy_characteristic_loop_matches_upstream_train_models() -> None:
    """Paper Figure 3a's loop: normalised importance weights per batch and over the validation draw."""
    fixture = _fixture()
    prefix = "loop_offpolicy"
    training = _training(epochs=8, patience=2, exact_condition=None, track_batch=False)
    explained = CharacteristicConfig("behaviour", hidden_widths=(64, 64))
    model = CharacteristicModel(characteristic_network(8, 4, explained), rngs=nnx.Rngs(0))
    assign(model.mlp, fixture, f"{prefix}_initial")
    actions = jnp.asarray(fixture[f"{prefix}_actions"])
    log_probabilities = jnp.asarray(fixture[f"{prefix}_log_probabilities"])
    rows = training.train_size
    base = _buffers(fixture)
    behaviour = Behaviour(actions[:rows], log_probabilities[:rows], actions[rows:], log_probabilities[rows:])
    buffers = Buffers(base.train, base.validation, base.exact, behaviour)
    context = characteristic_context(buffers, _agent(fixture), explained, training, 8, 32)
    importance = OffPolicyConfig(weighting=True)

    def learner(context: CharacteristicContext, frozen: None) -> Learner[CharacteristicModel, Draw]:
        del frozen
        return _recorded(characteristic_learner(context, training, 8, importance), fixture, prefix)

    sink, result = _replay(model, buffers, training, context, None, learner)
    _assert_rows(sink, fixture, prefix)
    assert_parameters(result.model.mlp, fixture, f"{prefix}_final", rtol=2e-4, atol=1e-5)


def test_shapley_loop_matches_upstream_train_models() -> None:
    fixture = _fixture()
    training = _training(epochs=8, patience=2, exact_condition=None, track_batch=False)
    explained = CharacteristicConfig("behaviour", hidden_widths=(64, 64))
    characteristic = CharacteristicModel(characteristic_network(8, 4, explained), rngs=nnx.Rngs(0))
    assign(characteristic.mlp, fixture, "loop_behaviour_final")
    layout = ShapleyLayout(8, 4)
    model = ShapleyModel(shapley_network(layout, (64, 64)), layout, rngs=nnx.Rngs(1))
    assign(model.mlp, fixture, "loop_shapley_initial")
    buffers = _buffers(fixture)
    config = ShapleyRunConfig(agent=Path("unused"), characteristic=Path("unused"), with_exact_char=True, train=training)
    function = MaskedCharacteristic(characteristic)
    context = shapley_context(buffers, _agent(fixture), function, explained, config, 8, 32)

    def learner(context: ShapleyContext, frozen: CharacteristicModel) -> Learner[ShapleyModel, Draw]:
        learner = shapley_learner(context, MaskedCharacteristic(frozen), config, 8)
        return _recorded(learner, fixture, "loop_shapley", single=True)

    sink, result = _replay(model, buffers, training, context, stack_seeds([characteristic]), learner)
    _assert_rows(sink, fixture, "loop_shapley")
    assert_parameters(result.model.mlp, fixture, "loop_shapley_final", rtol=2e-4, atol=1e-5)


def test_reloaded_epsilon_follows_the_upstream_save_rule(tmp_path: Path) -> None:
    """Upstream pickles epsilon after linear_schedule((steps - 1) * num_envs) (training.py:213, 289)."""
    summary = train_dqn.main([
        "mastermind222",
        "--hidden-widths",
        "4",
        "--train.num-envs",
        "2",
        "--train.total-timesteps",
        "40",
        "--train.buffer-size",
        "8",
        "--train.batch-size",
        "2",
        "--train.learning-starts",
        "0",
        "--train.exploration-fraction",
        "1.5",
        "--output",
        str(tmp_path),
    ])
    assert summary is not None
    folder = Path(summary.folder)
    config = load_settings(folder / "config.json")
    loaded = load_agent(folder, config.network, load_replay=False)
    last_step = jnp.asarray((config.train.vector_steps - 1) * config.train.num_envs, jnp.int32)
    assert 0.01 < loaded.epsilon < 1.0  # Mid-schedule, so the step choice matters.
    np.testing.assert_allclose(loaded.epsilon, epsilon_at(config.train, last_step), rtol=0, atol=1e-7)


def _replay[Model: nnx.Module, Context, Frozen](
    model: Model,
    buffers: Buffers,
    training: ExplanationTrainConfig,
    context: Context,
    frozen: Frozen,
    learner: Callable[[Context, Frozen], Learner[Model, Draw]],
) -> tuple[RecordingSink, ExplanationState[Model]]:
    """The production loop as a one-seed sweep, with recorded draws injected by `learner`."""
    sink = RecordingSink()
    state = stack_seeds([initialize_state(model, buffers, training, jax.random.key(0))])
    contexts = stack_seeds([context])
    result = nnx.jit(lambda state, contexts, frozen: train(state, contexts, frozen, learner, training, 1, sink))(
        state, contexts, frozen
    )
    return sink, seed_slice(result, 0)


def _recorded[Model: nnx.Module](
    learner: Learner[Model, Draw], fixture: dict[str, np.ndarray], prefix: str, *, single: bool = False
) -> Learner[Model, Draw]:
    """Replace random draws by upstream's recorded rows and coalitions, indexed by epoch and batch."""
    rows = jnp.asarray(fixture[f"{prefix}_train_rows"])
    coalitions = jnp.asarray(fixture[f"{prefix}_train_coalitions"])
    validation_rows = jnp.asarray(fixture[f"{prefix}_validation_rows"])
    validation_coalitions = jnp.asarray(fixture[f"{prefix}_validation_coalitions"])
    if single:  # Shapley draws one (1, F) coalition per batch.
        coalitions, validation_coalitions = coalitions[:, :, 0], validation_coalitions[:, 0]

    def train_draw(key: jax.Array, epoch: jax.Array, batch: jax.Array) -> Draw:
        del key
        return Draw(rows[epoch, batch], coalitions[epoch, batch])

    def validation_draw(key: jax.Array, epoch: jax.Array) -> Draw:
        del key
        return Draw(validation_rows[epoch], validation_coalitions[epoch])

    return learner._replace(train_draw=train_draw, validation_draw=validation_draw)


def _assert_rows(sink: RecordingSink, fixture: dict[str, np.ndarray], prefix: str) -> None:
    steps = fixture[f"{prefix}_epoch_loss_steps"]
    assert [row[1] for row in sink.rows] == steps.tolist(), "epochs run and logged steps must match"
    rows = np.asarray([row[2:5] for row in sink.rows])
    np.testing.assert_allclose(rows[:, 0], fixture[f"{prefix}_epoch_loss"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(rows[:, 1], fixture[f"{prefix}_val_loss"], rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(rows[:, 2], fixture[f"{prefix}_exact_loss"], rtol=RTOL, atol=ATOL)


def _training(
    *, epochs: int, patience: int, exact_condition: float | None, track_batch: bool
) -> ExplanationTrainConfig:
    # 40 buffered rows split 32/8, batches of 8; num_envs only constrains collection.
    return ExplanationTrainConfig(
        epochs=epochs,
        patience=patience,
        batch_size=8,
        buffer_size=40,
        num_envs=8,
        exact_condition=exact_condition,
        track_batch=track_batch,
    )


def _buffers(fixture: dict[str, np.ndarray]) -> Buffers:
    train = jnp.asarray(fixture["loop_train"])
    return Buffers(train, jnp.asarray(fixture["loop_validation"]), train)


def _agent(fixture: dict[str, np.ndarray]) -> FrozenAgent:
    agent = QNetwork(dqn_config(8, 4, hidden_widths=(64, 64)), rngs=nnx.Rngs(0))
    assign(agent.mlp, fixture, "loop_agent")
    return FrozenAgent(agent, jnp.asarray(0.2, jnp.float32))


def _fixture() -> dict[str, np.ndarray]:
    with np.load(FIXTURE, allow_pickle=False) as fixture:
        return {key: fixture[key] for key in fixture.files}


def _metadata() -> dict[str, object]:
    metadata: dict[str, object] = json.loads(FIXTURE.with_suffix(".json").read_text())
    return metadata
