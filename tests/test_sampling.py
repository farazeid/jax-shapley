"""Sampled characteristics against the unchanged source, and sampled explanation runs end to end."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from jax_shapley.algorithms.dqn import DQNConfig
from jax_shapley.characteristic_models.behaviour import CharacteristicConfig, SampledPool, sampled_values
from jax_shapley.characteristic_models.outcome import OutcomeConfig, OutcomeTrainConfig
from jax_shapley.common.supervised_learning import SupervisedConfig
from jax_shapley.envs import Mastermind
from jax_shapley.experiment import ExperimentConfig
from jax_shapley.main import run
from jax_shapley.shapley_models.behaviour import ShapleyConfig

FIXTURE = Path(__file__).parent / "fixtures" / "fastsverl_sampling.npz"
ENV = Mastermind(2, 2, 2)  # Episodes end within the guess limit, so exact values are finite.
AGENT = DQNConfig(
    env=ENV,
    hidden_widths=(8,),
    num_envs=2,
    total_timesteps=64,
    buffer_size=64,
    batch_size=4,
    learning_starts=0,
    train_frequency=1,
)
TRAINING = SupervisedConfig(epochs=3, batch_size=8, buffer_size=40, num_envs=4)
SAMPLED = CharacteristicConfig(
    env=ENV,
    hidden_widths=(8,),
    estimator="sampled",
    train=SupervisedConfig(epochs=3, batch_size=8, buffer_size=40, num_envs=4, exact_loss=False),
)
SHAPLEY = ShapleyConfig(env=ENV, hidden_widths=(8,), buffers="characteristic", train=TRAINING)
OUTCOME = OutcomeConfig(
    env=ENV,
    variant="on_policy",
    hidden_widths=(8,),
    train=OutcomeTrainConfig(batch_size=8, total_timesteps=256, num_envs=4),
)


def test_matched_rows_select_what_the_source_selects(monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture()
    for call in range(fixture["observations"].shape[0]):
        observations, coalition = fixture["observations"][call], fixture["coalitions"][call]
        pool, scores = fixture["pools"][call], fixture["scores"][call]
        _fix_scores(monkeypatch, scores)
        sampled = SampledPool(jnp.asarray(pool), jnp.asarray(fixture["pool_values"][call]), jnp.zeros(3))
        values = np.asarray(
            sampled_values(sampled, jax.random.key(0), jnp.asarray(observations), jnp.asarray(coalition))
        )
        has_match = ((observations[:, None, :] == pool[None]) | ~coalition).all(-1).any(-1)
        # Matches: the source's top-1 of where(matches, rand, MASK), so identical rows and values.
        np.testing.assert_array_equal(values[has_match], fixture["values"][call][has_match])
        # No match: every source score ties at MASK and its tie order picks a row of the freshly
        # permuted pool; the port takes the top score instead (deviation ledger).
        expected = fixture["pool_values"][call][np.argmax(scores[~has_match], axis=-1)]
        np.testing.assert_array_equal(values[~has_match], expected)
    # v(empty) is v_F's mean over the pool, in any order.
    np.testing.assert_allclose(fixture["pool_values"][0].mean(axis=0), fixture["null"], rtol=1e-6)


def test_sampled_values_are_uniform_among_matches() -> None:
    pool = jnp.asarray([[0, 1], [0, 2], [1, 1], [0, 1]], jnp.int32)
    indices = jnp.arange(4, dtype=jnp.float32)[:, None]  # Each pool row's value is its index.
    indexed = SampledPool(pool, indices, jnp.mean(indices, axis=0))
    coalition = jnp.asarray([True, False])
    keys = jax.random.split(jax.random.key(0), 4000)
    for observation, rows in (([[0, 5]], [0, 1, 3]), ([[7, 7]], [0, 1, 2, 3])):  # Unmatched: any pool row.
        observations = jnp.asarray(observation, jnp.int32)
        drawn = jax.vmap(sampled_values, in_axes=(None, 0, None, None))(indexed, keys, observations, coalition)
        counts = np.bincount(np.asarray(drawn).astype(int).reshape(-1), minlength=4)
        np.testing.assert_allclose(counts[rows] / keys.shape[0], 1 / len(rows), atol=0.03)
        assert counts.sum() == counts[rows].sum()
    values = jnp.asarray([[1.0], [2.0], [3.0], [4.0]], jnp.float32)
    sampled = SampledPool(pool, values, jnp.mean(values, axis=0))
    # Grand coalition on pool rows: the row itself or an identical one.
    grand = np.asarray(sampled_values(sampled, jax.random.key(1), pool, jnp.ones(2, jnp.bool_)))[:, 0]
    np.testing.assert_array_equal(grand[1:3], [2.0, 3.0])
    assert {grand[0], grand[3]} <= {1.0, 4.0}
    # Like the source's Cs[0], every row uses the first coalition.
    shared = jnp.asarray([[True, True], [False, False], [False, False], [False, False]])
    np.testing.assert_array_equal(np.asarray(sampled_values(sampled, jax.random.key(1), pool, shared))[:, 0], grand)


def test_sampled_runs_explain_their_own_pool() -> None:
    config = ExperimentConfig(name="test_sampled", algorithm=AGENT, characteristic=SAMPLED, shapley=(SHAPLEY,))
    result = run(config)
    assert result.characteristic_metrics is None  # A sampled characteristic trains nothing.
    buffers = result.characteristic_state.buffers
    np.testing.assert_array_equal(np.asarray(buffers.exact), np.asarray(buffers.train))
    metrics = result.shapley[0][1]
    assert metrics is not None
    active = np.asarray(metrics.active)
    assert active.sum() == 3
    for name in ("train_loss", "validation_loss", "exact_loss"):
        assert np.isfinite(np.asarray(getattr(metrics, name))[active]).all()
    invalid = (
        ShapleyConfig(env=ENV, hidden_widths=(8,), buffers="fresh", train=TRAINING),
        ShapleyConfig(env=ENV, hidden_widths=(8,), buffers="characteristic", with_exact_char=False, train=TRAINING),
    )
    for shapley in invalid:
        with pytest.raises(ValueError, match="sampled characteristic"):
            run(ExperimentConfig(name="test_sampled", algorithm=AGENT, characteristic=SAMPLED, shapley=(shapley,)))


def test_on_policy_outcome_conditions_on_a_sampled_behaviour() -> None:
    config = ExperimentConfig(
        name="test_sampled_outcome", algorithm=AGENT, characteristic=SAMPLED, outcome=OUTCOME, shapley=()
    )
    metrics = run(config).outcome_metrics
    assert metrics is not None
    active = np.asarray(metrics.active)
    assert active.any() and np.isfinite(np.asarray(metrics.exact_loss)[active]).all()
    for outcome in (
        OutcomeConfig(env=ENV, variant="off_policy", hidden_widths=(8,), train=OUTCOME.train),
        OutcomeConfig(env=ENV, variant="on_policy", with_exact_char=False, hidden_widths=(8,), train=OUTCOME.train),
    ):
        with pytest.raises(ValueError, match="sampled behaviour"):
            run(
                ExperimentConfig(
                    name="test_sampled_outcome", algorithm=AGENT, characteristic=SAMPLED, outcome=outcome, shapley=()
                )
            )


def _fix_scores(monkeypatch: pytest.MonkeyPatch, scores: np.ndarray) -> None:
    """The source's uniform score of every (row, pool row) pair."""

    def uniform(key: Array, shape: tuple[int, ...], dtype: jnp.dtype) -> Array:
        del key, dtype
        assert shape == scores.shape
        return jnp.asarray(scores)

    monkeypatch.setattr(jax.random, "uniform", uniform)


def _fixture() -> dict[str, np.ndarray]:
    metadata = json.loads(FIXTURE.with_suffix(".json").read_text())
    assert metadata["num_samples"] == 1
    with np.load(FIXTURE) as data:
        return {name: data[name] for name in data.files}
