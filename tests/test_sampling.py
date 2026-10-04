"""Sampled characteristics against the unchanged source, and sampled Shapley runs end to end."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jax_shapley import train_dqn
from jax_shapley.explanation_io import ExplanationSummary, load_characteristic, load_run_settings, prepare_context
from jax_shapley.sampling import SampledCharacteristic, sample_rows, sampled_pool, select_matches
from jax_shapley.shapley import (
    ShapleyLayout,
    ShapleyModel,
    contribution_loss,
    exact_shapley_loss,
    normalise,
    shapley_network,
)
from jax_shapley.train_explanations import (
    ShapleyRunConfig,
    main,
    parse_args,
    sampled_shapley_context,
    sampled_shapley_learner,
)

FIXTURE = Path(__file__).parent / "fixtures" / "fastsverl_sampling.npz"


def test_matched_rows_select_what_the_source_selects() -> None:
    fixture = _fixture()
    for call in range(fixture["observations"].shape[0]):
        observations, coalition = fixture["observations"][call], fixture["coalitions"][call]
        pool, scores = fixture["pools"][call], fixture["scores"][call]
        rows = np.asarray(
            select_matches(jnp.asarray(observations), jnp.asarray(coalition), jnp.asarray(pool), jnp.asarray(scores))
        )
        matched = ((observations[:, None, :] == pool[None]) | ~coalition).all(-1)
        values = fixture["pool_values"][call][rows]
        has_match = matched.any(-1)
        # Matches: the source's top-1 of where(matches, rand, MASK), so identical rows and values.
        np.testing.assert_array_equal(values[has_match], fixture["values"][call][has_match])
        assert matched[np.arange(rows.shape[0]), rows][has_match].all()
        # No match: every source score ties at MASK and its tie order picks a row of the freshly
        # permuted pool; the port draws one uniformly instead (deviation ledger).
        expected = np.argmax(scores[~has_match], axis=-1)
        np.testing.assert_array_equal(rows[~has_match], expected)
    np.testing.assert_allclose(
        np.asarray(sampled_pool(jnp.asarray(fixture["pool"]), jnp.asarray(_v_f_pool(fixture))).null),
        fixture["null"],
        rtol=1e-6,
    )


def test_sampled_values_are_uniform_among_matches() -> None:
    pool = jnp.asarray([[0, 1], [0, 2], [1, 1], [0, 1]], jnp.int32)
    values = jnp.asarray([[1.0], [2.0], [3.0], [4.0]], jnp.float32)
    observations = jnp.asarray([[0, 5]], jnp.int32)
    coalition = jnp.asarray([True, False])
    keys = jax.random.split(jax.random.key(0), 4000)
    rows = np.asarray(jax.vmap(lambda key: sample_rows(key, observations, coalition, pool)[0])(keys))
    counts = np.bincount(rows, minlength=4)
    assert counts[2] == 0  # Feature 0 differs from the observation.
    np.testing.assert_allclose(counts[[0, 1, 3]] / rows.shape[0], 1 / 3, atol=0.03)
    unmatched = np.asarray(
        jax.vmap(lambda key: sample_rows(key, jnp.asarray([[7, 7]], jnp.int32), coalition, pool)[0])(keys)
    )
    np.testing.assert_allclose(np.bincount(unmatched, minlength=4) / rows.shape[0], 1 / 4, atol=0.03)
    characteristic = SampledCharacteristic(sampled_pool(pool, values), jax.random.key(1))
    # Grand coalition on pool rows: the row itself or an identical one, so v_F exactly.
    np.testing.assert_array_equal(np.asarray(characteristic.values(pool, jnp.ones(2, jnp.bool_))), values)
    np.testing.assert_allclose(np.asarray(characteristic.null(pool)), np.full((4, 1), 2.5))
    # Like the source's Cs[0], every row uses the first coalition.
    shared = jnp.asarray([[True, True], [False, False], [False, False], [False, False]])
    np.testing.assert_array_equal(np.asarray(characteristic.values(pool, shared)), values)


def test_sampled_shapley_runs_explain_their_own_pool(tmp_path: Path, agent: Path) -> None:
    characteristic = _summaries(main(_characteristic(agent, tmp_path / "characteristic")))[0]
    folder = Path(characteristic.folder)
    assert characteristic.epochs == 0 and not characteristic.stopped_early
    loaded = load_characteristic(folder)
    np.testing.assert_array_equal(np.asarray(loaded.buffers.exact), np.asarray(loaded.buffers.train))
    assert (folder / "metrics.csv").read_text().count("\n") == 1  # Header only: nothing trains.
    shapley = _summaries(main(_shapley(agent, folder, tmp_path / "shapley")))[0]
    rows = (Path(shapley.folder) / "metrics.csv").read_text().splitlines()[1:]
    assert len(rows) == 3 and all(np.isfinite([float(cell) for cell in row.split(",")]).all() for row in rows)
    saved = load_run_settings(Path(shapley.folder) / "config.json")
    assert saved.characteristic_sha256 == loaded.sha256
    for invalid in (("--buffers", "fresh"), ("--no-with-exact-char",)):
        with pytest.raises(ValueError, match="sampled characteristics"):
            main([*_shapley(agent, folder, tmp_path / "invalid"), *invalid])


def test_sampled_exact_loss_matches_the_sampled_characteristic(tmp_path: Path, agent: Path) -> None:
    # The exact loss normalises with precomputed v_F; the sampled grand coalition gives the same.
    characteristic = _summaries(main(_characteristic(agent, tmp_path / "characteristic")))[0]
    loaded = load_characteristic(Path(characteristic.folder))
    config = parse_args(_shapley(agent, Path(characteristic.folder), tmp_path / "shapley"))
    assert isinstance(config, ShapleyRunConfig)
    context = prepare_context(config)
    explained = context.explained
    assert explained.estimator == "sampled"
    sampled = sampled_shapley_context(loaded.buffers, context.agent, explained, config, context.features, 64)
    learner = sampled_shapley_learner(sampled, config, context.features)
    layout = ShapleyLayout(context.features, context.actions)
    model = ShapleyModel(shapley_network(layout, (8,)), layout, rngs=nnx.Rngs(3))
    exact = sampled.exact
    assert exact is not None and sampled.grand is not None
    function = SampledCharacteristic(sampled.pool, jax.random.key(5))
    np.testing.assert_allclose(
        float(learner.exact_loss(model)),
        float(exact_shapley_loss(model, function, exact.states, exact.state_valid, exact.values)),
        rtol=1e-6,
    )
    contributions = normalise(model(exact.states).contributions, sampled.grand, function.null(exact.states))
    assert float(contribution_loss(contributions, exact.state_valid, exact.values)) == pytest.approx(
        float(learner.exact_loss(model)), rel=1e-6
    )


def test_on_policy_outcome_conditions_on_a_sampled_behaviour(tmp_path: Path, agent: Path) -> None:
    characteristic = Path(_summaries(main(_characteristic(agent, tmp_path / "characteristic")))[0].folder)
    outcome = _summaries(main(_outcome(agent, characteristic, tmp_path / "outcome")))[0]
    rows = (Path(outcome.folder) / "metrics.csv").read_text().splitlines()[1:]
    assert rows and all(np.isfinite(float(row.split(",")[4])) for row in rows if row.split(",")[4])
    for invalid in (("--variant", "off_policy"), ("--no-with-exact-char",)):
        with pytest.raises(ValueError, match="sampled behaviour"):
            main([*_outcome(agent, characteristic, tmp_path / "invalid"), *invalid])


@pytest.fixture(scope="module")
def agent(tmp_path_factory: pytest.TempPathFactory) -> Path:
    summary = train_dqn.main(_dqn(tmp_path_factory.mktemp("agent")))
    assert summary is not None
    return Path(summary.folder)


def _fixture() -> dict[str, np.ndarray]:
    metadata = json.loads(FIXTURE.with_suffix(".json").read_text())
    assert metadata["num_samples"] == 1
    with np.load(FIXTURE) as data:
        return {name: data[name] for name in data.files}


def _v_f_pool(fixture: dict[str, np.ndarray]) -> np.ndarray:
    """v_F of the unpermuted pool, recovered from the first call's permuted pool values."""
    pool, permuted, values = fixture["pool"], fixture["pools"][0], fixture["pool_values"][0]
    order = [int(np.flatnonzero((permuted == row).all(-1))[0]) for row in pool]
    return values[order]


def _summaries(result: ExplanationSummary | list[ExplanationSummary] | None) -> list[ExplanationSummary]:
    assert result is not None
    return result if isinstance(result, list) else [result]


def _dqn(output: Path) -> list[str]:
    return [
        "gridworld",
        "--hidden-widths",
        "8",
        "--chunk-steps",
        "4",
        "--train.num-envs",
        "2",
        "--train.total-timesteps",
        "64",
        "--train.buffer-size",
        "64",
        "--train.batch-size",
        "4",
        "--train.learning-starts",
        "0",
        "--train.train-frequency",
        "1",
        "--output",
        str(output),
    ]


def _characteristic(agent: Path, output: Path) -> list[str]:
    return [
        "characteristic",
        "--agent",
        str(agent),
        "--estimator",
        "sampled",
        "--hidden-widths",
        "8",
        "--train.no-exact-loss",
        "--train.epochs",
        "3",
        "--train.buffer-size",
        "40",
        "--train.num-envs",
        "4",
        "--train.batch-size",
        "8",
        "--output",
        str(output),
    ]


def _outcome(agent: Path, characteristic: Path, output: Path) -> list[str]:
    return [
        "outcome",
        "--agent",
        str(agent),
        "--characteristic",
        str(characteristic),
        "--variant",
        "on_policy",
        "--with-exact-char",
        "--hidden-widths",
        "8",
        "--train.total-timesteps",
        "256",
        "--train.num-envs",
        "4",
        "--train.batch-size",
        "8",
        "--output",
        str(output),
    ]


def _shapley(agent: Path, characteristic: Path, output: Path) -> list[str]:
    return [
        "shapley",
        "--agent",
        str(agent),
        "--characteristic",
        str(characteristic),
        "--buffers",
        "characteristic",
        "--with-exact-char",
        "--hidden-widths",
        "8",
        "--train.epochs",
        "3",
        "--train.buffer-size",
        "40",
        "--train.num-envs",
        "4",
        "--train.batch-size",
        "8",
        "--output",
        str(output),
    ]
