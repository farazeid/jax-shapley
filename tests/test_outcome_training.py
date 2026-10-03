"""Off-policy outcome runs: the CLI pipeline, exact-loss logging, sweeps and exact resume."""

import csv
from contextlib import suppress
from math import nan
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jax_shapley import train_dqn
from jax_shapley.characteristics import exact_characteristic
from jax_shapley.coalitions import all_coalitions
from jax_shapley.explanation_io import (
    ExplanationSnapshot,
    ExplanationStore,
    ExplanationSummary,
    ExplanationWriter,
    load_characteristic,
    load_outcome,
    load_run_settings,
    prepare_context,
    restore_on_policy,
)
from jax_shapley.outcome import (
    ConditionedReplay,
    OutcomeCharacteristic,
    exact_outcome,
    exact_outcome_loss,
    learnt_behaviour_table,
)
from jax_shapley.run_io import CallbackToken, load_agent, load_replay, load_settings
from jax_shapley.shapley import (
    ShapleyLayout,
    ShapleyModel,
    characteristic_table,
    exact_shapley,
    exact_shapley_loss,
    shapley_network,
)
from jax_shapley.tabular import transition_model
from jax_shapley.train_explanations import (
    ExplainedOutcome,
    FrozenAgent,
    OnPolicyState,
    OutcomeFrozen,
    OutcomeRunConfig,
    OutcomeShapleyFrozen,
    conditioned_policy,
    main,
    outcome_exact,
    outcome_function,
    parse_args,
    template_state,
)


@pytest.fixture(scope="module")
def explained(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A small Gridworld agent and an explore-policy behaviour characteristic of it."""
    root = tmp_path_factory.mktemp("explained")
    agent = train_dqn.main(_dqn(root / "agent"))
    assert agent is not None
    characteristic = main([
        "characteristic",
        "--agent",
        agent.folder,
        "--policy",
        "explore",
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
        str(root / "characteristic"),
    ])
    assert isinstance(characteristic, ExplanationSummary)
    return Path(agent.folder), Path(characteristic.folder)


def test_outcome_run_logs_the_exact_loss_of_its_restored_model(tmp_path: Path, explained: tuple[Path, Path]) -> None:
    agent_folder, characteristic_folder = explained
    summary = _single(main(_outcome(agent_folder, characteristic_folder, tmp_path, "--no-with-exact-char")))
    folder = Path(summary.folder)
    assert summary.stage == "outcome" and summary.epochs == 4
    rows = _rows(folder)
    assert [row[0] for row in rows] == [1, 2, 3, 4]
    assert [row[1] for row in rows] == [8, 12, 16, 20]  # track_batch: (epoch + 1) * batches.
    outcome = load_outcome(folder)
    assert outcome.characteristic_sha256 == load_characteristic(characteristic_folder).sha256
    # Recompute the ground truth eagerly from the saved inputs and score the restored best model.
    settings = load_settings(agent_folder / "config.json")
    loaded = load_agent(agent_folder, settings.network, load_replay=False)
    behaviour = load_characteristic(characteristic_folder)
    frozen = OutcomeFrozen(FrozenAgent(loaded.model, jnp.asarray(loaded.epsilon, jnp.float32)), behaviour.model)
    policy = conditioned_policy(frozen, behaviour.config.characteristic)
    tabular = transition_model(settings.env)
    coalitions = all_coalitions(2)
    observations = behaviour.buffers.exact
    exact_behaviour = exact_characteristic(
        observations, policy.agent(observations), coalitions, max_unique=observations.shape[0]
    )
    conditioned = learnt_behaviour_table(policy.behaviour, exact_behaviour.states, coalitions)
    exact = exact_outcome(
        tabular,
        exact_behaviour.states,
        exact_behaviour.state_valid,
        conditioned,
        policy.agent(tabular.states),
        1.0,
    )
    eager = exact_outcome_loss(OutcomeCharacteristic(outcome.networks, policy, "off_policy"), exact, coalitions)
    logged = {row[0]: row[4] for row in rows}
    np.testing.assert_allclose(logged[summary.best_epoch], eager, rtol=1e-5)


def test_outcome_buffers_are_the_agent_replay_prefixes(tmp_path: Path, explained: tuple[Path, Path]) -> None:
    agent_folder, characteristic_folder = explained
    summary = _single(main(_outcome(agent_folder, characteristic_folder, tmp_path)))
    replay = load_replay(agent_folder).data
    store = ExplanationStore(Path(summary.folder), read_only=True)
    try:
        template = (jnp.zeros((32, 2), jnp.int32), jnp.zeros((8, 2), jnp.int32), jnp.zeros((32, 2), jnp.int32))
        train_rows, validation_rows, exact_rows = store.arrays("buffers", template)
    finally:
        store.close()
    # The source's overlapping prefixes: train is replay[:train_size], validation replay[:validation_size].
    np.testing.assert_array_equal(train_rows, replay.observations[:32])
    np.testing.assert_array_equal(validation_rows, replay.observations[:8])
    np.testing.assert_array_equal(exact_rows, load_characteristic(characteristic_folder).buffers.exact)


def test_outcome_sweep_resumes_to_the_uninterrupted_result(
    tmp_path: Path, explained: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    agent_folder, characteristic_folder = explained
    arguments = [*_outcome(agent_folder, characteristic_folder, tmp_path / "baseline"), "--seeds", "1", "2"]
    baseline = _summaries(main(arguments))
    single = _single(main([*_outcome(agent_folder, characteristic_folder, tmp_path / "single"), "--seed", "2"]))
    np.testing.assert_allclose(_rows(Path(baseline[1].folder)), _rows(Path(single.folder)), rtol=1e-5, atol=1e-6)
    original = ExplanationWriter.checkpoint

    def checkpoint(self: ExplanationWriter, snapshot: ExplanationSnapshot) -> CallbackToken:
        token = original(self, snapshot)
        self.store.wait()
        if int(snapshot.epoch) == 2 and self.folder.name == "seed-1":
            raise OSError("injected interruption mid-sweep")
        return token

    with monkeypatch.context() as patch:
        patch.setattr(ExplanationWriter, "checkpoint", checkpoint)
        with pytest.raises(Exception, match="injected interruption"):
            main([*_outcome(agent_folder, characteristic_folder, tmp_path / "interrupted"), "--seeds", "1", "2"])
        with suppress(RuntimeError):
            jax.effects_barrier()  # Clear the expected failed callback effect.
    root = next((tmp_path / "interrupted").iterdir())
    resumed = _summaries(main(["resume", "--run", str(root)]))
    for left, right in zip(resumed, baseline, strict=True):
        assert _rows(Path(left.folder)) == _rows(Path(right.folder))
        assert load_outcome(Path(left.folder)).sha256 == load_outcome(Path(right.folder)).sha256
        saved = load_run_settings(Path(left.folder) / "config.json")
        assert saved.characteristic_sha256 == load_characteristic(characteristic_folder).sha256


@pytest.mark.parametrize("with_exact_char", [True, False])
def test_shapley_explains_an_outcome_run(tmp_path: Path, explained: tuple[Path, Path], with_exact_char: bool) -> None:
    agent_folder, characteristic_folder = explained
    outcome = _single(main(_outcome(agent_folder, characteristic_folder, tmp_path / "outcome")))
    flag = "--with-exact-char" if with_exact_char else "--no-with-exact-char"
    summary = _single(main([*_shapley(agent_folder, Path(outcome.folder), tmp_path / "shapley"), flag]))
    folder = Path(summary.folder)
    assert summary.stage == "shapley" and "shapley-outcome" in folder.name
    saved = load_run_settings(folder / "config.json")
    assert saved.characteristic_sha256 == load_outcome(Path(outcome.folder)).sha256
    # Score the restored best Shapley model eagerly against the ground truth the run should use.
    loaded = load_outcome(Path(outcome.folder))
    behaviour = load_characteristic(characteristic_folder)
    settings = load_settings(agent_folder / "config.json")
    agent = load_agent(agent_folder, settings.network, load_replay=False)
    frozen = OutcomeShapleyFrozen(
        FrozenAgent(agent.model, jnp.asarray(agent.epsilon, jnp.float32)),
        ExplainedOutcome(loaded.networks, behaviour.model),
    )
    function = outcome_function(frozen, behaviour.config.characteristic, loaded.config)
    exact = outcome_exact(
        behaviour.buffers.exact,
        function.policy,
        transition_model(settings.env),
        loaded.config.with_exact_char,
        loaded.config.gamma,
    )
    every = all_coalitions(2)
    values = exact.values[..., None] if with_exact_char else characteristic_table(function, exact.states, every)
    model = _shapley_model(folder)
    eager = exact_shapley_loss(model, function, exact.states, exact.state_valid, exact_shapley(values))
    logged = {row[0]: row[4] for row in _rows(folder)}
    np.testing.assert_allclose(logged[summary.best_epoch], eager, rtol=1e-5)
    with pytest.raises(ValueError, match="fresh buffers"):
        main([*_shapley(agent_folder, Path(outcome.folder), tmp_path / "reuse"), "--buffers", "characteristic"])


def test_on_policy_run_follows_train_v_on_policy(tmp_path: Path, explained: tuple[Path, Path]) -> None:
    agent_folder, characteristic_folder = explained
    summary = _single(main(_on_policy(agent_folder, characteristic_folder, tmp_path)))
    folder = Path(summary.folder)
    assert summary.epochs == 50 and summary.best_validation_loss is None and not summary.stopped_early
    rows = _csv(folder)
    # Source logging: global_step % 100 < num_envs among multiples of 8.
    assert [int(row["step"]) for row in rows] == [0, 104, 200, 304]
    assert all(row["validation_loss"] == "" and float(row["exact_loss"]) > 0 for row in rows)
    replay, steps, updates = _restored_on_policy(folder)
    assert steps == updates == 50 and int(replay.size) == 400
    coalitions = np.asarray(replay.coalitions).reshape(50, 8, 2)
    assert (coalitions == coalitions[:, :1]).all()  # One coalition shared by every environment per decision.
    pool = {tuple(row) for row in np.asarray(load_characteristic(characteristic_folder).buffers.train).tolist()}
    assert {tuple(row) for row in np.asarray(replay.explained).tolist()} <= pool
    observations = np.asarray(replay.transitions.observations).reshape(50, 8, 2)
    following = np.asarray(replay.transitions.next_observations).reshape(50, 8, 2)
    terminated = np.asarray(replay.transitions.terminated).reshape(50, 8)
    continuing = ~terminated[:-1]
    # Without termination the next decision starts from the stored next observation (no truncation here).
    np.testing.assert_array_equal(observations[1:][continuing], following[:-1][continuing])
    assert terminated.any()


def test_on_policy_resume_and_shapley(
    tmp_path: Path, explained: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    agent_folder, characteristic_folder = explained
    baseline = _single(main(_on_policy(agent_folder, characteristic_folder, tmp_path / "baseline")))
    original = ExplanationWriter.checkpoint

    def checkpoint(self: ExplanationWriter, snapshot: ExplanationSnapshot) -> CallbackToken:
        token = original(self, snapshot)
        self.store.wait()
        if int(snapshot.epoch) >= 20:
            raise OSError("injected interruption")
        return token

    with monkeypatch.context() as patch:
        patch.setattr(ExplanationWriter, "checkpoint", checkpoint)
        with pytest.raises(Exception, match="injected interruption"):
            main(_on_policy(agent_folder, characteristic_folder, tmp_path / "interrupted"))
        with suppress(RuntimeError):
            jax.effects_barrier()  # Clear the expected failed callback effect.
    folder = next((tmp_path / "interrupted").iterdir())
    resumed = _single(main(["resume", "--run", str(folder)]))
    np.testing.assert_array_equal(_rows(Path(resumed.folder)), _rows(Path(baseline.folder)))  # NaN-aware.
    assert load_outcome(Path(resumed.folder)).sha256 == load_outcome(Path(baseline.folder)).sha256
    shapley = _single(main(_shapley(agent_folder, Path(baseline.folder), tmp_path / "shapley")))
    assert shapley.epochs == 3 and all(row[4] >= 0 for row in _rows(Path(shapley.folder)))


def test_outcome_validation(tmp_path: Path, explained: tuple[Path, Path]) -> None:
    agent_folder, characteristic_folder = explained
    for invalid in (("--variant", "outcome"), ("--tau", "0"), ("--train.num-envs", "0")):
        with pytest.raises(SystemExit):  # tyro reports configuration errors and exits.
            parse_args([*_outcome(agent_folder, characteristic_folder, tmp_path), *invalid])
    with pytest.raises(ValueError, match="transitions in the agent's replay"):
        main([*_outcome(agent_folder, characteristic_folder, tmp_path), "--train.train-size", "128"])
    with pytest.raises(ValueError, match="different agent"):
        other = train_dqn.main([*_dqn(tmp_path / "other"), "--seed", "7"])
        assert other is not None
        main(_outcome(Path(other.folder), characteristic_folder, tmp_path))
    assert not list(tmp_path.glob("*outcome*"))  # Validation fails before any run artifacts exist.


def _single(result: ExplanationSummary | list[ExplanationSummary] | None) -> ExplanationSummary:
    summaries = _summaries(result)
    assert len(summaries) == 1
    return summaries[0]


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


def _outcome(agent: Path, characteristic: Path, output: Path, *extra: str) -> list[str]:
    return [
        "outcome",
        "--agent",
        str(agent),
        "--characteristic",
        str(characteristic),
        "--hidden-widths",
        "8",
        "--tau",
        "0.01",
        "--train.epochs",
        "4",
        "--train.train-size",
        "32",
        "--train.validation-size",
        "8",
        "--train.batch-size",
        "8",
        "--output",
        str(output),
        *extra,
    ]


def _shapley(agent: Path, characteristic: Path, output: Path) -> list[str]:
    return [
        "shapley",
        "--agent",
        str(agent),
        "--characteristic",
        str(characteristic),
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


def _shapley_model(folder: Path) -> ShapleyModel:
    layout = ShapleyLayout(2, 1)
    model = ShapleyModel(shapley_network(layout, (8,)), layout, rngs=nnx.Rngs(0))
    state = nnx.state(model)
    store = ExplanationStore(folder, read_only=True)
    try:
        values = store.arrays("model", tuple(jax.tree.leaves(state)))
    finally:
        store.close()
    nnx.update(model, jax.tree.unflatten(jax.tree.structure(state), values))
    return model


def _on_policy(agent: Path, characteristic: Path, output: Path) -> list[str]:
    return [
        "outcome",
        "--variant",
        "on_policy",
        "--agent",
        str(agent),
        "--characteristic",
        str(characteristic),
        "--hidden-widths",
        "8",
        "--tau",
        "0.01",
        "--train.total-timesteps",
        "400",
        "--train.num-envs",
        "8",
        "--train.batch-size",
        "16",
        "--chunk-epochs",
        "7",
        "--output",
        str(output),
    ]


def _restored_on_policy(folder: Path) -> tuple[ConditionedReplay, int, int]:
    """The checkpointed replay, completed vector steps and optimizer updates."""
    config = load_run_settings(folder / "config.json").config
    assert isinstance(config, OutcomeRunConfig)
    context = prepare_context(config)
    template = template_state(context, config)
    assert isinstance(template, OnPolicyState)
    store = ExplanationStore(folder, read_only=True)
    try:
        state = restore_on_policy(store, template, config, context)[0]
    finally:
        store.close()
    return state.replay, int(state.step), int(state.optimizer.step[...])


def _csv(folder: Path) -> list[dict[str, str]]:
    with (folder / "metrics.csv").open() as source:
        return list(csv.DictReader(source))


def _rows(folder: Path) -> list[list[float]]:
    """Metric rows without timing; losses a run does not compute are NaN."""
    keys = ("epoch", "step", "train_loss", "validation_loss", "exact_loss", "best_epoch")
    return [[nan if row[key] == "" else float(row[key]) for key in keys] for row in _csv(folder)]
