"""Off-policy outcome runs: the CLI pipeline, exact-loss logging, sweeps and exact resume."""

import csv
from contextlib import suppress
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
)
from jax_shapley.outcome import OutcomeCharacteristic, exact_outcome, exact_outcome_loss, learnt_behaviour_table
from jax_shapley.run_io import CallbackToken, load_agent, load_replay, load_settings
from jax_shapley.tabular import transition_model
from jax_shapley.train_explanations import (
    FrozenAgent,
    OutcomeFrozen,
    conditioned_policy,
    main,
    parse_args,
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


def test_outcome_validation(tmp_path: Path, explained: tuple[Path, Path]) -> None:
    agent_folder, characteristic_folder = explained
    for invalid in (("--variant", "on_policy"), ("--tau", "0")):
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


def _rows(folder: Path) -> list[list[float]]:
    with (folder / "metrics.csv").open() as source:
        return [
            [float(row[key]) for key in ("epoch", "step", "train_loss", "validation_loss", "exact_loss", "best_epoch")]
            for row in csv.DictReader(source)
        ]
