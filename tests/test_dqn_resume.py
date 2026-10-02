"""Same-folder interruption recovery and read-only inspection of complete snapshots."""

import csv
import json
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax import Array

from jax_shapley.run_io import (
    CallbackToken,
    CheckpointStore,
    RunWriter,
    Snapshot,
    load_agent,
    load_settings,
    restore_run,
)
from jax_shapley.train_dqn import (
    DQNRunConfig,
    EnvironmentConfig,
    Mastermind222Config,
    ResumeConfig,
    initialize_run,
    main,
    make_snapshot,
    parse_args,
)
from jax_shapley.wrappers import EpisodeStatisticsWrapper


def test_resume_cli_saved_settings_and_read_only_modes(tmp_path: Path) -> None:
    options = parse_args(["resume", "--run", str(tmp_path)])
    assert isinstance(options, ResumeConfig)
    with pytest.raises(SystemExit):
        parse_args(["resume", "--run", str(tmp_path), "--train.gamma", "0.5"])
    result = main(_args(tmp_path))
    assert result is not None
    folder = Path(result.folder)
    config = load_settings(folder / "config.json")
    assert isinstance(config, DQNRunConfig)
    before = _files(folder)
    assert main(["resume", "--run", str(folder), "--validate-only"]) is None
    assert main(["resume", "--run", str(folder), "--compile-only"]) is None
    assert _files(folder) == before
    (folder / "summary.json").unlink()
    metrics = (folder / "metrics.csv").read_bytes()
    recovered = main(["resume", "--run", str(folder)])
    assert recovered is not None and recovered.environment_steps == 32
    assert (folder / "summary.json").exists()
    assert (folder / "metrics.csv").read_bytes() == metrics
    assert len(list((folder / "checkpoints").glob("[0-9]*"))) == 1
    assert load_agent(folder, config.network, load_replay=False).replay is None


@pytest.mark.parametrize("stage", ["training", "evaluation"])
def test_interruption_restores_without_duplicate_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    baseline = main(_args(tmp_path / "baseline"))
    assert baseline is not None
    original_checkpoint = RunWriter.checkpoint
    original_evaluation = RunWriter.evaluation

    def checkpoint(self: RunWriter[EnvironmentConfig], snapshot: Snapshot) -> CallbackToken:
        token = original_checkpoint(self, snapshot)
        self.store.wait()
        if stage == "training" and int(snapshot.vector_step) == 3 and int(snapshot.phase) == 0:
            raise OSError("injected interruption after committed training checkpoint")
        return token

    def evaluation(
        self: RunWriter[EnvironmentConfig], returns: Array, completed: Array, first_index: Array
    ) -> CallbackToken:
        token = original_evaluation(self, returns, completed, first_index)
        if stage == "evaluation":
            raise OSError("injected interruption after uncommitted evaluation rows")
        return token

    with monkeypatch.context() as patch:
        patch.setattr(RunWriter, "checkpoint", checkpoint)
        patch.setattr(RunWriter, "evaluation", evaluation)
        with pytest.raises(Exception, match="injected interruption"):
            main(_args(tmp_path / "interrupted"))
        _clear_failed_effects()
    folder = next((tmp_path / "interrupted").iterdir())
    assert not (folder / "summary.json").exists()
    with (folder / "metrics.csv").open("a") as output:
        output.write("uncommitted tail\n")
    result = main(["resume", "--run", str(folder)])
    assert result is not None and result.environment_steps == baseline.environment_steps
    assert _rows(folder) == _rows(Path(baseline.folder))
    config = load_settings(folder / "config.json")
    actual = load_agent(folder, config.network)
    expected = load_agent(Path(baseline.folder), config.network)
    from flax import nnx

    for left, right in zip(
        jax.tree.leaves(nnx.state((actual.model, actual.target))),
        jax.tree.leaves(nnx.state((expected.model, expected.target))),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)
    assert isinstance(config.env, Mastermind222Config)
    env = EpisodeStatisticsWrapper(config.env.make())
    complete_states = []
    for saved_folder in (folder, Path(baseline.folder)):
        saved_config = load_settings(saved_folder / "config.json")
        template = initialize_run(env, env.default_params, saved_config.network, saved_config.train, jax.random.key(99))
        store = CheckpointStore(saved_folder, read_only=True)
        try:
            state, _ = restore_run(store, template, saved_config)
            complete_states.append(state)
        finally:
            store.close()
    left_state, right_state = complete_states
    for left, right in zip(
        jax.tree.leaves(nnx.state(left_state.optimizer)), jax.tree.leaves(nnx.state(right_state.optimizer)), strict=True
    ):
        np.testing.assert_array_equal(left, right)
    for left, right in zip(
        jax.tree.leaves((left_state.observations, left_state.env_state, left_state.updates, left_state.target_updates)),
        jax.tree.leaves((
            right_state.observations,
            right_state.env_state,
            right_state.updates,
            right_state.target_updates,
        )),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(jax.random.key_data(left_state.key), jax.random.key_data(right_state.key))
    assert actual.replay is not None and expected.replay is not None
    for left, right in zip(jax.tree.leaves(actual.replay), jax.tree.leaves(expected.replay), strict=True):
        np.testing.assert_array_equal(left, right)


def test_resume_rejects_changed_prefix_and_config_before_mutation(tmp_path: Path) -> None:
    result = main(_args(tmp_path))
    assert result is not None
    folder = Path(result.folder)
    metrics = folder / "metrics.csv"
    original = metrics.read_bytes()
    metrics.write_bytes(b"X" + original[1:])
    before = _files(folder)
    with pytest.raises(ValueError, match="prefix"):
        main(["resume", "--run", str(folder)])
    assert _files(folder) == before
    metrics.write_bytes(original)
    settings = json.loads((folder / "config.json").read_text())
    settings["config"]["train"]["gamma"] = 0.5
    (folder / "config.json").write_text(json.dumps(settings))
    before = _files(folder)
    with pytest.raises(ValueError, match="identical"):
        main(["resume", "--run", str(folder)])
    assert _files(folder) == before


def test_required_checkpoint_failure_prevents_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = CheckpointStore.wait

    def fail(self: CheckpointStore) -> None:
        original(self)
        if self.sequence is not None:
            raise OSError("injected asynchronous completion failure")

    with monkeypatch.context() as patch:
        patch.setattr(CheckpointStore, "wait", fail)
        with pytest.raises(Exception, match="completion failure"):
            main(_args(tmp_path))
        _clear_failed_effects()
    folder = next(tmp_path.iterdir())
    assert not (folder / "summary.json").exists()
    assert load_agent(folder, load_settings(folder / "config.json").network, load_replay=False).vector_step >= 1


@pytest.mark.parametrize("implementation", ["rbg", "unsafe_rbg"])
def test_resume_uses_saved_prng_after_default_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, implementation: str
) -> None:
    original = RunWriter.checkpoint

    def interrupt(self: RunWriter[EnvironmentConfig], snapshot: Snapshot) -> CallbackToken:
        token = original(self, snapshot)
        assert snapshot.key_implementation == implementation
        self.store.wait()
        if int(snapshot.vector_step) == 2 and int(snapshot.phase) == 0:
            raise OSError("injected PRNG interruption")
        return token

    with jax.default_prng_impl(implementation):
        baseline = main([*_args(tmp_path / "baseline"), "--train.total-timesteps", "8"])
        assert baseline is not None
        with monkeypatch.context() as patch:
            patch.setattr(RunWriter, "checkpoint", interrupt)
            with pytest.raises(Exception, match="PRNG interruption"):
                main([*_args(tmp_path / "interrupted"), "--train.total-timesteps", "8"])
            _clear_failed_effects()
    folder = next((tmp_path / "interrupted").iterdir())
    with jax.default_prng_impl("threefry2x32"):
        before = _files(folder)
        assert main(["resume", "--run", str(folder), "--validate-only"]) is None
        assert main(["resume", "--run", str(folder), "--compile-only"]) is None
        assert _files(folder) == before
        resumed = main(["resume", "--run", str(folder)])
    assert resumed is not None and resumed.environment_steps == baseline.environment_steps
    assert _rows(folder) == _rows(Path(baseline.folder))
    config = load_settings(folder / "config.json")
    actual, expected = load_agent(folder, config.network), load_agent(Path(baseline.folder), config.network)
    for left, right in zip(
        jax.tree.leaves(nnx.state((actual.model, actual.target))),
        jax.tree.leaves(nnx.state((expected.model, expected.target))),
        strict=True,
    ):
        np.testing.assert_array_equal(left, right)
    assert actual.replay is not None and expected.replay is not None
    for left, right in zip(jax.tree.leaves(actual.replay), jax.tree.leaves(expected.replay), strict=True):
        np.testing.assert_array_equal(left, right)
    store = CheckpointStore(folder, read_only=True)
    try:
        assert store.metadata().key_implementation == implementation
    finally:
        store.close()


def test_legacy_full_run_settings_restore_without_rewriting_config(tmp_path: Path) -> None:
    baseline = main([*_args(tmp_path), "--train.total-timesteps", "8"])
    assert baseline is not None
    folder = Path(baseline.folder)
    config = load_settings(folder / "config.json")
    env = EpisodeStatisticsWrapper(config.env.make())
    template = initialize_run(env, env.default_params, config.network, config.train, jax.random.key(99))
    store = CheckpointStore(folder)
    legacy = json.loads((folder / "config.json").read_text())
    legacy["config"]["checkpoint_mode"] = "run"
    legacy_text = json.dumps(legacy, indent=2) + "\n"
    try:
        restored, metadata = restore_run(store, template, config)
        snapshot = make_snapshot(
            restored,
            jnp.asarray(metadata.phase, jnp.int32),
            jnp.asarray(metadata.evaluation_episodes, jnp.int32),
            config.train,
        )
        store.save(snapshot, replace(metadata, settings=legacy_text))
        store.wait()
    finally:
        store.close()
    (folder / "config.json").write_text(legacy_text)
    (folder / "summary.json").unlink()
    before = _files(folder)
    assert main(["resume", "--run", str(folder), "--validate-only"]) is None
    assert main(["resume", "--run", str(folder), "--compile-only"]) is None
    assert _files(folder) == before
    recovered = main(["resume", "--run", str(folder)])
    assert recovered == baseline
    assert (folder / "config.json").read_text() == legacy_text
    assert (folder / "metrics.csv").read_bytes() == before["metrics.csv"][0]


def test_every_run_has_complete_recoverable_snapshot(tmp_path: Path) -> None:
    result = main([*_args(tmp_path), "--train.total-timesteps", "4", "--train.eval-episodes", "0"])
    assert result is not None
    folder = Path(result.folder)
    config = load_settings(folder / "config.json")
    store = CheckpointStore(folder, read_only=True)
    try:
        env = EpisodeStatisticsWrapper(config.env.make())
        template = initialize_run(env, env.default_params, config.network, config.train, jax.random.key(99))
        restored, metadata = restore_run(store, template, config)
    finally:
        store.close()
    assert int(restored.vector_step) == 2
    assert int(restored.replay.size) == 4
    assert metadata.phase == 2
    metrics = (folder / "metrics.csv").read_bytes()
    recovered = main(["resume", "--run", str(folder)])
    assert recovered is not None and recovered.environment_steps == 4
    assert (folder / "metrics.csv").read_bytes() == metrics


def _args(output: Path) -> list[str]:
    return [
        "mastermind222",
        "--train.total-timesteps",
        "32",
        "--train.num-envs",
        "2",
        "--train.buffer-size",
        "7",
        "--train.batch-size",
        "3",
        "--train.learning-starts",
        "0",
        "--train.train-frequency",
        "1",
        "--hidden-widths",
        "4",
        "--chunk-steps",
        "3",
        "--train.eval-episodes",
        "2",
        "--output",
        str(output),
    ]


def _files(folder: Path) -> dict[str, tuple[bytes, int]]:
    return {
        str(path.relative_to(folder)): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in folder.rglob("*")
        if path.is_file()
    }


def _rows(folder: Path) -> list[dict[str, str]]:
    with (folder / "metrics.csv").open() as source:
        return [{key: value for key, value in row.items() if key != "sps"} for row in csv.DictReader(source)]


def _clear_failed_effects() -> None:
    # Clear the expected failed callback effect, already asserted by its test.
    with suppress(RuntimeError):
        jax.effects_barrier()
