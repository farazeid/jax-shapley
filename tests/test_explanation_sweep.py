"""Seed sweeps: one compiled program for many seeds, with per-seed folders equal to single runs."""

import csv
import json
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp
import pytest

from jax_shapley import train_dqn
from jax_shapley.explanation_io import (
    ExplanationSnapshot,
    ExplanationSummary,
    ExplanationWriter,
    load_characteristic,
    load_run_settings,
    sweep_folders,
)
from jax_shapley.run_io import CallbackToken, load_settings
from jax_shapley.train_explanations import main, parse_args

SEEDS = ("1", "2", "3")
ITEMS = ("model", "best", "optimizer", "buffers", "runtime")


@pytest.fixture(scope="module")
def agent(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("agent")
    summary = train_dqn.main(_dqn(output))
    assert summary is not None
    return Path(summary.folder)


def test_sweep_seeds_equal_single_runs(tmp_path: Path, agent: Path) -> None:
    sweep = _summaries(main([*_characteristic(agent, tmp_path / "sweep"), "--seeds", *SEEDS]))
    assert [summary.folder.rsplit("/", 1)[-1] for summary in sweep] == [f"seed-{seed}" for seed in SEEDS]
    for seed, summary in zip(SEEDS, sweep, strict=True):
        single = _summaries(main([*_characteristic(agent, tmp_path / f"single-{seed}"), "--seed", seed]))[0]
        swept, alone = Path(summary.folder), Path(single.folder)
        assert summary.epochs == single.epochs and summary.best_epoch == single.best_epoch
        # Each seed folder records ordinary single-run settings (only the output parent differs).
        swept_settings, alone_settings = (
            load_run_settings(swept / "config.json"),
            load_run_settings(alone / "config.json"),
        )
        assert replace(swept_settings.config, output=Path()) == replace(alone_settings.config, output=Path())
        assert swept_settings.agent_sha256 == alone_settings.agent_sha256
        left, right = _rows(swept), _rows(alone)
        assert [row[:2] for row in left] == [row[:2] for row in right]
        np.testing.assert_allclose(np.asarray(left)[:, 2:], np.asarray(right)[:, 2:], rtol=1e-5, atol=1e-7)
        swept_arrays, alone_arrays = _arrays(swept), _arrays(alone)
        np.testing.assert_array_equal(swept_arrays["buffers"][0], alone_arrays["buffers"][0])
        for item in ("model", "best"):
            for left_array, right_array in zip(swept_arrays[item], alone_arrays[item], strict=True):
                np.testing.assert_allclose(left_array, right_array, rtol=1e-5, atol=1e-6)


def test_matched_shapley_sweep_pairs_characteristic_seeds(tmp_path: Path, agent: Path) -> None:
    characteristics = _summaries(main([*_characteristic(agent, tmp_path / "characteristic"), "--seeds", *SEEDS]))
    root = Path(characteristics[0].folder).parent
    assert sweep_folders(root) == [Path(summary.folder) for summary in characteristics]
    shapley = _summaries(main([*_shapley(agent, root, tmp_path / "shapley"), "--seeds", *SEEDS]))
    for characteristic, summary in zip(characteristics, shapley, strict=True):
        saved = load_run_settings(Path(summary.folder) / "config.json")
        assert saved.characteristic_sha256 == load_characteristic(Path(characteristic.folder)).sha256
    shared = _summaries(
        main([*_shapley(agent, Path(characteristics[0].folder), tmp_path / "shared"), "--seeds", "1", "2"])
    )
    digests = {load_run_settings(Path(summary.folder) / "config.json").characteristic_sha256 for summary in shared}
    assert len(digests) == 1


def test_interrupted_sweep_resumes_to_the_uninterrupted_result(
    tmp_path: Path, agent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = _summaries(main([*_characteristic(agent, tmp_path / "baseline"), "--seeds", *SEEDS]))
    original = ExplanationWriter.checkpoint

    def checkpoint(self: ExplanationWriter, snapshot: ExplanationSnapshot) -> CallbackToken:
        token = original(self, snapshot)
        self.store.wait()
        # Interrupt part-way through a sweep checkpoint: seeds 1-2 committed epoch 3, seed 3 did not.
        if int(snapshot.epoch) == 3 and self.folder.name == "seed-2":
            raise OSError("injected interruption mid-sweep")
        return token

    with monkeypatch.context() as patch:
        patch.setattr(ExplanationWriter, "checkpoint", checkpoint)
        with pytest.raises(Exception, match="injected interruption"):
            main([*_characteristic(agent, tmp_path / "interrupted"), "--seeds", *SEEDS])
        with suppress(RuntimeError):
            jax.effects_barrier()  # Clear the expected failed callback effect.
    root = next((tmp_path / "interrupted").iterdir())
    resumed = _summaries(main(["resume", "--run", str(root)]))
    for left, right in zip(resumed, baseline, strict=True):
        assert _rows(Path(left.folder)) == _rows(Path(right.folder))
        assert _bytes(Path(left.folder)) == _bytes(Path(right.folder))


def test_sweep_pairs_seeds_with_agents_from_a_directory(tmp_path: Path) -> None:
    agents = tmp_path / "agents"
    for seed in ("1", "2"):
        assert train_dqn.main([*_dqn(agents), "--seed", seed]) is not None
    sweep = _summaries(main([*_characteristic(agents, tmp_path / "characteristic"), "--seeds", "2", "1"]))
    by_seed = {load_settings(folder / "config.json").seed: folder for folder in agents.iterdir()}
    for seed, summary in zip((2, 1), sweep, strict=True):
        saved = load_run_settings(Path(summary.folder) / "config.json")
        assert saved.config.agent == by_seed[seed]
    digests = {load_run_settings(Path(summary.folder) / "config.json").agent_sha256 for summary in sweep}
    assert len(digests) == 2
    with pytest.raises(ValueError, match="no agent for seeds"):
        main([*_characteristic(agents, tmp_path / "missing"), "--seeds", "1", "3"])


def test_seed_folders_stand_alone_and_cli_validation(tmp_path: Path, agent: Path) -> None:
    sweep = _summaries(main([*_characteristic(agent, tmp_path), "--seeds", "4", "5"]))
    folder = Path(sweep[1].folder)
    assert load_characteristic(folder).buffers.train.shape == (32, 8)
    assert _summaries(main(["resume", "--run", str(folder)]))[0].epochs == sweep[1].epochs
    assert json.loads((folder.parent / "sweep.json").read_text())["seeds"] == [4, 5]
    with pytest.raises(SystemExit):
        parse_args([*_characteristic(agent, tmp_path), "--seeds", "1", "1"])


def _summaries(result: ExplanationSummary | list[ExplanationSummary] | None) -> list[ExplanationSummary]:
    assert result is not None
    return result if isinstance(result, list) else [result]


def _dqn(output: Path) -> list[str]:
    return [
        "mastermind222",
        "--hidden-widths",
        "8",
        "--chunk-steps",
        "4",
        "--train.num-envs",
        "2",
        "--train.total-timesteps",
        "64",
        "--train.buffer-size",
        "16",
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
        "--hidden-widths",
        "8",
        "--train.epochs",
        "6",
        "--train.buffer-size",
        "40",
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
        "--hidden-widths",
        "8",
        "--train.epochs",
        "4",
        "--train.buffer-size",
        "40",
        "--train.num-envs",
        "4",
        "--train.batch-size",
        "8",
        "--output",
        str(output),
    ]


def _rows(folder: Path) -> list[list[float]]:
    with (folder / "metrics.csv").open() as source:
        return [
            [float(row[key]) for key in ("epoch", "step", "train_loss", "validation_loss", "exact_loss")]
            for row in csv.DictReader(source)
        ]


def _arrays(folder: Path) -> dict[str, list[np.ndarray]]:
    options = ocp.CheckpointManagerOptions(read_only=True)
    manager = ocp.CheckpointManager(folder.resolve() / "checkpoints", options=options)
    try:
        restored = manager.restore(
            manager.latest_step(), args=ocp.args.Composite(**{item: ocp.args.StandardRestore() for item in ITEMS})
        )
        return {item: [np.asarray(value) for _, value in sorted(dict(restored[item]).items())] for item in ITEMS}
    finally:
        manager.close()


def _bytes(folder: Path) -> dict[str, list[bytes]]:
    return {item: [array.tobytes() for array in arrays] for item, arrays in _arrays(folder).items()}
