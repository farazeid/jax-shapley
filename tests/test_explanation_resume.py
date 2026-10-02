"""Explanation CLI artifacts, exact interruption recovery and rejection of changed inputs."""

import csv
import json
from contextlib import suppress
from pathlib import Path
from typing import cast

import jax
import numpy as np
import orbax.checkpoint as ocp
import pytest

from jax_shapley import train_dqn
from jax_shapley.explanation_io import (
    ExplanationSnapshot,
    ExplanationWriter,
    load_characteristic,
    load_run_settings,
)
from jax_shapley.run_io import CallbackToken
from jax_shapley.train_explanations import CharacteristicRunConfig, ShapleyRunConfig, main, parse_args

ITEMS = ("model", "best", "optimizer", "buffers", "runtime")


@pytest.fixture(scope="module")
def agent(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("agent")
    summary = train_dqn.main([
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
    ])
    assert summary is not None
    return Path(summary.folder)


def test_cli_artifacts_and_inspection_modes(tmp_path: Path, agent: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Validation reads no artifacts, so recipes can validate chained stages before they exist.
    assert main([*_characteristic(Path("missing"), tmp_path), "--validate-only"]) is None
    assert json.loads(capsys.readouterr().out)["stage"] == "characteristic"
    assert main([*_characteristic(agent, tmp_path), "--compile-only"]) is None
    assert not tmp_path.exists() or not any(tmp_path.iterdir())
    summary = main(_characteristic(agent, tmp_path))
    assert summary is not None and summary.stage == "characteristic" and summary.epochs == 6
    folder = Path(summary.folder)
    assert {path.name for path in folder.iterdir()} >= {"config.json", "metrics.csv", "summary.json", "checkpoints"}
    settings = load_run_settings(folder / "config.json")
    assert isinstance(settings.config, CharacteristicRunConfig) and settings.characteristic_sha256 is None
    assert len(_rows(folder)) == 6
    loaded = load_characteristic(folder)
    assert loaded.buffers.train.shape == (32, 8) and loaded.buffers.exact.shape == (32, 8)
    metrics = (folder / "metrics.csv").read_bytes()
    (folder / "summary.json").unlink()
    assert main(["resume", "--run", str(folder)]) is not None
    assert (folder / "metrics.csv").read_bytes() == metrics and (folder / "summary.json").exists()
    for buffers in ("fresh", "characteristic"):
        shapley = main(_shapley(agent, folder, tmp_path / buffers, buffers))
        assert shapley is not None and shapley.stage == "shapley"
        saved = load_run_settings(Path(shapley.folder) / "config.json")
        assert isinstance(saved.config, ShapleyRunConfig) and saved.characteristic_sha256 == loaded.sha256


def test_interruption_resumes_to_the_uninterrupted_result(
    tmp_path: Path, agent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = main(_characteristic(agent, tmp_path / "baseline"))
    assert baseline is not None
    original = ExplanationWriter.checkpoint

    def checkpoint(self: ExplanationWriter, snapshot: ExplanationSnapshot) -> CallbackToken:
        token = original(self, snapshot)
        self.store.wait()
        if int(snapshot.epoch) == 3:
            raise OSError("injected interruption after committed checkpoint")
        return token

    with monkeypatch.context() as patch:
        patch.setattr(ExplanationWriter, "checkpoint", checkpoint)
        with pytest.raises(Exception, match="injected interruption"):
            main(_characteristic(agent, tmp_path / "interrupted"))
        with suppress(RuntimeError):
            jax.effects_barrier()  # Clear the expected failed callback effect.
    folder = next((tmp_path / "interrupted").iterdir())
    assert not (folder / "summary.json").exists()
    with (folder / "metrics.csv").open("a") as output:
        output.write("uncommitted tail\n")
    resumed = main(["resume", "--run", str(folder)])
    assert resumed is not None and resumed.epochs == baseline.epochs
    assert _rows(folder) == _rows(Path(baseline.folder))
    assert _arrays(folder) == _arrays(Path(baseline.folder))


def test_resume_and_shapley_reject_changed_inputs(tmp_path: Path, agent: Path) -> None:
    summary = main(_characteristic(agent, tmp_path / "characteristic"))
    assert summary is not None
    folder = Path(summary.folder)
    settings = (folder / "config.json").read_text()
    tampered = json.loads(settings)
    tampered["agent_sha256"] = "0" * 64
    (folder / "config.json").write_text(json.dumps(tampered, indent=2) + "\n")
    with pytest.raises(ValueError, match="changed"):
        main(["resume", "--run", str(folder)])
    tampered = json.loads(settings)
    tampered["config"]["train"]["epochs"] = "6"
    (folder / "config.json").write_text(json.dumps(tampered))
    with pytest.raises(ValueError):
        main(["resume", "--run", str(folder)])
    (folder / "config.json").write_text(settings)
    other = main(["characteristic", *_characteristic(agent, tmp_path)[1:], "--kind", "prediction"])
    assert other is not None
    with pytest.raises(ValueError, match="identical train/validation sizes"):
        main([*_shapley(agent, folder, tmp_path / "shapley", "characteristic"), "--train.buffer-size", "80"])
    with pytest.raises(SystemExit):
        parse_args(["resume", "--run", str(folder), "--train.epochs", "3"])


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


def _shapley(agent: Path, characteristic: Path, output: Path, buffers: str) -> list[str]:
    return [
        "shapley",
        "--agent",
        str(agent),
        "--characteristic",
        str(characteristic),
        "--buffers",
        buffers,
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


def _rows(folder: Path) -> list[dict[str, str]]:
    with (folder / "metrics.csv").open() as source:
        return [{key: value for key, value in row.items() if key != "seconds"} for row in csv.DictReader(source)]


def _arrays(folder: Path) -> dict[str, list[bytes]]:
    """Every saved numerical array; equality means an exact continuation."""
    options = ocp.CheckpointManagerOptions(read_only=True)
    manager = ocp.CheckpointManager(folder.resolve() / "checkpoints", options=options)
    try:
        items = {item: ocp.args.StandardRestore() for item in ITEMS}
        restored = manager.restore(manager.latest_step(), args=ocp.args.Composite(**items))
        return {item: _bytes(restored[item]) for item in ITEMS}
    finally:
        manager.close()


def _bytes(item: object) -> list[bytes]:
    assert isinstance(item, dict)
    arrays = cast(dict[str, object], item)  # Orbax restores named arrays.
    return [np.asarray(arrays[name]).tobytes() for name in sorted(arrays)]
