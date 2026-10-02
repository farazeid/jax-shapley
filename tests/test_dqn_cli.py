"""CLI parsing, preflight, typed artifacts, and tiny host-orchestrated runs."""

import csv
import json
import os
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- fixed local module argv in CLI regression.
import sys
from pathlib import Path

import pytest

from jax_shapley.train_dqn import DQNRunConfig, HypercubeConfig, checkpoint_due, main, parse_args


def test_subcommands_tuple_settings_and_shell_override() -> None:
    config = parse_args([
        "hypercube",
        "--env.side-length",
        "6",
        "--env.dimension",
        "5",
        "--hidden-widths",
        "128",
        "128",
        "--train.gamma",
        "0.9",
        "--seed",
        "1",
        "--seed",
        "2",
    ])
    assert isinstance(config, DQNRunConfig)
    assert isinstance(config.env, HypercubeConfig)
    assert config.env.dimension == 5
    assert config.hidden_widths == (128, 128)
    assert config.train.gamma == pytest.approx(0.9)
    assert config.seed == 2
    assert config.network.input_features == 5
    mastermind = parse_args(["mastermind222"])
    assert isinstance(mastermind, DQNRunConfig)
    assert mastermind.network.input_features == 8


def test_help_and_invalid_settings(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as result:
        parse_args(["gridworld", "--help"])
    assert result.value.code == 0
    assert "--train.learning-rate" in capsys.readouterr().out
    with pytest.raises((ValueError, SystemExit)):
        parse_args(["gridworld", "--train.gamma", "nan"])
    with pytest.raises(SystemExit):
        parse_args(["gridworld", "--unknown"])


def test_validate_and_compile_only_create_no_artifacts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "absent"
    assert main(["gridworld", "--validate-only", "--output", str(output)]) is None
    assert "reference_commit" in capsys.readouterr().out
    assert not output.exists()
    assert (
        main([
            "gridworld",
            "--compile-only",
            "--hidden-widths",
            "4",
            "--train.buffer-size",
            "7",
            "--chunk-steps",
            "3",
            "--output",
            str(output),
        ])
        is None
    )
    assert "Compiled" in capsys.readouterr().out
    assert not output.exists()


def test_preflight_failure_stops_before_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from jax_shapley import train_dqn

    def fail(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("compiler failure")

    monkeypatch.setattr(train_dqn, "compile_program", fail)
    output = tmp_path / "absent"
    with pytest.raises(RuntimeError, match="compiler failure"):
        main(["gridworld", "--output", str(output)])
    assert not output.exists()


def test_tiny_training_artifacts_and_evaluation(tmp_path: Path) -> None:
    result = main([
        "mastermind222",
        "--train.total-timesteps",
        "17",
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
        str(tmp_path),
    ])
    assert result is not None
    assert result.environment_steps == 16  # Upstream floor division.
    assert result.updates == 7
    assert result.target_updates == 7
    assert result.evaluation_episodes >= 2
    folder = Path(result.folder)
    assert {"config.json", "summary.json", "metrics.csv", "checkpoints"} <= {file.name for file in folder.iterdir()}
    with (folder / "metrics.csv").open() as source:
        rows = list(csv.DictReader(source))
    assert {row["kind"] for row in rows} >= {"episode", "evaluation"}
    settings = json.loads((folder / "config.json").read_text())
    assert settings["config"]["train"]["total_timesteps"] == 17
    assert settings["network"]["input_features"] == 8


def test_optional_artifacts_and_checkpoint_cadence(tmp_path: Path) -> None:
    result = main([
        "gridworld",
        "--train.total-timesteps",
        "4",
        "--train.buffer-size",
        "3",
        "--train.batch-size",
        "2",
        "--hidden-widths",
        "4",
        "--chunk-steps",
        "3",
        "--checkpoint-mode",
        "none",
        "--output",
        str(tmp_path),
    ])
    assert result is not None
    assert not (Path(result.folder) / "checkpoints").exists()
    assert checkpoint_due(1, 1, 4)
    assert checkpoint_due(5_000, 1, 50_000)
    assert not checkpoint_due(4_999, 1, 50_000)
    assert checkpoint_due(625, 16, 100_000)


def test_module_entrypoint_training_and_completed_resume(tmp_path: Path) -> None:
    # Exercise python -m in fresh interpreters: host adapters must share the
    # trainer's class identities for CLI parsing and saved-settings restoration.
    environment = dict(os.environ)
    environment["JAX_PLATFORMS"] = "cpu"
    command = [sys.executable, "-m", "jax_shapley.train_dqn"]
    trained = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] -- fixed local module argv.
        [
            *command,
            "mastermind222",
            "--train.total-timesteps",
            "8",
            "--train.num-envs",
            "2",
            "--train.learning-starts",
            "0",
            "--train.train-frequency",
            "1",
            "--train.buffer-size",
            "7",
            "--train.batch-size",
            "3",
            "--train.eval-episodes",
            "2",
            "--hidden-widths",
            "4",
            "--chunk-steps",
            "3",
            "--output",
            str(tmp_path),
        ],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert trained.returncode == 0, trained.stdout + trained.stderr
    folder = next(tmp_path.iterdir())
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["environment_steps"] == 8
    assert summary["updates"] == 3
    assert summary["evaluation_episodes"] >= 2
    original_metrics = (folder / "metrics.csv").read_bytes()
    (folder / "summary.json").unlink()
    resumed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] -- fixed local module argv.
        [*command, "resume", "--run", str(folder)],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert json.loads((folder / "summary.json").read_text()) == summary
    assert (folder / "metrics.csv").read_bytes() == original_metrics
