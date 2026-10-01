"""Execute shell orchestration against a capture-only uv, never a training job."""

import json
import os
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- fixed local recipe argv, no shell interpolation.
import sys
from pathlib import Path
from typing import cast

import pytest

from jax_shapley.config import HypercubeConfig
from jax_shapley.train_dqn import parse_args

ROOT = Path(__file__).parents[1]


def test_gridworld_and_mastermind_source_settings(tmp_path: Path) -> None:
    grid = parse_args(_capture(tmp_path, "gridworld.sh")[0][4:])
    assert (grid.train.num_envs, grid.train.total_timesteps, grid.train.buffer_size) == (1, 50_000, 50_000)
    assert grid.train.learning_starts == 10_000
    assert grid.hidden_widths == (64, 64)
    mastermind = parse_args(_capture(tmp_path, "mastermind222.sh", ("--seed", "3"))[0][4:])
    assert (mastermind.train.num_envs, mastermind.train.total_timesteps) == (16, 100_000)
    assert mastermind.train.exploration_fraction == pytest.approx(0.25)
    assert mastermind.seed == 3


def test_all_hypercube_settings_and_seed_selection(tmp_path: Path) -> None:
    commands = _capture(tmp_path, "hypercube.sh")
    assert len(commands) == 400
    # Expected values transcribed from the pinned upstream timestep table.
    expected = {
        (2, 2): 100_000,
        (3, 2): 50_000,
        (4, 2): 50_000,
        (5, 2): 50_000,
        (6, 2): 50_000,
        (2, 3): 50_000,
        (3, 3): 50_000,
        (4, 3): 50_000,
        (5, 3): 100_000,
        (6, 3): 100_000,
        (2, 4): 50_000,
        (3, 4): 50_000,
        (4, 4): 100_000,
        (5, 4): 100_000,
        (6, 4): 500_000,
        (2, 5): 50_000,
        (3, 5): 100_000,
        (4, 5): 500_000,
        (5, 5): 500_000,
        (6, 5): 1_000_000,
    }
    for index, command in enumerate(commands):
        config = parse_args(command[4:])
        assert isinstance(config.env, HypercubeConfig)
        timesteps = expected[config.env.side_length, config.env.dimension]
        assert config.train.total_timesteps == config.train.buffer_size == timesteps
        assert config.train.learning_starts == timesteps // 50
        assert config.train.num_envs == 16
        assert config.hidden_widths == (128, 128)
        assert config.train.end_e == pytest.approx(0.05)
        assert config.seed == index % 20 + 1
    selected = _capture(tmp_path, "hypercube.sh", ("--seed", "3", "--env.side-length", "6", "--env.dimension", "5"))
    assert len(selected) == 1
    assert parse_args(selected[0][4:]).seed == 3


def test_smoke_recipe_is_short_and_has_updates(tmp_path: Path) -> None:
    config = parse_args(_capture(tmp_path, "smoke.sh")[0][4:])
    assert config.train.total_timesteps == 32
    assert config.train.learning_starts == 0
    assert config.train.eval_episodes == 2


def _capture(tmp_path: Path, script: str, extra: tuple[str, ...] = ()) -> list[list[str]]:
    binary = tmp_path / "bin"
    binary.mkdir(exist_ok=True)
    ledger = tmp_path / "commands.jsonl"
    if ledger.exists():
        ledger.unlink()
    stub = binary / "uv"
    stub.write_text(
        f"#!{sys.executable}\nimport json, os, sys\nwith open(os.environ['DQN_TEST_CAPTURE'], 'a') as f:\n    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
    )
    stub.chmod(0o755)
    environment = dict(os.environ)
    environment["PATH"] = str(binary) + os.pathsep + environment["PATH"]
    environment["DQN_TEST_CAPTURE"] = str(ledger)
    bash = shutil.which("bash")
    if bash is None:
        raise RuntimeError("bash is required for recipe validation")
    subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] -- trusted local script argv.
        [bash, str(ROOT / "experiments/dqn" / script), "--validate-only", *extra],
        env=environment,
        check=True,
        cwd=tmp_path,
    )
    commands: list[list[str]] = []
    for line in ledger.read_text().splitlines():
        value: object = json.loads(line)
        if not isinstance(value, list) or not all(isinstance(arg, str) for arg in value):
            raise ValueError("capture must contain string arguments")
        commands.append(cast(list[str], value))  # Validated JSON boundary.
    return commands
