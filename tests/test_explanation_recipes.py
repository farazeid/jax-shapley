"""Execute explanation recipes against a capture-only uv, never a training job."""

import json
import os
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- fixed local recipe argv, no shell interpolation.
import sys
from pathlib import Path
from typing import cast

import pytest

from jax_shapley import train_dqn
from jax_shapley.train_explanations import CharacteristicRunConfig, ShapleyRunConfig, parse_args

ROOT = Path(__file__).parents[1]
AGENT = ("--agent", "runs/agent")


@pytest.mark.parametrize(
    "script,behaviour,prediction,shapley,patience",
    [("gridworld.sh", 50, 30, 20, None), ("mastermind222.sh", 50, 80, 50, 10_000)],
)
def test_single_agent_recipes_encode_source_settings(
    tmp_path: Path, script: str, behaviour: int, prediction: int, shapley: int, patience: int | None
) -> None:
    for kind, epochs in (("behaviour", behaviour), ("prediction", prediction)):
        commands = _capture(tmp_path, script, ("characteristic", "--kind", kind, *AGENT))
        assert len(commands) == 20
        for seed, command in enumerate(commands, start=1):
            config = parse_args(command[4:])
            assert isinstance(config, CharacteristicRunConfig)
            assert (config.kind, config.policy, config.seed, config.hidden_widths) == (kind, "greedy", seed, (64, 64))
            assert (config.train.epochs, config.train.patience) == (epochs, patience)
            assert (config.train.train_size, config.train.validation_size, config.train.num_envs) == (8000, 2000, 16)
            assert config.train.exact_loss and config.train.exact_buffer_size is None
    commands = _capture(tmp_path, script, ("shapley", *AGENT, "--characteristic", "runs/c", "--seed", "4"))
    configs = [parse_args(command[4:]) for command in commands]
    assert [config.with_exact_char for config in configs if isinstance(config, ShapleyRunConfig)] == [True, False]
    for config in configs:
        assert isinstance(config, ShapleyRunConfig)
        assert (config.seed, config.buffers, config.train.epochs, config.train.patience) == (
            4,
            "fresh",
            shapley,
            patience,
        )


def test_characteristic_recipes_require_a_kind(tmp_path: Path) -> None:
    with pytest.raises(subprocess.CalledProcessError):
        _capture(tmp_path, "gridworld.sh", ("characteristic", *AGENT))


def test_hypercube_chain_settings(tmp_path: Path) -> None:
    selected = ("--kind", "prediction", "--seed", "2", "--env.side-length", "3", "--env.dimension", "5")
    agent, characteristic, shapley = _capture(tmp_path, "hypercube.sh", selected)
    dqn = train_dqn.parse_args(agent[4:])
    assert isinstance(dqn, train_dqn.DQNRunConfig) and isinstance(dqn.env, train_dqn.HypercubeConfig)
    assert (dqn.env.side_length, dqn.env.dimension, dqn.seed, dqn.train.total_timesteps) == (3, 5, 2, 100_000)
    char = parse_args(characteristic[4:])
    assert isinstance(char, CharacteristicRunConfig)
    assert (char.kind, char.hidden_widths, char.train.epochs) == ("prediction", (128, 128), 10_000)
    assert (char.train.train_size, char.train.validation_size, char.train.exact_size) == (9840, 160, 100_000)
    assert char.train.exact_condition == pytest.approx(0.01) and char.train.track_batch
    explained = parse_args(shapley[4:])
    assert isinstance(explained, ShapleyRunConfig)
    assert (explained.with_exact_char, explained.buffers) == (False, "characteristic")
    assert explained.train.exact_condition == pytest.approx(0.001)
    assert explained.characteristic.parent == char.output and explained.agent.parent == dqn.output
    sweep = _capture(tmp_path, "hypercube.sh", ("--kind", "behaviour", "--seed", "1"))
    assert len(sweep) == 20 * 3


def test_smoke_recipe_chains_three_short_stages(tmp_path: Path) -> None:
    agent, characteristic, shapley = _capture(tmp_path, "smoke.sh", ())
    assert train_dqn.parse_args(agent[4:]) is not None
    char, explained = parse_args(characteristic[4:]), parse_args(shapley[4:])
    assert isinstance(char, CharacteristicRunConfig) and isinstance(explained, ShapleyRunConfig)
    assert char.train.epochs == explained.train.epochs == 4


def _capture(tmp_path: Path, script: str, extra: tuple[str, ...]) -> list[list[str]]:
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
    # Stage names lead; the inspection flag follows so recipes never start gates or jobs.
    stage = extra[:1] if extra[:1] in (("characteristic",), ("shapley",)) else ()
    subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] -- trusted local script argv.
        [bash, str(ROOT / "experiments/explanations" / script), *stage, "--validate-only", *extra[len(stage) :]],
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
    return [command for command in commands if "--validate-only" in command]
