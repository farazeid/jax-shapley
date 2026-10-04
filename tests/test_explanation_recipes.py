"""Execute explanation recipes against a capture-only uv, never a training job."""

import json
import os
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- fixed local recipe argv, no shell interpolation.
import sys
from pathlib import Path
from typing import cast

import pytest

from jax_shapley import train_continual, train_dqn
from jax_shapley.train_explanations import CharacteristicRunConfig, OutcomeRunConfig, ShapleyRunConfig, parse_args

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
        # All 20 paper seeds train in one compiled sweep.
        (command,) = _capture(tmp_path, script, ("characteristic", "--kind", kind, *AGENT))
        config = parse_args(command[4:])
        assert isinstance(config, CharacteristicRunConfig)
        assert config.sweep == tuple(range(1, 21))
        assert (config.kind, config.policy, config.hidden_widths) == (kind, "greedy", (64, 64))
        assert (config.train.epochs, config.train.patience) == (epochs, patience)
        assert (config.train.train_size, config.train.validation_size, config.train.num_envs) == (8000, 2000, 16)
        assert config.train.exact_loss and config.train.exact_buffer_size is None
        (single,) = _capture(tmp_path, script, ("characteristic", "--kind", kind, *AGENT, "--seed", "7"))
        selected = parse_args(single[4:])
        assert isinstance(selected, CharacteristicRunConfig) and selected.sweep == (7,)
    commands = _capture(tmp_path, script, ("shapley", *AGENT, "--characteristic", "runs/c", "--seed", "4"))
    configs = [parse_args(command[4:]) for command in commands]
    assert [config.with_exact_char for config in configs if isinstance(config, ShapleyRunConfig)] == [True, False]
    for config in configs:
        assert isinstance(config, ShapleyRunConfig)
        assert (config.sweep, config.buffers, config.train.epochs, config.train.patience) == (
            (4,),
            "fresh",
            shapley,
            patience,
        )
    swept = [
        parse_args(command[4:])
        for command in _capture(tmp_path, script, ("shapley", *AGENT, "--characteristic", "runs/c"))
    ]
    assert all(isinstance(config, ShapleyRunConfig) and config.sweep == tuple(range(1, 21)) for config in swept)


@pytest.mark.parametrize(
    "script,tau,num_envs,total,train_size,shapley",
    [("gridworld.sh", 0.001, 8, 100_000, 50_000, 20), ("mastermind222.sh", 0.01, 16, 200_000, 100_000, 100)],
)
def test_outcome_recipes_encode_source_settings(
    tmp_path: Path, script: str, tau: float, num_envs: int, total: int, train_size: int, shapley: int
) -> None:
    for variant in ("on_policy", "off_policy"):
        selected = ("outcome", "--variant", variant, *AGENT, "--characteristic", "runs/c")
        commands = _capture(tmp_path, script, selected)
        configs = [parse_args(command[4:]) for command in commands]
        assert [config.with_exact_char for config in configs if isinstance(config, OutcomeRunConfig)] == [True, False]
        for config in configs:
            assert isinstance(config, OutcomeRunConfig)
            assert (config.variant, config.sweep, config.hidden_widths) == (variant, tuple(range(1, 21)), (64, 64))
            assert (config.tau, config.gamma) == (pytest.approx(tau), 1.0)
            training = config.train
            assert (training.batch_size, training.learning_rate) == (64, pytest.approx(2.5e-4))
            assert (training.total_timesteps, training.num_envs, training.train_frequency) == (total, num_envs, 1)
            assert training.target_network_frequency == 1 and training.exact_loss and training.track_batch
            # Source: validation = 10 * ExpArgs.num_envs, epochs = total * batch // train_size = patience.
            assert (training.train_size, training.validation_size) == (train_size, 10 * num_envs)
            assert (training.epochs, training.patience_epochs) == (128, 128)
    with pytest.raises(subprocess.CalledProcessError):
        _capture(tmp_path, script, ("outcome", *AGENT, "--characteristic", "runs/c"))
    commands = _capture(tmp_path, script, ("outcome-shapley", *AGENT, "--characteristic", "runs/o", "--seed", "3"))
    configs = [parse_args(command[4:]) for command in commands]
    assert [config.with_exact_char for config in configs if isinstance(config, ShapleyRunConfig)] == [True, False]
    for config in configs:
        assert isinstance(config, ShapleyRunConfig)
        assert (config.sweep, config.buffers, config.train.epochs, config.train.patience) == (
            (3,),
            "fresh",
            shapley,
            None,
        )
        assert (config.train.train_size, config.train.validation_size, config.train.num_envs) == (8000, 2000, 16)


@pytest.mark.parametrize(
    "kind,characteristic_epochs,shapley_epochs", [("behaviour", 1000, 300), ("prediction", 2000, 500)]
)
def test_mastermind333_recipe_encodes_source_settings(
    tmp_path: Path, kind: str, characteristic_epochs: int, shapley_epochs: int
) -> None:
    (command,) = _capture(tmp_path, "mastermind.sh", ("characteristic", "--kind", kind, *AGENT), leading=("333",))
    config = parse_args(command[4:])
    assert isinstance(config, CharacteristicRunConfig) and config.kind == kind
    assert (config.sweep, config.hidden_widths, config.policy) == (tuple(range(1, 21)), (128, 128), "greedy")
    assert (config.train.epochs, config.train.patience, config.train.batch_size) == (characteristic_epochs, 10_000, 128)
    assert (config.train.train_size, config.train.validation_size, config.train.exact_loss) == (8000, 2000, True)
    selected = ("shapley", "--kind", kind, *AGENT, "--characteristic", "runs/c")
    shapleys = [parse_args(command[4:]) for command in _capture(tmp_path, "mastermind.sh", selected, leading=("333",))]
    assert [config.with_exact_char for config in shapleys if isinstance(config, ShapleyRunConfig)] == [True, False]
    for shapley in shapleys:
        assert isinstance(shapley, ShapleyRunConfig)
        assert (shapley.train.epochs, shapley.hidden_widths, shapley.buffers) == (shapley_epochs, (128, 128), "fresh")


@pytest.mark.parametrize("size", ["443", "453", "463"])
def test_large_mastermind_recipe_encodes_source_settings(tmp_path: Path, size: str) -> None:
    (command,) = _capture(tmp_path, "mastermind.sh", ("characteristic", "--kind", "behaviour", *AGENT), leading=(size,))
    config = parse_args(command[4:])
    assert isinstance(config, CharacteristicRunConfig)
    assert (config.sweep, config.train.epochs, config.train.patience) == (tuple(range(1, 11)), 50_000, 5_000)
    assert not config.train.exact_loss and config.hidden_widths == (128, 128)
    (shapley,) = _capture(tmp_path, "mastermind.sh", ("shapley", *AGENT, "--characteristic", "runs/c"), leading=(size,))
    explained = parse_args(shapley[4:])
    assert isinstance(explained, ShapleyRunConfig) and not explained.train.exact_loss
    assert (explained.train.epochs, explained.train.patience, explained.sweep) == (50_000, 5_000, tuple(range(1, 11)))


@pytest.mark.parametrize("size", ["443", "453", "463"])
def test_large_mastermind_outcome_recipe_encodes_source_settings(tmp_path: Path, size: str) -> None:
    arguments = ("outcome", *AGENT, "--characteristic", "runs/c")
    (command,) = _capture(tmp_path, "mastermind.sh", arguments, leading=(size,))
    config = parse_args(command[4:])
    assert isinstance(config, OutcomeRunConfig) and config.variant == "off_policy" and not config.with_exact_char
    assert (config.sweep, config.tau, config.gamma, config.hidden_widths) == ((1,), 0.01, 1.0, (128, 128))
    training = config.train
    # epochs = total_timesteps * batch_size // train_buffer_size over the agent's whole 10M replay.
    assert (training.train_size, training.validation_size, training.batch_size) == (10_000_000, 160, 64)
    assert (training.epochs, training.patience_epochs) == (64, 64)
    assert training.track_batch and not training.exact_loss
    arguments = ("outcome-shapley", *AGENT, "--characteristic", "runs/o")
    (shapley,) = _capture(tmp_path, "mastermind.sh", arguments, leading=(size,))
    explained = parse_args(shapley[4:])
    assert isinstance(explained, ShapleyRunConfig) and not explained.train.exact_loss and not explained.with_exact_char
    assert (explained.train.epochs, explained.train.patience, explained.sweep) == (50_000, 5_000, tuple(range(1, 11)))


@pytest.mark.parametrize(
    "domain,timesteps,num_envs,groups",
    [
        ("mastermind222", 30_000, 16, ["exact", "wo_weight", "w_weight"]),
        ("gridworld", 15_000, 8, ["exact", "wo_weight", "unnormalised", "normalised", "0.99", "0.995", "0.998"]),
    ],
)
def test_offpolicy_recipes_encode_source_settings(
    tmp_path: Path, domain: str, timesteps: int, num_envs: int, groups: list[str]
) -> None:
    commands = _capture(tmp_path, "offpolicy.sh", ("--kind", "behaviour", "--num-seeds", "3"), leading=(domain,))
    agents = [train_dqn.parse_args(command[4:]) for command in commands if "jax_shapley.train_dqn" in command]
    assert len(agents) == 3
    for agent in agents:
        assert isinstance(agent, train_dqn.DQNRunConfig)
        assert (agent.train.total_timesteps, agent.train.buffer_size, agent.train.num_envs) == (
            timesteps,
            timesteps,
            num_envs,
        )
        assert (agent.train.learning_starts, agent.train.train_frequency) == (10_000, 10)
    characteristics = [parse_args(command[4:]) for command in commands if "jax_shapley.train_explanations" in command]
    assert len(characteristics) == len(groups)
    labels = []
    for config in characteristics:
        assert isinstance(config, CharacteristicRunConfig) and config.sweep == (1, 2, 3)
        training = config.train
        assert (training.train_size, training.validation_size) == (timesteps * 4 // 5, timesteps // 5)
        assert training.exact_buffer_size == timesteps * 4 // 5 and training.exact_loss
        assert training.epochs == training.patience == (2000 if domain == "mastermind222" else 50)
        if config.data == "rollouts":
            labels.append("exact")
        elif not config.off_policy:
            labels.append("wo_weight")
        elif not config.weighting:
            labels.append("unnormalised")
        elif config.clip_coef != float("inf"):
            labels.append(str(config.clip_coef))
        else:
            labels.append("w_weight" if domain == "mastermind222" else "normalised")
    assert labels == groups


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
    # Source hypercube_value.py stops prediction Shapley models at 0.01; behaviour uses 0.001.
    assert explained.train.exact_condition == pytest.approx(0.01)
    # Every seed's agent trains into one directory; the sweeps pair seed s with agent and characteristic s.
    assert char.agent == explained.agent == dqn.output and char.sweep == explained.sweep == (2,)
    assert explained.characteristic.parent == char.output
    sweep = _capture(tmp_path, "hypercube.sh", ("--kind", "behaviour", "--seed", "1"))
    assert len(sweep) == 20 * 3
    behaviour = parse_args(sweep[2][4:])
    assert isinstance(behaviour, ShapleyRunConfig) and behaviour.train.exact_condition == pytest.approx(0.001)


def test_smoke_recipe_chains_three_short_stages(tmp_path: Path) -> None:
    agent, characteristic, shapley = _capture(tmp_path, "smoke.sh", ())
    assert train_dqn.parse_args(agent[4:]) is not None
    char, explained = parse_args(characteristic[4:]), parse_args(shapley[4:])
    assert isinstance(char, CharacteristicRunConfig) and isinstance(explained, ShapleyRunConfig)
    assert char.train.epochs == explained.train.epochs == 4


@pytest.mark.parametrize(
    ("domain", "kind", "width", "characteristic_epochs", "shapley_epochs", "patience"),
    [
        ("mastermind222", "behaviour", 64, 50, 50, 10_000),
        ("mastermind222", "prediction", 64, 80, 50, 10_000),
        ("mastermind333", "behaviour", 128, 1000, 300, 10_000),
        ("mastermind333", "prediction", 128, 2000, 500, 10_000),
        ("gridworld", "behaviour", 64, 50, 20, None),
        ("gridworld", "prediction", 64, 30, 20, None),
    ],
)
def test_sampling_recipe_encodes_source_settings(
    tmp_path: Path,
    domain: str,
    kind: str,
    width: int,
    characteristic_epochs: int,
    shapley_epochs: int,
    patience: int | None,
) -> None:
    root = ("--output-root", "runs/sampling")
    model, sampled = _capture(tmp_path, "sampling.sh", ("characteristic", "--kind", kind, *AGENT, *root), (domain,))
    learnt, pool = parse_args(model[4:]), parse_args(sampled[4:])
    assert isinstance(learnt, CharacteristicRunConfig) and isinstance(pool, CharacteristicRunConfig)
    assert (learnt.kind, learnt.estimator, learnt.policy, learnt.hidden_widths) == (
        kind,
        "model",
        "greedy",
        (width,) * 2,
    )
    assert (learnt.train.epochs, learnt.train.patience) == (characteristic_epochs, patience or characteristic_epochs)
    assert (
        learnt.train.exact_loss and learnt.sweep == tuple(range(1, 21)) and learnt.output == Path("runs/sampling/model")
    )
    assert (pool.kind, pool.estimator, pool.train.exact_loss, pool.train.exact_size) == (kind, "sampled", False, 8000)
    for config in (learnt, pool):
        training = config.train
        assert (training.train_size, training.validation_size, training.batch_size, training.num_envs) == (
            8000,
            2000,
            128,
            16,
        )
    sources = ("--model", "runs/model", "--sampled", "runs/sampled")
    commands = _capture(tmp_path, "sampling.sh", ("shapley", "--kind", kind, *AGENT, *sources, *root), (domain,))
    groups = [parse_args(command[4:]) for command in commands]
    expected = [("runs/model", "fresh", True), ("runs/model", "fresh", False), ("runs/sampled", "characteristic", True)]
    for config, (characteristic, buffers, with_exact_char), group in zip(
        groups, expected, ("w_exact_model", "wo_exact_model", "w_exact_sampling"), strict=True
    ):
        assert isinstance(config, ShapleyRunConfig)
        assert (config.characteristic, config.buffers, config.with_exact_char) == (
            Path(characteristic),
            buffers,
            with_exact_char,
        )
        assert (config.train.epochs, config.train.patience) == (shapley_epochs, patience or shapley_epochs)
        assert config.train.exact_loss and config.hidden_widths == (width,) * 2
        assert config.output == Path("runs/sampling") / group and config.sweep == tuple(range(1, 21))


@pytest.mark.parametrize(
    ("domain", "tau", "total"), [("mastermind222", 0.01, 1_000_000), ("gridworld", 0.001, 100_000)]
)
def test_sampling_outcome_recipe_encodes_source_settings(tmp_path: Path, domain: str, tau: float, total: int) -> None:
    sources = ("--model", "runs/model", "--sampled", "runs/sampled", "--output-root", "runs/outcome")
    commands = _capture(tmp_path, "sampling.sh", ("outcome", *AGENT, *sources), (domain,))
    expected = [("runs/model", True), ("runs/model", False), ("runs/sampled", True)]
    for command, (characteristic, with_exact_char) in zip(commands, expected, strict=True):
        config = parse_args(command[4:])
        assert isinstance(config, OutcomeRunConfig) and config.variant == "on_policy"
        assert (config.characteristic, config.with_exact_char, config.tau) == (
            Path(characteristic),
            with_exact_char,
            tau,
        )
        training = config.train
        assert (training.total_timesteps, training.num_envs, training.batch_size) == (total, 16, 64)
        assert training.exact_loss and config.hidden_widths == (64, 64) and config.sweep == tuple(range(1, 21))


@pytest.mark.parametrize(("domain", "exploration"), [("mastermind222", 0.25), ("gridworld", 0.5)])
@pytest.mark.parametrize("kind", ["behaviour", "prediction", "outcome"])
def test_continual_recipe_encodes_source_settings(tmp_path: Path, domain: str, exploration: float, kind: str) -> None:
    commands = _capture(tmp_path, "continual.sh", (), (domain, kind))
    configs = [train_continual.parse_args(command[4:]) for command in commands]
    assert [config.explain.update_ratio for config in configs] == [1, 2, 10, 50]
    for config in configs:
        training, explain = config.train, config.explain
        assert config.seeds == tuple(range(1, 21)) and config.hidden_widths == (64, 64)
        assert (training.num_envs, training.total_timesteps, training.buffer_size, training.batch_size) == (
            16,
            100_000,
            100_000,
            64,
        )
        assert (training.exploration_fraction, training.learning_starts, training.train_frequency) == (
            exploration,
            0,
            10,
        )
        assert (training.tau, training.target_network_frequency, training.gamma) == (0.01, 1, 1.0)
        assert (explain.kind, explain.with_exact_char, explain.outcome_with_exact_char) == (kind, True, True)
        assert explain.exact_frequency == (5_000 if kind == "outcome" else 1_000)
        assert (explain.exact_buffer_size, explain.hidden_widths, explain.tau) == (10_000, (64, 64), 0.01)


def _capture(tmp_path: Path, script: str, extra: tuple[str, ...], leading: tuple[str, ...] = ()) -> list[list[str]]:
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
    stages = (("characteristic",), ("shapley",), ("outcome",), ("outcome-shapley",))
    stage = (*leading, *(extra[:1] if extra[:1] in stages else ()))
    subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] -- trusted local script argv.
        [
            bash,
            str(ROOT / "experiments/explanations" / script),
            *stage,
            "--validate-only",
            *extra[len(stage) - len(leading) :],
        ],
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
