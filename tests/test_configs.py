"""Every registered experiment config loads, and every run script supplies what its config leaves open.

Each `experiments/*.sh` runs with a stand-in `uv` that records its commands. Every recorded
`jax_shapley.main` (or `.continual`) command must parse into a complete, valid config:
sweep scripts supply each of their config's MISSING fields (and only real fields).
"""

import dataclasses
import os
import shlex
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- fixed local argv, no shell interpolation.
from pathlib import Path

import pytest
import tyro

from jax_shapley.envs import Hypercube, Mastermind
from jax_shapley.experiment import ContinualConfig, ExperimentConfig, load_configs

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(path for path in (ROOT / "experiments").glob("*.sh") if path.name != "smoke.sh")
SEEDS = tuple(range(1, 21))
TEN = tuple(range(1, 11))


def test_every_config_has_a_run_script() -> None:
    assert {path.stem for path in SCRIPTS} == set(load_configs(ExperimentConfig)) | set(load_configs(ContinualConfig))


@pytest.mark.parametrize("script", SCRIPTS, ids=[path.stem for path in SCRIPTS])
def test_run_scripts_complete_their_configs(script: Path, tmp_path: Path) -> None:
    configs = _capture(script, tmp_path)
    assert configs
    # Every sweep point compiles first, then runs with the same settings.
    assert [config.compile_only for config in configs] == [True, False] * (len(configs) // 2)
    for compiled, trained in zip(configs[::2], configs[1::2], strict=True):
        assert dataclasses.replace(compiled, compile_only=False) == trained
    for config in configs:
        assert config.name == script.stem
        assert not _missing(config), _missing(config)


@pytest.mark.parametrize(
    ("domain", "behaviour", "prediction", "shapley", "patience"),
    [("gridworld", 50, 30, 20, None), ("mastermind222", 50, 80, 50, 10_000)],
)
def test_single_agent_experiments_encode_source_settings(
    tmp_path: Path, domain: str, behaviour: int, prediction: int, shapley: int, patience: int | None
) -> None:
    for kind, epochs in (("behaviour", behaviour), ("prediction", prediction)):
        config = _experiments(f"{domain}_{kind}", tmp_path)[0]
        agent = config.algorithm
        if domain == "gridworld":
            assert (agent.num_envs, agent.total_timesteps, agent.buffer_size) == (1, 50_000, 50_000)
            assert agent.exploration_fraction == pytest.approx(0.5)
        else:
            assert (agent.num_envs, agent.total_timesteps, agent.buffer_size) == (16, 100_000, 100_000)
            assert agent.exploration_fraction == pytest.approx(0.25)
        assert (agent.learning_starts, agent.train_frequency, agent.hidden_widths, agent.seeds) == (
            10_000,
            10,
            (64, 64),
            (1,),
        )
        characteristic = config.characteristic
        assert characteristic.seeds == SEEDS
        assert (characteristic.kind, characteristic.policy, characteristic.hidden_widths) == (kind, "greedy", (64, 64))
        training = characteristic.train
        assert (training.epochs, training.patience) == (epochs, patience)
        assert (training.train_size, training.validation_size, training.num_envs) == (8000, 2000, 16)
        assert training.exact_loss and training.exact_buffer_size is None
        # Every Shapley seed explains characteristic seed 1, with both exact-loss ground truths.
        assert config.explained_seed == 1
        assert [stage.with_exact_char for stage in config.shapley] == [True, False]
        for stage in config.shapley:
            assert (stage.seeds, stage.buffers, stage.train.epochs, stage.train.patience) == (
                SEEDS,
                "fresh",
                shapley,
                patience,
            )


@pytest.mark.parametrize(
    ("domain", "tau", "num_envs", "total", "train_size", "shapley"),
    [("gridworld", 0.001, 8, 100_000, 50_000, 20), ("mastermind222", 0.01, 16, 200_000, 100_000, 100)],
)
def test_outcome_experiments_encode_source_settings(
    tmp_path: Path, domain: str, tau: float, num_envs: int, total: int, train_size: int, shapley: int
) -> None:
    configs = _experiments(f"{domain}_outcome__sweep", tmp_path)
    outcomes = [config.outcome for config in configs]
    assert [(outcome.variant, outcome.with_exact_char) for outcome in outcomes if outcome is not None] == [
        ("off_policy", True),
        ("off_policy", False),
        ("on_policy", True),
        ("on_policy", False),
    ]
    for config in configs:
        outcome = config.outcome
        assert outcome is not None
        assert (outcome.seeds, outcome.hidden_widths, outcome.tau, outcome.gamma) == (SEEDS, (64, 64), tau, 1.0)
        training = outcome.train
        assert (training.batch_size, training.learning_rate) == (64, pytest.approx(2.5e-4))
        assert (training.total_timesteps, training.num_envs, training.train_frequency) == (total, num_envs, 1)
        assert training.target_network_frequency == 1 and training.exact_loss and training.track_batch
        # Source: validation = 10 * ExpArgs.num_envs, epochs = total * batch // train_size = patience.
        assert (training.train_size, training.validation_size) == (train_size, 10 * num_envs)
        assert (training.epochs, training.patience_epochs) == (128, 128)
        assert config.characteristic.kind == "behaviour" and config.explained_seed == 1
        assert [stage.with_exact_char for stage in config.shapley] == [True, False]
        for stage in config.shapley:
            assert (stage.seeds, stage.buffers, stage.train.epochs, stage.train.patience) == (
                SEEDS,
                "fresh",
                shapley,
                None,
            )
            assert (stage.train.train_size, stage.train.validation_size, stage.train.num_envs) == (8000, 2000, 16)


@pytest.mark.parametrize(
    ("kind", "characteristic_epochs", "shapley_epochs"), [("behaviour", 1000, 300), ("prediction", 2000, 500)]
)
def test_mastermind333_experiments_encode_source_settings(
    tmp_path: Path, kind: str, characteristic_epochs: int, shapley_epochs: int
) -> None:
    config = _experiments(f"mastermind333_{kind}", tmp_path)[0]
    assert config.algorithm.env == Mastermind(3, 3, 3)
    assert (config.algorithm.total_timesteps, config.algorithm.hidden_widths) == (1_000_000, (128, 128))
    characteristic = config.characteristic
    assert (characteristic.kind, characteristic.seeds, characteristic.hidden_widths, characteristic.policy) == (
        kind,
        SEEDS,
        (128, 128),
        "greedy",
    )
    training = characteristic.train
    assert (training.epochs, training.patience, training.batch_size) == (characteristic_epochs, 10_000, 128)
    assert (training.train_size, training.validation_size, training.exact_loss) == (8000, 2000, True)
    assert [stage.with_exact_char for stage in config.shapley] == [True, False]
    for stage in config.shapley:
        assert (stage.train.epochs, stage.hidden_widths, stage.buffers) == (shapley_epochs, (128, 128), "fresh")


@pytest.mark.parametrize("kind", ["behaviour", "prediction", "outcome"])
def test_large_mastermind_experiments_encode_source_settings(tmp_path: Path, kind: str) -> None:
    configs = _experiments(f"mastermind_large_{kind}__sweep", tmp_path)
    assert [config.algorithm.env for config in configs] == [Mastermind(4, guesses, 3) for guesses in (4, 5, 6)]
    for config in configs:
        agent = config.algorithm
        assert (agent.total_timesteps, agent.buffer_size, agent.num_envs) == (10_000_000, 10_000_000, 16)
        assert (agent.learning_starts, agent.train_frequency, agent.hidden_widths) == (10_000, 10, (128, 128))
        assert agent.exploration_fraction == pytest.approx(0.25) and agent.end_e == pytest.approx(0.01)
        characteristic = config.characteristic
        assert (characteristic.seeds, characteristic.train.epochs, characteristic.train.patience) == (
            TEN,
            50_000,
            5_000,
        )
        assert not characteristic.train.exact_loss and characteristic.hidden_widths == (128, 128)
        (stage,) = config.shapley
        assert not stage.train.exact_loss
        assert (stage.train.epochs, stage.train.patience, stage.seeds) == (50_000, 5_000, TEN)
        if kind != "outcome":
            assert config.outcome is None
            continue
        outcome = config.outcome
        assert outcome is not None and outcome.variant == "off_policy" and not outcome.with_exact_char
        assert (outcome.seeds, outcome.tau, outcome.gamma, outcome.hidden_widths) == ((1,), 0.01, 1.0, (128, 128))
        training = outcome.train
        # epochs = total_timesteps * batch_size // train_buffer_size over the agent's whole 10M replay.
        assert (training.train_size, training.validation_size, training.batch_size) == (10_000_000, 160, 64)
        assert (training.epochs, training.patience_epochs) == (64, 64)
        assert training.track_batch and not training.exact_loss
        assert not stage.with_exact_char


@pytest.mark.parametrize(
    ("domain", "timesteps", "num_envs", "groups"),
    [
        ("mastermind222", 30_000, 16, ["exact", "wo_weight", "w_weight"]),
        ("gridworld", 15_000, 8, ["exact", "wo_weight", "unnormalised", "normalised", "0.99", "0.995", "0.998"]),
    ],
)
@pytest.mark.parametrize("kind", ["behaviour", "prediction"])
def test_offpolicy_experiments_encode_source_settings(
    tmp_path: Path, domain: str, kind: str, timesteps: int, num_envs: int, groups: list[str]
) -> None:
    configs = _experiments(f"{domain}_offpolicy_{kind}__sweep", tmp_path)
    labels = []
    for config in configs:
        agent = config.algorithm
        assert agent.seeds == SEEDS  # One agent per seed, explained by characteristic seed s.
        assert (agent.total_timesteps, agent.buffer_size, agent.num_envs) == (timesteps, timesteps, num_envs)
        assert (agent.learning_starts, agent.train_frequency) == (10_000, 10)
        characteristic = config.characteristic
        assert characteristic.seeds == SEEDS and characteristic.kind == kind and not config.shapley
        training = characteristic.train
        assert (training.train_size, training.validation_size) == (timesteps * 4 // 5, timesteps // 5)
        assert training.exact_buffer_size == timesteps * 4 // 5 and training.exact_loss
        # Source offpolicy/{off_policy,weighting,clipping}/dqn_{policy,value}_*.py; patience = epochs.
        epochs = 2000 if domain == "mastermind222" else (50 if kind == "behaviour" else 100)
        assert training.epochs == training.patience_epochs == epochs
        if characteristic.data == "rollouts":
            labels.append("exact")
        elif not characteristic.off_policy:
            labels.append("wo_weight")
        elif not characteristic.weighting:
            labels.append("unnormalised")
        elif characteristic.clip_coef != float("inf"):
            labels.append(str(characteristic.clip_coef))
        else:
            labels.append("w_weight" if domain == "mastermind222" else "normalised")
    assert labels == groups


@pytest.mark.parametrize("kind", ["behaviour", "prediction"])
def test_hypercube_experiments_encode_source_settings(tmp_path: Path, kind: str) -> None:
    configs = _experiments(f"hypercube_{kind}__sweep", tmp_path)
    assert len(configs) == 20
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
    assert {
        (config.algorithm.env.side_length, config.algorithm.env.dimension)
        for config in configs
        if isinstance(config.algorithm.env, Hypercube)
    } == set(expected)
    for config in configs:
        agent = config.algorithm
        assert isinstance(agent.env, Hypercube)
        timesteps = expected[agent.env.side_length, agent.env.dimension]
        assert agent.total_timesteps == agent.buffer_size == timesteps
        assert agent.learning_starts == timesteps // 50
        assert (agent.num_envs, agent.hidden_widths, agent.seeds) == (16, (128, 128), SEEDS)
        assert agent.end_e == pytest.approx(0.05)
        characteristic = config.characteristic
        assert (characteristic.kind, characteristic.hidden_widths, characteristic.train.epochs) == (
            kind,
            (128, 128),
            10_000,
        )
        training = characteristic.train
        assert (training.train_size, training.validation_size, training.exact_size) == (9840, 160, 100_000)
        assert training.exact_condition == pytest.approx(0.01) and training.track_batch
        # Every seed pairs agent, characteristic and Shapley model s.
        assert config.explained_seed is None
        (stage,) = config.shapley
        assert (stage.with_exact_char, stage.buffers) == (False, "characteristic")
        # Source hypercube_value.py stops prediction Shapley models at 0.01; behaviour uses 0.001.
        assert stage.train.exact_condition == pytest.approx(0.001 if kind == "behaviour" else 0.01)


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
def test_sampling_experiments_encode_source_settings(
    tmp_path: Path,
    domain: str,
    kind: str,
    width: int,
    characteristic_epochs: int,
    shapley_epochs: int,
    patience: int | None,
) -> None:
    model = _experiments(f"{domain}_sampling_{kind}", tmp_path)[0]
    sampled = _experiments(f"{domain}_sampled_{kind}", tmp_path)[0]
    learnt, pool = model.characteristic, sampled.characteristic
    assert (learnt.kind, learnt.estimator, learnt.policy, learnt.hidden_widths) == (
        kind,
        "model",
        "greedy",
        (width,) * 2,
    )
    assert (learnt.train.epochs, learnt.train.patience) == (characteristic_epochs, patience)
    assert learnt.train.exact_loss and learnt.seeds == SEEDS
    assert (pool.kind, pool.estimator, pool.train.exact_loss, pool.train.exact_size) == (kind, "sampled", False, 8000)
    for characteristic in (learnt, pool):
        training = characteristic.train
        assert (training.train_size, training.validation_size, training.batch_size, training.num_envs) == (
            8000,
            2000,
            128,
            16,
        )
    # Shapley seed s explains characteristic seed s: w_exact_model, wo_exact_model, then w_exact_sampling.
    groups = [(stage.buffers, stage.with_exact_char) for stage in (*model.shapley, *sampled.shapley)]
    assert groups == [("fresh", True), ("fresh", False), ("characteristic", True)]
    assert model.explained_seed is None and sampled.explained_seed is None
    for stage in (*model.shapley, *sampled.shapley):
        assert (stage.train.epochs, stage.train.patience) == (shapley_epochs, patience)
        assert stage.train.exact_loss and stage.hidden_widths == (width,) * 2 and stage.seeds == SEEDS


@pytest.mark.parametrize(
    ("domain", "tau", "total"), [("mastermind222", 0.01, 1_000_000), ("gridworld", 0.001, 100_000)]
)
def test_sampling_outcome_experiments_encode_source_settings(
    tmp_path: Path, domain: str, tau: float, total: int
) -> None:
    configs = [
        *_experiments(f"{domain}_sampling_outcome__sweep", tmp_path),
        *_experiments(f"{domain}_sampled_outcome", tmp_path),
    ]
    groups = [(config.characteristic.estimator, config.outcome.with_exact_char) for config in configs if config.outcome]
    assert groups == [("model", True), ("model", False), ("sampled", True)]
    for config in configs:
        outcome = config.outcome
        assert outcome is not None and outcome.variant == "on_policy" and outcome.tau == tau
        training = outcome.train
        assert (training.total_timesteps, training.num_envs, training.batch_size) == (total, 16, 64)
        assert training.exact_loss and outcome.hidden_widths == (64, 64) and outcome.seeds == SEEDS
        assert config.explained_seed is None and not config.shapley


@pytest.mark.parametrize(("domain", "exploration"), [("mastermind222", 0.25), ("gridworld", 0.5)])
def test_continual_experiments_encode_source_settings(tmp_path: Path, domain: str, exploration: float) -> None:
    configs = [config for config in _capture(ROOT / "experiments" / f"continual_{domain}__sweep.sh", tmp_path)]
    trained = [config for config in configs if isinstance(config, ContinualConfig) and not config.compile_only]
    assert [(config.explain.kind, config.explain.update_ratio) for config in trained] == [
        (kind, ratio) for kind in ("behaviour", "prediction", "outcome") for ratio in (1, 2, 10, 50)
    ]
    for config in trained:
        training, explain = config.algorithm, config.explain
        assert training.seeds == SEEDS and training.hidden_widths == (64, 64)
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
        assert (explain.with_exact_char, explain.outcome_with_exact_char) == (True, True)
        assert explain.exact_frequency == (5_000 if explain.kind == "outcome" else 1_000)
        assert (explain.exact_buffer_size, explain.hidden_widths, explain.tau) == (10_000, (64, 64), 0.01)


def _experiments(name: str, tmp_path: Path) -> list[ExperimentConfig]:
    """The configs an experiment script trains (its compile-only runs left out)."""
    configs = _capture(ROOT / "experiments" / f"{name}.sh", tmp_path)
    trained = [config for config in configs if isinstance(config, ExperimentConfig) and not config.compile_only]
    assert len(trained) == len(configs) // 2
    return trained


def _capture(script: Path, tmp_path: Path) -> list[ExperimentConfig | ContinualConfig]:
    """Run a script against a stand-in `uv` and parse every command it issues."""
    directory = tmp_path / script.stem
    record = directory / "commands"
    fake = directory / "bin" / "uv"
    fake.parent.mkdir(parents=True)
    fake.write_text(
        f'#!/usr/bin/env bash\nprintf "%q " "$@" >> {shlex.quote(str(record))}\necho >> {shlex.quote(str(record))}\n'
    )
    fake.chmod(0o755)
    environment = {**os.environ, "PATH": f"{fake.parent}{os.pathsep}{os.environ['PATH']}"}
    bash = shutil.which("bash")
    assert bash is not None
    subprocess.run([bash, str(script)], env=environment, check=True, cwd=ROOT)  # ruff: ignore[subprocess-without-shell-equals-true] -- fixed local argv.
    configs: list[ExperimentConfig | ContinualConfig] = []
    for line in record.read_text().splitlines():
        command = shlex.split(line)
        assert command[:3] == ["run", "python", "-m"], command
        if command[3] == "jax_shapley.main":
            configs.append(tyro.extras.overridable_config_cli(load_configs(ExperimentConfig), args=command[4:]))
        else:
            assert command[3] == "jax_shapley.continual", command
            configs.append(tyro.extras.overridable_config_cli(load_configs(ContinualConfig), args=command[4:]))
    return configs


def _missing(value: object, path: str = "") -> list[str]:
    """Dotted paths of every field still MISSING in a (nested) config."""
    if value is tyro.MISSING:
        return [path]
    if isinstance(value, tuple):
        return [found for index, item in enumerate(value) for found in _missing(item, f"{path}.{index}")]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [
            found
            for field in dataclasses.fields(value)
            for found in _missing(getattr(value, field.name), f"{path}.{field.name}")
        ]
    return []
