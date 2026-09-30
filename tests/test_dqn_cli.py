"""CLI parsing, preflight, typed artifacts, and tiny host-orchestrated runs."""

import csv
import json
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jax_shapley.algorithms.dqn import initialize_run, run_chunk
from jax_shapley.artifacts import describe, load_checkpoint, save_checkpoint
from jax_shapley.config import DQNRunConfig, DQNTrainConfig, GridworldConfig, HypercubeConfig
from jax_shapley.envs import Gridworld
from jax_shapley.train_dqn import checkpoint_due, main, parse_args
from jax_shapley.wrappers import EpisodeStatisticsWrapper


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
    assert isinstance(config.env, HypercubeConfig)
    assert config.env.dimension == 5
    assert config.hidden_widths == (128, 128)
    assert config.train.gamma == pytest.approx(0.9)
    assert config.seed == 2
    assert config.network.input_features == 5
    assert parse_args(["mastermind222"]).network.input_features == 8


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


def test_checkpoint_round_trip_and_corruption(tmp_path: Path) -> None:
    config = DQNRunConfig(
        GridworldConfig(),
        train=DQNTrainConfig(total_timesteps=12, buffer_size=7, batch_size=3, learning_starts=0, train_frequency=1),
        hidden_widths=(4,),
        chunk_steps=4,
    )
    env = EpisodeStatisticsWrapper(Gridworld())
    state = initialize_run(env, env.default_params, config.network, config.train, jax.random.key(0))
    state, _ = nnx.jit(partial(run_chunk, env=env, params=env.default_params, config=config.train, chunk_steps=4))(
        state, jnp.asarray(4, jnp.int32)
    )
    save_checkpoint(tmp_path, state, config)
    loaded = load_checkpoint(tmp_path, config.network)
    assert loaded.vector_step == 4
    assert loaded.replay is not None
    assert 0 <= loaded.epsilon <= 1
    for left, right in zip(
        jax.tree.leaves(nnx.state(state.model)), jax.tree.leaves(nnx.state(loaded.model)), strict=True
    ):
        np.testing.assert_array_equal(left, right)
    for left, right in zip(jax.tree.leaves(state.replay), jax.tree.leaves(loaded.replay), strict=True):
        np.testing.assert_array_equal(left, right)
    assert load_checkpoint(tmp_path, config.network, load_replay=False).replay is None
    assert "GridworldConfig" in describe(config)
    with np.load(tmp_path / "replay.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["vector_step"] = np.asarray(3, np.int32)
    np.savez_compressed(tmp_path / "replay.npz", allow_pickle=False, **arrays)
    with pytest.raises(ValueError, match="steps disagree"):
        load_checkpoint(tmp_path, config.network)


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
    assert {"config.json", "summary.json", "metrics.csv", "model.npz", "replay.npz"} <= {
        file.name for file in folder.iterdir()
    }
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
        "--no-save-model",
        "--no-save-replay",
        "--output",
        str(tmp_path),
    ])
    assert result is not None
    assert not (Path(result.folder) / "model.npz").exists()
    assert not (Path(result.folder) / "replay.npz").exists()
    assert checkpoint_due(1, 1, 4)
    assert checkpoint_due(5_000, 1, 50_000)
    assert not checkpoint_due(4_999, 1, 50_000)
    assert checkpoint_due(625, 16, 100_000)
