"""Tracking: live logs equal the returned metrics, stage checkpoints restore fully, and tracked runs hand off.

Backend runs execute in subprocesses: trackio fixes its storage directory at import time.
The wandb test writes to the account's cloud, so it only runs with JAX_SHAPLEY_WANDB_TEST=1.
"""

import json
import os
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- fixed local argv, no shell interpolation.
import sys
import tempfile
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import trackio
import wandb
from flax import nnx

from jax_shapley import backends
from jax_shapley.algorithms import dqn
from jax_shapley.characteristic_models import behaviour as characteristic
from jax_shapley.characteristic_models.behaviour import CharacteristicConfig
from jax_shapley.common import checkpoints
from jax_shapley.common.supervised_learning import EPOCH_BLOCK, EpochMetrics, SupervisedConfig, seed_slice, stack_seeds
from jax_shapley.envs import Mastermind
from jax_shapley.experiment import ExperimentConfig
from jax_shapley.main import ExperimentResult, main, run
from jax_shapley.shapley_models import behaviour as shapley
from jax_shapley.shapley_models.behaviour import ShapleyConfig

SEEDS = (1, 2)
TRAINING = SupervisedConfig(epochs=3, batch_size=16, buffer_size=160, num_envs=4)
ENV = Mastermind(2, 2, 2)
CONFIG = ExperimentConfig(
    name="test_tracking",
    algorithm=dqn.DQNConfig(
        env=ENV, num_envs=4, total_timesteps=1024, buffer_size=1024, learning_starts=128, batch_size=16
    ),
    characteristic=CharacteristicConfig(env=ENV, hidden_widths=(16, 16), seeds=SEEDS, train=TRAINING),
    shapley=(ShapleyConfig(env=ENV, hidden_widths=(16, 16), seeds=SEEDS, train=TRAINING),),
)
# The registered config, shrunk: the CLI only accepts registered configs, overridden.
SMALL_RUN = [
    "mastermind222_behaviour",
    "--algorithm.total-timesteps",
    "20000",
    "--algorithm.learning-starts",
    "1000",
    "--characteristic.seeds",
    "1",
    "2",
    "--characteristic.train.epochs",
    "3",
    "--shapley.0.seeds",
    "1",
    "2",
    "--shapley.0.train.epochs",
    "3",
    "--shapley.1.seeds",
    "1",
    "2",
    "--shapley.1.train.epochs",
    "3",
]
READ_GROUPS = """
import json, sys
from jax_shapley import backends
track, project, entity = sys.argv[1], sys.argv[2], sys.argv[3] or None
groups = backends.groups(track, project=project, entity=entity, prefix="mastermind222_behaviour")
print(json.dumps({group: {name: run.rows for name, run in backends.runs(track, project=project, entity=entity, group=group).items()} for group in groups}))
"""


def test_live_logs_equal_the_returned_metrics() -> None:
    """Live logs arrive one block at a time; together the blocks are exactly the returned metrics."""
    blocks: list[dqn.StepMetrics] = []
    config = CONFIG.algorithm
    program = nnx.jit(partial(dqn.train, config, blocks.append, None))
    state, metrics = program(jnp.stack([jax.random.key(seed) for seed in config.seeds]))
    jax.effects_barrier()
    assert len(blocks) == -(-config.vector_steps // dqn.STEP_BLOCK)
    for name in ("global_step", "loss", "updated", "episode_return", "completed"):
        logged = np.concatenate([np.asarray(getattr(block, name)) for block in blocks])  # (steps, seeds)
        np.testing.assert_array_equal(np.moveaxis(logged, 0, 1), np.asarray(getattr(metrics, name)))
    stage = CONFIG.characteristic
    keys = jnp.stack([jax.random.key(seed) for seed in stage.seeds])
    agents = stack_seeds([dqn.trained_agent(config, seed_slice(state, 0))] * len(stage.seeds))
    keys, buffers = nnx.jit(partial(characteristic.collect, stage))(keys, agents, None)
    max_unique = max(int(np.unique(rows, axis=0).shape[0]) for rows in np.asarray(buffers.exact))
    epochs: list[EpochMetrics] = []
    _, returned = nnx.jit(
        partial(characteristic.fit, stage, characteristic.targets, characteristic.head, max_unique, epochs.append)
    )(keys, agents, buffers)
    jax.effects_barrier()
    assert returned is not None
    assert len(epochs) == -(-int(np.asarray(returned.epoch).max()) // EPOCH_BLOCK)
    for name in ("epoch", "step", "train_loss", "validation_loss", "exact_loss", "active"):
        logged = np.concatenate([np.asarray(getattr(block, name)) for block in epochs])[: stage.train.epochs]
        np.testing.assert_array_equal(logged, np.asarray(getattr(returned, name)))


def test_stage_checkpoints_restore_the_whole_state(tmp_path: Path, result: ExperimentResult) -> None:
    stages = (
        ("agent", result.dqn_state, stack_seeds([dqn.initialize(CONFIG.algorithm, jax.random.key(0))])),
        (
            "characteristic",
            result.characteristic_state,
            characteristic.initialize(CONFIG.characteristic, characteristic.head),
        ),
        (
            "shapley",
            result.shapley[0][0],
            shapley.initialize(CONFIG.shapley[0], CONFIG.characteristic, characteristic.head),
        ),
    )
    for name, state, template in stages:
        checkpoints.save(tmp_path / name, state, {"stage": name})
        restored = checkpoints.load(tmp_path / name, template, {"stage": name})
        _assert_same_state(state, restored)
        with pytest.raises(ValueError, match="different settings"):
            checkpoints.load(tmp_path / name, template, {"stage": "other"})
    # A restored characteristic seed explains exactly as the trained one does.
    trained = seed_slice(result.characteristic_state, 0)
    restored = seed_slice(checkpoints.load(tmp_path / "characteristic", stages[1][2], {"stage": "characteristic"}), 0)
    observations = trained.buffers.validation
    np.testing.assert_array_equal(
        np.asarray(trained.model(observations).values), np.asarray(restored.model(observations).values)
    )


def test_trackio_runs_hand_off_through_artifacts_and_load_stages(tmp_path: Path) -> None:
    environment = {**os.environ, "TRACKIO_DIR": str(tmp_path / "trackio")}
    _main([*SMALL_RUN, "--track", "trackio", "--project", "test"], environment)
    first = _read("trackio", "test", environment)
    (group,) = first
    runs = first[group]
    prefix = "mastermind222_behaviour"
    assert sorted(runs) == sorted([
        f"{prefix}--algorithm--dqn--seed-1",
        *(
            f"{prefix}--{stage}--seed-{seed}"
            for stage in (
                "characteristic--behaviour",
                "shapley--behaviour--with-exact",
                "shapley--behaviour--without-exact",
            )
            for seed in (1, 2)
        ),
    ])
    assert {"agent/episodic_return", "agent/DQN_loss", "_step"} <= {
        key for row in runs[f"{prefix}--algorithm--dqn--seed-1"] for key in row
    }
    characteristic_rows = runs[f"{prefix}--characteristic--behaviour--seed-1"]
    assert "model/PolicyCharacteristicModel_exact_loss" in {key for row in characteristic_rows for key in row}
    assert [row["_step"] for row in characteristic_rows] == [1.0, 2.0, 3.0]
    # Loading the agent skips training it (its run logs nothing), and every later stage reproduces the first run.
    agent = f"{group}--algorithm--dqn:latest"
    _main([*SMALL_RUN, "--track", "trackio", "--project", "test", "--algorithm.checkpoint", agent], environment)
    second = _read("trackio", "test", environment)
    (loaded,) = [name for name in second if name != group]
    # trackio keeps no record of a run that logged nothing.
    assert second[loaded].get(f"{prefix}--algorithm--dqn--seed-1", []) == []
    for name, rows in second[loaded].items():
        if name != f"{prefix}--algorithm--dqn--seed-1":
            assert rows == runs[name], name
    # A checkpoint saved under different settings is refused.
    failed = _main(
        [
            *SMALL_RUN,
            "--track",
            "trackio",
            "--project",
            "test",
            "--algorithm.total-timesteps",
            "40000",
            "--algorithm.checkpoint",
            agent,
        ],
        environment,
        check=False,
    )
    assert failed.returncode != 0 and "different settings" in failed.stderr


def test_compile_only_runs_and_tracks_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("compile-only must not start a tracking run")

    monkeypatch.setattr(wandb, "init", refuse)
    monkeypatch.setattr(trackio, "init", refuse)
    assert main([*SMALL_RUN, "--compile-only"]) == ""


def test_checkpoint_uploads_retry_transient_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A busy backend at upload time must not discard a finished stage: transient errors retry."""
    monkeypatch.setenv("TRACKIO_DIR", str(tmp_path / "trackio"))
    run = trackio.init(project="upload-retry", name="stage")
    attempts: list[int] = []
    original = run.log_artifact

    def flaky(artifact_or_path: str, name: str | None = None, type: str | None = None) -> trackio.Artifact:
        attempts.append(len(attempts))
        if len(attempts) < 3:
            raise wandb.errors.CommError("the service process is busy and did not respond in time")
        return original(artifact_or_path, name=name, type=type)

    monkeypatch.setattr(run, "log_artifact", flaky)
    monkeypatch.setattr(backends.time, "sleep", lambda seconds: None)
    folder = tmp_path / "checkpoint"
    folder.mkdir()
    (folder / "metadata.json").write_text("{}")
    backends.upload(run, folder, "group--stage", "agent")
    assert len(attempts) == 3

    def down(artifact_or_path: str, name: str | None = None, type: str | None = None) -> trackio.Artifact:
        raise OSError("down")

    monkeypatch.setattr(run, "log_artifact", down)
    with pytest.raises(OSError, match="down"):
        backends.upload(run, folder, "group--stage", "agent")
    run.finish()


@pytest.mark.skipif(os.environ.get("JAX_SHAPLEY_WANDB_TEST") != "1", reason="writes to the wandb cloud")
def test_wandb_keeps_nothing_locally() -> None:
    before = set(Path(tempfile.gettempdir()).glob("jax-shapley-*"))
    _main([*SMALL_RUN, "--track", "wandb", "--project", "jax-shapley-test"], dict(os.environ))
    assert set(Path(tempfile.gettempdir()).glob("jax-shapley-*")) == before
    groups = _read("wandb", "jax-shapley-test", dict(os.environ))
    latest = groups[max(groups)]
    rows = latest["mastermind222_behaviour--shapley--behaviour--with-exact--seed-1"]
    assert "model/PolicyShapley_exact_loss" in {key for row in rows for key in row}


@pytest.fixture(scope="module")
def result() -> ExperimentResult:
    return run(CONFIG)


def _record(items: list[dqn.StepMetrics], seed: jax.Array, metrics: dqn.StepMetrics) -> None:
    del seed
    items.append(metrics)


def _main(arguments: list[str], environment: dict[str, str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] -- fixed local argv.
        [sys.executable, "-m", "jax_shapley.main", *arguments],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and completed.returncode != 0:
        raise AssertionError(completed.stderr[-3000:])
    return completed


def _read(track: str, project: str, environment: dict[str, str]) -> dict[str, dict[str, list[dict[str, float]]]]:
    completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] -- fixed local argv.
        [sys.executable, "-c", READ_GROUPS, track, project, "fastsverl" if track == "wandb" else ""],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    groups: dict[str, dict[str, list[dict[str, float]]]] = json.loads(completed.stdout.strip().splitlines()[-1])
    return groups


def _assert_same_state(expected: object, actual: object) -> None:
    left, right = jax.tree.leaves(nnx.state(expected)), jax.tree.leaves(nnx.state(actual))
    assert len(left) == len(right)
    for a, b in zip(left, right, strict=True):
        if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
