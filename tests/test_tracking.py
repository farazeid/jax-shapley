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
from flax import nnx

from new_jax_shapley.algorithms import dqn
from new_jax_shapley.characteristic_models import behaviour as characteristic
from new_jax_shapley.characteristic_models.behaviour import CharacteristicConfig
from new_jax_shapley.common import checkpoints
from new_jax_shapley.common.supervised_learning import EpochMetrics, SupervisedConfig, seed_slice, stack_seeds
from new_jax_shapley.envs import Mastermind
from new_jax_shapley.experiment import ExperimentConfig
from new_jax_shapley.main import ExperimentResult, run
from new_jax_shapley.shapley_models import behaviour as shapley
from new_jax_shapley.shapley_models.behaviour import ShapleyConfig

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
from new_jax_shapley import backends
track, project, entity = sys.argv[1], sys.argv[2], sys.argv[3] or None
groups = backends.groups(track, project=project, entity=entity, prefix="mastermind222_behaviour")
print(json.dumps({group: backends.history(track, project=project, entity=entity, group=group) for group in groups}))
"""


def test_live_logs_equal_the_returned_metrics() -> None:
    steps: list[dqn.StepMetrics] = []
    config = CONFIG.algorithm
    state, metrics = nnx.jit(partial(dqn.train, config, steps.append))(jax.random.key(config.seed))
    jax.effects_barrier()
    expected = np.asarray(metrics.completed).any(axis=1) | (
        np.asarray(metrics.updated) & (np.asarray(metrics.global_step) % dqn.LOG_STEPS < config.num_envs)
    )
    logged = [int(step.global_step) for step in steps]
    assert logged == np.asarray(metrics.global_step)[expected].tolist()
    for step in steps:
        index = int(step.global_step) // config.num_envs
        np.testing.assert_array_equal(np.asarray(step.episode_return), np.asarray(metrics.episode_return)[index])
        np.testing.assert_array_equal(np.asarray(step.loss), np.asarray(metrics.loss)[index])
    stage = CONFIG.characteristic
    keys = jnp.stack([jax.random.key(seed) for seed in stage.seeds])
    agents = stack_seeds([dqn.trained_agent(config, state)] * len(stage.seeds))
    keys, buffers = nnx.jit(partial(characteristic.collect, stage))(keys, agents)
    max_unique = max(int(np.unique(rows, axis=0).shape[0]) for rows in np.asarray(buffers.exact))
    epochs: list[EpochMetrics] = []
    _, returned = nnx.jit(partial(characteristic.fit, stage, max_unique, epochs.append))(keys, agents, buffers)
    jax.effects_barrier()
    active = np.asarray(returned.active).any(axis=1)
    assert len(epochs) == int(active.sum())
    for index, epoch in enumerate(epochs):
        for name in ("epoch", "step", "train_loss", "validation_loss", "exact_loss", "active"):
            np.testing.assert_array_equal(np.asarray(getattr(epoch, name)), np.asarray(getattr(returned, name))[index])


def test_stage_checkpoints_restore_the_whole_state(tmp_path: Path, result: ExperimentResult) -> None:
    stages = (
        ("agent", result.dqn_state, dqn.initialize(CONFIG.algorithm, jax.random.key(0))),
        ("characteristic", result.characteristic_state, characteristic.initialize(CONFIG.characteristic)),
        ("shapley", result.shapley[0][0], shapley.initialize(CONFIG.shapley[0], CONFIG.characteristic)),
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
        f"{prefix}--algorithm--dqn",
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
        key for row in runs[f"{prefix}--algorithm--dqn"] for key in row
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
    assert second[loaded].get(f"{prefix}--algorithm--dqn", []) == []
    for name, rows in second[loaded].items():
        if name != f"{prefix}--algorithm--dqn":
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


def _main(arguments: list[str], environment: dict[str, str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true] -- fixed local argv.
        [sys.executable, "-m", "new_jax_shapley.main", *arguments],
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
