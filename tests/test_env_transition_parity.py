"""Exhaustive transition parity against captured FastSVERL behavior."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_shapley.envs import (
    Gridworld,
    GridworldState,
    Hypercube,
    HypercubeState,
    Mastermind222,
    Mastermind222State,
)


class _RawStep(Protocol):
    def __call__(
        self, key: jax.Array, state: object, action: jax.Array, params: object
    ) -> tuple[jax.Array, object, jax.Array, jax.Array, dict[str, jax.Array]]: ...


type RawStepOutput = tuple[jax.Array, object, jax.Array, jax.Array, dict[str, jax.Array]]

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "fastsverl_transitions.npz"


@pytest.mark.slow
def test_exhaustive_transition_parity(request: pytest.FixtureRequest) -> None:
    output = request.config.getoption("--parity-output")
    assert_zero_mismatches(measure_parity(output))


@pytest.mark.gpu
@pytest.mark.slow
def test_gpu_exhaustive_transition_parity(request: pytest.FixtureRequest) -> None:
    jax_logger = logging.getLogger("jax._src.xla_bridge")
    logger_was_disabled = jax_logger.disabled
    jax_logger.disabled = True
    try:
        try:
            gpu_devices = jax.devices("gpu")
        except RuntimeError:
            gpu_devices = []
    finally:
        jax_logger.disabled = logger_was_disabled
    if not gpu_devices:
        pytest.skip("GPU parity skipped: this test requires an available JAX GPU backend")
    assert jax.default_backend() == "gpu"
    output = request.config.getoption("--parity-output")
    assert_zero_mismatches(measure_parity(output))


def measure_parity(output: Path | None) -> dict[str, dict[str, int]]:
    results = transition_parity_counts(FIXTURE_PATH)
    if output is not None:
        write_parity_csv(output, parity_rows(jax.default_backend(), results))
    return results


def assert_zero_mismatches(results: dict[str, dict[str, int]]) -> None:
    assert sum(row["tested"] for row in results.values()) == 143_136
    failures = {
        name: {field: value for field, value in row.items() if field != "tested" and value}
        for name, row in results.items()
    }
    failures = {name: row for name, row in failures.items() if row}
    assert not failures, failures


def transition_parity_counts(fixture_path: Path) -> dict[str, dict[str, int]]:
    """Return component-level mismatch counts for captured reference transitions."""
    fixture = np.load(fixture_path)
    results: dict[str, dict[str, int]] = {}

    env = Gridworld()
    count = fixture["grid_action"].size
    state = GridworldState(
        position=jnp.asarray(fixture["grid_state"]),
        terminated=jnp.zeros(count, dtype=jnp.bool_),
        time=jnp.zeros(count, dtype=jnp.int32),
    )
    output = _batched_raw_step(
        cast(_RawStep, env.step_env),
        jax.random.split(jax.random.key(0), count),
        state,
        jnp.asarray(fixture["grid_action"]),
        env.default_params,
    )
    obs, next_state, reward, done, info = jax.device_get(output)
    results["gridworld"] = _result_row(
        count,
        observation=_difference_mask(obs, fixture["grid_next_obs"]),
        state=_difference_mask(next_state.position, fixture["grid_next_obs"])
        | _difference_mask(next_state.terminated, fixture["grid_terminated"])
        | _difference_mask(next_state.time, np.ones(count, dtype=np.int32)),
        reward=_difference_mask(reward, fixture["grid_reward"]),
        terminated=_difference_mask(done, fixture["grid_terminated"])
        | _difference_mask(info["terminated"], fixture["grid_terminated"]),
        truncated=_difference_mask(info["truncated"], fixture["grid_truncated"]),
    )

    for side_length in range(2, 7):
        for dimension in range(2, 6):
            name = f"hypercube_{side_length}x{dimension}"
            config = fixture["cube_config"]
            indices = np.flatnonzero((config[:, 0] == side_length) & (config[:, 1] == dimension))
            count = indices.size
            env = Hypercube(side_length, dimension)
            state = HypercubeState(
                position=jnp.asarray(fixture["cube_state"][indices, :dimension]),
                terminated=jnp.zeros(count, dtype=jnp.bool_),
                time=jnp.zeros(count, dtype=jnp.int32),
            )
            output = _batched_raw_step(
                cast(_RawStep, env.step_env),
                jax.random.split(jax.random.key(side_length * 10 + dimension), count),
                state,
                jnp.asarray(fixture["cube_action"][indices]),
                env.default_params,
            )
            obs, next_state, reward, done, info = jax.device_get(output)
            expected_obs = fixture["cube_next_obs"][indices, :dimension]
            results[name] = _result_row(
                count,
                observation=_difference_mask(obs, expected_obs),
                state=_difference_mask(next_state.position, expected_obs)
                | _difference_mask(next_state.terminated, fixture["cube_terminated"][indices])
                | _difference_mask(next_state.time, np.ones(count, dtype=np.int32)),
                reward=_difference_mask(reward, fixture["cube_reward"][indices]),
                terminated=_difference_mask(done, fixture["cube_terminated"][indices])
                | _difference_mask(info["terminated"], fixture["cube_terminated"][indices]),
                truncated=_difference_mask(info["truncated"], fixture["cube_truncated"][indices]),
            )

    env = Mastermind222()
    count = fixture["mm_action"].size
    state = Mastermind222State(
        board=jnp.asarray(fixture["mm_board"]),
        goal=jnp.asarray(fixture["mm_goal"]),
        guess_index=jnp.asarray(fixture["mm_guess_index"]),
        terminated=jnp.zeros(count, dtype=jnp.bool_),
        time=jnp.asarray(fixture["mm_guess_index"]),
    )
    output = _batched_raw_step(
        cast(_RawStep, env.step_env),
        jax.random.split(jax.random.key(999), count),
        state,
        jnp.asarray(fixture["mm_action"]),
        env.default_params,
    )
    obs, next_state, reward, done, info = jax.device_get(output)
    expected_board = fixture["mm_next_board"]
    results["mastermind_222"] = _result_row(
        count,
        observation=_difference_mask(obs, expected_board.reshape(count, -1)),
        state=_difference_mask(next_state.board, expected_board)
        | _difference_mask(next_state.goal, fixture["mm_goal"])
        | _difference_mask(next_state.guess_index, fixture["mm_guess_index"] + np.asarray(1, dtype=np.int32))
        | _difference_mask(next_state.terminated, fixture["mm_terminated"])
        | _difference_mask(next_state.time, fixture["mm_guess_index"] + np.asarray(1, dtype=np.int32)),
        reward=_difference_mask(reward, fixture["mm_reward"]),
        terminated=_difference_mask(done, fixture["mm_terminated"])
        | _difference_mask(info["terminated"], fixture["mm_terminated"]),
        truncated=_difference_mask(info["truncated"], fixture["mm_truncated"]),
    )
    return results


def _batched_raw_step(
    step: _RawStep,
    keys: jax.Array,
    state: object,
    actions: jax.Array,
    params: object,
) -> RawStepOutput:
    transformed = jax.jit(jax.vmap(step, in_axes=(0, 0, 0, None)))
    return cast(RawStepOutput, transformed(keys, state, actions, params))


def _result_row(tested: int, **components: np.ndarray) -> dict[str, int]:
    masks = {name: np.asarray(mask, dtype=np.bool_) for name, mask in components.items()}
    record_mask = np.logical_or.reduce(tuple(masks.values()))
    return {
        "tested": tested,
        "mismatched": int(np.count_nonzero(record_mask)),
        **{name: int(np.count_nonzero(mask)) for name, mask in masks.items()},
    }


def _difference_mask(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.ndim <= 1:
        return actual != expected
    axes = tuple(range(1, actual.ndim))
    return np.any(actual != expected, axis=axes)


def parity_rows(
    backend: str,
    results: dict[str, dict[str, int]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [
        {
            "backend": backend,
            "environment": environment,
            **counts,
            "mismatch_proportion": counts["mismatched"] / counts["tested"],
        }
        for environment, counts in results.items()
    ]
    count_fields = (
        "tested",
        "mismatched",
        "observation",
        "state",
        "reward",
        "terminated",
        "truncated",
    )
    aggregate = {key: sum(row[key] for row in results.values()) for key in count_fields}
    rows.append({
        "backend": backend,
        "environment": "aggregate",
        **aggregate,
        "mismatch_proportion": aggregate["mismatched"] / aggregate["tested"],
    })
    return rows


def write_parity_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "backend",
        "environment",
        "tested",
        "mismatched",
        "mismatch_proportion",
        "observation",
        "state",
        "reward",
        "terminated",
        "truncated",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
