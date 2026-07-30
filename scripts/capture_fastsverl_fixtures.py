"""Capture exhaustive transition fixtures from a local FastSVERL checkout.

This developer tool deliberately does not fetch or install FastSVERL. Pass a
checkout pinned to the commit recorded in ``EXPECTED_COMMIT``. The generated
fixtures are consumed by offline tests.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import numpy as np

EXPECTED_COMMIT = "25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b"
SOURCE_FILES = {
    "gridworld": Path("fastsverl/fastsverl/envs/gwb.py"),
    "hypercube": Path("fastsverl/fastsverl/envs/hypercube.py"),
    "mastermind": Path("fastsverl/fastsverl/envs/mastermind.py"),
}


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def capture_gridworld(module: ModuleType) -> dict[str, np.ndarray]:
    env = module.GWB(seed=0)
    states = np.asarray(list(env.states_decode.values()), dtype=np.int32)
    inputs: list[np.ndarray] = []
    actions: list[int] = []
    next_obs: list[np.ndarray] = []
    rewards: list[float] = []
    terminated: list[bool] = []
    truncated: list[bool] = []
    for state in states:
        for action in range(env.action_space.n):
            env.reset(start_state=state)
            obs, reward, done, trunc, _ = env.step(action)
            inputs.append(state.copy())
            actions.append(action)
            next_obs.append(obs)
            rewards.append(reward)
            terminated.append(done)
            truncated.append(trunc)
    return {
        "grid_state": np.asarray(inputs, dtype=np.int32),
        "grid_action": np.asarray(actions, dtype=np.int32),
        "grid_next_obs": np.asarray(next_obs, dtype=np.int32),
        "grid_reward": np.asarray(rewards, dtype=np.float32),
        "grid_terminated": np.asarray(terminated, dtype=np.bool_),
        "grid_truncated": np.asarray(truncated, dtype=np.bool_),
    }


def capture_hypercubes(module: ModuleType) -> dict[str, np.ndarray]:
    configs: list[tuple[int, int]] = []
    states: list[np.ndarray] = []
    actions: list[int] = []
    next_obs: list[np.ndarray] = []
    rewards: list[float] = []
    terminated: list[bool] = []
    truncated: list[bool] = []

    for side_length in range(2, 7):
        for dimension in range(2, 6):
            env = module.Hypercube(n=side_length, d=dimension, p=0, seed=0)
            for state in env.states:
                for action in range(env.action_space.n):
                    env.pos = state.copy()
                    obs, reward, done, trunc, _ = env.step(action)
                    padded_state = np.full(5, -1, dtype=np.int32)
                    padded_next = np.full(5, -1, dtype=np.int32)
                    padded_state[:dimension] = state
                    padded_next[:dimension] = obs
                    configs.append((side_length, dimension))
                    states.append(padded_state)
                    actions.append(action)
                    next_obs.append(padded_next)
                    rewards.append(reward)
                    terminated.append(done)
                    truncated.append(trunc)

    return {
        "cube_config": np.asarray(configs, dtype=np.int32),
        "cube_state": np.asarray(states, dtype=np.int32),
        "cube_action": np.asarray(actions, dtype=np.int32),
        "cube_next_obs": np.asarray(next_obs, dtype=np.int32),
        "cube_reward": np.asarray(rewards, dtype=np.float32),
        "cube_terminated": np.asarray(terminated, dtype=np.bool_),
        "cube_truncated": np.asarray(truncated, dtype=np.bool_),
    }


def capture_mastermind(module: ModuleType) -> dict[str, np.ndarray]:
    env = module.Mastermind(code_size=2, num_guesses=2, num_pegs=2, seed=0)
    boards: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    guess_indices: list[int] = []
    actions: list[int] = []
    next_boards: list[np.ndarray] = []
    rewards: list[float] = []
    terminated: list[bool] = []
    truncated: list[bool] = []

    def record(source_env: object, action: int) -> object:
        transition_env = deepcopy(source_env)
        boards.append(transition_env.grid.copy())
        goals.append(transition_env.goal.copy())
        guess_indices.append(transition_env.count)
        actions.append(action)
        _, reward, done, trunc, _ = transition_env.step(action)
        next_boards.append(transition_env.grid.copy())
        rewards.append(reward)
        terminated.append(done)
        truncated.append(trunc)
        return transition_env

    for goal in env.index_to_guess:
        env.reset()
        env.goal = goal.copy()
        for first_action in range(env.action_space.n):
            after_first = record(env, first_action)
            if not terminated[-1]:
                for second_action in range(env.action_space.n):
                    record(after_first, second_action)

    return {
        "mm_board": np.asarray(boards, dtype=np.int32),
        "mm_goal": np.asarray(goals, dtype=np.int32),
        "mm_guess_index": np.asarray(guess_indices, dtype=np.int32),
        "mm_action": np.asarray(actions, dtype=np.int32),
        "mm_next_board": np.asarray(next_boards, dtype=np.int32),
        "mm_reward": np.asarray(rewards, dtype=np.float32),
        "mm_terminated": np.asarray(terminated, dtype=np.bool_),
        "mm_truncated": np.asarray(truncated, dtype=np.bool_),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/fastsverl_transitions.npz"),
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    commit = git_commit(source_root)
    if commit != EXPECTED_COMMIT:
        raise SystemExit(f"Expected FastSVERL {EXPECTED_COMMIT}, found {commit}")

    paths = {name: source_root / relative for name, relative in SOURCE_FILES.items()}
    modules = {name: load_module(f"fastsverl_fixture_{name}", path) for name, path in paths.items()}

    arrays: dict[str, np.ndarray] = {}
    arrays.update(capture_gridworld(modules["gridworld"]))
    arrays.update(capture_hypercubes(modules["hypercube"]))
    arrays.update(capture_mastermind(modules["mastermind"]))

    counts = {
        "gridworld": int(arrays["grid_action"].size),
        "hypercube": int(arrays["cube_action"].size),
        "mastermind_222": int(arrays["mm_action"].size),
    }
    counts["total"] = sum(counts.values())
    if counts != {
        "gridworld": 20,
        "hypercube": 143_052,
        "mastermind_222": 64,
        "total": 143_136,
    }:
        raise RuntimeError(f"Unexpected fixture counts: {counts}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    provenance = {
        "fastsverl_commit": commit,
        "source_sha256": {name: sha256(path) for name, path in paths.items()},
        "transition_counts": counts,
        "fixture_sha256": sha256(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
