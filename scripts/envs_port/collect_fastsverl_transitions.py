"""Capture transition fixtures from a local FastSVERL checkout.

This developer tool deliberately does not fetch or install FastSVERL. Pass a
checkout pinned to the commit exposed as the ``expected_commit`` CLI option. The generated
fixtures are consumed by offline tests:

- ``transitions``: exhaustive Gridworld, Hypercube and Mastermind-222 transitions.
- ``mastermind``: the paper's larger Mastermind sizes (3-3-3, 4-4-3, 4-5-3, 4-6-3). Their
  histories are too many to enumerate, so the fixture holds every (first guess, goal)
  pair, which pins the clue logic, and seeded episodes that win at every guess index or
  lose at the last guess, which pin termination and rewards.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json

# Fixture provenance invokes Git with fixed argv and no shell.
import subprocess  # ruff: ignore[suspicious-subprocess-import]
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Literal, Protocol, cast

import numpy as np
import numpy.typing as npt
import tyro

ArrayLike = npt.ArrayLike


# (code size, guesses, pegs) reported in the paper beyond 2-2-2.
MASTERMIND_SIZES = ((3, 3, 3), (4, 4, 3), (4, 5, 3), (4, 6, 3))
EPISODES_PER_GUESS = 100
LOSING_EPISODES = 200
EXPECTED_MASTERMIND_COUNTS = {"mm333": 1907, "mm443": 8342, "mm453": 9044, "mm463": 9842}


@dataclass(frozen=True)
class Args:
    """Configuration for fixture capture from a pinned local checkout."""

    source_root: Path
    output: Path = Path("tests/fixtures/fastsverl_transitions.npz")
    expected_commit: str = "25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b"
    gridworld_source: Path = Path("fastsverl/fastsverl/envs/gwb.py")
    hypercube_source: Path = Path("fastsverl/fastsverl/envs/hypercube.py")
    mastermind_source: Path = Path("fastsverl/fastsverl/envs/mastermind.py")
    mastermind_output: Path = Path("tests/fixtures/fastsverl_mastermind.npz")
    fixtures: Literal["all", "transitions", "mastermind"] = "all"


class _ActionSpace(Protocol):
    n: int


class _GridworldFactory(Protocol):
    def __call__(self, *, seed: int) -> _GridworldInstance: ...


class _HypercubeFactory(Protocol):
    def __call__(self, *, n: int, d: int, p: int, seed: int) -> _HypercubeInstance: ...


class _MastermindFactory(Protocol):
    def __call__(self, *, code_size: int, num_guesses: int, num_pegs: int, seed: int) -> _MastermindInstance: ...


class _SavezCompressed(Protocol):
    def __call__(self, file: Path, **arrays: ArrayLike) -> None: ...


class _GridworldInstance(Protocol):
    states_decode: dict[str, np.ndarray]
    action_space: _ActionSpace

    def reset(self, start_state: np.ndarray | None = None) -> object: ...

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]: ...


class _HypercubeInstance(Protocol):
    states: list[np.ndarray]
    action_space: _ActionSpace
    pos: np.ndarray

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]: ...


class _MastermindInstance(Protocol):
    index_to_guess: np.ndarray
    num_guesses: int
    action_space: _ActionSpace
    grid: np.ndarray
    goal: np.ndarray
    count: int

    def reset(self) -> object: ...

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]: ...


def main() -> None:
    args = tyro.cli(Args)
    source_root = args.source_root.resolve()
    commit = git_commit(source_root)
    if commit != args.expected_commit:
        raise SystemExit(f"Expected FastSVERL {args.expected_commit}, found {commit}")

    relative_paths = {
        "gridworld": args.gridworld_source,
        "hypercube": args.hypercube_source,
        "mastermind": args.mastermind_source,
    }
    paths = {name: source_root / relative for name, relative in relative_paths.items()}
    modules = {name: load_module(f"fastsverl_fixture_{name}", path) for name, path in paths.items()}
    if args.fixtures in ("all", "mastermind"):
        write_mastermind_sizes(modules, args.mastermind_output, commit, paths["mastermind"])
    if args.fixtures == "mastermind":
        return
    arrays = capture_transitions(modules)
    counts = transition_counts(arrays)
    expected_counts = {
        "gridworld": 20,
        "hypercube": 143_052,
        "mastermind_222": 64,
        "total": 143_136,
    }
    if counts != expected_counts:
        raise RuntimeError(f"Unexpected fixture counts: {counts}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cast(_SavezCompressed, np.savez_compressed)(args.output, **arrays)
    provenance = {
        "fastsverl_commit": commit,
        "source_sha256": {name: sha256(path) for name, path in paths.items()},
        "transition_counts": counts,
        "fixture_sha256": sha256(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps(provenance, indent=2, sort_keys=True))


def capture_transitions(modules: dict[str, ModuleType]) -> dict[str, ArrayLike]:
    gridworld = cast(_GridworldFactory, modules["gridworld"].GWB)
    hypercube = cast(_HypercubeFactory, modules["hypercube"].Hypercube)
    mastermind = cast(_MastermindFactory, modules["mastermind"].Mastermind)
    arrays: dict[str, ArrayLike] = {}
    arrays.update(capture_gridworld(gridworld(seed=0)))
    arrays.update(capture_hypercubes(hypercube))
    arrays.update(capture_mastermind(mastermind(code_size=2, num_guesses=2, num_pegs=2, seed=0)))
    return arrays


def write_mastermind_sizes(modules: dict[str, ModuleType], output: Path, commit: str, source: Path) -> None:
    factory = cast(_MastermindFactory, modules["mastermind"].Mastermind)
    arrays: dict[str, ArrayLike] = {}
    counts: dict[str, int] = {}
    for code_size, num_guesses, num_pegs in MASTERMIND_SIZES:
        name = f"mm{code_size}{num_guesses}{num_pegs}"
        env = factory(code_size=code_size, num_guesses=num_guesses, num_pegs=num_pegs, seed=0)
        captured = capture_mastermind_sizes(env, np.random.default_rng(code_size * 100 + num_guesses * 10 + num_pegs))
        arrays.update({key.replace("mm_", f"{name}_", 1): value for key, value in captured.items()})
        counts[name] = int(np.asarray(captured["mm_action"]).size)
    if counts != EXPECTED_MASTERMIND_COUNTS:
        raise RuntimeError(f"Unexpected Mastermind fixture counts: {counts}")
    output.parent.mkdir(parents=True, exist_ok=True)
    cast(_SavezCompressed, np.savez_compressed)(output, **arrays)
    provenance = {
        "fastsverl_commit": commit,
        "source_sha256": {"mastermind": sha256(source)},
        "transition_counts": counts,
        "episodes_per_guess_index": EPISODES_PER_GUESS,
        "losing_episodes": LOSING_EPISODES,
        "fixture_sha256": sha256(output),
    }
    output.with_suffix(".json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(json.dumps(provenance, indent=2, sort_keys=True))


def capture_mastermind_sizes(env: _MastermindInstance, rng: np.random.Generator) -> dict[str, ArrayLike]:
    """Every first guess against every goal, then winning and losing episodes from seeded draws."""
    rows = _MastermindRows()
    # Native dtype: upstream's clue cache keys on the goal's raw int64 bytes.
    goals = [np.asarray(goal) for goal in env.index_to_guess]
    actions = env.action_space.n
    for goal in goals:
        for action in range(actions):
            rows.record(_fresh(env, goal), action)
    for win_at in range(env.num_guesses):
        for _ in range(EPISODES_PER_GUESS):
            goal_index = int(rng.integers(actions))
            current = _fresh(env, goals[goal_index])
            for _ in range(win_at):
                # A wrong guess: every other action, uniformly.
                action = int(rng.integers(actions - 1))
                current = rows.record(current, action if action < goal_index else action + 1)
            rows.record(current, goal_index)
    for _ in range(LOSING_EPISODES):
        current = _fresh(env, goals[int(rng.integers(actions))])
        while True:
            current = rows.record(current, int(rng.integers(actions)))
            if rows.terminated[-1]:
                break
    return rows.arrays()


def _fresh(env: _MastermindInstance, goal: np.ndarray) -> _MastermindInstance:
    fresh = deepcopy(env)
    fresh.reset()
    fresh.goal = goal.copy()
    return fresh


class _MastermindRows:
    """Transitions recorded from deep copies, so every source state stays intact."""

    def __init__(self) -> None:
        self.boards: list[np.ndarray] = []
        self.goals: list[np.ndarray] = []
        self.guess_indices: list[int] = []
        self.actions: list[int] = []
        self.next_boards: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.terminated: list[bool] = []
        self.truncated: list[bool] = []

    def record(self, source: _MastermindInstance, action: int) -> _MastermindInstance:
        env = deepcopy(source)
        self.boards.append(np.asarray(env.grid.copy(), dtype=np.int8))
        self.goals.append(np.asarray(env.goal.copy(), dtype=np.int8))
        self.guess_indices.append(int(env.count))
        self.actions.append(action)
        _, reward, done, trunc, _ = env.step(action)
        self.next_boards.append(np.asarray(env.grid.copy(), dtype=np.int8))
        self.rewards.append(float(reward))
        self.terminated.append(done)
        self.truncated.append(trunc)
        return env

    def arrays(self) -> dict[str, ArrayLike]:
        return {
            "mm_board": np.asarray(self.boards, dtype=np.int8),
            "mm_goal": np.asarray(self.goals, dtype=np.int8),
            "mm_guess_index": np.asarray(self.guess_indices, dtype=np.int32),
            "mm_action": np.asarray(self.actions, dtype=np.int32),
            "mm_next_board": np.asarray(self.next_boards, dtype=np.int8),
            "mm_reward": np.asarray(self.rewards, dtype=np.float32),
            "mm_terminated": np.asarray(self.terminated, dtype=np.bool_),
            "mm_truncated": np.asarray(self.truncated, dtype=np.bool_),
        }


def transition_counts(arrays: dict[str, ArrayLike]) -> dict[str, int]:
    arrays_nd = {key: np.asarray(value) for key, value in arrays.items()}
    counts = {
        "gridworld": int(arrays_nd["grid_action"].size),
        "hypercube": int(arrays_nd["cube_action"].size),
        "mastermind_222": int(arrays_nd["mm_action"].size),
    }
    counts["total"] = sum(counts.values())
    return counts


def git_commit(root: Path) -> str:
    return subprocess.check_output(  # ruff: ignore[subprocess-without-shell-equals-true]
        ["git", "-C", str(root), "rev-parse", "HEAD"],  # ruff: ignore[start-process-with-partial-path]
        text=True,
    ).strip()


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def capture_gridworld(env: _GridworldInstance) -> dict[str, ArrayLike]:
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
            next_obs.append(np.asarray(obs, dtype=np.int32))
            rewards.append(float(reward))
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


def capture_hypercubes(factory: _HypercubeFactory) -> dict[str, ArrayLike]:
    states: list[np.ndarray] = []
    configs: list[tuple[int, int]] = []
    actions: list[int] = []
    next_obs: list[np.ndarray] = []
    rewards: list[float] = []
    terminated: list[bool] = []
    truncated: list[bool] = []

    for side_length in range(2, 7):
        for dimension in range(2, 6):
            env = factory(n=side_length, d=dimension, p=0, seed=0)
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
                    rewards.append(float(reward))
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


def capture_mastermind(env: _MastermindInstance) -> dict[str, ArrayLike]:
    boards: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    guess_indices: list[int] = []
    actions: list[int] = []
    next_boards: list[np.ndarray] = []
    rewards: list[float] = []
    terminated: list[bool] = []
    truncated: list[bool] = []

    def record(source_env: _MastermindInstance, action: int) -> _MastermindInstance:
        transition_env = deepcopy(source_env)
        boards.append(np.asarray(transition_env.grid.copy(), dtype=np.int32))
        goals.append(np.asarray(transition_env.goal.copy(), dtype=np.int32))
        guess_indices.append(int(transition_env.count))
        actions.append(action)
        _, reward, done, trunc, _ = transition_env.step(action)
        next_boards.append(np.asarray(transition_env.grid.copy(), dtype=np.int32))
        rewards.append(float(reward))
        terminated.append(done)
        truncated.append(trunc)
        return transition_env

    for goal in env.index_to_guess:
        env.reset()
        env.goal = np.asarray(goal.copy(), dtype=np.int32)
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


if __name__ == "__main__":
    main()
