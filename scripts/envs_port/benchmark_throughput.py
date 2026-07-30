"""Benchmark original FastSVERL Gymnasium environments against JAX rollouts."""

from __future__ import annotations

import csv
import gc
import importlib.util
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Literal

import numpy as np
import tyro


@dataclass(frozen=True)
class Args:
    """Configuration for an original-Gymnasium or JAX throughput run."""

    backend: Literal["original", "jax"] = "jax"
    source_root: Path | None = None
    output: Path = Path("docs/data/envs-port--throughput--jax-cpu.csv")
    block_steps: int = 1_024
    samples: int = 5
    min_sample_seconds: float = 0.5
    num_envs: tuple[int, ...] = tuple(2**power for power in range(12))
    environments: tuple[str, ...] = (
        "gridworld",
        "hypercube_2x2",
        "hypercube_6x5",
        "mastermind_222",
    )
    seed: int = 0
    available_memory_fraction: float = 0.25
    construction_seconds_limit: float = 120.0


def main() -> None:
    args = tyro.cli(Args)
    if args.backend == "original":
        if args.source_root is None:
            raise SystemExit("--source-root is required when --backend original")
        rows = benchmark_original(args, args.source_root.resolve())
    else:
        rows = benchmark_jax(args)
    write_rows(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


def benchmark_original(args: Args, source_root: Path) -> list[dict[str, object]]:
    modules = source_modules(source_root)
    safe_6x5, safe_note = hypercube_6x5_safe_count(args, source_root)
    rows: list[dict[str, object]] = []
    for environment_name in args.environments:
        for num_envs in args.num_envs:
            if environment_name == "hypercube_6x5" and num_envs > safe_6x5:
                rows.append(infeasible_row(args, environment_name, num_envs, safe_note))
                continue
            print(f"original {environment_name} n={num_envs}", flush=True)
            rows.extend(benchmark_original_point(args, environment_name, num_envs, modules))
    return rows


def source_modules(source_root: Path) -> dict[str, ModuleType]:
    base = source_root / "fastsverl" / "fastsverl" / "envs"
    return {
        "gridworld": load_module("bench_fastsverl_gwb", base / "gwb.py"),
        "hypercube": load_module("bench_fastsverl_hypercube", base / "hypercube.py"),
        "mastermind": load_module("bench_fastsverl_mastermind", base / "mastermind.py"),
    }


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hypercube_6x5_safe_count(args: Args, source_root: Path) -> tuple[int, str]:
    """Estimate a safe original-worker cap using a clean subprocess RSS delta."""
    code = f"""
import importlib.util, json, psutil, time
from pathlib import Path
path = Path({str(source_root)!r}) / 'fastsverl/fastsverl/envs/hypercube.py'
spec = importlib.util.spec_from_file_location('memory_probe_hypercube', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
process = psutil.Process()
before = process.memory_info().rss
started = time.perf_counter()
env = module.Hypercube(n=6, d=5, p=0, seed={args.seed})
elapsed = time.perf_counter() - started
after = process.memory_info().rss
print(json.dumps({{'bytes': max(after-before, 1), 'seconds': elapsed}}))
"""
    output = subprocess.check_output([sys.executable, "-c", code], text=True)

    import psutil

    measurement = json.loads(output)
    budget = int(psutil.virtual_memory().available * args.available_memory_fraction)
    memory_cap = max(1, budget // measurement["bytes"])
    time_cap = max(
        1,
        int(args.construction_seconds_limit / max(measurement["seconds"], 1e-6)),
    )
    cap = min(max(args.num_envs), memory_cap, time_cap)
    safe_power = max(value for value in args.num_envs if value <= cap)
    note = (
        f"one 6x5 source env: {measurement['bytes']} bytes RSS, "
        f"{measurement['seconds']:.3f}s construction; safe cap {safe_power}"
    )
    return safe_power, note


def infeasible_row(
    args: Args, environment_name: str, num_envs: int, note: str
) -> dict[str, object]:
    return {
        "environment": environment_name,
        "backend": "gymnasium_cpu",
        "device": "cpu",
        "num_envs": num_envs,
        "sample": -1,
        "block_steps": args.block_steps,
        "repeated_blocks": 0,
        "elapsed_seconds": "",
        "steps_per_second": "",
        "status": "memory_infeasible",
        "note": note,
    }


def benchmark_original_point(
    args: Args,
    environment_name: str,
    num_envs: int,
    modules: dict[str, ModuleType],
) -> list[dict[str, object]]:
    import gymnasium as gym

    envs = gym.vector.SyncVectorEnv(
        [original_thunk(environment_name, modules, args.seed + index) for index in range(num_envs)]
    )
    envs.reset(seed=args.seed)
    actions = action_schedule(args, environment_name, num_envs)

    def run_block() -> None:
        for action in actions:
            envs.step(action)

    run_block()
    started = time.perf_counter()
    run_block()
    repeated_blocks = calibrate(args, time.perf_counter() - started)

    rows = []
    for sample in range(args.samples):
        started = time.perf_counter()
        for _ in range(repeated_blocks):
            run_block()
        elapsed = time.perf_counter() - started
        transitions = num_envs * args.block_steps * repeated_blocks
        rows.append(
            result_row(
                args,
                environment_name,
                "gymnasium_cpu",
                "cpu",
                num_envs,
                sample,
                repeated_blocks,
                elapsed,
                transitions,
            )
        )
    envs.close()
    gc.collect()
    return rows


def original_thunk(
    environment_name: str, modules: dict[str, ModuleType], seed: int
) -> Callable[[], object]:
    import gymnasium as gym

    def thunk() -> object:
        if environment_name == "gridworld":
            env = modules["gridworld"].GWB(seed=seed)
        elif environment_name == "hypercube_2x2":
            env = modules["hypercube"].Hypercube(n=2, d=2, p=0, seed=seed)
        elif environment_name == "hypercube_6x5":
            env = modules["hypercube"].Hypercube(n=6, d=5, p=0, seed=seed)
        elif environment_name == "mastermind_222":
            env = modules["mastermind"].Mastermind(
                code_size=2, num_guesses=2, num_pegs=2, seed=seed
            )
        else:
            raise ValueError(environment_name)
        return gym.wrappers.RecordEpisodeStatistics(env)

    return thunk


def benchmark_jax(args: Args) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for environment_name in args.environments:
        for num_envs in args.num_envs:
            print(f"jax {environment_name} n={num_envs}", flush=True)
            rows.extend(benchmark_jax_point(args, environment_name, num_envs))
    return rows


def benchmark_jax_point(
    args: Args, environment_name: str, num_envs: int
) -> list[dict[str, object]]:
    import jax
    import jax.numpy as jnp

    env = jax_environment(environment_name)
    params = env.default_params
    keys = jax.random.split(jax.random.key(args.seed), num_envs)
    batched_reset = jax.vmap(env.reset, in_axes=(0, None))
    batched_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    obs, state = batched_reset(keys, params)
    actions = jax.device_put(jnp.asarray(action_schedule(args, environment_name, num_envs)))

    def rollout(carry: tuple, rollout_actions: jax.Array) -> tuple:
        def scan_step(current: tuple, action: jax.Array) -> tuple:
            _, current_state, current_keys = current
            split = jax.vmap(jax.random.split)(current_keys)
            next_keys, step_keys = split[:, 0], split[:, 1]
            next_obs, next_state, reward, done, _ = batched_step(
                step_keys, current_state, action, params
            )
            return (next_obs, next_state, next_keys), (reward, done)

        return jax.lax.scan(scan_step, carry, rollout_actions)

    compiled = jax.jit(rollout)
    carry = (obs, state, keys)
    carry, outputs = compiled(carry, actions)
    jax.block_until_ready(outputs)
    started = time.perf_counter()
    carry, outputs = compiled(carry, actions)
    jax.block_until_ready(outputs)
    repeated_blocks = calibrate(args, time.perf_counter() - started)

    rows = []
    for sample in range(args.samples):
        started = time.perf_counter()
        for _ in range(repeated_blocks):
            carry, outputs = compiled(carry, actions)
        jax.block_until_ready(outputs)
        elapsed = time.perf_counter() - started
        transitions = num_envs * args.block_steps * repeated_blocks
        rows.append(
            result_row(
                args,
                environment_name,
                f"jax_{jax.default_backend()}",
                str(jax.devices()[0]),
                num_envs,
                sample,
                repeated_blocks,
                elapsed,
                transitions,
            )
        )
    return rows


def jax_environment(environment_name: str) -> object:
    from jax_shapley.envs import Gridworld, Hypercube, Mastermind222

    if environment_name == "gridworld":
        return Gridworld()
    if environment_name == "hypercube_2x2":
        return Hypercube(2, 2)
    if environment_name == "hypercube_6x5":
        return Hypercube(6, 5)
    if environment_name == "mastermind_222":
        return Mastermind222()
    raise ValueError(environment_name)


def action_schedule(args: Args, environment_name: str, num_envs: int) -> np.ndarray:
    actions = {
        "gridworld": 4,
        "hypercube_2x2": 4,
        "hypercube_6x5": 10,
        "mastermind_222": 4,
    }[environment_name]
    time_indices = np.arange(args.block_steps, dtype=np.int32)[:, None]
    env_indices = np.arange(num_envs, dtype=np.int32)[None, :]
    return (time_indices + env_indices) % actions


def calibrate(args: Args, probe_seconds: float) -> int:
    return max(1, math.ceil(args.min_sample_seconds / max(probe_seconds, 1e-9)))


def result_row(
    args: Args,
    environment_name: str,
    backend: str,
    device: str,
    num_envs: int,
    sample: int,
    repeated_blocks: int,
    elapsed: float,
    transitions: int,
) -> dict[str, object]:
    return {
        "environment": environment_name,
        "backend": backend,
        "device": device,
        "num_envs": num_envs,
        "sample": sample,
        "block_steps": args.block_steps,
        "repeated_blocks": repeated_blocks,
        "elapsed_seconds": elapsed,
        "steps_per_second": transitions / elapsed,
        "status": "ok",
        "note": "",
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "environment",
        "backend",
        "device",
        "num_envs",
        "sample",
        "block_steps",
        "repeated_blocks",
        "elapsed_seconds",
        "steps_per_second",
        "status",
        "note",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
