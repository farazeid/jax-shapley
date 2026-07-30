"""Summarize raw environment-throughput measurements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tyro


@dataclass(frozen=True)
class Args:
    """Inputs and statistics settings for the throughput summary."""

    gymnasium_cpu_data: Path = Path("docs/data/envs-port--throughput--gymnasium-cpu.csv")
    jax_cpu_data: Path = Path("docs/data/envs-port--throughput--jax-cpu.csv")
    jax_gpu_data: Path = Path("docs/data/envs-port--throughput--jax-gpu.csv")
    output: Path = Path("docs/data/envs-port--throughput--summary.csv")
    confidence_level: float = 0.95
    bootstrap_resamples: int = 10_000
    bootstrap_seed: int = 0


def main() -> None:
    args = tyro.cli(Args)
    data = load_throughput(args)
    summary = summarize(args, data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"Wrote {len(summary)} rows to {args.output}")


def load_throughput(args: Args) -> pd.DataFrame:
    paths = (args.gymnasium_cpu_data, args.jax_cpu_data, args.jax_gpu_data)
    return pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)


def summarize(args: Args, data: pd.DataFrame) -> pd.DataFrame:
    valid = data[data["status"] == "ok"].copy()
    rng = np.random.default_rng(args.bootstrap_seed)
    rows = []
    groups = valid.groupby(["environment", "backend", "num_envs"])["steps_per_second"]
    for (environment, backend, num_envs), group in groups:
        values = group.to_numpy(dtype=np.float64)
        ci_lower, ci_upper = bootstrap_median_interval(args, values, rng)
        rows.append(
            {
                "environment": environment,
                "backend": backend,
                "num_envs": num_envs,
                "samples": values.size,
                "median": np.median(values),
                "q1": np.quantile(values, 0.25),
                "q3": np.quantile(values, 0.75),
                "median_ci95_lower": ci_lower,
                "median_ci95_upper": ci_upper,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_median_interval(
    args: Args,
    values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if not 0.0 < args.confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    if args.bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")
    indices = rng.integers(
        values.size,
        size=(args.bootstrap_resamples, values.size),
    )
    bootstrap_medians = np.median(values[indices], axis=1)
    tail = (1.0 - args.confidence_level) / 2.0
    lower, upper = np.quantile(bootstrap_medians, (tail, 1.0 - tail))
    return float(lower), float(upper)


if __name__ == "__main__":
    main()
