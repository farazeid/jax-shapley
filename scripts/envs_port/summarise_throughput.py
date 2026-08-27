"""Summarize raw environment-throughput measurements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tyro
from scipy import stats


@dataclass(frozen=True)
class Args:
    """Inputs for mean throughput and fixed 95% Student t intervals."""

    gymnasium_cpu_data: Path = Path("docs/data/envs-port--throughput--gymnasium-cpu.csv")
    jax_cpu_data: Path = Path("docs/data/envs-port--throughput--jax-cpu.csv")
    jax_gpu_data: Path = Path("docs/data/envs-port--throughput--jax-gpu.csv")
    output: Path = Path("docs/data/envs-port--throughput--summary.csv")


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
    rows = []
    groups = valid.groupby(["environment", "backend", "num_envs"])["steps_per_second"]
    for (environment, backend, num_envs), group in groups:
        values = group.to_numpy(dtype=np.float64)
        mean, standard_deviation, ci_lower, ci_upper = mean_95_interval(values)
        rows.append({
            "environment": environment,
            "backend": backend,
            "num_envs": num_envs,
            "samples": values.size,
            "mean": mean,
            "sample_standard_deviation": standard_deviation,
            "mean_ci95_lower": ci_lower,
            "mean_ci95_upper": ci_upper,
        })
    return pd.DataFrame(rows)


def mean_95_interval(values: np.ndarray) -> tuple[float, float, float, float]:
    if values.size < 2:
        raise ValueError("at least two samples are required for a Student t interval")
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1))
    standard_error = standard_deviation / np.sqrt(values.size)
    critical_value = stats.t.ppf(0.975, df=values.size - 1)
    margin = float(critical_value * standard_error)
    return mean, standard_deviation, mean - margin, mean + margin


if __name__ == "__main__":
    main()
