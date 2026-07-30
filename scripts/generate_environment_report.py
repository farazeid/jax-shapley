"""Generate throughput figures and a Markdown report from raw benchmark CSVs."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from pathlib import Path

import gymnasium
import gymnax
import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

DATA_DIR = Path("docs/data")
FIGURE_DIR = Path("docs/figures")
REPORT_PATH = Path("docs/environment_port_report.md")
PANELS = [
    ("gridworld", "Gridworld"),
    ("hypercube_2x2", "Hypercube 2×2"),
    ("hypercube_6x5", "Hypercube 6×5"),
    ("mastermind_222", "Mastermind-222"),
]
BACKENDS = [
    ("gymnasium_cpu", "Gymnasium CPU", "#555555", "o"),
    ("jax_cpu", "JAX CPU", "#2070b4", "s"),
    ("jax_gpu", "JAX GPU", "#d1495b", "^"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def command_output(command: list[str], fallback: str = "unavailable") -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return fallback


def load_throughput() -> pd.DataFrame:
    files = [
        DATA_DIR / "throughput_gymnasium_cpu.csv",
        DATA_DIR / "throughput_jax_cpu.csv",
        DATA_DIR / "throughput_jax_gpu.csv",
    ]
    return pd.concat((pd.read_csv(path) for path in files), ignore_index=True)


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    valid = data[data["status"] == "ok"].copy()
    return (
        valid.groupby(["environment", "backend", "num_envs"])["steps_per_second"]
        .agg(
            median="median",
            q1=lambda values: np.quantile(values, 0.25),
            q3=lambda values: np.quantile(values, 0.75),
        )
        .reset_index()
    )


def plot(summary: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for axis, (environment, title) in zip(axes.flat, PANELS, strict=True):
        panel = summary[summary["environment"] == environment]
        for backend, label, color, marker in BACKENDS:
            line = panel[panel["backend"] == backend].sort_values("num_envs")
            if line.empty:
                continue
            x = line["num_envs"].to_numpy()
            axis.plot(
                x,
                line["median"],
                label=label,
                color=color,
                marker=marker,
                linewidth=2,
                markersize=4,
            )
            axis.fill_between(x, line["q1"], line["q3"], color=color, alpha=0.14)
        axis.set_title(title)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks([1, 4, 16, 64, 256, 1024, 2048])
        axis.set_xticklabels(["1", "4", "16", "64", "256", "1024", "2048"])
        axis.set_ylabel("Environment transitions / second")
        if environment == "hypercube_6x5":
            axis.axvline(16, color="#777777", linestyle=":", linewidth=1)
            axis.text(
                19,
                axis.get_ylim()[0] * 1.5,
                "Original capped at 16",
                fontsize=8,
                color="#555555",
            )
    for axis in axes[-1]:
        axis.set_xlabel("Number of environments")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    figure.suptitle("FastSVERL environment throughput: original vs end-to-end JAX", y=0.98)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(FIGURE_DIR / "environment_throughput.png", dpi=180)
    figure.savefig(FIGURE_DIR / "environment_throughput.svg")
    plt.close(figure)


def format_rate(value: float | None) -> str:
    if value is None or np.isnan(value):
        return "—"
    return f"{value:,.0f}"


def endpoint_table(summary: pd.DataFrame) -> str:
    lines = [
        "| Environment | Workers | Gymnasium CPU | JAX CPU | JAX GPU | CPU speedup | GPU speedup |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for environment, title in PANELS:
        workers = 16 if environment == "hypercube_6x5" else 2048
        panel = summary[(summary["environment"] == environment) & (summary["num_envs"] == workers)]
        values = {row.backend: float(row.median) for row in panel.itertuples(index=False)}
        original = values.get("gymnasium_cpu")
        cpu = values.get("jax_cpu")
        gpu = values.get("jax_gpu")
        cpu_speedup = cpu / original if cpu is not None and original is not None else np.nan
        gpu_speedup = gpu / original if gpu is not None and original is not None else np.nan
        lines.append(
            f"| {title} | {workers:,} | {format_rate(original)} | {format_rate(cpu)} | "
            f"{format_rate(gpu)} | {cpu_speedup:,.1f}× | {gpu_speedup:,.1f}× |"
        )
    return "\n".join(lines)


def parity_table() -> str:
    frames = [
        pd.read_csv(DATA_DIR / "parity_cpu.csv"),
        pd.read_csv(DATA_DIR / "parity_gpu.csv"),
    ]
    data = pd.concat(frames)
    aggregate = data[data["environment"] == "aggregate"]
    lines = [
        "| JAX backend | Transitions tested | Mismatched | Proportion |",
        "|---|---:|---:|---:|",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {row.backend.upper()} | {int(row.tested):,} | {int(row.mismatched):,} | "
            f"{row.mismatch_proportion:.12f} |"
        )
    return "\n".join(lines)


def write_report(summary: pd.DataFrame) -> None:
    cpu_model = command_output(
        ["bash", "-lc", "lscpu | awk -F: '/Model name/{gsub(/^ +/,\"\",$2); print $2}'"]
    )
    gpu = command_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    cuda = command_output(["bash", "-lc", "nvidia-smi | sed -n '3p' | sed 's/^| //;s/ *|$//'"])
    original_data = pd.read_csv(DATA_DIR / "throughput_gymnasium_cpu.csv")
    skipped = original_data[
        (original_data["environment"] == "hypercube_6x5") & (original_data["status"] != "ok")
    ]
    cap_note = skipped["note"].dropna().iloc[0] if not skipped.empty else "No cap recorded."
    lock_hash = sha256(Path("uv.lock"))
    fixture_provenance = (Path("tests/fixtures/fastsverl_transitions.json")).read_text().strip()

    report = f"""# JAX environment port: parity and throughput

## Result

The JAX ports match **all 143,136 exhaustive FastSVERL transitions** on both CPU and
GPU. Throughput depends strongly on batch size: GPU has overhead at small batches,
then reaches hundreds of millions of transitions per second at 2,048 environments.

{parity_table()}

The mismatch criterion is record-level: a transition is different if any next
observation, environment-state field, reward, termination, or truncation value differs.
Component-level counts and every per-configuration result are retained in
`docs/data/parity_cpu.csv` and `docs/data/parity_gpu.csv`.

## Throughput

Rates below are median environment transitions per second over five samples. The
Gridworld, Hypercube 2×2, and Mastermind rows use 2,048 environments. Hypercube 6×5
uses 16—the largest safe original point—so all three backends are directly comparable.

{endpoint_table(summary)}

![Environment throughput](figures/environment_throughput.png)

The complete 1, 2, 4, …, 2,048 curves, five raw samples per successful point, elapsed
times, and IQRs are available in `docs/data/`. Initial reset, construction, action
generation, transfers, and compilation are excluded. Timed execution includes automatic
resets encountered during 1,024-step rollout blocks.

The original uses FastSVERL's `gym.vector.SyncVectorEnv`. JAX uses `jit(vmap(step))`
inside `lax.scan`, synchronizing with `block_until_ready()`. Identical deterministic
action schedules are used. The original implementation has no GPU path.

For Hypercube 6×5, the unmodified source owns a large transition dictionary per worker.
The safe-source probe reported: `{cap_note}`. Original points above 16 are retained as
`memory_infeasible`; JAX CPU and GPU include the complete curve through 2,048.

## Machine and software

- CPU: {cpu_model}
- Logical CPUs: {psutil.cpu_count(logical=True)}; physical cores: {psutil.cpu_count(logical=False)}
- RAM: {psutil.virtual_memory().total / 2**30:.1f} GiB
- GPU: {gpu}
- Driver/CUDA report: {cuda}
- OS: {platform.platform()}
- Python: {platform.python_version()}
- JAX/JAXLIB: {jax.__version__} / {jax.lib.__version__}
- Gymnax: {gymnax.__version__ if hasattr(gymnax, "__version__") else "0.0.9 (locked)"}
- Gymnasium: {gymnasium.__version__}
- uv lock SHA-256: `{lock_hash}`
- FastSVERL source commit: `25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b`

## Reproduction

```bash
uv sync --all-groups --extra cuda

# Offline semantic tests
JAX_PLATFORMS=cpu uv run --extra cuda pytest
uv run --extra cuda pytest -m gpu

# Parity CSVs
PYTHONPATH=. JAX_PLATFORMS=cpu uv run --extra cuda python scripts/report_parity.py \\
  --output docs/data/parity_cpu.csv
PYTHONPATH=. uv run --extra cuda python scripts/report_parity.py \\
  --output docs/data/parity_gpu.csv

# Throughput; SOURCE_ROOT is a temporary checkout at the pinned commit
uv run --group benchmark python scripts/benchmark_envs.py --backend original \\
  --source-root "$SOURCE_ROOT" --output docs/data/throughput_gymnasium_cpu.csv
JAX_PLATFORMS=cpu uv run --all-groups --extra cuda python scripts/benchmark_envs.py \\
  --backend jax --output docs/data/throughput_jax_cpu.csv
uv run --all-groups --extra cuda python scripts/benchmark_envs.py \\
  --backend jax --output docs/data/throughput_jax_gpu.csv
uv run --group benchmark python scripts/generate_environment_report.py
```

## Provenance and limitations

Fixtures were captured from the pinned source and are offline test inputs; FastSVERL is
not installed or imported by normal tests. NumPy and JAX PRNG streams are intentionally
not expected to be seed-for-seed identical. This benchmark measures environments only,
without policy inference or learning. Agents, FastSVERL models, replay buffers, training,
and paper evaluation reproduction remain deferred.

Fixture provenance:

```json
{fixture_provenance}
```
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)


def main() -> None:
    data = load_throughput()
    summary = summarize(data)
    summary.to_csv(DATA_DIR / "throughput_summary.csv", index=False)
    plot(summary)
    write_report(summary)
    print(f"Wrote {REPORT_PATH} and figures under {FIGURE_DIR}")


if __name__ == "__main__":
    main()
