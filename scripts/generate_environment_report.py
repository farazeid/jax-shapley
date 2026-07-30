"""Generate throughput figures and a Markdown report from raw benchmark CSVs."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

import gymnasium
import gymnax
import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import tyro


@dataclass(frozen=True)
class Args:
    """Inputs, outputs, and presentation settings for the environment-port report."""

    gymnasium_cpu_data: Path = Path("docs/data/envs-port--throughput--gymnasium-cpu.csv")
    jax_cpu_data: Path = Path("docs/data/envs-port--throughput--jax-cpu.csv")
    jax_gpu_data: Path = Path("docs/data/envs-port--throughput--jax-gpu.csv")
    parity_cpu_data: Path = Path("docs/data/envs-port--parity--cpu.csv")
    parity_gpu_data: Path = Path("docs/data/envs-port--parity--gpu.csv")
    summary_output: Path = Path("docs/data/envs-port--throughput--summary.csv")
    svg_output: Path = Path("docs/figures/envs-port--throughput.svg")
    png_output: Path = Path("docs/figures/envs-port--throughput.png")
    report_output: Path = Path(
        "docs/Quantitative Comparison between Gymnasium and JAX environment "
        "implementations of FastSVERL.md"
    )
    fixture_provenance: Path = Path("tests/fixtures/fastsverl_transitions.json")
    uv_lock: Path = Path("uv.lock")
    figure_width: float = 13.0
    figure_height: float = 9.0
    png_dpi: int = 180
    title_y: float = 0.99
    legend_y: float = 0.945
    subplot_top: float = 0.88
    figure_title: str = "FastSVERL environment throughput: original vs end-to-end JAX"
    report_title: str = (
        "Quantitative Comparison between Gymnasium and JAX environment implementations of FastSVERL"
    )
    panel_environments: tuple[str, ...] = (
        "gridworld",
        "hypercube_2x2",
        "hypercube_6x5",
        "mastermind_222",
    )
    panel_titles: tuple[str, ...] = (
        "Gridworld",
        "Hypercube 2×2",
        "Hypercube 6×5",
        "Mastermind-222",
    )
    backend_keys: tuple[str, ...] = ("gymnasium_cpu", "jax_cpu", "jax_gpu")
    backend_labels: tuple[str, ...] = ("Gymnasium CPU", "JAX CPU", "JAX GPU")
    backend_colors: tuple[str, ...] = ("#555555", "#2070b4", "#d1495b")
    backend_markers: tuple[str, ...] = ("o", "s", "^")
    endpoint_workers: int = 2_048
    hypercube_6x5_endpoint_workers: int = 16
    fastsverl_commit: str = "25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b"
    confidence_level: float = 0.95
    bootstrap_resamples: int = 10_000
    bootstrap_seed: int = 0
    confidence_band_alpha: float = 0.14
    confidence_note_y: float = 0.895


def main() -> None:
    args = tyro.cli(Args)
    data = load_throughput(args)
    summary = summarize(args, data)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    plot(args, summary)
    write_report(args, summary)
    print(f"Wrote {args.report_output}, {args.svg_output}, and {args.png_output}")


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


def plot(args: Args, summary: pd.DataFrame) -> None:
    panels = zip(args.panel_environments, args.panel_titles, strict=True)
    args.svg_output.parent.mkdir(parents=True, exist_ok=True)
    args.png_output.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(args.figure_width, args.figure_height),
        sharex=True,
    )
    for axis, (environment, title) in zip(axes.flat, panels, strict=True):
        panel = summary[summary["environment"] == environment]
        for backend, label, color, marker in zip(
            args.backend_keys,
            args.backend_labels,
            args.backend_colors,
            args.backend_markers,
            strict=True,
        ):
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
            axis.fill_between(
                x,
                line["median_ci95_lower"],
                line["median_ci95_upper"],
                color=color,
                alpha=args.confidence_band_alpha,
            )
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
    figure.suptitle(
        args.figure_title,
        y=args.title_y,
    )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, args.legend_y),
        ncol=3,
        frameon=False,
    )
    figure.text(
        0.5,
        args.confidence_note_y,
        f"Shading: {args.confidence_level:.0%} bootstrap confidence interval for median",
        horizontalalignment="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0, 1, args.subplot_top))
    figure.savefig(args.svg_output)
    figure.savefig(args.png_output, dpi=args.png_dpi)
    plt.close(figure)


def write_report(args: Args, summary: pd.DataFrame) -> None:
    cpu_model = cpu_model_name()
    gpu = command_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    cuda = command_output(["bash", "-lc", "nvidia-smi | sed -n '3p' | sed 's/^| //;s/ *|$//'"])
    original_data = pd.read_csv(args.gymnasium_cpu_data)
    skipped = original_data[
        (original_data["environment"] == "hypercube_6x5") & (original_data["status"] != "ok")
    ]
    cap_note = skipped["note"].dropna().iloc[0] if not skipped.empty else "No cap recorded."
    fixture_provenance = args.fixture_provenance.read_text().strip()
    report = report_markdown(
        args,
        summary,
        cpu_model,
        gpu,
        cuda,
        cap_note,
        sha256(args.uv_lock),
        fixture_provenance,
    )
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(report)


def report_markdown(
    args: Args,
    summary: pd.DataFrame,
    cpu_model: str,
    gpu: str,
    cuda: str,
    cap_note: str,
    lock_hash: str,
    fixture_provenance: str,
) -> str:
    return f"""# {args.report_title}

## Result

The JAX ports match **all 143,136 exhaustive FastSVERL transitions** on both CPU and
GPU. Throughput depends strongly on batch size: GPU has overhead at small batches,
then reaches hundreds of millions of transitions per second at 2,048 environments.

{parity_table(args)}

The mismatch criterion is record-level: a transition is different if any next
observation, environment-state field, reward, termination, or truncation value differs.
Component-level counts and every per-configuration result are retained in
`docs/data/envs-port--parity--cpu.csv` and `docs/data/envs-port--parity--gpu.csv`.

## Throughput

Rates below are median environment transitions per second over five samples. The
Gridworld, Hypercube 2×2, and Mastermind rows use 2,048 environments. Hypercube 6×5
uses 16—the largest safe original point—so all three backends are directly comparable.

{endpoint_table(args, summary)}

![Environment throughput](figures/envs-port--throughput.svg)

The complete 1, 2, 4, …, 2,048 curves, five raw samples per successful point, elapsed
times, IQRs, and 95% confidence bounds are available in `docs/data/`. The shaded
areas are deterministic percentile-bootstrap confidence intervals for the median,
using {args.bootstrap_resamples:,} resamples with seed {args.bootstrap_seed}. Initial
reset, construction, action
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
- FastSVERL source commit: `{args.fastsverl_commit}`

## Reproduction

```bash
uv sync --all-groups --extra cuda

# Offline semantic tests
JAX_PLATFORMS=cpu uv run --extra cuda pytest
uv run --extra cuda pytest -m gpu

{reproduction_commands()}

```

All script settings shown above—and the rollout sizes, sample counts, timing threshold,
figure dimensions, and output paths—are adjustable as `tyro` CLI options. Run any script
with `--help` to inspect them.

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


def reproduction_commands() -> str:
    commands = [
        "# Parity CSVs",
        (
            "PYTHONPATH=. JAX_PLATFORMS=cpu uv run --extra cuda python "
            "scripts/report_parity.py --output docs/data/envs-port--parity--cpu.csv"
        ),
        (
            "PYTHONPATH=. uv run --extra cuda python scripts/report_parity.py "
            "--output docs/data/envs-port--parity--gpu.csv"
        ),
        "",
        "# Throughput; SOURCE_ROOT is a temporary checkout at the pinned commit",
        (
            "uv run --group benchmark python scripts/benchmark_envs.py "
            '--backend original --source-root "$SOURCE_ROOT" --output '
            "docs/data/envs-port--throughput--gymnasium-cpu.csv"
        ),
        (
            "JAX_PLATFORMS=cpu uv run --all-groups --extra cuda python "
            "scripts/benchmark_envs.py --backend jax --output "
            "docs/data/envs-port--throughput--jax-cpu.csv"
        ),
        (
            "uv run --all-groups --extra cuda python scripts/benchmark_envs.py "
            "--backend jax --output docs/data/envs-port--throughput--jax-gpu.csv"
        ),
        "uv run --group benchmark python scripts/generate_environment_report.py",
    ]
    return "\n".join(commands)


def endpoint_table(args: Args, summary: pd.DataFrame) -> str:
    panels = zip(args.panel_environments, args.panel_titles, strict=True)
    lines = [
        "| Environment | Workers | Gymnasium CPU | JAX CPU | JAX GPU | CPU speedup | GPU speedup |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for environment, title in panels:
        workers = (
            args.hypercube_6x5_endpoint_workers
            if environment == "hypercube_6x5"
            else args.endpoint_workers
        )
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


def parity_table(args: Args) -> str:
    data = pd.concat((pd.read_csv(args.parity_cpu_data), pd.read_csv(args.parity_gpu_data)))
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


def format_rate(value: float | None) -> str:
    if value is None or np.isnan(value):
        return "—"
    return f"{value:,.0f}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def cpu_model_name() -> str:
    output = command_output(["lscpu"])
    for line in output.splitlines():
        if line.startswith("Model name:"):
            return line.partition(":")[2].strip()
    return "unavailable"


def command_output(command: list[str], fallback: str = "unavailable") -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return fallback


if __name__ == "__main__":
    main()
