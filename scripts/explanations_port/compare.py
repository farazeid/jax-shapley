"""Compare PyTorch FastSVERL and JAX explanation runs: exact-loss outcomes and training speed.

Acceptance rule (stated before the runs): for every configuration, a two-sided Welch
t-test on log10 of the final exact loss across seeds must give p >= 0.01. Hypercube
runs apply the same test to the number of batches until the exact-loss stopping
condition. The ports use independent random streams, so the agents, buffers and
coalitions differ: the comparison is distributional, not seed-for-seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tyro
from scipy import stats

KEYS = ["environment", "kind", "stage", "exact_char"]


@dataclass(frozen=True)
class Args:
    upstream_runs: Path = Path("docs/data/explanations-port--upstream-runs.csv")
    jax_runs: Path = Path("docs/data/explanations-port--jax-runs.csv")
    upstream_timing: Path = Path("docs/data/explanations-port--upstream-timing.csv")
    jax_timing: Path = Path("docs/data/explanations-port--jax-timing.csv")
    summary: Path = Path("docs/data/explanations-port--summary.csv")
    speed: Path = Path("docs/data/explanations-port--speed.csv")
    curves_svg: Path = Path("docs/figures/explanations-port--exact-loss.svg")
    hypercube_svg: Path = Path("docs/figures/explanations-port--hypercube.svg")
    alpha: float = 0.01


def main() -> None:
    args = tyro.cli(Args)
    runs = pd.concat([pd.read_csv(args.upstream_runs), pd.read_csv(args.jax_runs)], ignore_index=True)
    runs["exact_char"] = runs["exact_char"].fillna("")
    timing = pd.concat([pd.read_csv(args.upstream_timing), pd.read_csv(args.jax_timing)], ignore_index=True)
    timing["exact_char"] = timing["exact_char"].fillna("")
    summary = summarize(runs, args.alpha)
    speed = speeds(timing)
    for path in (args.summary, args.speed, args.curves_svg, args.hypercube_svg):
        path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)
    speed.to_csv(args.speed, index=False)
    plot_curves(runs, args.curves_svg)
    plot_hypercube(summary, args.hypercube_svg)
    failed = summary[summary["p_value"] < args.alpha] if not summary.empty else summary
    print(summary.to_string(index=False))
    print(speed.to_string(index=False))
    print(f"{len(failed)} of {len(summary)} comparisons below p = {args.alpha}")


def summarize(runs: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Final exact loss per run (and batches to stopping for Hypercube), compared across frameworks."""
    last = runs.sort_values("step").groupby([*KEYS, "framework", "seed"], as_index=False).last()
    rows = []
    for group_key, group in last.groupby(KEYS):
        key = cast(tuple[str, str, str, str], group_key)  # One value per grouping column.
        metrics = [("final_exact_loss", "exact_loss")]
        if str(key[0]).startswith("hypercube"):
            metrics.append(("batches_to_stop", "step"))
        for metric, column in metrics:
            upstream = group[group["framework"] == "upstream"][column].to_numpy(np.float64)
            jax = group[group["framework"] == "jax"][column].to_numpy(np.float64)
            if upstream.size < 2 or jax.size < 2:
                continue
            p_value = float(stats.ttest_ind(np.log10(jax), np.log10(upstream), equal_var=False).pvalue)
            rows.append({
                **dict(zip(KEYS, key, strict=True)),
                "metric": metric,
                "upstream_runs": upstream.size,
                "jax_runs": jax.size,
                "upstream_geometric_mean": float(10 ** np.log10(upstream).mean()),
                "jax_geometric_mean": float(10 ** np.log10(jax).mean()),
                "ratio": float(10 ** (np.log10(jax).mean() - np.log10(upstream).mean())),
                "p_value": p_value,
                "consistent": p_value >= alpha,
            })
    return pd.DataFrame(rows)


def speeds(timing: pd.DataFrame) -> pd.DataFrame:
    """Per-epoch seconds (mean including checkpoints, and median steady state) and total process wall time.

    Upstream checkpoints only at the end of a run; JAX commits a full-state checkpoint every tenth of the
    epochs, which dominates the mean for runs of a few seconds.
    """
    loop = timing[(timing["measure"] == "loop") & (timing["epochs"] > 1)].copy()
    loop["seconds_per_epoch"] = loop["seconds"] / (loop["epochs"] - 1)
    keys = ["environment", "stage", "framework"]
    mean_epoch = loop.groupby(keys)["seconds_per_epoch"].median().unstack("framework").add_suffix("_mean_epoch")
    steady = timing[timing["measure"] == "epoch_median"].groupby(keys)["seconds"].median().unstack("framework")
    wall = timing[timing["measure"] == "process"].groupby(keys)["seconds"].sum().unstack("framework")
    table = (
        mean_epoch
        .join(steady.add_suffix("_median_epoch"), how="outer")
        .join(wall.add_suffix("_total_wall"), how="outer")
        .reset_index()
    )
    table.columns = [str(column) for column in table.columns]
    for measure in ("mean_epoch", "median_epoch", "total_wall"):
        table[f"{measure}_speedup"] = table[f"upstream_{measure}"] / table[f"jax_{measure}"]
    return pd.DataFrame(table)


def plot_curves(runs: pd.DataFrame, path: Path) -> None:
    """Gridworld and Mastermind-222: exact loss per epoch, geometric mean and 95% interval over seeds."""
    panels = [
        (environment, kind, stage, exact_char)
        for environment in ("gridworld", "mastermind222")
        for kind in ("behaviour", "prediction")
        for stage, exact_char in (("characteristic", ""), ("shapley", "with"), ("shapley", "without"))
    ]
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(4, 3, figsize=(13, 13))
    for axis, (environment, kind, stage, exact_char) in zip(axes.flat, panels, strict=True):
        selection = runs[
            (runs["environment"] == environment)
            & (runs["kind"] == kind)
            & (runs["stage"] == stage)
            & (runs["exact_char"] == exact_char)
        ]
        for framework, color, label in (("upstream", "#555555", "PyTorch (upstream)"), ("jax", "#2070b4", "JAX")):
            data = selection[selection["framework"] == framework].pivot_table(
                index="step", columns="seed", values="exact_loss"
            )
            if data.empty:
                continue
            logs = np.log10(data.to_numpy(np.float64))
            mean = logs.mean(axis=1)
            half = stats.t.ppf(0.975, logs.shape[1] - 1) * logs.std(axis=1, ddof=1) / np.sqrt(logs.shape[1])
            axis.plot(data.index, 10**mean, color=color, label=label, linewidth=1.8)
            axis.fill_between(data.index, 10 ** (mean - half), 10 ** (mean + half), color=color, alpha=0.15)
        title = f"{environment} {kind} {stage}" + (f" ({exact_char} exact char)" if exact_char else "")
        axis.set_title(title, fontsize=9)
        axis.set_yscale("log")
        axis.set_xlabel("epoch", fontsize=8)
        axis.set_ylabel("exact loss", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.text(
        0.5, 0.965, "Geometric mean over seeds; shading: 95% t interval of the log mean", ha="center", fontsize=9
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path)
    plt.close(figure)
    _normalize_svg(path)


def plot_hypercube(summary: pd.DataFrame, path: Path) -> None:
    """Hypercube: batches until the exact-loss condition, per cube, kind and stage."""
    if summary.empty:
        return
    data = summary[(summary["metric"] == "batches_to_stop")]
    if data.empty:
        return
    labels = [f"{row.environment.removeprefix('hypercube-')}\n{row.kind}\n{row.stage}" for row in data.itertuples()]
    x = np.arange(len(data))
    figure, axis = plt.subplots(figsize=(max(8, len(data) * 0.9), 5))
    axis.bar(x - 0.2, data["upstream_geometric_mean"], width=0.4, color="#555555", label="PyTorch (upstream)")
    axis.bar(x + 0.2, data["jax_geometric_mean"], width=0.4, color="#2070b4", label="JAX")
    axis.set_xticks(x, labels, fontsize=7)
    axis.set_yscale("log")
    axis.set_ylabel("batches until exact condition (geometric mean)")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    _normalize_svg(path)


def _normalize_svg(path: Path) -> None:
    lines = path.read_text().splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n")


if __name__ == "__main__":
    main()
