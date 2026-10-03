"""Compare PyTorch FastSVERL and JAX explanation runs: exact-loss outcomes and training speed.

Metrics are log10 of each run's final exact loss and, for Hypercube, of the batches
until the exact-loss stopping condition. The ports use independent random streams,
so agents, buffers and coalitions differ: the comparison is distributional.

Equivalence is tested in two steps, in this order:

1. `--mode margins` reads only the JAX runs. Each metric's margin is the 95th
   percentile of |mean(A) - mean(B)| over random 10-vs-10 splits of the JAX seeds:
   the difference two halves of one implementation show by chance. It has a floor
   of log10(1.02). The margins are committed before any upstream comparison.
2. `--mode compare` requires the committed margins. Each metric passes TOST at
   alpha 0.05 when the 90% Welch interval of the JAX-minus-upstream mean log10
   difference lies inside (-margin, margin). Welch difference p-values are kept
   for reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log10
from pathlib import Path
from typing import Literal, cast

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
    outcome_svg: Path = Path("docs/figures/explanations-port--outcome-exact-loss.svg")
    margins: Path = Path("docs/data/explanations-port--margins.csv")
    mode: Literal["margins", "compare"] = "compare"
    alpha: float = 0.01
    tost_alpha: float = 0.05
    bootstrap: int = 10_000
    margin_floor: float = log10(1.02)


def main() -> None:
    args = tyro.cli(Args)
    if args.mode == "margins":
        # Pilot: JAX runs only, before any cross-framework statistic exists.
        jax_runs = pd.read_csv(args.jax_runs)
        jax_runs["exact_char"] = jax_runs["exact_char"].fillna("")
        margins = margins_from(final_metrics(jax_runs), args.bootstrap, args.margin_floor)
        args.margins.parent.mkdir(parents=True, exist_ok=True)
        margins.to_csv(args.margins, index=False)
        print(margins.to_string(index=False))
        return
    if not args.margins.exists():
        raise SystemExit(f"commit the pilot margins first: {args.margins} (--mode margins)")
    runs = pd.concat([pd.read_csv(args.upstream_runs), pd.read_csv(args.jax_runs)], ignore_index=True)
    runs["exact_char"] = runs["exact_char"].fillna("")
    timing = pd.concat([pd.read_csv(args.upstream_timing), pd.read_csv(args.jax_timing)], ignore_index=True)
    timing["exact_char"] = timing["exact_char"].fillna("")
    margins = pd.read_csv(args.margins)
    margins["exact_char"] = margins["exact_char"].fillna("")
    summary = summarize(final_metrics(runs), margins, args.alpha, args.tost_alpha)
    speed = speeds(timing)
    for path in (args.summary, args.speed, args.curves_svg, args.hypercube_svg, args.outcome_svg):
        path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)
    speed.to_csv(args.speed, index=False)
    plot_curves(runs, args.curves_svg)
    plot_outcome_curves(runs, args.outcome_svg)
    plot_hypercube(summary, args.hypercube_svg)
    print(summary.to_string(index=False))
    print(speed.to_string(index=False))
    equivalent = int(summary["equivalent"].sum()) if not summary.empty else 0
    print(f"{equivalent} of {len(summary)} comparisons equivalent (TOST, alpha {args.tost_alpha})")


def final_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per run and metric: log10 of the final exact loss (and of batches to stopping for Hypercube).

    Runs whose final exact loss is not finite and positive are kept with a NaN value and counted as failures.
    """
    logged = runs.dropna(subset=["exact_loss"]).copy()
    # Characteristics have no exact-characteristic group; upstream Hypercube chains label them "without".
    logged.loc[logged["stage"] == "characteristic", "exact_char"] = ""
    last = logged.sort_values("step").groupby([*KEYS, "framework", "seed"], as_index=False).last()
    frames = []
    for metric, column in (("final_exact_loss", "exact_loss"), ("batches_to_stop", "step")):
        selected = last if metric == "final_exact_loss" else last[last["environment"].str.startswith("hypercube")]
        values = selected[column].to_numpy(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            logs = np.where(np.isfinite(values) & (values > 0), np.log10(values), np.nan)
        frame = selected[[*KEYS, "framework", "seed"]].copy()
        frame["metric"], frame["log10"] = metric, logs
        frames.append(frame)
    return pd.DataFrame(pd.concat(frames, ignore_index=True))


def margins_from(finals: pd.DataFrame, draws: int, floor: float) -> pd.DataFrame:
    """95th percentile of the absolute half-difference over random equal splits of each metric's seeds."""
    rows = []
    for group_key, group in finals.groupby([*KEYS, "metric"]):
        key = cast(tuple[str, str, str, str, str], group_key)  # One value per grouping column.
        values = group["log10"].to_numpy(np.float64)
        values = values[np.isfinite(values)]
        half = values.size // 2
        if half < 2:
            continue
        rng = np.random.default_rng(0)
        differences = np.empty(draws)
        for draw in range(draws):
            order = rng.permutation(values.size)
            differences[draw] = values[order[:half]].mean() - values[order[half : 2 * half]].mean()
        margin = max(float(np.quantile(np.abs(differences), 0.95)), floor)
        rows.append({
            **dict(zip([*KEYS, "metric"], key, strict=True)),
            "jax_runs": values.size,
            "split": f"{half} vs {half}",
            "margin_log10": margin,
            "margin_ratio": 10**margin,
        })
    return pd.DataFrame(rows)


def summarize(finals: pd.DataFrame, margins: pd.DataFrame, alpha: float, tost_alpha: float) -> pd.DataFrame:
    """TOST equivalence of mean log10 metrics within the pilot margins; Welch difference tests for reference."""
    margin_of = {tuple(row[[*KEYS, "metric"]]): float(row["margin_log10"]) for _, row in margins.iterrows()}
    rows = []
    for group_key, group in finals.groupby([*KEYS, "metric"]):
        key = cast(tuple[str, str, str, str, str], group_key)  # One value per grouping column.
        if key not in margin_of:
            continue
        upstream_all = group[group["framework"] == "upstream"]["log10"].to_numpy(np.float64)
        jax_all = group[group["framework"] == "jax"]["log10"].to_numpy(np.float64)
        upstream, jax = upstream_all[np.isfinite(upstream_all)], jax_all[np.isfinite(jax_all)]
        if upstream.size < 2 or jax.size < 2:
            continue
        margin = margin_of[key]
        difference = float(jax.mean() - upstream.mean())
        jax_variance, upstream_variance = jax.var(ddof=1) / jax.size, upstream.var(ddof=1) / upstream.size
        error = float(np.sqrt(jax_variance + upstream_variance))
        freedom = float(
            (jax_variance + upstream_variance) ** 2
            / (jax_variance**2 / (jax.size - 1) + upstream_variance**2 / (upstream.size - 1))
        )
        half_width = float(stats.t.ppf(1 - tost_alpha, freedom)) * error
        low, high = difference - half_width, difference + half_width
        tost_p = max(
            float(stats.t.sf((difference + margin) / error, freedom)),
            float(stats.t.cdf((difference - margin) / error, freedom)),
        )
        p_value = float(stats.ttest_ind(jax, upstream, equal_var=False).pvalue)
        rows.append({
            **dict(zip([*KEYS, "metric"], key, strict=True)),
            "upstream_runs": upstream.size,
            "jax_runs": jax.size,
            "upstream_failed": int(upstream_all.size - upstream.size),
            "jax_failed": int(jax_all.size - jax.size),
            "upstream_geometric_mean": float(10 ** upstream.mean()),
            "jax_geometric_mean": float(10 ** jax.mean()),
            "ratio": float(10**difference),
            "ratio_ci_low": float(10**low),
            "ratio_ci_high": float(10**high),
            "margin_ratio": float(10**margin),
            "tost_p_value": tost_p,
            "equivalent": low > -margin and high < margin,
            "p_value": p_value,
            "consistent": p_value >= alpha,
        })
    return pd.DataFrame(rows)


def speeds(timing: pd.DataFrame) -> pd.DataFrame:
    """Seconds per seed-epoch (mean including checkpoints, and median steady state) and total process wall time.

    A JAX sweep advances every seed's epoch in one compiled step, so its logged epoch interval is divided by
    the sweep's seed count; upstream runs one seed per process. Upstream checkpoints only at the end of a
    run; JAX commits a full-state checkpoint every tenth of the epochs, which dominates the mean for short runs.
    """
    timing = timing.copy()
    group = ["framework", "environment", "kind", "stage", "exact_char"]
    epochs = timing["measure"].isin(["loop", "epoch_median"])
    seeds = timing[epochs].groupby(group)["seed"].transform("nunique")
    timing.loc[epochs, "seconds"] = timing.loc[epochs, "seconds"] / seeds.where(
        timing.loc[epochs, "framework"] == "jax", 1
    )
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


def plot_outcome_curves(runs: pd.DataFrame, path: Path) -> None:
    """Outcome characteristics and Shapley models: exact loss per logged step, geometric mean over seeds."""
    panels = [
        (f"{case}-outcome", kind, stage, exact_char)
        for case, explained in (("gridworld", "on_policy"), ("mastermind222", "off_policy"))
        for kind, stage, exact_char in (
            ("on_policy", "outcome", "with"),
            ("on_policy", "outcome", "without"),
            ("off_policy", "outcome", "with"),
            ("off_policy", "outcome", "without"),
            (explained, "shapley", "w_policy"),
            (explained, "shapley", "wo_policy"),
            (explained, "shapley", "wo_perf"),
        )
    ]
    if runs[runs["environment"].str.endswith("-outcome")].empty:
        return
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 7, figsize=(26, 7.5))
    for axis, (environment, kind, stage, exact_char) in zip(axes.flat, panels, strict=True):
        selection = runs[
            (runs["environment"] == environment)
            & (runs["kind"] == kind)
            & (runs["stage"] == stage)
            & (runs["exact_char"] == exact_char)
        ].dropna(subset=["exact_loss"])
        for framework, color, label in (("upstream", "#555555", "PyTorch (upstream)"), ("jax", "#2070b4", "JAX")):
            data = selection[selection["framework"] == framework].pivot_table(
                index="step", columns="seed", values="exact_loss"
            )
            data = data[(data > 0).all(axis=1) & np.isfinite(data).all(axis=1)]
            if data.empty or data.shape[1] < 2:
                continue
            logs = np.log10(data.to_numpy(np.float64))
            mean = logs.mean(axis=1)
            half = stats.t.ppf(0.975, logs.shape[1] - 1) * logs.std(axis=1, ddof=1) / np.sqrt(logs.shape[1])
            axis.plot(data.index, 10**mean, color=color, label=label, linewidth=1.5)
            axis.fill_between(data.index, 10 ** (mean - half), 10 ** (mean + half), color=color, alpha=0.15)
        title = f"{environment.removesuffix('-outcome')} {kind.replace('_', '-')} {stage} ({exact_char})"
        axis.set_title(title, fontsize=9)
        axis.set_yscale("log")
        axis.set_xlabel("global step" if stage == "outcome" and kind == "on_policy" else "logged step", fontsize=8)
        axis.set_ylabel("exact loss", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
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
