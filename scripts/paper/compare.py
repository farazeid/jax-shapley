"""Compare regenerated JAX figures and tables with the published FastSVERL results.

Applies `docs/data/paper--criteria.md` unchanged:

- curves: at 25% and 100% of each published series' x range, the JAX mean passes when
  it is within a factor of 2 of the published mean, or within 3 combined standard errors
  (`se_published` is the half-width of the published band), or both are below 1e-6;
- Hypercube bars: the same with a factor of 1.5;
- tables: a factor of 2 or 3 combined standard errors, for updates and final loss;
- claims (`paper--claims.csv`): each paper finding, evaluated on the JAX results alone.

Every figure, including the port figures `envs-port--throughput.svg` and
`explanations-port--exact-loss.svg`, uses one figure language: a fixed colour per concept,
JAX solid with filled markers, FastSVERL dashed with open markers, 95% t intervals.

Inputs come from `figures.py` (`paper--jax-curves.csv`, `paper--jax-bars.csv`) and
`tables.py` (`paper--jax-table_*.csv`, `paper--jax-table_2-nullity.csv`). Writes
`paper--comparison.csv`, `paper--claims-results.csv` and overlay plots under
`docs/figures/paper/`.

Usage: uv run --group benchmark python scripts/paper/compare.py [--data docs/data]
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from math import log10
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import tyro
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
from scipy import stats

type FloatArray = npt.NDArray[np.float64]
type Record = dict[str, str]

CURVE_RATIO = log10(2)
BAR_RATIO = log10(1.5)
TABLE_RATIO = log10(2)
Z_LIMIT = 3.0
ZERO = 1e-6
CHECKPOINTS = (0.25, 1.0)
TABLES = ("table_1_behaviour", "table_3_prediction", "table_4_outcome_offpolicy")

# The shared figure language (Okabe-Ito palette): one colour per concept, marker shape for on- (circle) or
# off-policy (triangle) data; JAX solid with filled markers, FastSVERL dashed with open markers.
BLACK, ORANGE, SKY, GREEN = "#000000", "#E69F00", "#56B4E9", "#009E73"
YELLOW, BLUE, VERMILLION, PURPLE = "#F0E442", "#0072B2", "#D55E00", "#CC79A7"
MUTED = "#555555"
ON_POLICY, OFF_POLICY = "o", "^"
CONCEPTS: dict[str, tuple[str, str]] = {
    "Exact characteristic": (BLUE, ON_POLICY),
    "Approx characteristic": (ORANGE, ON_POLICY),
    "On-policy, exact behaviour": (BLUE, ON_POLICY),
    "On-policy, approx behaviour": (ORANGE, ON_POLICY),
    "Off-policy, exact behaviour": (BLUE, OFF_POLICY),
    "Off-policy, approx behaviour": (ORANGE, OFF_POLICY),
    "Approx outcome, exact behaviour": (BLUE, ON_POLICY),
    "Approx outcome, approx behaviour": (ORANGE, ON_POLICY),
    "Exact outcome, exact behaviour": (GREEN, ON_POLICY),
    "On-policy": (BLACK, ON_POLICY),
    "Off-policy (without IS)": (VERMILLION, OFF_POLICY),
    "Without IS": (VERMILLION, OFF_POLICY),
    "Off-policy (with IS)": (SKY, OFF_POLICY),
    "Normalised IS": (SKY, OFF_POLICY),
    "With IS (clip = 0.99)": ("#A6D9F5", OFF_POLICY),
    "With IS (clip = 0.995)": ("#2B8CC4", OFF_POLICY),
    "With IS (clip = 0.998)": ("#1A5A85", OFF_POLICY),
    "With IS (no clipping)": (SKY, OFF_POLICY),
    "Unnormalised IS": (PURPLE, OFF_POLICY),
    "Exact": (BLUE, ON_POLICY),
    "Model": (ORANGE, ON_POLICY),
    "Sample": (PURPLE, ON_POLICY),
    "1:1": (YELLOW, ON_POLICY),
    "2:1": ("#C9B800", ON_POLICY),
    "10:1": ("#8A7F00", ON_POLICY),
    "50:1": ("#4D4700", ON_POLICY),
    "Mastermind-443": ("#B0B0B0", ON_POLICY),
    "Mastermind-453": ("#707070", ON_POLICY),
    "Mastermind-463": ("#303030", ON_POLICY),
}
# Upstream draws single-series figures without a label; the extracted series is named by its colour.
SINGLE_SERIES = "#1f77b4"
SINGLE_SERIES_LABELS = {
    "_policy_char": "Behaviour characteristic",
    "_value_char": "Prediction characteristic",
    "_return": "Episodic return",
    "_dqn_loss": "DQN loss",
}
KINDS = {"policy": "behaviour", "value": "prediction", "perf": "outcome"}
FIGURE_ENVIRONMENTS = {
    "mastermind_222": "Mastermind-222",
    "mastermind_333": "Mastermind-333",
    "gwb": "Gridworld",
    "large": "Mastermind-443/453/463",
}
TITLE_PATTERNS = (
    (r"(policy|value|perf)_char", "{} · characteristic"),
    (r"(policy|value|perf)_shapley", "{} · Shapley"),
    (r"(policy|value)_sampling", "{} · Shapley on a sampled characteristic"),
    (r"perf_sampling", "outcome · characteristic on a sampled behaviour"),
    (r"offpolicy_(policy|value)", "{} · characteristic from off-policy data"),
    (r"offpolicy_clipping_(policy|value)", "{} · off-policy characteristic, IS clipping"),
    (r"offpolicy_weighting_(policy|value)", "{} · off-policy characteristic, IS weighting"),
    (r"parallel_(policy|value)_approx", "{} · continual Shapley"),
    (r"parallel_perf_approx_behaviour", "outcome · continual Shapley"),
    (r"parallel_return", "continual · agent return"),
    (r"parallel_dqn_loss", "continual · agent DQN loss"),
    (r"perf_offpolicy_shapley", "outcome · off-policy Shapley"),
)
# Upstream's plot() shades mean +/- one standard error (fill="stderr") over its 20 runs.
PUBLISHED_T = float(stats.t.ppf(0.975, 19))
INTERVAL_NOTE = "Shaded: 95% t interval of the mean over runs (published: the source's ±1 SE band \N{MULTIPLICATION SIGN} t₁₉, 20 runs)."
OUTCOME_NOTE = (
    "Approx behaviour: scored against the outcome under the exact behaviour, so the error includes the "
    "behaviour model's. Exact behaviour: scored under the learnt behaviour, isolating the outcome model's error."
)
SAMPLING_NOTE = "Model starts after its characteristic's training (offset by the characteristic's updates)."
THROUGHPUT_PANELS = (
    ("gridworld", "Gridworld"),
    ("hypercube_2x2", "Hypercube 2\N{MULTIPLICATION SIGN}2"),
    ("hypercube_6x5", "Hypercube 6\N{MULTIPLICATION SIGN}5"),
    ("mastermind_222", "Mastermind-222"),
)
ENVIRONMENTS = {"gridworld": "Gridworld", "mastermind222": "Mastermind-222"}
# Shapley runs without with_exact_char are upstream's "Exact characteristic" series.
EXACT_LOSS_CONCEPTS = {
    ("characteristic", ""): ("characteristic", BLACK),
    ("shapley", "without"): ("Shapley, exact characteristic", BLUE),
    ("shapley", "with"): ("Shapley, approx characteristic", ORANGE),
}


@dataclass(frozen=True, slots=True)
class Args:
    data: Path = Path("docs/data")
    figures: Path = Path("docs/figures/paper")
    port_figures: Path = Path("docs/figures")


@dataclass(frozen=True, slots=True)
class Platform:
    """One throughput measurement file and how its line is drawn."""

    file: str
    label: str
    colour: str
    jax: bool


@dataclass(frozen=True, slots=True)
class Band:
    """A published series: mean line and the half-width of its standard-error band."""

    x: FloatArray
    mean: FloatArray
    half_width: FloatArray


@dataclass(frozen=True, slots=True)
class Curve:
    """A regenerated JAX series."""

    x: FloatArray
    mean: FloatArray
    standard_error: FloatArray
    runs: int


@dataclass(frozen=True, slots=True)
class Comparison:
    target: str
    series: str
    quantity: str
    published: float
    published_se: float
    jax: float
    jax_se: float
    ratio: float
    z: float
    passed: bool


@dataclass(frozen=True, slots=True)
class Figures:
    published: dict[tuple[str, str], Band]
    jax: dict[tuple[str, str], Curve]


# Throughput platforms; the Gymnasium source and JAX CPU share one Intel i9-14900K.
PLATFORMS = (
    Platform("gymnasium-cpu", "Gymnasium, CPU", BLACK, jax=False),
    Platform("jax-cpu", "JAX, CPU", BLUE, jax=True),
    Platform("jax-colab-t4", "JAX, T4 GPU", GREEN, jax=True),
    Platform("jax-colab-tpu-v5e", "JAX, TPU v5e-1", VERMILLION, jax=True),
    Platform("jax-gpu", "JAX, RTX 4080 SUPER", PURPLE, jax=True),
)


def main(args: Args) -> None:
    figures = Figures(published_bands(args.data / "paper--published-curves.csv"), jax_curves(args.data))
    comparisons = compare_curves(figures) + compare_bars(args.data) + compare_tables(args.data)
    _write(args.data / "paper--comparison.csv", comparisons)
    claims = evaluate_claims(args.data, figures)
    with (args.data / "paper--claims-results.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("claim", "check", "values", "result"))
        writer.writerows(claims)
    args.figures.mkdir(parents=True, exist_ok=True)
    # Every published figure: JAX against published where rerun, published alone (labelled) where not.
    for figure in sorted({figure for figure, _ in figures.published} | {figure for figure, _ in figures.jax}):
        overlay(figure, figures, args.figures)
    for figure in ("hypercube_policy", "hypercube_value"):
        bar_overlay(figure, args.data, args.figures)
    throughput_figure(args.data, args.port_figures / "envs-port--throughput.svg")
    exact_loss_figure(args.data, args.port_figures / "explanations-port--exact-loss.svg")
    passed = sum(comparison.passed for comparison in comparisons)
    held = sum(row[3] == "pass" for row in claims)
    print(f"{passed}/{len(comparisons)} quantities pass; {held}/{len(claims)} claims hold")


def compare_curves(figures: Figures) -> list[Comparison]:
    comparisons = []
    for (figure, series), curve in sorted(figures.jax.items()):
        band = figures.published.get((figure, series))
        if band is None:
            continue
        low, high = float(band.x.min()), float(band.x.max())
        for fraction in CHECKPOINTS:
            point = low + fraction * (high - low)
            comparisons.append(
                _compare(
                    figure,
                    series,
                    f"mean at {fraction:.0%} of x",
                    float(np.interp(point, band.x, band.mean)),
                    float(np.interp(point, band.x, band.half_width)),
                    float(np.interp(point, curve.x, curve.mean)),
                    float(np.interp(point, curve.x, curve.standard_error)),
                    CURVE_RATIO,
                )
            )
    return comparisons


def compare_bars(data: Path) -> list[Comparison]:
    published = {
        (row["figure"], row["model"], int(row["states"])): row for row in _records(data / "paper--published-bars.csv")
    }
    comparisons = []
    for row in _records(data / "paper--jax-bars.csv"):
        model = "Characteristic" if row["model"] == "Characteristic" else "Shapley"
        target = published.get((row["figure"], model, int(row["states"])))
        if target is None:
            continue
        comparisons.append(
            _compare(
                row["figure"],
                f"{model} {row['side']}x{row['dimension']}",
                "updates to target loss",
                float(target["mean"]),
                float(target["standard_error"]),
                float(row["mean"]),
                float(row["standard_error"]),
                BAR_RATIO,
            )
        )
    return comparisons


def compare_tables(data: Path) -> list[Comparison]:
    published = {
        (row["table"], row["domain"], row["model"]): row for row in _records(data / "paper--published-tables.csv")
    }
    comparisons = []
    for table in TABLES:
        for row in _records(data / f"paper--jax-{table}.csv"):
            target = published.get((table, row["domain"], row["model"]))
            if target is None:
                continue
            comparisons.extend(
                _compare(
                    table,
                    f"{row['domain']} {row['model']}",
                    quantity,
                    float(target[f"{quantity}_mean"]),
                    float(target[f"{quantity}_se"]),
                    float(row[f"{quantity}_mean"]),
                    float(row[f"{quantity}_se"]),
                    TABLE_RATIO,
                )
                for quantity in ("updates", "loss")
            )
    return comparisons


def evaluate_claims(data: Path, figures: Figures) -> list[tuple[str, str, str, str]]:
    """Each claim's relation on the JAX results; 'missing' when its results do not exist yet."""
    results = []
    for claim in _records(data / "paper--claims.csv"):
        try:
            values, passed = _claim(claim, data, figures)
            results.append((claim["claim"], claim["check"], values, "pass" if passed else "fail"))
        except LookupError as error:
            results.append((claim["claim"], claim["check"], str(error), "missing"))
    return results


def published_bands(path: Path) -> dict[tuple[str, str], Band]:
    points: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
    for row in _records(path):
        points.setdefault((row["figure"], row["series"], row["kind"]), []).append((float(row["x"]), float(row["y"])))
    bands = {}
    for (figure, series, kind), mean in points.items():
        low, high = points.get((figure, series, "low")), points.get((figure, series, "high"))
        if kind != "mean" or low is None or high is None:
            continue
        x, y = _sorted(mean)
        low_x, low_y = _sorted(low)
        high_x, high_y = _sorted(high)
        half_width = np.abs(np.interp(x, high_x, high_y) - np.interp(x, low_x, low_y)) / 2
        bands[figure, series] = Band(x, y, half_width)
    return bands


def jax_curves(data: Path) -> dict[tuple[str, str], Curve]:
    points: dict[tuple[str, str], list[tuple[float, float, float, float]]] = {}
    for row in _records(data / "paper--jax-curves.csv"):
        point = (float(row["x"]), float(row["mean"]), float(row["standard_error"]), float(row["runs"]))
        points.setdefault((row["figure"], row["series"]), []).append(point)
    curves = {}
    for key, rows in points.items():
        ordered = np.asarray(sorted(rows), np.float64)
        curves[key] = Curve(ordered[:, 0], ordered[:, 1], ordered[:, 2], int(ordered[:, 3].min()))
    return curves


def overlay(figure: str, figures: Figures, folder: Path) -> None:
    """One figure in the shared style: JAX solid with filled markers, published dashed with open markers,
    one colour per concept and 95% t intervals. Series without a JAX rerun are labelled "not rerun"."""
    handle, axis = plt.subplots(figsize=(6, 4.4))
    names = {name for name_figure, name in figures.jax if name_figure == figure}
    names |= {name for name_figure, name in figures.published if name_figure == figure}
    values, concepts = [], []
    for name in sorted(names, key=_concept_order):
        label, colour, marker = series_style(figure, name)
        curve = figures.jax.get((figure, name))
        if curve is not None:
            values.append(curve.mean)
            half = stats.t.ppf(0.975, curve.runs - 1) * curve.standard_error if curve.runs > 1 else curve.standard_error
            _draw(axis, curve.x, curve.mean, half, colour, marker, jax=True)
        band = figures.published.get((figure, name))
        if band is not None:
            values.append(band.mean)
            _draw(axis, band.x, band.mean, PUBLISHED_T * band.half_width, colour, marker, jax=False)
        concepts.append(
            _handle(label + ("" if curve is not None else " (not rerun)"), colour, marker, curve is not None)
        )
    means = np.concatenate(values)
    if np.all(means > 0) and means.max() / means.min() > 100:
        axis.set_yscale("log")
    rerun = any(name_figure == figure for name_figure, _ in figures.jax)
    axis.set_title(figure_title(figure) + ("" if rerun else " (not rerun)"), fontsize=10)
    axis.set_xlabel(x_label(figure))
    axis.set_ylabel(y_label(figure))
    axis.legend(handles=[*concepts, *_sources()], fontsize=6, frameon=False)
    notes = [INTERVAL_NOTE, *figure_notes(figure)]
    handle.text(0.5, 0.005, "\n".join(notes), ha="center", va="bottom", fontsize=6, color=MUTED, wrap=True)
    handle.tight_layout(rect=(0, 0.035 * len(notes), 1, 1))
    _save(handle, folder / f"{figure}.svg")


def bar_overlay(figure: str, data: Path, folder: Path) -> None:
    """Hypercube updates to the target loss against the number of states, one panel per dimension:
    published (open markers, dashed) and JAX (filled, solid) mean with 95% t intervals, per model;
    cubes not rerun show the published value alone."""
    published = [row for row in _records(data / "paper--published-bars.csv") if row["figure"] == figure]
    jax_path = data / "paper--jax-bars.csv"
    rerun = [row for row in _records(jax_path) if row["figure"] == figure] if jax_path.exists() else []
    dimensions = (2, 3, 4, 5)
    handle, axes = plt.subplots(1, len(dimensions), figsize=(12, 3.8), sharey=True)
    for axis, dimension in zip(axes, dimensions, strict=True):
        # Published panel titles name their dimension first ("Dimension = 3 | ...").
        panel = [row for row in published if row["title"].startswith(f"Dimension = {dimension}")]
        for model, colour in (("Characteristic", BLUE), ("Shapley", ORANGE)):
            points = sorted(
                (int(row["states"]), float(row["mean"]), PUBLISHED_T * float(row["standard_error"]))
                for row in panel
                if row["model"] == model
            )
            if points:
                states, mean, error = (np.asarray(column) for column in zip(*points, strict=True))
                axis.errorbar(
                    states,
                    mean,
                    yerr=error,
                    color=colour,
                    marker="o",
                    markersize=4,
                    markerfacecolor="white",
                    linestyle="--",
                )
            ours = sorted(
                (
                    int(row["states"]),
                    float(row["mean"]),
                    stats.t.ppf(0.975, max(int(row["runs"]) - 1, 1)) * float(row["standard_error"]),
                )
                for row in rerun
                if int(row["dimension"]) == dimension and row["model"] == model
            )
            if ours:
                states, mean, error = (np.asarray(column) for column in zip(*ours, strict=True))
                axis.errorbar(states, mean, yerr=error, color=colour, marker="o", markersize=4, linestyle="-")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_xlabel("Number of states")
        rerun_dimension = any(int(row["dimension"]) == dimension for row in rerun)
        axis.set_title(f"Dimension {dimension}" + ("" if rerun_dimension else " (not rerun)"), fontsize=10)
    axes[0].set_ylabel("Updates to target loss")
    models = [_handle("Characteristic", BLUE, "o", True), _handle("Shapley", ORANGE, "o", True)]
    axes[-1].legend(handles=[*models, *_sources()], fontsize=6, frameon=False)
    handle.suptitle(figure_title(figure))
    note = "Error bars: 95% t interval of the mean over runs (published: the source's ±1 SE \N{MULTIPLICATION SIGN} t₁₉, 20 runs)."
    handle.text(0.5, 0.01, note, ha="center", va="bottom", fontsize=7, color=MUTED)
    handle.tight_layout(rect=(0, 0.04, 1, 1))
    _save(handle, folder / f"{figure}.svg")


def throughput_figure(data: Path, path: Path) -> None:
    """Environment transitions per second against the number of parallel environments, per platform:
    mean over samples with a 95% t interval; Gymnasium is the FastSVERL source (dashed, open markers)."""
    samples: dict[tuple[str, str, int], list[float]] = {}
    for platform in PLATFORMS:
        for row in _records(data / f"envs-port--throughput--{platform.file}.csv"):
            if row["status"] == "ok":
                key = (row["environment"], platform.label, int(row["num_envs"]))
                samples.setdefault(key, []).append(float(row["steps_per_second"]))
    handle, axes = plt.subplots(2, 2, figsize=(11, 8.4), sharex=True)
    counts = set()
    for axis, (environment, title) in zip(axes.flat, THROUGHPUT_PANELS, strict=True):
        for platform in PLATFORMS:
            points = sorted(
                (envs, np.asarray(values, np.float64))
                for (name, label, envs), values in samples.items()
                if name == environment and label == platform.label
            )
            if not points:
                continue
            counts |= {values.size for _, values in points}
            x = np.asarray([envs for envs, _ in points], np.float64)
            mean = np.asarray([values.mean() for _, values in points])
            half = np.asarray([
                stats.t.ppf(0.975, values.size - 1) * values.std(ddof=1) / np.sqrt(values.size) for _, values in points
            ])
            _draw(axis, x, mean, half, platform.colour, "o", jax=platform.jax)
        axis.set_title(title, fontsize=10)
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks([1, 4, 16, 64, 256, 1024, 2048], ["1", "4", "16", "64", "256", "1024", "2048"])
        axis.set_ylabel("Transitions per second")
        if environment == "hypercube_6x5":
            axis.axvline(16, color=MUTED, linestyle=":", linewidth=1)
            axis.text(19, axis.get_ylim()[0] * 1.5, "Gymnasium fits at most 16", fontsize=7, color=MUTED)
    for axis in axes[-1]:
        axis.set_xlabel("Parallel environments")
    platforms = [_handle(platform.label, platform.colour, "o", platform.jax) for platform in PLATFORMS]
    handle.suptitle("Environment throughput: Gymnasium (FastSVERL) against JAX", y=0.99)
    handle.legend(
        handles=platforms,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=len(PLATFORMS),
        frameon=False,
        fontsize=8,
    )
    note = (
        f"Shaded: 95% t interval of the mean over {'/'.join(str(count) for count in sorted(counts))} samples. "
        "TPU uses JAX's default reduced-precision float32 matmuls. T4 and TPU v5e-1 are Google Colab runtimes."
    )
    handle.text(0.5, 0.005, note, ha="center", va="bottom", fontsize=7, color=MUTED)
    handle.tight_layout(rect=(0, 0.02, 1, 0.92))
    _save(handle, path)


def exact_loss_figure(data: Path, path: Path) -> None:
    """Gridworld and Mastermind-222 exact loss per epoch, PyTorch (dashed, open) against JAX (solid, filled):
    geometric mean over seeds with a 95% t interval of the log mean, one colour per concept."""
    runs = _records(data / "explanations-port--upstream-runs.csv") + _records(data / "explanations-port--jax-runs.csv")
    losses: dict[tuple[str, str, str, str, str], dict[str, dict[int, float]]] = {}
    for row in runs:
        key = (row["framework"], row["environment"], row["kind"], row["stage"], row["exact_char"])
        losses.setdefault(key, {}).setdefault(row["seed"], {})[int(row["step"])] = float(row["exact_loss"])
    handle, axes = plt.subplots(4, 3, figsize=(13, 13.5))
    seeds = set()
    panels = [
        (environment, kind, stage, exact_char)
        for environment in ("gridworld", "mastermind222")
        for kind in ("behaviour", "prediction")
        for stage, exact_char in (("characteristic", ""), ("shapley", "without"), ("shapley", "with"))
    ]
    for axis, (environment, kind, stage, exact_char) in zip(axes.flat, panels, strict=True):
        concept, colour = EXACT_LOSS_CONCEPTS[stage, exact_char]
        for framework in ("upstream", "jax"):
            by_seed = losses.get((framework, environment, kind, stage, exact_char), {})
            if not by_seed:
                continue
            steps = sorted(set.intersection(*(set(values) for values in by_seed.values())))
            logs = np.log10([[values[step] for values in by_seed.values()] for step in steps])
            seeds.add(logs.shape[1])
            mean = logs.mean(axis=1)
            half = stats.t.ppf(0.975, logs.shape[1] - 1) * logs.std(axis=1, ddof=1) / np.sqrt(logs.shape[1])
            x = np.asarray(steps, np.float64)
            line = "-" if framework == "jax" else "--"
            face = colour if framework == "jax" else "white"
            axis.plot(
                x,
                10**mean,
                color=colour,
                linestyle=line,
                marker="o",
                markerfacecolor=face,
                markevery=_every(x),
                markersize=4,
                linewidth=1.4,
            )
            axis.fill_between(x, 10 ** (mean - half), 10 ** (mean + half), color=colour, alpha=0.15, linewidth=0)
        title = f"{ENVIRONMENTS[environment]} · {kind} · {concept}"
        axis.set_title(title, fontsize=9)
        axis.set_yscale("log")
        axis.set_xlabel("Epoch", fontsize=8)
        axis.set_ylabel("Exact loss (MSE)", fontsize=8)
    sources = [_handle("JAX", MUTED, "o", True), _handle("FastSVERL (PyTorch)", MUTED, "o", False)]
    handle.suptitle("Exact loss per epoch: FastSVERL (PyTorch) against JAX, same configurations", y=0.99)
    handle.legend(handles=sources, loc="upper center", bbox_to_anchor=(0.5, 0.97), ncol=2, frameon=False)
    note = (
        f"Geometric mean over {'/'.join(str(count) for count in sorted(seeds))} seeds; "
        "shaded: 95% t interval of the log mean."
    )
    handle.text(0.5, 0.005, note, ha="center", va="bottom", fontsize=8, color=MUTED)
    handle.tight_layout(rect=(0, 0.015, 1, 0.95))
    _save(handle, path)


def series_style(figure: str, name: str) -> tuple[str, str, str]:
    """A series' legend name, colour and marker; unlabelled single-series figures take their concept's name."""
    if name == SINGLE_SERIES:
        for suffix, label in SINGLE_SERIES_LABELS.items():
            if figure.endswith(suffix):
                return label, BLACK, ON_POLICY
        raise ValueError(f"no concept for the single series of {figure}")
    if name not in CONCEPTS:
        raise ValueError(f"no style for series {name!r} of {figure}")
    colour, marker = CONCEPTS[name]
    return name, colour, marker


def figure_title(figure: str) -> str:
    """'Environment · explanation kind · model' from the source's figure name."""
    if figure.startswith("hypercube_"):
        return f"Hypercube · {KINDS[figure.removeprefix('hypercube_')]} · updates to target loss"
    parts = figure.split("_", 1)[1]
    for key, environment in FIGURE_ENVIRONMENTS.items():
        if not parts.startswith(f"{key}_"):
            continue
        for pattern, template in TITLE_PATTERNS:
            if match := re.fullmatch(pattern, parts.removeprefix(f"{key}_")):
                kinds: list[str] = [KINDS[group] for group in match.groups()]
                title: str = template.format(*kinds)
                return f"{environment} · {title}"
    raise ValueError(f"no title for {figure}")


def x_label(figure: str) -> str:
    return "Environment steps (\N{MULTIPLICATION SIGN}0.1)" if "_parallel_" in figure else "Training updates"


def y_label(figure: str) -> str:
    if figure.endswith("_return"):
        return "Episodic return"
    if figure.endswith("_dqn_loss"):
        return "DQN loss"
    if figure.startswith("app_large_"):
        return "Training loss (MSE)"
    return "Exact loss (MSE)"


def figure_notes(figure: str) -> list[str]:
    notes = []
    if figure.endswith(("_perf_char", "_perf_shapley")):
        notes.append(OUTCOME_NOTE)
    if figure.endswith("_sampling"):
        notes.append(SAMPLING_NOTE)
    return notes


def _draw(
    axis: Axes, x: FloatArray, mean: FloatArray, half: FloatArray, colour: str, marker: str, *, jax: bool
) -> None:
    face = colour if jax else "white"
    line = "-" if jax else "--"
    axis.plot(
        x,
        mean,
        color=colour,
        linestyle=line,
        marker=marker,
        markerfacecolor=face,
        markevery=_every(x),
        markersize=4,
        linewidth=1.4,
    )
    axis.fill_between(x, mean - half, mean + half, color=colour, alpha=0.2 if jax else 0.1, linewidth=0)


def _every(x: FloatArray) -> int:
    """About eight markers per line, whatever its sampling."""
    return max(1, x.size // 8)


def _handle(label: str, colour: str, marker: str, jax: bool) -> Line2D:
    return Line2D(
        [],
        [],
        color=colour,
        linestyle="-" if jax else "--",
        marker=marker,
        markerfacecolor=colour if jax else "white",
        markersize=4,
        label=label,
    )


def _concept_order(name: str) -> int:
    return list(CONCEPTS).index(name) if name in CONCEPTS else -1


def _save(handle: Figure, path: Path) -> None:
    handle.savefig(path)
    plt.close(handle)
    lines = path.read_text().splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n")


def _sources() -> list[Line2D]:
    return [_handle("JAX", MUTED, "o", True), _handle("FastSVERL (published)", MUTED, "o", False)]


def _compare(
    target: str,
    series: str,
    quantity: str,
    published: float,
    published_se: float,
    jax: float,
    jax_se: float,
    ratio_limit: float,
) -> Comparison:
    ratio = abs(log10(jax / published)) if jax > 0 and published > 0 else float("inf")
    spread = (published_se**2 + jax_se**2) ** 0.5
    z = abs(jax - published) / spread if spread > 0 else float("inf")
    passed = ratio <= ratio_limit or z <= Z_LIMIT or (abs(jax) < ZERO and abs(published) < ZERO)
    return Comparison(target, series, quantity, published, published_se, jax, jax_se, ratio, z, passed)


def _claim(claim: Record, data: Path, figures: Figures) -> tuple[str, bool]:
    check, figure, name = claim["check"], claim["figure"], claim["claim"]
    if name.startswith("tables_"):
        return _table_claim(name, data)
    if name == "table2_nullity":
        rows = _records(data / "paper--jax-table_2-nullity.csv")
        if not rows:
            raise LookupError("no Table 2 results")
        empty, filled = float(rows[0]["mean_abs_empty_rows"]), float(rows[0]["mean_abs_filled_rows"])
        return f"empty {empty:.4g}, filled {filled:.4g}", empty <= 0.25 * filled
    if check.startswith("spearman"):
        return _scaling_claim(figure, data)
    if match := re.fullmatch(r"ratio\(final\((.+)\) / final\((.+)\)\) within \[([\d.]+); ([\d.]+)\]", check):
        first = _statistic(figures, figure, "final", match.group(1))
        second = _statistic(figures, figure, "final", match.group(2))
        ratio = first / second
        return f"{ratio:.4g}", float(match.group(3)) <= ratio <= float(match.group(4))
    if match := re.fullmatch(r"(final|max)\((.+)\) ([<>]) (final|max)\((.+)\)", check):
        left = _statistic(figures, figure, match.group(1), match.group(2))
        right = _statistic(figures, figure, match.group(4), match.group(5))
        holds = left < right if match.group(3) == "<" else left > right
        return f"{left:.4g} vs {right:.4g}", holds
    raise ValueError(f"unsupported claim check: {check}")


def _statistic(figures: Figures, figure: str, statistic: str, label: str) -> float:
    """'final': the JAX mean at the end of the published x range; 'max': its maximum over that range."""
    label = label.replace(";", ",")  # paper--claims.csv writes label commas as semicolons.
    curve = figures.jax.get((figure, label))
    if curve is None:
        raise LookupError(f"no JAX series {figure} / {label}")
    band = figures.published.get((figure, label))
    high = float(band.x.max()) if band is not None else float(curve.x.max())
    if statistic == "final":
        return float(np.interp(high, curve.x, curve.mean))
    return float(curve.mean[curve.x <= high].max())


def _scaling_claim(figure: str, data: Path) -> tuple[str, bool]:
    by_dimension: dict[int, list[tuple[int, float]]] = {}
    for row in _records(data / "paper--jax-bars.csv"):
        if row["figure"] == figure and row["model"] == "Characteristic":
            by_dimension.setdefault(int(row["dimension"]), []).append((int(row["states"]), float(row["mean"])))
    if len(by_dimension) < 4 or any(len(cubes) < 5 for cubes in by_dimension.values()):
        raise LookupError("incomplete Hypercube grid")
    correlations = []
    for _, cubes in sorted(by_dimension.items()):
        states = [state for state, _ in cubes]
        updates = [update for _, update in cubes]
        correlations.append(float(stats.spearmanr(states, updates).statistic))
    return ", ".join(f"{value:.2f}" for value in correlations), all(value >= 0.8 for value in correlations)


def _table_claim(claim: str, data: Path) -> tuple[str, bool]:
    rows: list[Record] = []
    for table in TABLES:
        records = _records(data / f"paper--jax-{table}.csv")
        if not records:
            raise LookupError(f"no {table}")
        rows.extend(records)
    if claim == "tables_converge":
        losses = np.asarray([float(row["loss_mean"]) for row in rows], np.float64)
        return f"{losses.min():.3g}-{losses.max():.3g}", bool(np.all(np.isfinite(losses) & (losses > 0)))
    groups: dict[str, list[float]] = {}
    for row in rows:
        groups.setdefault(f"{row['table']} {row['model']}", []).append(float(row["updates_mean"]))
    spreads = {name: max(updates) / min(updates) for name, updates in groups.items()}
    summary = ", ".join(f"{name} {value:.2f}" for name, value in spreads.items())
    return summary, all(value <= 3 for value in spreads.values())


def _records(path: Path) -> list[Record]:
    if not path.exists():
        return []
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def _sorted(points: list[tuple[float, float]]) -> tuple[FloatArray, FloatArray]:
    ordered = np.asarray(sorted(points), np.float64)
    return ordered[:, 0], ordered[:, 1]


def _write(path: Path, comparisons: list[Comparison]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow((
            "target",
            "series",
            "quantity",
            "published",
            "published_se",
            "jax",
            "jax_se",
            "log10_ratio",
            "z",
            "passed",
        ))
        writer.writerows(
            (
                item.target,
                item.series,
                item.quantity,
                f"{item.published:.6g}",
                f"{item.published_se:.6g}",
                f"{item.jax:.6g}",
                f"{item.jax_se:.6g}",
                f"{item.ratio:.4g}",
                f"{item.z:.4g}",
                item.passed,
            )
            for item in comparisons
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
