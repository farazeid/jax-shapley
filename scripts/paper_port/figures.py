"""Regenerate the paper's figures from JAX runs with the source's plot statistics.

Each figure spec follows its upstream plotting script (`experiments/**/plots/plot_*.py`
at FastSVERL 25b2cbe): the logged tag, the groups and labels, `start_epoch` offsets,
`n_batches` x scaling and `sem_correction`. `curve` ports `fastsverl.utils.plot`:
runs are merged on the logging step with `merge_asof(direction="nearest")`, linearly
interpolated, then summarised by the mean and standard error over runs. Hypercube bars
port `barcharts`: each run's last logged exact-loss step, mean and standard error.

Sources: the milestone-2 sweeps (`docs/data/explanations-port--jax-runs.csv`) for the
Mastermind-222 and Gridworld accuracy figures, and run folders under `runs/paper` for
the rest (written by `runs/paper/run_*.sh`). Figures whose runs do not exist yet are
listed as missing.

Usage: uv run --group benchmark python scripts/paper_port/figures.py [--paper runs/paper] [--output docs/data]
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import tyro

type FloatArray = npt.NDArray[np.float64]
type Run = tuple[FloatArray, FloatArray]
type Loader = Callable[[Sources], list[Run]]

# Batches per epoch in the source's plot scripts: (train fraction x buffer) // batch size.
BATCHES = (0.8 * 10_000) // 128
OFFPOLICY_MASTERMIND_BATCHES = (0.8 * 30_000) // 128
OFFPOLICY_GRIDWORLD_BATCHES = (0.8 * 15_000) // 128
# parallel/plots: "Scaling because models updated every 10 env steps."
PARALLEL_SCALE = 0.1
CUBES = tuple((side, dimension) for dimension in (2, 3, 4, 5) for side in (2, 3, 4, 5, 6))


@dataclass(frozen=True, slots=True)
class Args:
    paper: Path = Path("runs/paper")
    milestone: Path = Path("docs/data/explanations-port--jax-runs.csv")
    output: Path = Path("docs/data")


@dataclass(frozen=True, slots=True)
class Series:
    label: str
    runs: Loader
    start: float = 0.0
    """The source's `start_epoch`, added to every logging step before scaling."""


@dataclass(frozen=True, slots=True)
class FigureSpec:
    name: str
    series: tuple[Series, ...]
    n_batches: float = 1.0
    sem_correction: bool = False


@dataclass(frozen=True, slots=True)
class Curve:
    x: FloatArray
    mean: FloatArray
    standard_error: FloatArray
    runs: int


@dataclass
class Sources:
    """Run loaders; every run is (logging steps, values) with missing values dropped."""

    paper: Path
    milestone: Path
    _milestone: pd.DataFrame | None = field(default=None, repr=False)

    def milestone_runs(self, environment: str, kind: str, stage: str, exact_char: str) -> list[Run]:
        if self._milestone is None:
            self._milestone = pd.read_csv(self.milestone, keep_default_na=False)
        frame = self._milestone
        selected = frame[
            (frame["framework"] == "jax")
            & (frame["environment"] == environment)
            & (frame["kind"] == kind)
            & (frame["stage"] == stage)
            & (frame["exact_char"] == exact_char)
        ]
        runs = []
        for _, rows in selected.groupby("seed"):
            values = pd.to_numeric(rows["exact_loss"], errors="coerce")
            keep = values.notna()
            runs.append((_floats(rows["step"][keep]), _floats(values[keep])))
        return _required(runs, f"{environment} {kind} {stage} {exact_char}")

    def sweep(self, folder: Path, column: str, **settings: object) -> list[Run]:
        """Seed folders of the one sweep under `folder` whose saved settings match `settings`."""
        sweeps = [sweep for sweep in sorted(folder.glob("*")) if sweep.is_dir() and _matches(sweep, settings)]
        if len(sweeps) != 1:
            raise MissingRunsError(f"expected one sweep in {folder} matching {settings}, found {len(sweeps)}")
        seeds = sorted(sweeps[0].glob("seed-*")) or [sweeps[0]]
        if any(not (seed / "summary.json").exists() for seed in seeds):
            raise MissingRunsError(f"{sweeps[0]} has unfinished seeds")
        return _required([_metrics(seed / "metrics.csv", column) for seed in seeds], str(folder))

    def continual(self, domain: str, kind: str, ratio: int, column: str) -> list[Run]:
        seeds = self._continual_seeds(domain, kind, ratio)
        if column == "episode_return":
            return [_metrics(seed / "episodes.csv", column) for seed in seeds]
        return [_metrics(seed / "metrics.csv", column) for seed in seeds]

    def _continual_seeds(self, domain: str, kind: str, ratio: int) -> list[Path]:
        root = self.paper / "continual" / domain / kind
        for run in sorted(root.glob("*")):
            seeds = sorted(run.glob("seed-*"))
            if seeds:
                config = json.loads((seeds[0] / "config.json").read_text())["config"]
                if config["explain"]["update_ratio"] == ratio:
                    return seeds
        raise MissingRunsError(f"no {domain} {kind} continual run at ratio {ratio}")


class MissingRunsError(Exception):
    """The runs a figure needs do not exist (yet)."""


def main(args: Args) -> None:
    sources = Sources(args.paper, args.milestone)
    rows: list[tuple[str, str, float, float, float, int]] = []
    missing: list[str] = []
    for spec in figures():
        try:
            curves = figure_curves(spec, sources)
        except MissingRunsError as error:
            missing.append(f"{spec.name}: {error}")
            continue
        for label, curve in curves.items():
            rows.extend(
                (spec.name, label, float(x), float(mean), float(se), curve.runs)
                for x, mean, se in zip(curve.x, curve.mean, curve.standard_error, strict=True)
            )
    with (args.output / "paper--jax-curves.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("figure", "series", "x", "mean", "standard_error", "runs"))
        writer.writerows(_rounded(row) for row in rows)
    bars, missing_bars = hypercube_bars(args.paper)
    with (args.output / "paper--jax-bars.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("figure", "model", "side", "dimension", "states", "mean", "standard_error", "runs"))
        writer.writerows(bars)
    print(f"{len({row[0] for row in rows})} figures, {len(bars)} bars")
    for line in [*missing, *missing_bars]:
        print(f"missing {line}")


def figures() -> list[FigureSpec]:
    """Every published curve figure, in the source scripts' terms."""
    specs: list[FigureSpec] = []
    for domain, environment in (("mastermind_222", "mastermind222"), ("gwb", "gridworld")):
        for kind, name in (("behaviour", "policy"), ("prediction", "value")):
            specs.append(
                FigureSpec(
                    f"app_{domain}_{name}_char",
                    (Series("#1f77b4", _milestone(environment, kind, "characteristic", "")),),
                    BATCHES,
                )
            )
            specs.append(
                FigureSpec(
                    f"app_{domain}_{name}_shapley",
                    (
                        Series("Exact characteristic", _milestone(environment, kind, "shapley", "without")),
                        Series("Approx characteristic", _milestone(environment, kind, "shapley", "with")),
                    ),
                    BATCHES,
                )
            )
        outcome = f"{environment}-outcome"
        specs.append(
            FigureSpec(
                f"app_{domain}_perf_char",
                (
                    Series("On-policy, approx behaviour", _milestone(outcome, "on_policy", "outcome", "with")),
                    Series("On-policy, exact behaviour", _milestone(outcome, "on_policy", "outcome", "without")),
                    Series("Off-policy, approx behaviour", _milestone(outcome, "off_policy", "outcome", "with")),
                    Series("Off-policy, exact behaviour", _milestone(outcome, "off_policy", "outcome", "without")),
                ),
            )
        )
        variant = "off_policy" if environment == "mastermind222" else "on_policy"
        specs.append(
            FigureSpec(
                f"app_{domain}_perf_shapley",
                (
                    Series("Approx outcome, approx behaviour", _milestone(outcome, variant, "shapley", "w_policy")),
                    Series("Approx outcome, exact behaviour", _milestone(outcome, variant, "shapley", "wo_policy")),
                    Series("Exact outcome, exact behaviour", _milestone(outcome, variant, "shapley", "wo_perf")),
                ),
                BATCHES,
            )
        )
    specs += [
        _renamed(specs, "app_mastermind_222_policy_char", "main_mastermind_222_policy_char"),
        _renamed(specs, "app_mastermind_222_perf_char", "main_mastermind_222_perf_char"),
        _renamed(specs, "app_mastermind_222_perf_shapley", "main_mastermind_222_perf_shapley"),
    ]
    for kind, name in (("behaviour", "policy"), ("prediction", "value")):
        root = Path("mastermind-333") / kind
        specs.append(
            FigureSpec(
                f"app_mastermind_333_{name}_char",
                (Series("#1f77b4", _sweep(root / "characteristic", "exact_loss")),),
                BATCHES,
            )
        )
        specs.append(
            FigureSpec(
                f"app_mastermind_333_{name}_shapley",
                (
                    Series("Exact characteristic", _sweep(root / "shapley", "exact_loss", with_exact_char=False)),
                    Series("Approx characteristic", _sweep(root / "shapley", "exact_loss", with_exact_char=True)),
                ),
                BATCHES,
            )
        )
        specs.append(
            FigureSpec(
                f"app_large_{name}_char",
                tuple(
                    Series(
                        f"Mastermind-{size}", _sweep(Path(f"mastermind-{size}") / kind / "characteristic", "train_loss")
                    )
                    for size in ("443", "453", "463")
                ),
            )
        )
        specs.append(
            FigureSpec(
                f"app_large_{name}_shapley",
                tuple(
                    Series(f"Mastermind-{size}", _sweep(Path(f"mastermind-{size}") / kind / "shapley", "train_loss"))
                    for size in ("443", "453", "463")
                ),
                BATCHES,
            )
        )
    specs.append(
        FigureSpec(
            "app_large_perf_offpolicy_shapley",
            tuple(
                Series(f"Mastermind-{size}", _sweep(Path(f"mastermind-{size}") / "outcome" / "shapley", "train_loss"))
                for size in ("443", "453", "463")
            ),
            BATCHES,
        )
    )
    specs += offpolicy_figures() + continual_figures() + sampling_figures()
    return specs


def offpolicy_figures() -> list[FigureSpec]:
    """offpolicy/plots: every figure uses sem_correction."""
    specs = []
    for kind, name in (("behaviour", "policy"), ("prediction", "value")):
        root = Path("offpolicy/mastermind222") / kind
        groups = (
            ("Off-policy (without IS)", "wo_weight"),
            ("Off-policy (with IS)", "w_weight"),
            ("On-policy", "exact"),
        )
        series = tuple(Series(label, _sweep(root / group, "exact_loss")) for label, group in groups)
        specs.append(FigureSpec(f"app_mastermind_222_offpolicy_{name}", series, OFFPOLICY_MASTERMIND_BATCHES, True))
        root = Path("offpolicy/gridworld") / kind
        clipping = (
            ("Without IS", "wo_weight"),
            ("With IS (clip = 0.99)", "clip_0.99"),
            ("With IS (clip = 0.995)", "clip_0.995"),
            ("With IS (clip = 0.998)", "clip_0.998"),
            ("With IS (no clipping)", "normalised"),
            ("On-policy", "exact"),
        )
        weighting = (
            ("Unnormalised IS", "unnormalised"),
            ("Without IS", "wo_weight"),
            ("Normalised IS", "normalised"),
            ("On-policy", "exact"),
        )
        for figure, groups in (("clipping", clipping), ("weighting", weighting)):
            series = tuple(Series(label, _sweep(root / group, "exact_loss")) for label, group in groups)
            specs.append(FigureSpec(f"app_gwb_offpolicy_{figure}_{name}", series, OFFPOLICY_GRIDWORLD_BATCHES, True))
    specs.append(_renamed(specs, "app_mastermind_222_offpolicy_policy", "main_mastermind_222_offpolicy_policy"))
    return specs


def continual_figures() -> list[FigureSpec]:
    """parallel/plots: x is environment steps x 0.1; return and DQN loss come from one group."""
    specs = []
    for domain, environment in (("mastermind_222", "mastermind222"), ("gwb", "gridworld")):
        for kind, name in (("behaviour", "policy"), ("prediction", "value"), ("outcome", "perf")):
            figure = f"app_{domain}_parallel_{name}_approx" + ("_behaviour" if kind == "outcome" else "")
            series = tuple(
                Series(f"{ratio}:1", _continual(environment, kind, ratio, "shapley_exact_loss"))
                for ratio in (1, 2, 10, 50)
            )
            specs.append(FigureSpec(figure, series, PARALLEL_SCALE))
        for column, figure in (("episode_return", "return"), ("dqn_loss", "dqn_loss")):
            series = (Series("#1f77b4", _continual(environment, "behaviour", 1, column)),)
            specs.append(FigureSpec(f"app_{domain}_parallel_{figure}", series, PARALLEL_SCALE))
    specs.append(
        _renamed(specs, "app_mastermind_222_parallel_value_approx", "main_mastermind_222_parallel_value_approx")
    )
    return specs


def sampling_figures() -> list[FigureSpec]:
    """sampling/plots: the model group starts after its characteristic's epochs (start_epoch)."""
    specs = []
    offsets = {
        ("mastermind222", "behaviour"): 50,
        ("mastermind222", "prediction"): 50,  # The source's value figure also offsets by 50 (80 trained).
        ("mastermind333", "behaviour"): 1000,
        ("mastermind333", "prediction"): 2000,
        ("gridworld", "behaviour"): 50,
        ("gridworld", "prediction"): 30,
    }
    names = {"mastermind222": "mastermind_222", "mastermind333": "mastermind_333", "gridworld": "gwb"}
    for (domain, kind), offset in offsets.items():
        root = Path("sampling") / domain / kind / "shapley"
        series = (
            Series("Exact", _sweep(root / "wo_exact_model", "exact_loss")),
            Series("Sample", _sweep(root / "w_exact_sampling", "exact_loss")),
            Series("Model", _sweep(root / "w_exact_model", "exact_loss"), float(offset)),
        )
        name = "policy" if kind == "behaviour" else "value"
        specs.append(FigureSpec(f"app_{names[domain]}_{name}_sampling", series, BATCHES))
    for domain in ("mastermind222", "gridworld"):
        root = Path("sampling") / domain / "outcome"
        series = (
            Series("Exact", _sweep(root / "wo_exact_model", "exact_loss")),
            Series("Sample", _sweep(root / "w_exact_sampling", "exact_loss")),
            Series("Model", _sweep(root / "w_exact_model", "exact_loss"), 50 * (0.8 * 10_000) // 128),
        )
        specs.append(FigureSpec(f"app_{names[domain]}_perf_sampling", series))
    specs.append(_renamed(specs, "app_mastermind_222_policy_sampling", "main_mastermind_222_policy_sampling"))
    return specs


def figure_curves(spec: FigureSpec, sources: Sources) -> dict[str, Curve]:
    """Source plot(): per-group merged runs, optional SEM correction, then mean and standard error."""
    merged = [merge_runs(series.runs(sources), series.start) for series in spec.series]
    values = [group for _, group in merged]
    if spec.sem_correction:
        stacked = np.stack(values)  # The source's correction needs identical step grids.
        global_mean = stacked.mean(axis=(0, 2))
        treatment = stacked.mean(axis=0)
        values = [group + global_mean[:, None] - treatment for group in values]
    curves = {}
    for series, (steps, group) in zip(spec.series, zip([step for step, _ in merged], values, strict=True), strict=True):
        runs = group.shape[1]
        mean = group.mean(axis=1)
        standard_error = group.std(axis=1, ddof=1) / np.sqrt(runs) if runs > 1 else np.zeros_like(mean)
        curves[series.label] = Curve(steps * spec.n_batches, mean, standard_error, runs)
    return curves


def merge_runs(runs: list[Run], start: float) -> tuple[FloatArray, FloatArray]:
    """pandas merge_asof of every run onto the first run's steps, then linear interpolation."""
    frames = [pd.DataFrame({"step": steps + start, index: values}) for index, (steps, values) in enumerate(runs)]
    merged = frames[0]
    for frame in frames[1:]:
        merged = pd.merge_asof(merged.sort_values("step"), frame.sort_values("step"), on="step", direction="nearest")
    merged = merged.sort_values("step").interpolate(method="linear", axis=0).dropna()
    return _floats(merged["step"]), np.asarray(merged.drop(columns="step").to_numpy(), np.float64)


def hypercube_bars(paper: Path) -> tuple[list[tuple[object, ...]], list[str]]:
    """Source barcharts: each run's last logged exact-loss step; mean and standard error over runs."""
    bars: list[tuple[object, ...]] = []
    missing: list[str] = []
    for kind, figure in (("behaviour", "hypercube_policy"), ("prediction", "hypercube_value")):
        for side, dimension in CUBES:
            root = paper / "hypercube" / kind / f"{side}x{dimension}"
            for stage, model in (("characteristic", "Characteristic"), ("shapley", "Shapley")):
                try:
                    runs = Sources(paper, Path()).sweep(root / stage, "exact_loss")
                except MissingRunsError as error:
                    missing.append(f"{figure} {side}x{dimension} {stage}: {error}")
                    continue
                steps = np.asarray([steps[-1] for steps, _ in runs], np.float64)
                standard_error = float(steps.std(ddof=1) / np.sqrt(steps.size)) if steps.size > 1 else 0.0
                states = side**dimension - 1
                bars.append((figure, model, side, dimension, states, float(steps.mean()), standard_error, steps.size))
    return bars, missing


def _milestone(environment: str, kind: str, stage: str, exact_char: str) -> Loader:
    def load(sources: Sources) -> list[Run]:
        return sources.milestone_runs(environment, kind, stage, exact_char)

    return load


def _sweep(folder: Path, column: str, **settings: object) -> Loader:
    def load(sources: Sources) -> list[Run]:
        return sources.sweep(sources.paper / folder, column, **settings)

    return load


def _continual(domain: str, kind: str, ratio: int, column: str) -> Loader:
    def load(sources: Sources) -> list[Run]:
        return sources.continual(domain, kind, ratio, column)

    return load


def _renamed(specs: list[FigureSpec], source: str, name: str) -> FigureSpec:
    spec = next(spec for spec in specs if spec.name == source)
    return FigureSpec(name, spec.series, spec.n_batches, spec.sem_correction)


def _matches(sweep: Path, settings: dict[str, object]) -> bool:
    if not settings:
        return True
    configs = sorted(sweep.glob("seed-*/config.json")) or [sweep / "config.json"]
    if not configs[0].exists():
        return False
    config = json.loads(configs[0].read_text())["config"]
    return all(config.get(name) == value for name, value in settings.items())


def _metrics(path: Path, column: str) -> Run:
    frame = pd.read_csv(path)
    values = pd.to_numeric(frame[column], errors="coerce")
    keep = values.notna()
    return _floats(frame["step"][keep]), _floats(values[keep])


def _required(runs: list[Run], name: str) -> list[Run]:
    if not runs or any(steps.size == 0 for steps, _ in runs):
        raise MissingRunsError(f"no logged values for {name}")
    return runs


def _floats(values: object) -> FloatArray:
    return np.asarray(values, np.float64)


def _rounded(row: tuple[str, str, float, float, float, int]) -> tuple[str, str, str, str, str, int]:
    figure, label, x, mean, standard_error, runs = row
    return figure, label, f"{x:.6g}", f"{mean:.6g}", f"{standard_error:.6g}", runs


if __name__ == "__main__":
    main(tyro.cli(Args))
