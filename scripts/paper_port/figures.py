"""Regenerate the paper's figures from tracked JAX runs with the source's plot statistics.

Each figure spec follows its upstream plotting script (`experiments/**/plots/plot_*.py`
at FastSVERL 25b2cbe): the logged tag, the groups and labels, `start_epoch` offsets,
`n_batches` x scaling and `sem_correction`. `curve` ports `fastsverl.utils.plot`:
runs are merged on the logging step with `merge_asof(direction="nearest")`, linearly
interpolated, then summarised by the mean and standard error over runs. Hypercube bars
port `barcharts`: each run's last logged exact-loss step, mean and standard error.

Runs come only from the tracking backend: each series names a registered experiment
(`new_jax_shapley/configs/NAME.py`), a stage's runs within it and an upstream tag, read
from that experiment's latest run group. Figures whose experiments are not ported yet, or
have no tracked runs, are listed as missing.

Usage: uv run --group benchmark python scripts/paper_port/figures.py --track wandb|trackio [--output docs/data]
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import tyro

from new_jax_shapley import backends
from new_jax_shapley.experiment import load_configs

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
# Shapley stages by the source's series labels: its "Exact characteristic" runs set with_exact_char=False.
EXACT_CHARACTERISTIC = "without-exact"
APPROX_CHARACTERISTIC = "with-exact"


@dataclass(frozen=True, slots=True)
class Args:
    track: backends.Track
    project: str = "jax-shapley"
    entity: str | None = None
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
    """Tracked runs; every run is (logging steps, values) of one tag."""

    track: backends.Track
    project: str
    entity: str | None
    ported: frozenset[str]
    _histories: dict[str, dict[str, list[backends.Scalars]]] = field(default_factory=dict, repr=False)

    def runs(self, experiment: str, stage: str, tag: str) -> list[Run]:
        """Every seed's run of `stage` (`algorithm--dqn`, `characteristic--behaviour`,
        `shapley--behaviour--with-exact`, ...), in seed order."""
        history = self._history(experiment)
        prefix = f"{experiment}--{stage}"
        names = sorted((name for name in history if name == prefix or name.startswith(f"{prefix}--")), key=_seed)
        runs = []
        for name in names:
            logged = [(row["_step"], row[tag]) for row in history[name] if tag in row and "_step" in row]
            runs.append((_floats([step for step, _ in logged]), _floats([value for _, value in logged])))
        return _required(runs, f"{experiment} {stage} {tag}")

    def _history(self, experiment: str) -> dict[str, list[backends.Scalars]]:
        if experiment not in self.ported:
            raise MissingRunsError(f"experiment {experiment} is not ported")
        if experiment not in self._histories:
            found = backends.groups(self.track, project=self.project, entity=self.entity, prefix=f"{experiment}-")
            if not found:
                raise MissingRunsError(f"no tracked runs of {experiment} in {self.track} project {self.project}")
            self._histories[experiment] = backends.history(
                self.track, project=self.project, entity=self.entity, group=found[-1]
            )
        return self._histories[experiment]


class MissingRunsError(Exception):
    """The runs a figure needs do not exist (yet)."""


def main(args: Args) -> None:
    sources = Sources(args.track, args.project, args.entity, frozenset(load_configs()))
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
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "paper--jax-curves.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("figure", "series", "x", "mean", "standard_error", "runs"))
        writer.writerows(_rounded(row) for row in rows)
    bars, missing_bars = hypercube_bars(sources)
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
            experiment = f"{environment}_{kind}"
            specs.append(
                FigureSpec(
                    f"app_{domain}_{name}_char",
                    (Series("#1f77b4", _tracked(experiment, f"characteristic--{kind}", _tag(kind, "characteristic"))),),
                    BATCHES,
                )
            )
            specs.append(
                FigureSpec(
                    f"app_{domain}_{name}_shapley",
                    (
                        Series(
                            "Exact characteristic",
                            _tracked(experiment, f"shapley--{kind}--{EXACT_CHARACTERISTIC}", _tag(kind)),
                        ),
                        Series(
                            "Approx characteristic",
                            _tracked(experiment, f"shapley--{kind}--{APPROX_CHARACTERISTIC}", _tag(kind)),
                        ),
                    ),
                    BATCHES,
                )
            )
        outcome = f"{environment}_outcome"
        tag = _tag("outcome", "characteristic")
        specs.append(
            FigureSpec(
                f"app_{domain}_perf_char",
                (
                    Series("On-policy, approx behaviour", _tracked(f"{outcome}_on_policy", "outcome-with-policy", tag)),
                    Series("On-policy, exact behaviour", _tracked(f"{outcome}_on_policy", "outcome-exact-policy", tag)),
                    Series(
                        "Off-policy, approx behaviour", _tracked(f"{outcome}_off_policy", "outcome-with-policy", tag)
                    ),
                    Series(
                        "Off-policy, exact behaviour", _tracked(f"{outcome}_off_policy", "outcome-exact-policy", tag)
                    ),
                ),
            )
        )
        variant = f"{outcome}_off_policy" if environment == "mastermind222" else f"{outcome}_on_policy"
        specs.append(
            FigureSpec(
                f"app_{domain}_perf_shapley",
                (
                    Series(
                        "Approx outcome, approx behaviour", _tracked(variant, "shapley-with-policy", _tag("outcome"))
                    ),
                    Series(
                        "Approx outcome, exact behaviour", _tracked(variant, "shapley-exact-policy", _tag("outcome"))
                    ),
                    Series(
                        "Exact outcome, exact behaviour", _tracked(variant, "shapley-exact-outcome", _tag("outcome"))
                    ),
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
        experiment = f"mastermind333_{kind}"
        specs.append(
            FigureSpec(
                f"app_mastermind_333_{name}_char",
                (Series("#1f77b4", _tracked(experiment, f"characteristic--{kind}", _tag(kind, "characteristic"))),),
                BATCHES,
            )
        )
        specs.append(
            FigureSpec(
                f"app_mastermind_333_{name}_shapley",
                (
                    Series(
                        "Exact characteristic",
                        _tracked(experiment, f"shapley--{kind}--{EXACT_CHARACTERISTIC}", _tag(kind)),
                    ),
                    Series(
                        "Approx characteristic",
                        _tracked(experiment, f"shapley--{kind}--{APPROX_CHARACTERISTIC}", _tag(kind)),
                    ),
                ),
                BATCHES,
            )
        )
        specs.append(
            FigureSpec(
                f"app_large_{name}_char",
                tuple(
                    Series(
                        f"Mastermind-{size}",
                        _tracked(
                            f"mastermind{size}_{kind}", f"characteristic--{kind}", _tag(kind, "characteristic", "epoch")
                        ),
                    )
                    for size in ("443", "453", "463")
                ),
            )
        )
        specs.append(
            FigureSpec(
                f"app_large_{name}_shapley",
                tuple(
                    Series(
                        f"Mastermind-{size}",
                        _tracked(
                            f"mastermind{size}_{kind}",
                            f"shapley--{kind}--{APPROX_CHARACTERISTIC}",
                            _tag(kind, loss="epoch"),
                        ),
                    )
                    for size in ("443", "453", "463")
                ),
                BATCHES,
            )
        )
    specs.append(
        FigureSpec(
            "app_large_perf_offpolicy_shapley",
            tuple(
                Series(
                    f"Mastermind-{size}",
                    _tracked(f"mastermind{size}_outcome", "shapley-with-policy", _tag("outcome", loss="epoch")),
                )
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
        experiment = f"offpolicy_mastermind222_{kind}"
        groups = (
            ("Off-policy (without IS)", "wo_weight"),
            ("Off-policy (with IS)", "w_weight"),
            ("On-policy", "exact"),
        )
        series = tuple(Series(label, _tracked(experiment, group, _tag(kind))) for label, group in groups)
        specs.append(FigureSpec(f"app_mastermind_222_offpolicy_{name}", series, OFFPOLICY_MASTERMIND_BATCHES, True))
        experiment = f"offpolicy_gridworld_{kind}"
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
            series = tuple(Series(label, _tracked(experiment, group, _tag(kind))) for label, group in groups)
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
                Series(f"{ratio}:1", _tracked(f"continual_{environment}_{kind}_ratio{ratio}", "continual", _tag(kind)))
                for ratio in (1, 2, 10, 50)
            )
            specs.append(FigureSpec(figure, series, PARALLEL_SCALE))
        for tag, figure in (("agent/episodic_return", "return"), ("agent/DQN_loss", "dqn_loss")):
            series = (Series("#1f77b4", _tracked(f"continual_{environment}_behaviour_ratio1", "continual", tag)),)
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
        experiment = f"sampling_{domain}_{kind}"
        series = (
            Series("Exact", _tracked(experiment, "wo_exact_model", _tag(kind))),
            Series("Sample", _tracked(experiment, "w_exact_sampling", _tag(kind))),
            Series("Model", _tracked(experiment, "w_exact_model", _tag(kind)), float(offset)),
        )
        name = "policy" if kind == "behaviour" else "value"
        specs.append(FigureSpec(f"app_{names[domain]}_{name}_sampling", series, BATCHES))
    for domain in ("mastermind222", "gridworld"):
        experiment = f"sampling_{domain}_outcome"
        series = (
            Series("Exact", _tracked(experiment, "wo_exact_model", _tag("outcome"))),
            Series("Sample", _tracked(experiment, "w_exact_sampling", _tag("outcome"))),
            Series("Model", _tracked(experiment, "w_exact_model", _tag("outcome")), 50 * (0.8 * 10_000) // 128),
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


def hypercube_bars(sources: Sources) -> tuple[list[tuple[object, ...]], list[str]]:
    """Source barcharts: each run's last logged exact-loss step; mean and standard error over runs."""
    bars: list[tuple[object, ...]] = []
    missing: list[str] = []
    for kind, figure in (("behaviour", "hypercube_policy"), ("prediction", "hypercube_value")):
        for side, dimension in CUBES:
            experiment = f"hypercube{side}x{dimension}_{kind}"
            for stage, model, tag in (
                (f"characteristic--{kind}", "Characteristic", _tag(kind, "characteristic")),
                (f"shapley--{kind}--{APPROX_CHARACTERISTIC}", "Shapley", _tag(kind)),
            ):
                try:
                    runs = sources.runs(experiment, stage, tag)
                except MissingRunsError as error:
                    missing.append(f"{figure} {side}x{dimension} {model}: {error}")
                    continue
                steps = np.asarray([steps[-1] for steps, _ in runs], np.float64)
                standard_error = float(steps.std(ddof=1) / np.sqrt(steps.size)) if steps.size > 1 else 0.0
                states = side**dimension - 1
                bars.append((figure, model, side, dimension, states, float(steps.mean()), standard_error, steps.size))
    return bars, missing


def _tracked(experiment: str, stage: str, tag: str) -> Loader:
    def load(sources: Sources) -> list[Run]:
        return sources.runs(experiment, stage, tag)

    return load


def _tag(kind: str, model: str = "shapley", loss: str = "exact") -> str:
    """The source's `model/{ModelClass}_{exact,epoch}_loss` tag."""
    explained = "Policy" if kind == "behaviour" else "Value"
    return f"model/{explained}{'CharacteristicModel' if model == 'characteristic' else 'Shapley'}_{loss}_loss"


def _renamed(specs: list[FigureSpec], source: str, name: str) -> FigureSpec:
    spec = next(spec for spec in specs if spec.name == source)
    return FigureSpec(name, spec.series, spec.n_batches, spec.sem_correction)


def _seed(name: str) -> tuple[str, int]:
    """Run names `...--seed-N` ordered by seed number."""
    stage, _, seed = name.rpartition("--seed-")
    return stage, int(seed) if seed.isdigit() else -1


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
