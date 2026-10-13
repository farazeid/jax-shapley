"""Regenerate the paper's figures from tracked JAX runs with the source's plot statistics.

Each figure spec follows its upstream plotting script (`experiments/**/plots/plot_*.py`
at FastSVERL 25b2cbe): the logged tag, the groups and labels, `start_epoch` offsets,
`n_batches` x scaling and `sem_correction`. `curve` ports `fastsverl.utils.plot`:
runs are merged on the logging step with `merge_asof(direction="nearest")`, linearly
interpolated, then summarised by the mean and standard error over runs. Hypercube bars
port `barcharts`: each run's last logged exact-loss step, mean and standard error.

Runs come only from the tracking backend: each series names a registered experiment
(`new_jax_shapley/configs/NAME.py`), a stage's runs within it (`NAME--STAGE--...--seed-N`)
and an upstream tag, read from the experiment's latest run group, or for a sweep the latest
group whose matching run was configured with the series' settings (e.g. a Hypercube's side
and dimension). Figures whose experiments are not ported yet, or have no tracked runs, are
listed as missing.

Usage: uv run --group benchmark python scripts/paper_port/figures.py --track wandb|trackio [--output docs/data]
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import dataclass, field
from math import isclose
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import tyro

from new_jax_shapley import backends
from new_jax_shapley.experiment import load_configs

type FloatArray = npt.NDArray[np.float64]
type Run = tuple[FloatArray, FloatArray]
type Loader = Callable[[Sources], list[Run]]
type Where = tuple[str, dict[str, object]]
"""A sweep point: a stage whose run in the group was configured with these (dotted) settings."""

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
LARGE = (("443", 4), ("453", 5), ("463", 6))


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
    _groups: dict[str, list[str]] = field(default_factory=dict, repr=False)
    _runs: dict[str, dict[str, backends.TrackedRun]] = field(default_factory=dict, repr=False)

    def runs(self, experiment: str, stage: str, tag: str, where: Where | None = None) -> list[Run]:
        """Every seed's run of `stage` (`algorithm--dqn`, `characteristic--behaviour`,
        `shapley--behaviour--with-exact`, ...), in seed order, from the experiment's latest group
        (with `where`, the latest group holding a run of that stage with those settings)."""
        if experiment not in self.ported:
            raise MissingRunsError(f"experiment {experiment} is not ported")
        if experiment not in self._groups:
            self._groups[experiment] = backends.groups(
                self.track, project=self.project, entity=self.entity, prefix=f"{experiment}-"
            )
        for group in reversed(self._groups[experiment]):
            if group not in self._runs:
                self._runs[group] = backends.runs(self.track, project=self.project, entity=self.entity, group=group)
            tracked = self._runs[group]
            if where is not None and not any(
                name.startswith(f"{experiment}--{where[0]}") and _matches(run.config, where[1])
                for name, run in tracked.items()
            ):
                continue
            prefix = f"{experiment}--{stage}"
            names = sorted((name for name in tracked if name == prefix or name.startswith(f"{prefix}--")), key=_seed)
            runs = []
            for name in names:
                logged = [(row["_step"], row[tag]) for row in tracked[name].rows if tag in row and "_step" in row]
                runs.append((_floats([step for step, _ in logged]), _floats([value for _, value in logged])))
            return _required(runs, f"{experiment} {stage} {tag}")
        raise MissingRunsError(f"no tracked runs of {experiment}{'' if where is None else f' with {where}'}")


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
            characteristic = _tracked(experiment, f"characteristic--{kind}", _tag(kind, "characteristic"))
            specs.append(FigureSpec(f"app_{domain}_{name}_char", (Series("#1f77b4", characteristic),), BATCHES))
            exact = _tracked(experiment, f"shapley--{kind}--{EXACT_CHARACTERISTIC}", _tag(kind))
            approx = _tracked(experiment, f"shapley--{kind}--{APPROX_CHARACTERISTIC}", _tag(kind))
            specs.append(
                FigureSpec(
                    f"app_{domain}_{name}_shapley",
                    (Series("Exact characteristic", exact), Series("Approx characteristic", approx)),
                    BATCHES,
                )
            )
        # Outcome: "approx behaviour" runs set with_exact_char (as the source's groups do).
        outcome = f"{environment}_outcome__sweep"
        series = []
        for variant, label in (("on-policy", "On-policy"), ("off-policy", "Off-policy")):
            tag = f"model/PerformanceCharacteristic{'OnPolicy' if variant == 'on-policy' else 'OffPolicy'}_exact_loss"
            for exactness, behaviour in (("with-exact", "approx"), ("without-exact", "exact")):
                loader = _tracked(outcome, f"outcome--{variant}--{exactness}", tag)
                series.append(Series(f"{label}, {behaviour} behaviour", loader))
        specs.append(FigureSpec(f"app_{domain}_perf_char", tuple(series)))
        # The source's Shapley groups: w_policy, wo_policy and wo_perf of the paper's outcome variant.
        variant = "off-policy" if environment == "mastermind222" else "on-policy"
        groups = (
            ("Approx outcome, approx behaviour", "with-exact", "with-exact"),
            ("Approx outcome, exact behaviour", "with-exact", "without-exact"),
            ("Exact outcome, exact behaviour", "without-exact", "without-exact"),
        )
        series = [
            Series(
                label,
                _tracked(
                    outcome,
                    f"shapley--outcome--{shapley}",
                    "model/PerformanceShapley_exact_loss",
                    (f"outcome--{variant}--{characteristic}", {}),
                ),
            )
            for label, shapley, characteristic in groups
        ]
        specs.append(FigureSpec(f"app_{domain}_perf_shapley", tuple(series), BATCHES))
    specs += [
        _renamed(specs, "app_mastermind_222_policy_char", "main_mastermind_222_policy_char"),
        _renamed(specs, "app_mastermind_222_perf_char", "main_mastermind_222_perf_char"),
        _renamed(specs, "app_mastermind_222_perf_shapley", "main_mastermind_222_perf_shapley"),
    ]
    for kind, name in (("behaviour", "policy"), ("prediction", "value")):
        experiment = f"mastermind333_{kind}"
        characteristic = _tracked(experiment, f"characteristic--{kind}", _tag(kind, "characteristic"))
        specs.append(FigureSpec(f"app_mastermind_333_{name}_char", (Series("#1f77b4", characteristic),), BATCHES))
        exact = _tracked(experiment, f"shapley--{kind}--{EXACT_CHARACTERISTIC}", _tag(kind))
        approx = _tracked(experiment, f"shapley--{kind}--{APPROX_CHARACTERISTIC}", _tag(kind))
        specs.append(
            FigureSpec(
                f"app_mastermind_333_{name}_shapley",
                (Series("Exact characteristic", exact), Series("Approx characteristic", approx)),
                BATCHES,
            )
        )
        large = f"mastermind_large_{kind}__sweep"
        characteristics = tuple(
            Series(
                f"Mastermind-{size}",
                _tracked(
                    large,
                    f"characteristic--{kind}",
                    _tag(kind, "characteristic", "epoch"),
                    ("algorithm--dqn", {"env.num_guesses": guesses}),
                ),
            )
            for size, guesses in LARGE
        )
        specs.append(FigureSpec(f"app_large_{name}_char", characteristics))
        shapleys = tuple(
            Series(
                f"Mastermind-{size}",
                _tracked(
                    large,
                    f"shapley--{kind}--without-exact",
                    _tag(kind, loss="epoch"),
                    ("algorithm--dqn", {"env.num_guesses": guesses}),
                ),
            )
            for size, guesses in LARGE
        )
        specs.append(FigureSpec(f"app_large_{name}_shapley", shapleys, BATCHES))
    outcome_shapleys = tuple(
        Series(
            f"Mastermind-{size}",
            _tracked(
                "mastermind_large_outcome__sweep",
                "shapley--outcome--without-exact",
                "model/PerformanceShapley_epoch_loss",
                ("algorithm--dqn", {"env.num_guesses": guesses}),
            ),
        )
        for size, guesses in LARGE
    )
    specs.append(FigureSpec("app_large_perf_offpolicy_shapley", outcome_shapleys, BATCHES))
    specs += offpolicy_figures() + continual_figures() + sampling_figures()
    return specs


def offpolicy_figures() -> list[FigureSpec]:
    """offpolicy/plots: characteristics' exact losses; every figure uses sem_correction."""
    inf = float("inf")
    settings: dict[str, dict[str, object]] = {
        "exact": {"data": "rollouts"},
        "wo_weight": {"data": "replay", "off_policy": False},
        "w_weight": {"data": "replay", "off_policy": True, "weighting": True, "clip_coef": inf},
        "normalised": {"data": "replay", "off_policy": True, "weighting": True, "clip_coef": inf},
        "unnormalised": {"data": "replay", "off_policy": True, "weighting": False, "clip_coef": inf},
        **{
            f"clip_{coef}": {"data": "replay", "off_policy": True, "weighting": True, "clip_coef": coef}
            for coef in (0.99, 0.995, 0.998)
        },
    }
    specs = []
    for kind, name in (("behaviour", "policy"), ("prediction", "value")):
        tag = _tag(kind, "characteristic")
        stage = f"characteristic--{kind}"

        def series(experiment: str, groups: tuple[tuple[str, str], ...], stage: str, tag: str) -> tuple[Series, ...]:
            return tuple(
                Series(label, _tracked(experiment, stage, tag, (stage, settings[group]))) for label, group in groups
            )

        groups = (
            ("Off-policy (without IS)", "wo_weight"),
            ("Off-policy (with IS)", "w_weight"),
            ("On-policy", "exact"),
        )
        mastermind = series(f"mastermind222_offpolicy_{kind}__sweep", groups, stage, tag)
        specs.append(FigureSpec(f"app_mastermind_222_offpolicy_{name}", mastermind, OFFPOLICY_MASTERMIND_BATCHES, True))
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
            gridworld = series(f"gridworld_offpolicy_{kind}__sweep", groups, stage, tag)
            specs.append(FigureSpec(f"app_gwb_offpolicy_{figure}_{name}", gridworld, OFFPOLICY_GRIDWORLD_BATCHES, True))
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
    """sampling/plots: the model group starts after its characteristic's epochs (start_epoch).

    Exact: Shapley models against a learnt characteristic's exact values (wo_exact_model);
    Sample: against a sampled characteristic (w_exact_sampling); Model: against the exact
    characteristic (w_exact_model). Outcome groups compare outcome characteristics.
    """
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
        learnt, sampled = f"{domain}_sampling_{kind}", f"{domain}_sampled_{kind}"
        series = (
            Series("Exact", _tracked(learnt, f"shapley--{kind}--without-exact", _tag(kind))),
            Series("Sample", _tracked(sampled, f"shapley--{kind}--with-exact", _tag(kind))),
            Series("Model", _tracked(learnt, f"shapley--{kind}--with-exact", _tag(kind)), float(offset)),
        )
        name = "policy" if kind == "behaviour" else "value"
        specs.append(FigureSpec(f"app_{names[domain]}_{name}_sampling", series, BATCHES))
    tag = "model/PerformanceCharacteristicOnPolicy_exact_loss"
    for domain in ("mastermind222", "gridworld"):
        learnt, sampled = f"{domain}_sampling_outcome__sweep", f"{domain}_sampled_outcome"
        series = (
            Series("Exact", _tracked(learnt, "outcome--on-policy--without-exact", tag)),
            Series("Sample", _tracked(sampled, "outcome--on-policy--with-exact", tag)),
            Series("Model", _tracked(learnt, "outcome--on-policy--with-exact", tag), 50 * (0.8 * 10_000) // 128),
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
            cube: Where = ("algorithm--dqn", {"env.side_length": side, "env.dimension": dimension})
            for stage, model, tag in (
                (f"characteristic--{kind}", "Characteristic", _tag(kind, "characteristic")),
                (f"shapley--{kind}--without-exact", "Shapley", _tag(kind)),
            ):
                try:
                    runs = sources.runs(f"hypercube_{kind}__sweep", stage, tag, cube)
                except MissingRunsError as error:
                    missing.append(f"{figure} {side}x{dimension} {model}: {error}")
                    continue
                steps = np.asarray([steps[-1] for steps, _ in runs], np.float64)
                standard_error = float(steps.std(ddof=1) / np.sqrt(steps.size)) if steps.size > 1 else 0.0
                states = side**dimension - 1
                bars.append((figure, model, side, dimension, states, float(steps.mean()), standard_error, steps.size))
    return bars, missing


def _tracked(experiment: str, stage: str, tag: str, where: Where | None = None) -> Loader:
    def load(sources: Sources) -> list[Run]:
        return sources.runs(experiment, stage, tag, where)

    return load


def _tag(kind: str, model: str = "shapley", loss: str = "exact") -> str:
    """The source's `model/{ModelClass}_{exact,epoch}_loss` tag for behaviour and prediction models."""
    explained = "Policy" if kind == "behaviour" else "Value"
    return f"model/{explained}{'CharacteristicModel' if model == 'characteristic' else 'Shapley'}_{loss}_loss"


def _matches(config: dict[str, object], settings: dict[str, object]) -> bool:
    """Whether a logged (nested) run config has every dotted setting; floats compare as floats
    (backends may log infinity as text)."""
    for path, expected in settings.items():
        value: object = config
        for key in path.split("."):
            if not isinstance(value, dict):
                return False
            value = cast(dict[str, object], value).get(key)
        if isinstance(expected, float) and not isinstance(expected, bool):
            if not isinstance(value, int | float | str) or not isclose(float(value), expected):
                return False
        elif value != expected:
            return False
    return True


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
