"""Compare regenerated JAX figures and tables with the published FastSVERL results.

Applies `docs/data/paper--criteria.md` unchanged:

- curves: at 25% and 100% of each published series' x range, the JAX mean passes when
  it is within a factor of 2 of the published mean, or within 3 combined standard errors
  (`se_published` is the half-width of the published band), or both are below 1e-6;
- Hypercube bars: the same with a factor of 1.5;
- tables: a factor of 2 or 3 combined standard errors, for updates and final loss;
- claims (`paper--claims.csv`): each paper finding, evaluated on the JAX results alone.

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


@dataclass(frozen=True, slots=True)
class Args:
    data: Path = Path("docs/data")
    figures: Path = Path("docs/figures/paper")


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
    for figure in sorted({figure for figure, _ in figures.jax}):
        overlay(figure, figures, args.figures)
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
    points: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
    for row in _records(data / "paper--jax-curves.csv"):
        point = (float(row["x"]), float(row["mean"]), float(row["standard_error"]))
        points.setdefault((row["figure"], row["series"]), []).append(point)
    curves = {}
    for key, rows in points.items():
        ordered = np.asarray(sorted(rows), np.float64)
        curves[key] = Curve(ordered[:, 0], ordered[:, 1], ordered[:, 2])
    return curves


def overlay(figure: str, figures: Figures, folder: Path) -> None:
    """Published means and bands (dashed) and JAX mean +/- standard error (solid), one colour per series."""
    handle, axis = plt.subplots(figsize=(6, 4))
    series = sorted(name for name_figure, name in figures.jax if name_figure == figure)
    values = []
    for index, name in enumerate(series):
        colour = f"C{index % 10}"
        curve = figures.jax[figure, name]
        values.append(curve.mean)
        axis.plot(curve.x, curve.mean, color=colour, label=f"JAX {name}")
        lower, upper = curve.mean - curve.standard_error, curve.mean + curve.standard_error
        axis.fill_between(curve.x, lower, upper, color=colour, alpha=0.2)
        band = figures.published.get((figure, name))
        if band is not None:
            axis.plot(band.x, band.mean, color=colour, linestyle="--", label=f"Published {name}")
            axis.fill_between(band.x, band.mean - band.half_width, band.mean + band.half_width, color=colour, alpha=0.1)
    means = np.concatenate(values)
    if np.all(means > 0) and means.max() / means.min() > 100:
        axis.set_yscale("log")
    axis.set_title(figure)
    axis.set_xlabel("Training updates")
    axis.legend(fontsize=6)
    handle.tight_layout()
    handle.savefig(folder / f"{figure}.svg")
    plt.close(handle)


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
