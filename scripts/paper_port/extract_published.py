"""Extract the published FastSVERL figure data and tables from the pinned upstream checkout.

The paper's figures are the matplotlib vector PDFs committed under
`experiments/**/plots/` at FastSVERL `25b2cbe`; its large-Mastermind tables are the
`.tex` files beside them. This reads them without rendering:

- **Axes** are the clip rectangles data artists draw into. Tick marks and their
  label text calibrate a linear or log10 map from page points to data; an offset
  label such as `1e6` scales the axis.
- **Curves** are coloured stroked polylines (mean over runs); a same-colour filled
  polygon is the shaded band (upper edge forward, lower edge back). Legend swatches
  name each colour.
- **Bars** (Hypercube) are filled rectangles; with upstream's bar geometry
  (width 0.16 x states, characteristic left of the state count, Shapley right) each
  bar identifies its state count and model, and its error bar gives the standard error.

Outputs: `paper--published-curves.csv`, `paper--published-bars.csv` and
`paper--published-tables.csv`. `--self-test` renders synthetic plots with known
data in upstream's style and checks that extraction recovers them.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import numpy as np
import pypdf
import tyro
from pypdf.generic import ContentStream

type Point = tuple[float, float]
type Colour = tuple[float, float, float]
type Box = tuple[float, float, float, float]  # left, bottom, right, top

GREY_TOLERANCE = 0.02
CURVE_POINTS = 64
TICK_MIN_LENGTH = 3.0
TICK_MAX_LENGTH = 10.0
OFFSET = re.compile(r"1e(-?\d+)")
TICK_LABEL = re.compile(r"[0-9.eE+\-\N{MINUS SIGN}]+")


@dataclass(frozen=True)
class Args:
    source_root: Path
    """Pinned FastSVERL checkout (commit 25b2cbe)."""
    output: Path = Path("docs/data")
    self_test: bool = False


@dataclass(frozen=True, slots=True)
class Text:
    value: str
    x: float
    y: float
    rotated: bool


@dataclass(frozen=True, slots=True)
class Shape:
    """One painted path in page coordinates."""

    operator: str
    stroke: Colour
    fill: Colour
    subpaths: tuple[tuple[Point, ...], ...]
    clip: Box | None

    @property
    def points(self) -> tuple[Point, ...]:
        return tuple(point for subpath in self.subpaths for point in subpath)


@dataclass(frozen=True, slots=True)
class Axis:
    """Page coordinate to data value, linear or log10."""

    slope: float
    intercept: float
    log: bool

    def value(self, coordinate: float) -> float:
        raw = self.slope * coordinate + self.intercept
        return math.pow(10.0, raw) if self.log else raw


@dataclass(slots=True)
class Panel:
    box: Box
    x: Axis | None = None
    y: Axis | None = None
    titles: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CurveRow:
    figure: str
    panel: int
    title: str
    series: str
    kind: str
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class BarRow:
    figure: str
    panel: int
    title: str
    model: str
    states: float
    mean: float
    standard_error: float


# Figures and tables the paper reports (upstream plot outputs, relative to experiments/).
FIGURES = (
    "fastsverl/characteristics/plots/main_mastermind_222_policy_char.pdf",
    "fastsverl/characteristics/plots/main_mastermind_222_perf_char.pdf",
    "fastsverl/shapleys/plots/main_mastermind_222_perf_shapley.pdf",
    "fastsverl/characteristics/plots/app_mastermind_222_policy_char.pdf",
    "fastsverl/characteristics/plots/app_mastermind_222_value_char.pdf",
    "fastsverl/characteristics/plots/app_mastermind_222_perf_char.pdf",
    "fastsverl/characteristics/plots/app_mastermind_333_policy_char.pdf",
    "fastsverl/characteristics/plots/app_mastermind_333_value_char.pdf",
    "fastsverl/characteristics/plots/app_gwb_policy_char.pdf",
    "fastsverl/characteristics/plots/app_gwb_value_char.pdf",
    "fastsverl/characteristics/plots/app_gwb_perf_char.pdf",
    "fastsverl/characteristics/plots/app_large_policy_char.pdf",
    "fastsverl/characteristics/plots/app_large_value_char.pdf",
    "fastsverl/shapleys/plots/app_mastermind_222_policy_shapley.pdf",
    "fastsverl/shapleys/plots/app_mastermind_222_value_shapley.pdf",
    "fastsverl/shapleys/plots/app_mastermind_222_perf_shapley.pdf",
    "fastsverl/shapleys/plots/app_mastermind_333_policy_shapley.pdf",
    "fastsverl/shapleys/plots/app_mastermind_333_value_shapley.pdf",
    "fastsverl/shapleys/plots/app_gwb_policy_shapley.pdf",
    "fastsverl/shapleys/plots/app_gwb_value_shapley.pdf",
    "fastsverl/shapleys/plots/app_gwb_perf_shapley.pdf",
    "fastsverl/shapleys/plots/app_large_policy_shapley.pdf",
    "fastsverl/shapleys/plots/app_large_value_shapley.pdf",
    "fastsverl/shapleys/plots/app_large_perf_offpolicy_shapley.pdf",
    "offpolicy/plots/main_mastermind_222_offpolicy_policy.pdf",
    "offpolicy/plots/app_mastermind_222_offpolicy_policy.pdf",
    "offpolicy/plots/app_mastermind_222_offpolicy_value.pdf",
    "offpolicy/plots/app_gwb_offpolicy_weighting_policy.pdf",
    "offpolicy/plots/app_gwb_offpolicy_weighting_value.pdf",
    "offpolicy/plots/app_gwb_offpolicy_clipping_policy.pdf",
    "offpolicy/plots/app_gwb_offpolicy_clipping_value.pdf",
    "parallel/plots/main_mastermind_222_parallel_value_approx.pdf",
    "parallel/plots/app_mastermind_222_parallel_policy_approx.pdf",
    "parallel/plots/app_mastermind_222_parallel_value_approx.pdf",
    "parallel/plots/app_mastermind_222_parallel_perf_approx_behaviour.pdf",
    "parallel/plots/app_mastermind_222_parallel_return.pdf",
    "parallel/plots/app_mastermind_222_parallel_dqn_loss.pdf",
    "parallel/plots/app_gwb_parallel_policy_approx.pdf",
    "parallel/plots/app_gwb_parallel_value_approx.pdf",
    "parallel/plots/app_gwb_parallel_perf_approx_behaviour.pdf",
    "parallel/plots/app_gwb_parallel_return.pdf",
    "parallel/plots/app_gwb_parallel_dqn_loss.pdf",
    "sampling/shapleys/plots/main_mastermind_222_policy_sampling.pdf",
    "sampling/shapleys/plots/app_mastermind_222_policy_sampling.pdf",
    "sampling/shapleys/plots/app_mastermind_222_value_sampling.pdf",
    "sampling/shapleys/plots/app_mastermind_333_policy_sampling.pdf",
    "sampling/shapleys/plots/app_mastermind_333_value_sampling.pdf",
    "sampling/shapleys/plots/app_gwb_policy_sampling.pdf",
    "sampling/shapleys/plots/app_gwb_value_sampling.pdf",
    "sampling/characteristics/plots/app_mastermind_222_perf_sampling.pdf",
    "sampling/characteristics/plots/app_gwb_perf_sampling.pdf",
)
BAR_FIGURES = ("hypercube/plots/hypercube_policy.pdf", "hypercube/plots/hypercube_value.pdf")
TABLES = {
    "table_1_behaviour": "fastsverl/shapleys/plots/main_large_policy_shapley.tex",
    "table_3_prediction": "fastsverl/shapleys/plots/app_large_value_shapley.tex",
    "table_4_outcome_offpolicy": "fastsverl/shapleys/plots/app_large_perf_offpolicy_shapley.tex",
}


def main() -> None:
    args = tyro.cli(Args)
    if args.self_test:
        self_test()
        return
    experiments = args.source_root / "experiments"
    curves = [row for figure in FIGURES for row in resample(extract_curves(experiments / figure, Path(figure).stem))]
    bars = [row for figure in BAR_FIGURES for row in extract_bars(experiments / figure, Path(figure).stem)]
    tables = [row for name, path in TABLES.items() for row in parse_table(experiments / path, name)]
    args.output.mkdir(parents=True, exist_ok=True)
    _write(args.output / "paper--published-curves.csv", CurveRow.__slots__, [_fields(row) for row in curves])
    _write(args.output / "paper--published-bars.csv", BarRow.__slots__, [_fields(row) for row in bars])
    _write(args.output / "paper--published-tables.csv", TableRow.__slots__, [_fields(row) for row in tables])
    print(f"{len(FIGURES)} figures, {len(curves)} curve points; {len(bars)} bars; {len(tables)} table rows")


def extract_curves(path: Path, figure: str) -> list[CurveRow]:
    """Mean curves and band edges of every panel, labelled by legend."""
    texts, shapes = read_page(path)
    panels = calibrate(texts, shapes)
    legend = legend_labels(texts, shapes)
    rows: list[CurveRow] = []
    for index, panel in enumerate(panels):
        if panel.x is None or panel.y is None:
            raise ValueError(f"{figure}: panel {index} could not be calibrated")
        title = " | ".join(panel.titles)
        for shape in shapes:
            if shape.clip != panel.box or _is_grey(shape.stroke if shape.operator == "S" else shape.fill):
                continue
            if shape.operator == "S" and len(shape.points) > 3:
                series = legend.get(_key(shape.stroke), _name(shape.stroke))
                rows.extend(
                    CurveRow(figure, index, title, series, "mean", panel.x.value(x), panel.y.value(y))
                    for x, y in shape.points
                )
            elif shape.operator in ("f", "B", "F") and len(shape.points) > 8:
                series = legend.get(_key(shape.fill), _name(shape.fill))
                for kind, edge in zip(("high", "low"), _band_edges(shape.points), strict=True):
                    rows.extend(
                        CurveRow(figure, index, title, series, kind, panel.x.value(x), panel.y.value(y))
                        for x, y in edge
                    )
    return rows


def resample(rows: list[CurveRow], points: int = CURVE_POINTS) -> list[CurveRow]:
    """Each (panel, series, kind) on a uniform grid over its own x range, to six significant digits."""
    groups: dict[tuple[int, str, str, str], list[CurveRow]] = {}
    for row in rows:
        groups.setdefault((row.panel, row.title, row.series, row.kind), []).append(row)
    resampled: list[CurveRow] = []
    for (panel, title, series, kind), members in groups.items():
        ordered = sorted(members, key=lambda row: row.x)
        xs = np.asarray([row.x for row in ordered])
        ys = np.asarray([row.y for row in ordered])
        grid = np.linspace(xs[0], xs[-1], points)
        resampled.extend(
            CurveRow(ordered[0].figure, panel, title, series, kind, float(f"{x:.6g}"), float(f"{y:.6g}"))
            for x, y in zip(grid, np.interp(grid, xs, ys), strict=True)
        )
    return resampled


def extract_bars(path: Path, figure: str) -> list[BarRow]:
    """Hypercube bars: one per (dimension panel, state count, model), with standard errors."""
    texts, shapes = read_page(path)
    panels = calibrate(texts, shapes)
    rows: list[BarRow] = []
    for index, panel in enumerate(panels):
        if panel.x is None or panel.y is None:
            raise ValueError(f"{figure}: panel {index} could not be calibrated")
        x_axis, y_axis = panel.x, panel.y
        title = " | ".join(panel.titles)
        caps = [shape for shape in shapes if shape.clip == panel.box and shape.operator == "S" and _is_black(shape)]
        for shape in shapes:
            if shape.clip != panel.box or shape.operator not in ("f", "B") or _is_grey(shape.fill):
                continue
            xs = [point[0] for point in shape.points]
            ys = [point[1] for point in shape.points]
            left, right, top = x_axis.value(min(xs)), x_axis.value(max(xs)), y_axis.value(max(ys))
            # Upstream geometry: width = 0.16 * states; characteristic [0.84 S, S], Shapley [S, 1.16 S].
            ratio = right / left
            model, states = (
                ("Characteristic", right) if abs(ratio - 1 / 0.84) < abs(ratio - 1.16) else ("Shapley", left)
            )
            centre = (min(xs) + max(xs)) / 2
            error_tops = [
                max(point[1] for point in cap.points)
                for cap in caps
                if all(abs(point[0] - centre) < 0.5 for point in cap.points) and len(cap.points) == 2
            ]
            error = y_axis.value(max(error_tops)) - top if error_tops else math.nan
            rows.append(BarRow(figure, index, title, model, round(states), top, error))
    return rows


def read_page(path: Path) -> tuple[list[Text], list[Shape]]:
    reader = pypdf.PdfReader(path)
    page = reader.pages[0]
    texts: list[Text] = []

    def visit(text: str, transform: list[float], matrix: list[float], font: object, size: float) -> None:
        del matrix, font, size
        if text.strip():
            texts.append(Text(text.strip(), float(transform[4]), float(transform[5]), abs(transform[1]) > 0.5))

    page.extract_text(visitor_text=visit)
    return texts, _shapes(ContentStream(page.get_contents(), reader).operations)


def calibrate(texts: list[Text], shapes: list[Shape]) -> list[Panel]:
    """Panels from the clip rectangles data artists use, top to bottom, with axes from ticks and labels."""
    boxes = sorted({shape.clip for shape in shapes if shape.clip is not None}, key=lambda box: (-box[3], box[0]))
    panels = [Panel(box) for box in boxes if box[2] - box[0] > 50 and box[3] - box[1] > 30]
    for panel in panels:
        left, bottom, right, top = panel.box
        ticks_x = sorted({
            round(shape.points[0][0], 2)
            for shape in shapes
            if _tick(shape, vertical=True)
            and left - 0.5 <= shape.points[0][0] <= right + 0.5
            and bottom - 6 <= min(point[1] for point in shape.points) <= bottom + 0.5
        })
        ticks_y = sorted({
            round(shape.points[0][1], 2)
            for shape in shapes
            if _tick(shape, vertical=False)
            and bottom - 0.5 <= shape.points[0][1] <= top + 0.5
            and left - 6 <= min(point[0] for point in shape.points) <= left + 0.5
        })
        # Without tick marks (seaborn whitegrid), gridlines spanning the axes mark the ticks.
        ticks_x = ticks_x or _gridlines(shapes, panel.box, vertical=True)
        ticks_y = ticks_y or _gridlines(shapes, panel.box, vertical=False)
        # Axis offset labels such as "1e6" scale an axis; they are not tick labels.
        numeric = [text for text in texts if TICK_LABEL.fullmatch(text.value) and not OFFSET.fullmatch(text.value)]
        labels_x = sorted(
            (
                text
                for text in numeric
                if not text.rotated and bottom - 30 < text.y < bottom - 6 and left - 20 < text.x < right
            ),
            key=lambda text: text.x,
        )
        labels_y = sorted(
            (
                text
                for text in numeric
                if not text.rotated and left - 70 < text.x < left and bottom - 6 <= text.y < top + 8
            ),
            key=lambda text: text.y,
        )
        panel.x = _axis(ticks_x, labels_x, _offset(texts, panel, axis="x"))
        panel.y = _axis(ticks_y, labels_y, _offset(texts, panel, axis="y"))
        panel.titles = [
            text.value
            for text in texts
            if not text.rotated
            and left - 5 < text.x < right
            and top - 40 < text.y < top + 25
            and not re.fullmatch(r"[0-9.eE+\-\N{MINUS SIGN}]+", text.value)
        ]
    # Shared x axes (Hypercube) are labelled only on the bottom panel.
    for panel in panels:
        if panel.x is None:
            shared = [other.x for other in panels if other.x is not None and other.box[::2] == panel.box[::2]]
            panel.x = shared[0] if shared else None
    return panels


def legend_labels(texts: list[Text], shapes: list[Shape]) -> dict[str, str]:
    """Legend swatches (short horizontal strokes or small patches) named by the text to their right."""
    labels: dict[str, str] = {}
    for shape in shapes:
        points = shape.points
        if shape.clip is not None or len(points) < 2:
            continue
        ys = [point[1] for point in points]
        xs = [point[0] for point in points]
        if max(ys) - min(ys) > 12 or not 10 < max(xs) - min(xs) < 40:
            continue
        colour = shape.stroke if shape.operator == "S" else shape.fill
        if _is_grey(colour):
            continue
        centre_y = (max(ys) + min(ys)) / 2
        candidates = [text for text in texts if 0 < text.x - max(xs) < 20 and abs(text.y + 3 - centre_y) < 6]
        if candidates:
            labels.setdefault(_key(colour), min(candidates, key=lambda text: text.x).value)
    return labels


@dataclass(frozen=True, slots=True)
class TableRow:
    table: str
    domain: str
    model: str
    updates_mean: float
    updates_se: float
    loss_mean: float
    loss_se: float


def parse_table(path: Path, name: str) -> list[TableRow]:
    """Rows of upstream's `calculate_final_stats` LaTeX: `Group & (m \\pm s) \\times 10^{e} & ...`."""
    cell = re.compile(r"\(\s*([0-9.]+)\s*\\pm\s*([0-9.]+)\s*\)\s*\\times\s*10\^\{?(-?\d+)\}?")
    rows: list[TableRow] = []
    for line in path.read_text().splitlines():
        if "&" not in line or "\\pm" not in line:
            continue
        group, updates_text, loss_text = (column.strip() for column in line.rstrip("\\ ").split("&")[:3])
        group_match = re.fullmatch(r"(Mastermind-\d{3})\s+(.+)", group)
        updates, loss = cell.search(updates_text), cell.search(loss_text)
        if group_match is None or updates is None or loss is None:
            raise ValueError(f"{path}: unexpected table row {line!r}")
        rows.append(
            TableRow(
                name,
                group_match[1],
                group_match[2],
                float(updates[1]) * 10 ** int(updates[3]),
                float(updates[2]) * 10 ** int(updates[3]),
                float(loss[1]) * 10 ** int(loss[3]),
                float(loss[2]) * 10 ** int(loss[3]),
            )
        )
    return rows


def self_test() -> None:
    """Render synthetic plots in upstream's style and check that extraction recovers the data."""
    import tempfile

    import matplotlib as mpl

    mpl.use("pdf")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "curves.pdf"
        x = np.linspace(0, 300_000, 400)
        mean = 0.02 * np.exp(-x / 80_000) + 0.002
        error = 0.002 + 0.001 * rng.random(400)
        plt.style.use("seaborn-v0_8-whitegrid")
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.plot(x, mean, label="Normalised IS")
        axis.fill_between(x, mean - error, mean + error, alpha=0.2)
        axis.plot(x, 2 * mean, label="Without IS")
        axis.fill_between(x, 2 * mean - error, 2 * mean + error, alpha=0.2)
        axis.set_xlabel("Training updates")
        axis.set_ylabel("MSE")
        axis.legend()
        figure.savefig(path)
        plt.close(figure)
        rows = extract_curves(path, "synthetic")
        for label, factor in (("Normalised IS", 1.0), ("Without IS", 2.0)):
            got = [row for row in rows if row.series == label and row.kind == "mean"]
            gx = np.asarray([row.x for row in got])
            gy = np.asarray([row.y for row in got])
            np.testing.assert_allclose(gy, np.interp(gx, x, factor * mean), rtol=2e-3, atol=2e-5)
            high = [row for row in rows if row.series == label and row.kind == "high"]
            hx = np.asarray([row.x for row in high])
            np.testing.assert_allclose(
                [row.y for row in high], np.interp(hx, x, factor * mean + error), rtol=2e-3, atol=2e-5
            )
        log_path = Path(directory) / "log.pdf"
        figure, axis = plt.subplots(figsize=(6, 2))
        axis.plot([1, 10, 100, 1000], [3, 30, 300, 3000], color="tab:red", label="Line")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(1, 11_000)
        axis.legend()
        figure.savefig(log_path)
        plt.close(figure)
        line = [row for row in extract_curves(log_path, "log") if row.kind == "mean"]
        np.testing.assert_allclose([row.y / row.x for row in line], 3.0, rtol=2e-3)
    print("self-test passed: linear and log axes, bands and legend labels recovered")


def _shapes(operations: list[tuple[list[object], bytes]]) -> list[Shape]:
    page = _Painter()
    for operands, operator in operations:
        values = [float(value) for value in operands if isinstance(value, int | float)]
        page.apply(operator.decode(), values)
    return page.shapes


class _Painter:
    """A minimal PDF graphics-state machine: transforms, colours, clips and painted paths."""

    PAINT = frozenset({"S", "s", "f", "F", "f*", "B", "B*", "b", "b*"})

    def __init__(self) -> None:
        self.shapes: list[Shape] = []
        self.stack: list[tuple[list[float], Colour, Colour, Box | None]] = []
        self.ctm: list[float] = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        self.stroke: Colour = (0.0, 0.0, 0.0)
        self.fill: Colour = (0.0, 0.0, 0.0)
        self.clip: Box | None = None
        self.subpaths: list[list[Point]] = []
        self.pending_clip = False

    def apply(self, name: str, values: list[float]) -> None:
        if name in ("q", "Q", "cm"):
            self._state(name, values)
        elif name in ("RG", "G", "rg", "g"):
            self._colour(name, values)
        elif name in ("m", "l", "c", "re"):
            self._path(name, values)
        elif name in ("W", "W*"):
            self.pending_clip = True
        elif name == "n" or name in self.PAINT:
            self._finish(name)

    def _state(self, name: str, values: list[float]) -> None:
        if name == "q":
            self.stack.append((list(self.ctm), self.stroke, self.fill, self.clip))
        elif name == "Q" and self.stack:
            self.ctm, self.stroke, self.fill, self.clip = self.stack.pop()
        elif name == "cm" and len(values) == 6:
            a, b, c, d, e, f = values
            m = self.ctm
            self.ctm = [
                a * m[0] + b * m[2],
                a * m[1] + b * m[3],
                c * m[0] + d * m[2],
                c * m[1] + d * m[3],
                e * m[0] + f * m[2] + m[4],
                e * m[1] + f * m[3] + m[5],
            ]

    def _colour(self, name: str, values: list[float]) -> None:
        colour = (values[0], values[1], values[2]) if len(values) == 3 else (values[0],) * 3
        if name in ("RG", "G"):
            self.stroke = (colour[0], colour[1], colour[2])
        else:
            self.fill = (colour[0], colour[1], colour[2])

    def _path(self, name: str, values: list[float]) -> None:
        if name == "m" and len(values) == 2:
            self.subpaths.append([self._transform(values[0], values[1])])
        elif name == "l" and len(values) == 2 and self.subpaths:
            self.subpaths[-1].append(self._transform(values[0], values[1]))
        elif name == "c" and len(values) == 6 and self.subpaths:
            self.subpaths[-1].append(self._transform(values[4], values[5]))
        elif name == "re" and len(values) == 4:
            x, y, width, height = values
            corners = ((x, y), (x + width, y), (x + width, y + height), (x, y + height))
            self.subpaths.append([self._transform(cx, cy) for cx, cy in corners])

    def _finish(self, name: str) -> None:
        points = [point for subpath in self.subpaths for point in subpath]
        if self.pending_clip and points:
            xs, ys = [point[0] for point in points], [point[1] for point in points]
            self.clip = (round(min(xs), 2), round(min(ys), 2), round(max(xs), 2), round(max(ys), 2))
        elif name != "n" and points:
            subpaths = tuple(tuple(subpath) for subpath in self.subpaths)
            self.shapes.append(Shape(name.rstrip("*"), self.stroke, self.fill, subpaths, self.clip))
        self.subpaths, self.pending_clip = [], False

    def _transform(self, x: float, y: float) -> Point:
        m: list[float] = self.ctm
        return (float(m[0] * x + m[2] * y + m[4]), float(m[1] * x + m[3] * y + m[5]))


def _tick(shape: Shape, *, vertical: bool) -> bool:
    points = shape.points
    if shape.operator not in ("B", "S") or len(points) != 2 or not _is_black(shape):
        return False
    (x0, y0), (x1, y1) = points
    length = abs(y1 - y0) if vertical else abs(x1 - x0)
    straight = abs(x1 - x0) < 0.01 if vertical else abs(y1 - y0) < 0.01
    # Tick marks are a few points long; axes spines span the panel.
    return straight and TICK_MIN_LENGTH <= length <= TICK_MAX_LENGTH


def _gridlines(shapes: list[Shape], box: Box, *, vertical: bool) -> list[float]:
    left, bottom, right, top = box
    positions: set[float] = set()
    for shape in shapes:
        points = shape.points
        if (
            shape.clip != box
            or shape.operator != "S"
            or len(points) != 2
            or not _is_grey(shape.stroke)
            or max(shape.stroke) < 0.3
        ):
            continue
        (x0, y0), (x1, y1) = points
        if vertical and abs(x1 - x0) < 0.01 and abs(abs(y1 - y0) - (top - bottom)) < 1:
            positions.add(round(x0, 2))
        if not vertical and abs(y1 - y0) < 0.01 and abs(abs(x1 - x0) - (right - left)) < 1:
            positions.add(round(y0, 2))
    return sorted(positions)


def _axis(ticks: list[float], labels: list[Text], scale: float) -> Axis | None:
    """The map whose label reading fits the tick positions: plain numbers (linear) or powers of ten (log)."""
    plain = _plain(labels)
    if plain is not None and len(plain) == len(ticks) >= 2:
        fitted = _fit(ticks, [value * scale for value in plain], log=False)
        if fitted is not None:
            return fitted
    powers = _powers(labels)
    if powers is not None and len(powers) == len(ticks) >= 2:
        return _fit(ticks, [10.0**exponent for exponent in powers], log=True)
    return None


def _fit(ticks: list[float], values: list[float], *, log: bool) -> Axis | None:
    coordinates = np.asarray(ticks, np.float64)
    data = np.log10(np.asarray(values, np.float64)) if log else np.asarray(values, np.float64)
    slope, intercept = np.polyfit(coordinates, data, 1)
    error = float(np.max(np.abs(slope * coordinates + intercept - data)))
    if error > 1e-3 * max(float(np.ptp(data)), 1e-12):
        return None
    return Axis(float(slope), float(intercept), log=log)


def _plain(labels: list[Text]) -> list[float] | None:
    """Labels as numbers in axis order; restores minus signs that text extraction drops.

    Matplotlib draws negative ticks with U+2212, which upstream's embedded fonts lose
    in extraction: "-2, -1, 0, 1" reads as "2, 1, 0, 1". Values must increase along
    the axis, so labels before a zero that decrease towards it are negative.
    """
    try:
        values = [float(label.value.replace("\N{MINUS SIGN}", "-")) for label in labels]
    except ValueError:
        return None
    if values and all(later > earlier for earlier, later in pairwise(values)):
        return values
    if 0.0 in values:
        zero = values.index(0.0)
        repaired = [-abs(value) for value in values[:zero]] + values[zero:]
        if all(later > earlier for earlier, later in pairwise(repaired)):
            return repaired
    return values


def _powers(labels: list[Text]) -> list[int] | None:
    """Mathtext `10^k` labels, extracted either as "10k" or as "10" followed by a raised "k"."""
    exponents: list[int] = []
    index = 0
    while index < len(labels):
        text = labels[index].value.replace("\N{MINUS SIGN}", "-")
        follower = labels[index + 1] if index + 1 < len(labels) else None
        # The exponent run starts at (or just above and right of) the base's origin.
        if (
            text == "10"
            and follower is not None
            and re.fullmatch(r"-?\d+", follower.value.replace("\N{MINUS SIGN}", "-"))
            and abs(follower.x - labels[index].x) < 15
            and -1 < follower.y - labels[index].y < 8
        ):
            exponents.append(int(follower.value.replace("\N{MINUS SIGN}", "-")))
            index += 2
            continue
        match = re.fullmatch(r"10(-?\d+)", text)
        if match is None:
            return None
        exponents.append(int(match[1]))
        index += 1
    return exponents


def _offset(texts: list[Text], panel: Panel, *, axis: str) -> float:
    left, bottom, right, top = panel.box
    for text in texts:
        match = OFFSET.fullmatch(text.value)
        if not match:
            continue
        near_x = axis == "x" and right - 40 < text.x < right + 10 and bottom - 40 < text.y < bottom
        near_y = axis == "y" and left - 10 < text.x < left + 40 and top < text.y < top + 20
        if near_x or near_y:
            return 10.0 ** int(match[1])
    return 1.0


def _band_edges(points: tuple[Point, ...]) -> tuple[list[Point], list[Point]]:
    """fill_between polygons run forward along one edge and back along the other."""
    xs = [point[0] for point in points]
    turn = int(np.argmax(xs))
    first, second = list(points[: turn + 1]), list(reversed(points[turn + 1 :]))
    first_mean = np.mean([point[1] for point in first]) if first else 0.0
    second_mean = np.mean([point[1] for point in second]) if second else 0.0
    return (first, second) if first_mean >= second_mean else (second, first)


def _is_grey(colour: Colour) -> bool:
    return max(colour) - min(colour) < GREY_TOLERANCE


def _is_black(shape: Shape) -> bool:
    return max(shape.stroke) < 0.05


def _key(colour: Colour) -> str:
    return ",".join(f"{channel:.3f}" for channel in colour)


def _name(colour: Colour) -> str:
    return "#" + "".join(f"{round(channel * 255):02x}" for channel in colour)


def _fields(row: CurveRow | BarRow | TableRow) -> list[object]:
    return [getattr(row, name) for name in row.__slots__]


def _write(path: Path, columns: tuple[str, ...], rows: list[list[object]]) -> None:
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(columns)
        writer.writerows(rows)


if __name__ == "__main__":
    main()
