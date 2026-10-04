"""Paper tables from completed runs: the source's calculate_final_stats and explanation boards.

Large-scale tables (paper Tables 1, 3 and 4) summarise each group of runs by its last
logged epoch, as `fastsverl.utils.calculate_final_stats` does over W&B histories:

- updates = that epoch's logging step x batches per epoch, so the patience tail counts;
- final loss = that epoch's training loss (the source's `*_epoch_loss` tag);
- standard errors are std(ddof=1) / sqrt(n).

The source's table scripts pass n_batches = (0.8 * 10_000) // 128 = 62 for every row.
Explanation boards (paper Table 2) follow `plot_explanations.py`: one output's normalised
Shapley values on the board grid, last guess row first, scaled by the largest magnitude.
"""

import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float

SOURCE_BATCHES = (0.8 * 10_000) // 128
"""62.0: the n_batches the source's large-scale table scripts pass."""

type BoardValues = Float[np.ndarray, "guesses columns"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type FeatureValues = Float[np.ndarray, "features"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.
type RunValues = Float[np.ndarray, "runs"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.


@dataclass(frozen=True, slots=True)
class FinalEpoch:
    """A run's last logged epoch: the source's logging step and that epoch's training loss."""

    step: int
    train_loss: float


@dataclass(frozen=True, slots=True)
class GroupStats:
    """One table row: mean and standard error of updates and final loss over a group's runs."""

    label: str
    updates_mean: float
    updates_se: float
    loss_mean: float
    loss_se: float
    runs: int


def group_stats(label: str, runs: Sequence[Path], n_batches: float = SOURCE_BATCHES) -> GroupStats:
    """Source calculate_final_stats for one group of completed run folders."""
    if len(runs) < 2:
        raise ValueError("standard errors need at least two runs")
    finals = [final_epoch(run) for run in runs]
    updates = np.asarray([final.step for final in finals], np.float64) * n_batches
    losses = np.asarray([final.train_loss for final in finals], np.float64)
    return GroupStats(
        label, float(updates.mean()), standard_error(updates), float(losses.mean()), standard_error(losses), len(runs)
    )


def latex_table(groups: Sequence[GroupStats]) -> str:
    """The source's `DataFrame.to_latex(index=False, escape=False)` layout."""
    rows = [("Group", "Updates", "Final Loss")]
    rows += [
        (
            group.label,
            format_scientific(group.updates_mean, group.updates_se),
            format_scientific(group.loss_mean, group.loss_se),
        )
        for group in groups
    ]
    lines = [" & ".join(row) + " \\\\" for row in rows]
    return "\n".join((
        "\\begin{tabular}{lll}",
        "\\toprule",
        lines[0],
        "\\midrule",
        *lines[1:],
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ))


def write_table_csv(
    path: Path, table: str, domains: Sequence[str], models: Sequence[str], groups: Sequence[GroupStats]
) -> None:
    """Rows in the schema of docs/data/paper--published-tables.csv."""
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("table", "domain", "model", "updates_mean", "updates_se", "loss_mean", "loss_se", "runs"))
        for domain, model, group in zip(domains, models, groups, strict=True):
            writer.writerow((
                table,
                domain,
                model,
                group.updates_mean,
                group.updates_se,
                group.loss_mean,
                group.loss_se,
                group.runs,
            ))


def board_values(contributions: FeatureValues, guesses: int) -> BoardValues:
    """Source plot_explanations: reshape to the board, last guess first, scaled into [-1, 1]."""
    values = np.flipud(np.reshape(np.asarray(contributions, np.float64), (guesses, -1)))
    if values.max() - values.min() != 0:
        values = values / max(abs(values.max()), abs(values.min()))
    return values


def format_scientific(mean: float, standard_error: float) -> str:
    """Source _format_scientific: `$(m \\pm s) \\times 10^{e}$` with e = floor(log10|mean|)."""
    exponent = 0 if mean == 0 else math.floor(math.log10(abs(mean)))
    scale = 10.0**exponent
    return f"$({mean / scale:.3f} \\pm {standard_error / scale:.3f}) \\times 10^{{{exponent}}}$"


def final_epoch(folder: Path) -> FinalEpoch:
    """The last row of a run's metrics.csv."""
    with (folder / "metrics.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"{folder} has no logged epochs")
    return FinalEpoch(int(rows[-1]["step"]), float(rows[-1]["train_loss"]))


def standard_error(values: RunValues) -> float:
    return float(np.std(values, ddof=1) / np.sqrt(values.shape[0]))
