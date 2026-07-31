"""Plot summarized environment-throughput measurements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tyro


@dataclass(frozen=True)
class Args:
    """Input, outputs, and presentation settings for the throughput figure."""

    summary: Path = Path("docs/data/envs-port--throughput--summary.csv")
    svg_output: Path = Path("docs/figures/envs-port--throughput.svg")
    png_output: Path = Path("docs/figures/envs-port--throughput.png")
    figure_width: float = 13.0
    figure_height: float = 9.0
    png_dpi: int = 180
    title_y: float = 0.99
    legend_y: float = 0.945
    subplot_top: float = 0.88
    figure_title: str = "FastSVERL environment throughput: original vs end-to-end JAX"
    panel_environments: tuple[str, ...] = (
        "gridworld",
        "hypercube_2x2",
        "hypercube_6x5",
        "mastermind_222",
    )
    panel_titles: tuple[str, ...] = (
        "Gridworld",
        "Hypercube 2x2",
        "Hypercube 6x5",
        "Mastermind-222",
    )
    backend_keys: tuple[str, ...] = ("gymnasium_cpu", "jax_cpu", "jax_gpu")
    backend_labels: tuple[str, ...] = ("Gymnasium CPU", "JAX CPU", "JAX GPU")
    backend_colors: tuple[str, ...] = ("#555555", "#2070b4", "#d1495b")
    backend_markers: tuple[str, ...] = ("o", "s", "^")
    confidence_band_alpha: float = 0.14
    confidence_note_y: float = 0.895
    original_hypercube_6x5_cap: int = 16


def main() -> None:
    args = tyro.cli(Args)
    summary = pd.read_csv(args.summary)
    plot(args, summary)
    print(f"Wrote {args.svg_output} and {args.png_output}")


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
                line["mean"],
                label=label,
                color=color,
                marker=marker,
                linewidth=2,
                markersize=4,
            )
            axis.fill_between(
                x,
                line["mean_ci95_lower"],
                line["mean_ci95_upper"],
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
            axis.axvline(
                args.original_hypercube_6x5_cap,
                color="#777777",
                linestyle=":",
                linewidth=1,
            )
            axis.text(
                19,
                axis.get_ylim()[0] * 1.5,
                f"Original capped at {args.original_hypercube_6x5_cap}",
                fontsize=8,
                color="#555555",
            )
    for axis in axes[-1]:
        axis.set_xlabel("Number of environments")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle(args.figure_title, y=args.title_y)
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
        "Shading: 95% Student's t-distribution confidence interval for mean",
        horizontalalignment="center",
        fontsize=9,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0, 1, args.subplot_top))
    figure.savefig(args.svg_output)
    figure.savefig(args.png_output, dpi=args.png_dpi)
    plt.close(figure)
    normalize_svg(args.svg_output)


def normalize_svg(path: Path) -> None:
    lines = path.read_text().splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n")


if __name__ == "__main__":
    main()
