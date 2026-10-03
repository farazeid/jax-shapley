"""Run the JAX explanation recipes with the comparison protocol and export CSVs.

The protocol matches run_upstream.py: Gridworld and Mastermind-222 use one agent
(seed 1), characteristic seeds 1..N, and Shapley seeds 1..N, all explaining the
first characteristic with and without the exact characteristic. Each Hypercube
seed trains its own agent, characteristic and Shapley model. Settings come only
from the shell recipes. The caller runs ./check.sh once and sets
JAX_SHAPLEY_GATES_PASSED=1, so individual recipes skip the gates.
"""

import argparse
import csv
import itertools
import json
import os
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- fixed local recipe argv.
import time
from dataclasses import dataclass
from pathlib import Path

from jax_shapley.explanation_io import sweep_folders

HYPERCUBES = ((3, 2), (3, 3), (2, 5))
KINDS = ("behaviour", "prediction")
RUN_COLUMNS = ("framework", "environment", "kind", "stage", "exact_char", "seed", "step", "train_loss", "exact_loss")
TIMING_COLUMNS = ("framework", "environment", "kind", "stage", "exact_char", "seed", "measure", "seconds", "epochs")


@dataclass(frozen=True, slots=True)
class Invocation:
    """One recipe call: its wall time and the explanation runs it created."""

    environment: str
    kind: str
    stage: str
    seed: int
    seconds: float
    runs: tuple[Path, ...]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/explanations-port"))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--cases", nargs="+", default=["gridworld", "mastermind222", "hypercube"])
    parser.add_argument("--runs", type=Path, default=Path("docs/data/explanations-port--jax-runs.csv"))
    parser.add_argument("--timing", type=Path, default=Path("docs/data/explanations-port--jax-timing.csv"))
    args = parser.parse_args()
    if os.environ.get("JAX_SHAPLEY_GATES_PASSED") != "1":
        raise SystemExit("run ./check.sh first and set JAX_SHAPLEY_GATES_PASSED=1")
    root = Path(args.root)
    invocations: list[Invocation] = []
    for case in args.cases:
        if case == "hypercube":
            invocations.extend(hypercube(root / "hypercube", args.seeds))
        else:
            invocations.extend(single_agent(root / case, case, args.seeds))
    export(invocations, Path(args.runs), Path(args.timing))


def single_agent(root: Path, case: str, seeds: int) -> list[Invocation]:
    agent = _invoke(["experiments/dqn/" + f"{case}.sh", "--seed", "1", "--output", str(root / "agent")], root / "agent")
    agent_run = agent.runs[0]
    invocations = [Invocation(case, "", "agent", 1, agent.seconds, agent.runs)]
    recipe = f"experiments/explanations/{case}.sh"
    sweep = ["--seeds", *(str(seed) for seed in range(1, seeds + 1))]
    for kind in KINDS:
        # One compiled sweep per stage: every seed trains in the same program.
        output = root / kind / "characteristic"
        command = [recipe, "characteristic", "--kind", kind, "--agent", str(agent_run), *sweep]
        result = _invoke([*command, "--output", str(output)], output)
        invocations.append(Invocation(case, kind, "characteristic", 0, result.seconds, result.runs))
        characteristic_root = result.runs[0].parent
        # Paper protocol (every seed explains the first characteristic), then the matched protocol
        # (naming the characteristic sweep pairs seed s with characteristic seed s).
        for label, characteristic in ((case, result.runs[0]), (f"{case}-matched", characteristic_root)):
            output = root / kind / f"{label}-shapley"
            command = [recipe, "shapley", "--agent", str(agent_run), "--characteristic", str(characteristic), *sweep]
            shapley = _invoke([*command, "--output", str(output)], output)
            invocations.append(Invocation(label, kind, "shapley", 0, shapley.seconds, shapley.runs))
    return invocations


def hypercube(root: Path, seeds: int) -> list[Invocation]:
    """Per kind and cube: every seed's agent, then one characteristic sweep and one Shapley sweep."""
    invocations: list[Invocation] = []
    for kind in KINDS:
        for side, dimension in HYPERCUBES:
            command = [
                "experiments/explanations/hypercube.sh",
                "--kind",
                kind,
                "--num-seeds",
                str(seeds),
                "--env.side-length",
                str(side),
                "--env.dimension",
                str(dimension),
                "--output-root",
                str(root),
            ]
            cube = root / kind / f"{side}x{dimension}"
            started = time.perf_counter()
            _run(command)
            runs = (*sweep_folders(_single(cube / "characteristic")), *sweep_folders(_single(cube / "shapley")))
            elapsed = time.perf_counter() - started
            invocations.append(Invocation(f"hypercube-{side}x{dimension}", kind, "combined", 0, elapsed, runs))
    return invocations


def export(invocations: list[Invocation], runs_path: Path, timing_path: Path) -> None:
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    with runs_path.open("w", newline="") as rows_file, timing_path.open("w", newline="") as timing_file:
        rows, timing = csv.writer(rows_file), csv.writer(timing_file)
        rows.writerow(RUN_COLUMNS)
        timing.writerow(TIMING_COLUMNS)
        for invocation in invocations:
            timing.writerow((
                "jax",
                invocation.environment,
                invocation.kind,
                invocation.stage,
                "",
                invocation.seed,
                "process",
                round(invocation.seconds, 3),
                "",
            ))
            if invocation.stage == "agent":
                continue
            for folder in invocation.runs:
                settings = json.loads((folder / "config.json").read_text())
                stage = settings["stage"]
                config = settings["config"]
                seed = int(config["seed"])
                exact_char = "" if stage == "characteristic" else ("with" if config["with_exact_char"] else "without")
                with (folder / "metrics.csv").open() as source:
                    metrics = list(csv.DictReader(source))
                for row in metrics:
                    rows.writerow((
                        "jax",
                        invocation.environment,
                        invocation.kind,
                        stage,
                        exact_char,
                        seed,
                        row["step"],
                        row["train_loss"],
                        row["exact_loss"],
                    ))
                seconds = [float(row["seconds"]) for row in metrics]
                timing.writerow(_epoch_median("jax", invocation, stage, exact_char, seed, seconds))
                timing.writerow((
                    "jax",
                    invocation.environment,
                    invocation.kind,
                    stage,
                    exact_char,
                    seed,
                    "loop",
                    round(seconds[-1] - seconds[0], 6) if len(seconds) > 1 else 0.0,
                    len(seconds),
                ))


def _epoch_median(
    framework: str, invocation: Invocation, stage: str, exact_char: str, seed: int, seconds: list[float]
) -> tuple[object, ...]:
    """Median epoch duration: steady-state compute, robust to checkpoint epochs."""
    gaps = sorted(later - earlier for earlier, later in itertools.pairwise(seconds))
    median = gaps[len(gaps) // 2] if gaps else 0.0
    return (
        framework,
        invocation.environment,
        invocation.kind,
        stage,
        exact_char,
        seed,
        "epoch_median",
        round(median, 6),
        len(seconds),
    )


def _invoke(command: list[str], output: Path) -> Invocation:
    """Run a recipe; its runs are the seed folders of every sweep (or run folder) it created, in seed order."""
    started = time.perf_counter()
    _run(command)
    runs: list[Path] = []
    for created in sorted(path for path in output.iterdir() if path.is_dir()):
        runs.extend(sweep_folders(created))
    return Invocation("", "", "", 0, time.perf_counter() - started, tuple(runs))


def _run(command: list[str]) -> None:
    bash = shutil.which("bash")
    if bash is None:
        raise RuntimeError("bash is required")
    print("[jax]", " ".join(command), flush=True)
    subprocess.run([bash, *command], check=True)  # ruff: ignore[subprocess-without-shell-equals-true] -- fixed recipe argv.


def _single(folder: Path) -> Path:
    runs = [path for path in folder.iterdir() if path.is_dir()]
    if len(runs) != 1:
        raise RuntimeError(f"expected one run in {folder}")
    return runs[0]


if __name__ == "__main__":
    main()
