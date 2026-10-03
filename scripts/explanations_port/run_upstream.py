"""Run the unchanged FastSVERL experiment scripts for the explanation-port comparison.

Run through scripts/explanations_port/run.sh, which supplies an isolated PyTorch
environment with the pinned checkout installed in editable mode. Each upstream run
script executes unchanged in a subprocess. A two-line harness first imports
`fastsverl.envs`: the committed run scripts never import it, so without it gym
cannot find the registered environments. Arguments are the experiment drivers'
own dataclass defaults with the paper's per-domain values. Weights & Biases
tracking is disabled, the seed count is reduced, and explanation saving is off.
TensorBoard scalars are exported to CSV.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import itertools
import json
import runpy
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import] -- fixed argv to the verified pinned checkout.
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

REFERENCE_COMMIT = "25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b"
HARNESS = "import runpy, sys\nimport fastsverl.envs\nsys.argv = sys.argv[1:]\nrunpy.run_path(sys.argv[0], run_name='__main__')"
SMALL_ARCH = {
    "input": ["Linear", [None, 64]],
    "input_activation": ["ReLU", []],
    "hidden1": ["Linear", [64, 64]],
    "hidden1_activation": ["ReLU", []],
    "output": ["Linear", [64, None]],
}
HYPERCUBES = ((3, 2, 50_000), (3, 3, 50_000), (2, 5, 50_000))
CLASSES = {"behaviour": ("Policy", "policy"), "prediction": ("Value", "value")}
RUN_COLUMNS = ("framework", "environment", "kind", "stage", "exact_char", "seed", "step", "train_loss", "exact_loss")
# Outcome runs with a non-terminating conditioned policy never finish upstream; see outcome().
OUTCOME_TIMEOUT = 3600.0
TIMING_COLUMNS = ("framework", "environment", "kind", "stage", "exact_char", "seed", "measure", "seconds", "epochs")


@dataclass(frozen=True, slots=True)
class Run:
    environment: str
    kind: str
    stage: str
    exact_char: str
    seed: int
    folder: Path
    wall_seconds: float


class Scalar(Protocol):
    step: int
    value: float
    wall_time: float


class Accumulator(Protocol):
    def Reload(self) -> object: ...  # ruff: ignore[invalid-function-name] -- TensorBoard API.
    def Tags(self) -> dict[str, list[str]]: ...  # ruff: ignore[invalid-function-name] -- TensorBoard API.
    def Scalars(self, tag: str) -> list[Scalar]: ...  # ruff: ignore[invalid-function-name] -- TensorBoard API.


class EventAccumulatorModule(Protocol):
    def EventAccumulator(self, path: str, size_guidance: dict[str, int]) -> Accumulator: ...  # ruff: ignore[invalid-function-name] -- TensorBoard API.


class Exporter:
    """Appends each finished run's losses and timing, so an interrupted sweep keeps completed runs."""

    def __init__(self, runs_path: Path, timing_path: Path) -> None:
        self.runs_path, self.timing_path = runs_path, timing_path
        self.module = cast(
            EventAccumulatorModule, importlib.import_module("tensorboard.backend.event_processing.event_accumulator")
        )
        runs_path.parent.mkdir(parents=True, exist_ok=True)
        for path, columns in ((runs_path, RUN_COLUMNS), (timing_path, TIMING_COLUMNS)):
            with path.open("w", newline="") as output:
                csv.writer(output).writerow(columns)

    def failure(self, environment: str, kind: str, stage: str, exact_char: str, seed: int, reason: str) -> None:
        """A run that could not complete, kept in the timing CSV so the comparison can account for it."""
        print(f"[upstream] FAILED {environment} {kind} {stage} {exact_char} seed={seed}: {reason}", flush=True)
        with self.timing_path.open("a", newline="") as timing_file:
            csv.writer(timing_file).writerow(("upstream", environment, kind, stage, exact_char, seed, reason, "", ""))

    def record(self, run: Run) -> None:
        """Long-format exact/train losses per logged step, and the run's timing."""
        with self.runs_path.open("a", newline="") as rows_file, self.timing_path.open("a", newline="") as timing_file:
            rows, timing = csv.writer(rows_file), csv.writer(timing_file)
            timing.writerow((
                "upstream",
                run.environment,
                run.kind,
                run.stage,
                run.exact_char,
                run.seed,
                "process",
                round(run.wall_seconds, 3),
                "",
            ))
            accumulator = self.module.EventAccumulator(str(run.folder), {"scalars": 0})
            accumulator.Reload()
            tags = accumulator.Tags()["scalars"]
            stages = ("Characteristic", "Shapley") if run.stage == "combined" else ("",)
            if run.stage == "agent":
                return
            # train_V_on_policy logs its loss every 100 global steps, not per epoch.
            train_suffix = "OnPolicy_loss" if run.kind == "on_policy" and run.stage == "outcome" else "_epoch_loss"
            for stage in stages:
                label = run.stage if not stage else ("characteristic" if stage == "Characteristic" else "shapley")
                exact = _series(accumulator, tags, stage, "_exact_loss")
                train = _series(accumulator, tags, stage, train_suffix)
                for step in sorted(exact.keys() | train.keys()):
                    rows.writerow((
                        "upstream",
                        run.environment,
                        run.kind,
                        label,
                        run.exact_char,
                        run.seed,
                        step,
                        train.get(step, ("", 0.0))[0],
                        exact.get(step, ("", 0.0))[0],
                    ))
                times = sorted(time for _, time in train.values())
                gaps = sorted(later - earlier for earlier, later in itertools.pairwise(times))
                timing.writerow((
                    "upstream",
                    run.environment,
                    run.kind,
                    label,
                    run.exact_char,
                    run.seed,
                    "epoch_median",
                    round(gaps[len(gaps) // 2], 6) if gaps else 0.0,
                    len(times),
                ))
                timing.writerow((
                    "upstream",
                    run.environment,
                    run.kind,
                    label,
                    run.exact_char,
                    run.seed,
                    "loop",
                    round(times[-1] - times[0], 6) if len(times) > 1 else 0.0,
                    len(times),
                ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["gridworld", "mastermind222", "hypercube", "gridworld-outcome", "mastermind222-outcome"],
    )
    parser.add_argument("--runs", type=Path, default=Path("docs/data/explanations-port--upstream-runs.csv"))
    parser.add_argument("--timing", type=Path, default=Path("docs/data/explanations-port--upstream-timing.csv"))
    args = parser.parse_args()
    root = Path(args.source_root).resolve()
    _verify(root)
    exporter = Exporter(Path(args.runs), Path(args.timing))
    for case in args.cases:
        if case == "hypercube":
            hypercube(root, args.seeds, exporter)
        elif case.endswith("-outcome"):
            outcome(root, case.removesuffix("-outcome"), args.seeds, exporter)
        else:
            single_agent(root, case, args.seeds, exporter)


def single_agent(root: Path, case: str, seeds: int, exporter: Exporter) -> None:
    """One agent; characteristic seeds; Shapley seeds under the paper and matched protocols."""
    experiments = root / "experiments/fastsverl"
    suffix = "gwb" if case == "gridworld" else "mastermind"
    driver = _driver(experiments / f"agents/dqn_{suffix}.py")
    exp, agent, env = _instance(driver, "ExpArgs"), _instance(driver, "AgentArgs"), _instance(driver, "EnvArgs")
    if case == "mastermind222":
        agent.update(critic_arch=SMALL_ARCH, total_timesteps=100_000, buffer_size=100_000)
        env.update(code_size=2, num_guesses=2, num_pegs=2)
    agent_run = _launch(
        exporter, experiments / "agents", "run.py", 1, [exp, agent, env], ["DQN"], case, "", "agent", ""
    )
    for kind, (prefix, name) in CLASSES.items():
        char_driver = _driver(experiments / f"characteristics/dqn_{name}_{suffix}.py")
        char_exp, char_args = _instance(char_driver, "ExpArgs"), _instance(char_driver, "CharacteristicArgs")
        char_exp["agent_args_f"] = agent_run.folder.name
        if case == "mastermind222":
            arch = dict(SMALL_ARCH, output_activation=["Softmax", [-1]]) if kind == "behaviour" else SMALL_ARCH
            char_args.update(model_arch=arch, epochs=50 if kind == "behaviour" else 80, save_val=False)
        characteristics = [
            _launch(
                exporter,
                experiments / "characteristics",
                "run_policy_prediction.py",
                seed,
                [char_exp, char_args],
                ["DQN", f"{prefix}CharacteristicModel"],
                case,
                kind,
                "characteristic",
                "",
            )
            for seed in range(1, seeds + 1)
        ]
        shapley_driver = _driver(experiments / f"shapleys/dqn_{name}_{suffix}.py")
        shapley_exp, shapley_args = _instance(shapley_driver, "ExpArgs"), _instance(shapley_driver, "ShapleyArgs")
        shapley_exp.update(agent_args_f=agent_run.folder.name, char_args_f=characteristics[0].folder.name)
        shapley_args["save_explanations"] = False
        if case == "mastermind222":
            shapley_args.update(model_arch=SMALL_ARCH, epochs=50, save_val=False)
        # Paper protocol (every seed explains the first characteristic), then the matched protocol
        # (seed s explains characteristic s), which makes seeds independent replicates of both stages.
        for label, matched in ((case, False), (f"{case}-matched", True)):
            for seed in range(1, seeds + 1):
                shapley_exp["char_args_f"] = characteristics[seed - 1 if matched else 0].folder.name
                for with_exact in (True, False):
                    shapley_args["with_exact_char"] = with_exact
                    _launch(
                        exporter,
                        experiments / "shapleys",
                        "run_policy_prediction.py",
                        seed,
                        [shapley_exp, shapley_args],
                        ["DQN", f"{prefix}CharacteristicModel", f"{prefix}Shapley"],
                        label,
                        kind,
                        "shapley",
                        "with" if with_exact else "without",
                    )


def outcome(root: Path, case: str, seeds: int, exporter: Exporter) -> None:
    """Matched replicates: seed s trains agent s, behaviour characteristic s, outcome characteristics s, Shapley s.

    Outcome characteristics cover both variants, with and without the exact behaviour
    characteristic. Shapley models follow dqn_performance_*.py: the variant the driver
    explains (on-policy for Gridworld, off-policy for Mastermind) in its three groups.
    """
    experiments = root / "experiments/fastsverl"
    suffix = "gwb" if case == "gridworld" else "mastermind"
    label = f"{case}-outcome"
    agent_driver = _driver(experiments / f"agents/dqn_{suffix}.py")
    policy_driver = _driver(experiments / f"characteristics/dqn_policy_{suffix}.py")
    outcome_driver = _driver(experiments / f"characteristics/dqn_performance_{suffix}.py")
    shapley_driver = _driver(experiments / f"shapleys/dqn_performance_{suffix}.py")
    for seed in range(1, seeds + 1):
        exp, agent, env = (_instance(agent_driver, name) for name in ("ExpArgs", "AgentArgs", "EnvArgs"))
        if case == "mastermind222":
            agent.update(critic_arch=SMALL_ARCH, total_timesteps=100_000, buffer_size=100_000)
            env.update(code_size=2, num_guesses=2, num_pegs=2)
        agent_run = _launch(
            exporter, experiments / "agents", "run.py", seed, [exp, agent, env], ["DQN"], label, "", "agent", ""
        )
        policy_exp, policy_args = _instance(policy_driver, "ExpArgs"), _instance(policy_driver, "CharacteristicArgs")
        policy_exp["agent_args_f"] = agent_run.folder.name
        if case == "mastermind222":
            policy_args.update(
                model_arch=dict(SMALL_ARCH, output_activation=["Softmax", [-1]]), epochs=50, save_val=False
            )
        behaviour = _launch(
            exporter,
            experiments / "characteristics",
            "run_policy_prediction.py",
            seed,
            [policy_exp, policy_args],
            ["DQN", "PolicyCharacteristicModel"],
            label,
            "behaviour",
            "characteristic",
            "",
        )
        outcomes: dict[tuple[str, bool], Run] = {}
        for variant, cls in (
            ("on_policy", "PerformanceCharacteristicOnPolicy"),
            ("off_policy", "PerformanceCharacteristicOffPolicy"),
        ):
            for with_exact in (True, False):
                outcome_exp = _instance(outcome_driver, "ExpArgs")
                outcome_args = _instance(outcome_driver, "CharacteristicArgs")
                outcome_exp["agent_args_f"] = agent_run.folder.name
                outcome_args.update(policy_char_args_f=behaviour.folder.name, with_exact_char=with_exact)
                exact_char = "with" if with_exact else "without"
                try:
                    outcomes[variant, with_exact] = _launch(
                        exporter,
                        experiments / "characteristics",
                        "run_performance.py",
                        seed,
                        [outcome_exp, outcome_args],
                        ["DQN", cls],
                        label,
                        variant,
                        "outcome",
                        exact_char,
                        timeout=OUTCOME_TIMEOUT,
                    )
                except subprocess.TimeoutExpired:
                    # Upstream value iteration never converges when the conditioned policy can loop forever.
                    exporter.failure(label, variant, "outcome", exact_char, seed, "timeout")
        shapley_args = _instance(shapley_driver, "ShapleyArgs")
        explained = "off_policy" if shapley_args["off_policy_char"] else "on_policy"
        shapley_args["save_explanations"] = False
        # The driver's groups: w_policy and wo_policy explain exact outcome values; wo_perf the learnt one.
        for group, char_exact, shapley_exact in (
            ("w_policy", True, True),
            ("wo_policy", False, True),
            ("wo_perf", False, False),
        ):
            if (explained, char_exact) not in outcomes:
                exporter.failure(label, explained, "shapley", group, seed, "missing outcome characteristic")
                continue
            shapley_exp = _instance(shapley_driver, "ExpArgs")
            shapley_exp.update(
                agent_args_f=agent_run.folder.name, char_args_f=outcomes[explained, char_exact].folder.name
            )
            shapley_args["with_exact_char"] = shapley_exact
            _launch(
                exporter,
                experiments / "shapleys",
                "run_performance.py",
                seed,
                [shapley_exp, shapley_args],
                ["DQN"],
                label,
                explained,
                "shapley",
                group,
            )


def hypercube(root: Path, seeds: int, exporter: Exporter) -> None:
    """Paper protocol: every seed trains its own agent, characteristic and Shapley model in one process."""
    folder = root / "experiments/hypercube"
    for kind, (_, name) in CLASSES.items():
        driver = _driver(folder / f"hypercube_{name}.py")
        exp, agent, env = _instance(driver, "ExpArgs"), _instance(driver, "AgentArgs"), _instance(driver, "EnvArgs")
        char_args, shapley_args = _instance(driver, "CharacteristicArgs"), _instance(driver, "ShapleyArgs")
        for side, dimension, timesteps in HYPERCUBES:
            agent.update(total_timesteps=timesteps, buffer_size=timesteps, learning_starts=int(timesteps / 50))
            env.update(n=side, d=dimension)
            for seed in range(1, seeds + 1):
                _launch(
                    exporter,
                    folder,
                    "run.py",
                    seed,
                    [exp, agent, env, char_args, shapley_args],
                    [],
                    f"hypercube-{side}x{dimension}",
                    kind,
                    "combined",
                    "without",
                )


def _series(accumulator: Accumulator, tags: list[str], stage: str, suffix: str) -> dict[int, tuple[float, float]]:
    """Scalars whose tag ends with suffix (and names the stage, in combined runs), keyed by logged step."""
    selected = [tag for tag in tags if tag.startswith("model/") and tag.endswith(suffix) and stage in tag]
    if len(selected) > 1:
        raise ValueError(f"ambiguous tags {selected}")
    return {
        int(item.step): (float(item.value), float(item.wall_time))
        for tag in selected
        for item in accumulator.Scalars(tag)
    }


def _launch(
    exporter: Exporter,
    folder: Path,
    script: str,
    seed: int,
    arguments: list[dict[str, object]],
    classes: list[str],
    environment: str,
    kind: str,
    stage: str,
    exact_char: str,
    timeout: float | None = None,
) -> Run:
    runs = folder / "runs"
    before = set(runs.iterdir()) if runs.exists() else set()
    started = time.perf_counter()
    argv = [sys.executable, "-c", HARNESS, script, str(seed), *(json.dumps(value) for value in arguments), *classes]
    print(f"[upstream] {environment} {kind or 'agent'} {stage} {exact_char} seed={seed}", flush=True)
    subprocess.run(argv, cwd=folder, check=True, timeout=timeout)  # ruff: ignore[subprocess-without-shell-equals-true] -- fixed argv, verified checkout.
    created = sorted(set(runs.iterdir()) - before)
    if len(created) != 1:
        raise RuntimeError(f"expected one new run folder in {runs}, found {created}")
    run = Run(environment, kind, stage, exact_char, seed, created[0], time.perf_counter() - started)
    exporter.record(run)
    return run


def _driver(path: Path) -> dict[str, object]:
    """The experiment driver's module globals; its __main__ block (which launches runs) is not executed."""
    module = runpy.run_path(str(path), run_name="upstream_driver")
    return {name: value for name, value in module.items()}


def _instance(driver: dict[str, object], name: str) -> dict[str, object]:
    """A fresh default instance's __dict__, which the drivers serialize as JSON, with tracking off."""
    cls = driver[name]
    if not isinstance(cls, type):
        raise TypeError(f"{name} is not a class")
    values: dict[str, object] = {key: value for key, value in vars(cls()).items()}
    if name == "ExpArgs":
        values.update(track=False, num_runs=1)
    return values


def _verify(root: Path) -> None:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required")
    commit = subprocess.check_output([git, "-C", str(root), "rev-parse", "HEAD"], text=True).strip()  # ruff: ignore[subprocess-without-shell-equals-true] -- read-only fixed argv.
    status = subprocess.check_output([git, "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True)  # ruff: ignore[subprocess-without-shell-equals-true] -- read-only fixed argv.
    if commit != REFERENCE_COMMIT or status.strip():
        raise ValueError("expected a clean checkout of the pinned FastSVERL commit")


if __name__ == "__main__":
    main()
