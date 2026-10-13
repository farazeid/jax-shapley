"""Regenerate the paper's large-Mastermind tables (Tables 1-4) from tracked JAX runs.

Tables 1, 3 and 4 follow the source's table scripts (`plot_main.py`, `plot_app.py`) and
`fastsverl.utils.calculate_final_stats`: each group of runs is summarised by its last
logged epoch, with 62 batches per epoch:

- updates = that epoch's logging step x batches per epoch, so the patience tail counts;
- final loss = that epoch's training loss (the source's `*_epoch_loss` tag);
- standard errors are std(ddof=1) / sqrt(n).

Table 2 follows `plot_explanations.py` on the paper's Mastermind-463 board: the
behaviour Shapley values (Shapley seed 1) for the agent's greedy action, scaled by the
largest magnitude. The script checks that the action is the paper's CCCB.

Runs come only from the tracking backend: each size's experiment
(`mastermind{SIZE}_{behaviour,prediction,outcome}`) is read from its latest run group,
and Table 2 restores the agent, characteristic and Shapley checkpoints from that group's
artifacts (as a `table-2` run that records their use). Tables whose experiments are not
ported yet, or have no tracked runs, are reported missing.

Usage: uv run python scripts/paper_port/tables.py --track wandb|trackio [--output docs/data]
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from html import escape
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import trackio
import tyro
import wandb
from jaxtyping import Float

from new_jax_shapley import backends
from new_jax_shapley.algorithms import dqn
from new_jax_shapley.algorithms.dqn import TRACKING_FIELDS
from new_jax_shapley.characteristic_models import behaviour as characteristic
from new_jax_shapley.common import checkpoints
from new_jax_shapley.common.supervised_learning import seed_slice, stack_seeds
from new_jax_shapley.envs import Mastermind
from new_jax_shapley.envs.mastermind import MastermindState
from new_jax_shapley.experiment import ExperimentConfig, load_configs
from new_jax_shapley.shapley_models import behaviour as shapley
from new_jax_shapley.shapley_models.behaviour import ShapleyModel, explain, masked_null, masked_values

SIZES = ("443", "453", "463")  # code size, guesses, pegs
SOURCE_BATCHES = (0.8 * 10_000) // 128
"""62.0: the n_batches the source's large-scale table scripts pass."""
LETTERS = {-1: "", 1: "A", 2: "B", 3: "C"}
# Paper Table 2, first guess first: [Clue 1 (misplaced), pegs..., Clue 2 (exact)], A=1, B=2, C=3.
TABLE_2_BOARD = (
    (0, 1, 1, 3, 1, 1),
    (0, 3, 3, 3, 3, 3),
    (2, 3, 2, 3, 3, 2),
    (2, 2, 3, 3, 3, 2),
    (-1, -1, -1, -1, -1, -1),
    (-1, -1, -1, -1, -1, -1),
)
TABLE_2_ACTION = "CCCB"
TABLE_2_EXPERIMENT = "mastermind_large_behaviour__sweep"
TABLE_2_SHAPLEY_SEED = 1

type BoardValues = Float[np.ndarray, "guesses columns"]  # ruff: ignore[forward-annotation-syntax-error] -- jaxtyping shape grammar.
type FeatureValues = Float[np.ndarray, "features"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.
type RunValues = Float[np.ndarray, "runs"]  # ruff: ignore[undefined-name] -- jaxtyping symbolic axis.


@dataclass(frozen=True, slots=True)
class Args:
    track: backends.Track
    project: str = "jax-shapley"
    entity: str | None = None
    output: Path = Path("docs/data")
    tables: bool = True
    """Tables 1, 3 and 4."""
    table_2: bool = True


@dataclass(frozen=True, slots=True)
class GroupStats:
    """One table row: mean and standard error of updates and final loss over a group's runs."""

    label: str
    updates_mean: float
    updates_se: float
    loss_mean: float
    loss_se: float
    runs: int


@dataclass(frozen=True, slots=True)
class Explainer:
    """The explained agent's Q-network, its behaviour characteristic and Shapley model."""

    agent: dqn.QNetwork
    characteristic: characteristic.CharacteristicModel
    shapley: ShapleyModel
    env: Mastermind


@dataclass(frozen=True, slots=True)
class Board:
    """Table 2: the flipped board's cells, scaled Shapley values and the greedy guess."""

    cells: list[list[str]]
    values: BoardValues
    action: str
    action_row: int


class MissingRunsError(Exception):
    """The runs a table needs do not exist (yet)."""


def main(args: Args) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    configs = {name: config for name, (_, config) in load_configs(ExperimentConfig).items()}
    for table, kind, models in (
        ()
        if not args.tables
        else (
            ("table_1_behaviour", "behaviour", ("Characteristic", "Shapley")),
            ("table_3_prediction", "prediction", ("Characteristic", "Shapley")),
            ("table_4_outcome_offpolicy", "outcome", ("Shapley",)),
        )
    ):
        domains, labels, groups = [], [], []
        try:
            for size in SIZES:
                experiment = f"mastermind_large_{kind}__sweep"
                group = _group(args, configs, experiment, int(size[1]))
                tracked = backends.runs(args.track, project=args.project, entity=args.entity, group=group)
                for model in models:
                    domains.append(f"Mastermind-{size}")
                    labels.append(model)
                    label = f"Mastermind-{size} {model}"
                    # The characteristic sweep, or the experiment's first Shapley sweep.
                    shapley_config = configs[experiment].shapley[0]
                    exactness = "with-exact" if shapley_config.with_exact_char else "without-exact"
                    stage = f"characteristic--{kind}" if model == "Characteristic" else f"shapley--{kind}--{exactness}"
                    prefix = f"{experiment}--{stage}--"
                    runs = [run.rows for name, run in sorted(tracked.items()) if name.startswith(prefix)]
                    explained = {"behaviour": "Policy", "prediction": "Value", "outcome": "Performance"}[kind]
                    tag = f"model/{explained}{'CharacteristicModel' if model == 'Characteristic' else 'Shapley'}_epoch_loss"
                    # Source calculate_final_stats: each run's last logged epoch of the training loss.
                    finals = [
                        [(row["_step"], row[tag]) for row in rows if tag in row and "_step" in row] for rows in runs
                    ]
                    if len(finals) < 2 or any(not logged for logged in finals):
                        raise MissingRunsError(f"{label}: needs two or more runs that logged {tag}")
                    updates = np.asarray([logged[-1][0] for logged in finals], np.float64) * SOURCE_BATCHES
                    losses = np.asarray([logged[-1][1] for logged in finals], np.float64)
                    groups.append(
                        GroupStats(
                            label,
                            float(updates.mean()),
                            standard_error(updates),
                            float(losses.mean()),
                            standard_error(losses),
                            len(runs),
                        )
                    )
        except MissingRunsError as error:
            print(f"missing {table}: {error}")
            continue
        _write_table(args.output, table, domains, labels, groups)
    if args.table_2:
        try:
            explainer = load_explainer(args, configs)
        except MissingRunsError as error:
            print(f"missing table_2: {error}")
            return
        paper = explain_board(explainer, np.asarray(TABLE_2_BOARD, np.int32))
        boards, wins = greedy_boards(explainer, guesses_made=4)
        analogue = explain_board(explainer, boards[int(np.argmax(wins))])
        for name, board in (("paper-board", paper), ("analogue", analogue)):
            (args.output / f"paper--jax-table_2-{name}.html").write_text(board_html(board))
            _write_board_csv(args.output / f"paper--jax-table_2-{name}.csv", board)
        # Nullity: every non-terminal greedy state with 1 to num_guesses - 1 guesses on the board.
        reached = np.concatenate([greedy_boards(explainer, made)[0] for made in range(1, explainer.env.num_guesses)])
        empty, filled = nullity(explainer, reached)
        with (args.output / "paper--jax-table_2-nullity.csv").open("w", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(("states", "mean_abs_empty_rows", "mean_abs_filled_rows"))
            writer.writerow((len(reached), f"{empty:.6f}", f"{filled:.6f}"))
        print(f"Table 2: greedy guess on the paper's board {paper.action} (paper {TABLE_2_ACTION})")
        print(f"Nullity over {len(reached)} greedy states: mean |value| {empty:.4f} empty, {filled:.4f} filled")


def load_explainer(args: Args, configs: dict[str, ExperimentConfig]) -> Explainer:
    """The Mastermind-463 agent, behaviour characteristic and Shapley seed 1 behind Table 2."""
    group = _group(args, configs, TABLE_2_EXPERIMENT, len(TABLE_2_BOARD))
    # The sweep config leaves the guesses open: Mastermind-463, as its run script set them.
    sweep = configs[TABLE_2_EXPERIMENT]
    board = replace(sweep.algorithm.env, num_guesses=len(TABLE_2_BOARD))
    config = replace(
        sweep,
        algorithm=replace(sweep.algorithm, env=board),
        characteristic=replace(sweep.characteristic, env=board),
        shapley=tuple(replace(stage, env=board) for stage in sweep.shapley),
    )
    env = config.algorithm.env
    if not isinstance(env, Mastermind) or env.num_guesses != len(TABLE_2_BOARD):
        raise ValueError("Table 2 explains a Mastermind-463 agent")
    algorithm, explained, stage = config.algorithm, config.characteristic, config.shapley[0]
    exactness = "with-exact" if stage.with_exact_char else "without-exact"
    # The settings each stage's checkpoint was saved with, as the stages write them.
    algorithm_settings = {n: v for n, v in asdict(algorithm).items() if n not in TRACKING_FIELDS}
    explained_settings = {n: v for n, v in asdict(explained).items() if n not in TRACKING_FIELDS}
    shapley_settings = {n: v for n, v in asdict(stage).items() if n not in TRACKING_FIELDS}
    with backends.session(args.track, args.entity) as root:
        tracking = wandb if args.track == "wandb" else trackio
        run = tracking.init(
            project=args.project,
            name=f"{TABLE_2_EXPERIMENT}--report--table-2",
            group=group,
            config={"report": "table_2"},
        )
        try:
            directory = run.use_artifact(f"{group}--algorithm--dqn:latest").download(root=str(root / "agent"))
            agent_state = checkpoints.load(
                Path(directory),
                stack_seeds([dqn.initialize(algorithm, jax.random.key(0))] * len(algorithm.seeds)),
                {"stage": "algorithm", "env": env.name, "config": algorithm_settings},
            )
            reference = f"{group}--characteristic--{explained.kind}:latest"
            directory = run.use_artifact(reference).download(root=str(root / "characteristic"))
            characteristics = checkpoints.load(
                Path(directory),
                characteristic.initialize(explained, characteristic.head),
                {"stage": "characteristic", "env": env.name, "config": explained_settings},
            )
            reference = f"{group}--shapley--{explained.kind}--{exactness}:latest"
            directory = run.use_artifact(reference).download(root=str(root / "shapley"))
            shapleys = checkpoints.load(
                Path(directory),
                shapley.initialize(stage, explained, characteristic.head),
                {"stage": "shapley", "env": env.name, "explained": explained_settings, "config": shapley_settings},
            )
        finally:
            run.finish()
    chosen = seed_slice(characteristics, explained.seeds.index(config.explained_seed))
    model = seed_slice(shapleys, stage.seeds.index(TABLE_2_SHAPLEY_SEED)).model
    agents = dqn.trained_agent(algorithm, agent_state)
    agent = seed_slice(agents, explained.seeds.index(config.explained_seed) if len(algorithm.seeds) > 1 else 0)
    return Explainer(agent.model, chosen.model, model, env)


def explain_board(explainer: Explainer, board: np.ndarray) -> Board:
    """Source plot_explanations for one board: the greedy action's scaled, flipped Shapley values."""
    observation = jnp.asarray(board, jnp.int32).reshape(1, -1)
    action = int(jnp.argmax(explainer.agent(observation).q_values[0]))
    contributions = explain(masked_values, masked_null, explainer.shapley, explainer.characteristic, observation)
    env = explainer.env
    values = board_values(np.asarray(contributions[0, :, action]), env.num_guesses)
    guess = "".join(LETTERS[int(peg)] for peg in env.decode_action(action))
    cells = [
        [
            ("" if value == -1 else str(value)) if column in (0, len(row) - 1) else LETTERS[int(value)]
            for column, value in enumerate(row)
        ]
        for row in np.flipud(board).tolist()
    ]
    empty_row = next(index for index, row in enumerate(board.tolist()) if row[0] == -1)
    return Board(cells, values, guess, env.num_guesses - 1 - empty_row)


def greedy_boards(explainer: Explainer, guesses_made: int) -> tuple[np.ndarray, np.ndarray]:
    """Every code's board after the greedy agent's first guesses, when none of them won.

    The greedy policy is deterministic, so these are all the states it reaches with
    `guesses_made` guesses on the board. `wins` marks boards whose next greedy guess wins.
    """
    env = explainer.env
    params = env.default_params
    codes = jax.vmap(env.decode_action)(jnp.arange(env.num_actions))
    count = codes.shape[0]
    states = MastermindState(
        time=jnp.zeros(count, jnp.int32),
        board=jnp.full((count, env.num_guesses, env.code_size + 2), -1, jnp.int32),
        goal=codes,
        guess_index=jnp.zeros(count, jnp.int32),
        terminated=jnp.zeros(count, jnp.bool_),
    )
    won = jnp.zeros(codes.shape[0], jnp.bool_)
    for _ in range(guesses_made):
        actions = jnp.argmax(explainer.agent(states.board.reshape(codes.shape[0], -1)).q_values, axis=-1)
        _, states, _, done, _ = jax.vmap(env.step_env, in_axes=(None, 0, 0, None))(
            jax.random.key(0), states, actions, params
        )
        won = won | done
    boards = np.asarray(states.board)[~np.asarray(won)]
    goals = np.asarray(codes)[~np.asarray(won)]
    actions = jnp.argmax(explainer.agent(jnp.asarray(boards.reshape(boards.shape[0], -1))).q_values, axis=-1)
    wins = np.all(np.asarray(jax.vmap(env.decode_action)(actions)) == goals, axis=-1)
    return boards, wins


def nullity(explainer: Explainer, boards: np.ndarray) -> tuple[float, float]:
    """Mean |scaled value| of the greedy action over empty and filled board rows."""
    observations = jnp.asarray(boards.reshape(boards.shape[0], -1))
    actions = np.asarray(jnp.argmax(explainer.agent(observations).q_values, axis=-1))
    contributions = np.asarray(
        explain(masked_values, masked_null, explainer.shapley, explainer.characteristic, observations)
    )
    empty, filled = [], []
    for board, values, action in zip(boards, contributions, actions, strict=True):
        scaled = np.flipud(board_values(values[:, action], explainer.env.num_guesses))
        rows = board[:, 0] == -1
        empty.append(np.abs(scaled[rows]).mean())
        filled.append(np.abs(scaled[~rows]).mean())
    return float(np.mean(empty)), float(np.mean(filled))


def board_html(board: Board) -> str:
    """A standalone shaded table: blue for positive contributions, red for negative."""
    guesses = len(board.cells)
    header = "".join(f"<th>{name}</th>" for name in ("Guess", "Clue 1", "Pos 1", "Pos 2", "Pos 3", "Pos 4", "Clue 2"))
    rows = []
    for index, (cells, values) in enumerate(zip(board.cells, board.values, strict=True)):
        shown = list(cells)
        if index == board.action_row:
            shown[1:-1] = [f'<span class="action">{letter}</span>' for letter in board.action]
        else:
            shown = [escape(cell) for cell in shown]
        tds = "".join(
            f'<td style="background:{_shade(float(value))}" title="{value:+.3f}">{cell}</td>'
            for cell, value in zip(shown, values, strict=True)
        )
        rows.append(f"<tr><th>{guesses - index}</th>{tds}</tr>")
    return (
        "<!doctype html><meta charset=utf-8><title>Mastermind-463 behaviour explanation</title>"
        "<style>body{font-family:sans-serif;background:#fff}table{border-collapse:collapse}"
        "td,th{border:1px solid #444;padding:6px 10px;text-align:center}.action{color:#16bd31;font-weight:bold}</style>"
        f"<table><tr>{header}</tr>{''.join(rows)}</table>\n"
    )


def board_values(contributions: FeatureValues, guesses: int) -> BoardValues:
    """Source plot_explanations: reshape to the board, last guess first, scaled into [-1, 1]."""
    values = np.flipud(np.reshape(np.asarray(contributions, np.float64), (guesses, -1)))
    if values.max() - values.min() != 0:
        values = values / max(abs(values.max()), abs(values.min()))
    return values


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


def format_scientific(mean: float, standard_error: float) -> str:
    """Source _format_scientific: `$(m \\pm s) \\times 10^{e}$` with e = floor(log10|mean|)."""
    exponent = 0 if mean == 0 else math.floor(math.log10(abs(mean)))
    scale = 10.0**exponent
    return f"$({mean / scale:.3f} \\pm {standard_error / scale:.3f}) \\times 10^{{{exponent}}}$"


def standard_error(values: RunValues) -> float:
    return float(np.std(values, ddof=1) / np.sqrt(values.shape[0]))


def _group(args: Args, configs: dict[str, ExperimentConfig], experiment: str, guesses: int) -> str:
    """The latest run group of a large-Mastermind sweep at `guesses` guesses (its agent's env)."""
    if experiment not in configs:
        raise MissingRunsError(f"experiment {experiment} is not ported")
    found = backends.groups(args.track, project=args.project, entity=args.entity, prefix=f"{experiment}-")
    for group in reversed(found):
        tracked = backends.runs(args.track, project=args.project, entity=args.entity, group=group)
        for name, run in tracked.items():
            env = run.config.get("env")
            dqn_run = name.startswith(f"{experiment}--algorithm--dqn") and isinstance(env, dict)
            if dqn_run and cast(dict[str, object], env).get("num_guesses") == guesses:
                return group
    raise MissingRunsError(f"no tracked runs of {experiment} with {guesses} guesses in {args.track} {args.project}")


def _write_table(output: Path, table: str, domains: list[str], models: list[str], groups: list[GroupStats]) -> None:
    """Rows in the schema of docs/data/paper--published-tables.csv, and the source's LaTeX."""
    with (output / f"paper--jax-{table}.csv").open("w", newline="") as file:
        writer = csv.writer(file)
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
    (output / f"paper--jax-{table}.tex").write_text(latex_table(groups))


def _write_board_csv(path: Path, board: Board) -> None:
    columns = ("clue_1", "pos_1", "pos_2", "pos_3", "pos_4", "clue_2")
    with path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("guess", "column", "cell", "value", "action"))
        for index, (cells, values) in enumerate(zip(board.cells, board.values, strict=True)):
            for column, (cell, value) in enumerate(zip(cells, values, strict=True)):
                action = board.action[column - 1] if index == board.action_row and 0 < column < len(columns) - 1 else ""
                writer.writerow((len(board.cells) - index, columns[column], cell, f"{value:.6f}", action))


def _shade(value: float) -> str:
    colour = (33, 102, 172) if value >= 0 else (178, 24, 43)
    return f"rgba({colour[0]},{colour[1]},{colour[2]},{min(abs(value), 1.0):.3f})"


if __name__ == "__main__":
    main(tyro.cli(Args))
