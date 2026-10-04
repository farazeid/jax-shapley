"""Regenerate the paper's large-Mastermind tables (Tables 1-4) from completed JAX runs.

Tables 1, 3 and 4 follow the source's table scripts (`plot_main.py`, `plot_app.py`):
`calculate_final_stats` over 10 seeds per group, with 62 batches per epoch. Each
size's runs live under `ROOT/mastermind-SIZE/{behaviour,prediction,outcome}/...`, as
`runs/paper/run_large.sh` writes them.

Table 2 follows `plot_explanations.py` on the paper's Mastermind-463 board: the
behaviour Shapley values (Shapley seed 1) for the agent's greedy action, scaled by
the largest magnitude. The script checks that the action is the paper's CCCB.

Usage: uv run python scripts/paper_port/tables.py [--root runs/paper] [--output docs/data]
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from html import escape
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from jax_shapley.envs.mastermind import Mastermind, MastermindState
from jax_shapley.explanation_io import load_characteristic, load_shapley, sweep_folders
from jax_shapley.networks import QNetwork
from jax_shapley.reporting import BoardValues, GroupStats, board_values, group_stats, latex_table, write_table_csv
from jax_shapley.run_io import load_agent, load_settings
from jax_shapley.shapley import MaskedCharacteristic, ShapleyModel, explain
from jax_shapley.train_dqn import MastermindConfig

SIZES = ("443", "453", "463")
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


@dataclass(frozen=True, slots=True)
class Args:
    root: Path = Path("runs/paper")
    output: Path = Path("docs/data")
    tables: bool = True
    """Tables 1, 3 and 4."""
    table_2: bool = True


@dataclass(frozen=True, slots=True)
class Explainer:
    """The explained agent's Q-network, its behaviour characteristic and Shapley model."""

    agent: QNetwork
    characteristic: MaskedCharacteristic
    shapley: ShapleyModel
    env: Mastermind


@dataclass(frozen=True, slots=True)
class Board:
    """Table 2: the flipped board's cells, scaled Shapley values and the greedy guess."""

    cells: list[list[str]]
    values: BoardValues
    action: str
    action_row: int


def main(args: Args) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
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
        for size in SIZES:
            for model in models:
                label = f"Mastermind-{size} {model}"
                domains.append(f"Mastermind-{size}")
                labels.append(model)
                groups.append(group_stats(label, sweep_folders(_sweep(args.root, size, kind, model))))
        _write_table(args.output, table, domains, labels, groups)
    if args.table_2:
        explainer = load_explainer(args.root)
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


def load_explainer(root: Path) -> Explainer:
    """The Mastermind-463 agent, behaviour characteristic and Shapley seed 1 behind Table 2."""
    shapley = load_shapley(sweep_folders(_sweep(root, "463", "behaviour", "Shapley"))[0])
    characteristic = load_characteristic(shapley.config.characteristic)
    settings = load_settings(shapley.config.agent / "config.json")
    if not isinstance(settings.env, MastermindConfig) or settings.env.num_guesses != len(TABLE_2_BOARD):
        raise ValueError("Table 2 explains a Mastermind-463 agent")
    agent = load_agent(shapley.config.agent, settings.network, load_replay=False)
    return Explainer(agent.model, MaskedCharacteristic(characteristic.model), shapley.model, settings.env.make())


def explain_board(explainer: Explainer, board: np.ndarray) -> Board:
    """Source plot_explanations for one board: the greedy action's scaled, flipped Shapley values."""
    observation = jnp.asarray(board, jnp.int32).reshape(1, -1)
    action = int(jnp.argmax(explainer.agent(observation).q_values[0]))
    contributions = explain(explainer.shapley, explainer.characteristic, observation)
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
    contributions = np.asarray(explain(explainer.shapley, explainer.characteristic, observations))
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


def _sweep(root: Path, size: str, kind: str, model: str) -> Path:
    stage = "characteristic" if model == "Characteristic" else "shapley"
    matches = sorted((root / f"mastermind-{size}" / kind / stage).glob("*sweep*"))
    if len(matches) != 1:
        raise ValueError(f"expected one {kind} {stage} sweep for Mastermind-{size}, found {len(matches)}")
    return matches[0]


def _write_table(output: Path, table: str, domains: list[str], models: list[str], groups: list[GroupStats]) -> None:
    write_table_csv(output / f"paper--jax-{table}.csv", table, domains, models, groups)
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
