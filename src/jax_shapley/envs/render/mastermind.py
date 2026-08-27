"""Host-side renderer for Mastermind-222."""

from __future__ import annotations

import numpy as np

from jax_shapley.envs.mastermind import Mastermind222State
from jax_shapley.envs.render.common import (
    PALETTE,
    ImageDrawType,
    ImageFontType,
    RenderConfig,
    create_canvas,
    draw_text,
    finish_canvas,
    host_array,
    host_scalar,
    resolve_config,
    scaled_box,
)


def render_mastermind(
    state: Mastermind222State,
    *,
    num_guesses: int,
    code_size: int,
    config: RenderConfig | None,
    transparent: bool = False,
) -> np.ndarray:
    """Render the public Mastermind board without revealing its secret by default."""
    config = resolve_config(config)
    board = host_array(state.board, "state.board", ndim=2)
    goal = host_array(state.goal, "state.goal", ndim=1)
    guess_index = int(host_scalar(state.guess_index, "state.guess_index"))
    terminated = bool(host_scalar(state.terminated, "state.terminated"))
    time = int(host_scalar(state.time, "state.time"))
    if board.shape != (num_guesses, code_size + 2):
        raise ValueError(f"Mastermind board must have shape {(num_guesses, code_size + 2)}, got {board.shape}")
    if goal.shape != (code_size,):
        raise ValueError(f"Mastermind goal must have shape {(code_size,)}, got {goal.shape}")

    image, draw, font, scale = create_canvas(config)
    left = 30.0
    top = 88.0
    row_height = min(118.0, (config.height - 180.0) / num_guesses)
    board_right = config.width * 0.64
    peg_radius = min(28.0, row_height * 0.27)

    draw.rounded_rectangle(
        scaled_box((18, 10, config.width * 0.64, 68), scale),
        radius=9 * scale,
        fill=PALETTE.panel,
        outline=PALETTE.grid,
    )
    draw_text(draw, (left, 25), "Mastermind-222", font=font, scale=scale, anchor="lm")
    draw_text(
        draw,
        (left, 50),
        "2 positions | values {1,2} | at most 2 guesses | secret hidden",
        font=font,
        scale=scale,
        fill=PALETTE.muted,
        anchor="lm",
    )
    for row_index in range(num_guesses):
        row_top = top + row_index * row_height
        row_bottom = row_top + row_height - 12
        used = bool(np.all(board[row_index] >= 0))
        active = not terminated and row_index == min(guess_index, num_guesses - 1)
        fill = "#FFF4D6" if active else PALETTE.panel
        draw.rounded_rectangle(
            scaled_box((left, row_top, board_right, row_bottom), scale),
            radius=12 * scale,
            fill=fill,
            outline=PALETTE.agent if active else PALETTE.grid,
            width=max(2, (4 if active else 2) * scale),
        )
        draw_text(
            draw,
            (left + 15, (row_top + row_bottom) / 2),
            f"{row_index + 1}",
            font=font,
            scale=scale,
            fill=PALETTE.muted,
            anchor="mm",
        )
        for peg_index in range(code_size):
            centre_x = left + 78 + peg_index * (peg_radius * 2.8)
            centre_y = (row_top + row_bottom) / 2
            peg = int(board[row_index, 1 + peg_index]) if used else -1
            _draw_peg(draw, font, scale, centre_x, centre_y, peg_radius, peg)
        clue_left = board_right - 118
        if used:
            _draw_clues(
                draw,
                font,
                scale,
                clue_left,
                (row_top + row_bottom) / 2,
                exact=int(board[row_index, -1]),
                misplaced=int(board[row_index, 0]),
            )
        else:
            draw_text(
                draw,
                (clue_left + 48, (row_top + row_bottom) / 2),
                "unused row",
                font=font,
                scale=scale,
                fill=PALETTE.muted,
                anchor="mm",
            )

    _draw_legend(
        draw,
        font,
        scale,
        left=config.width * 0.67,
        top=70,
        right=config.width - 18,
    )
    if config.reveal_secret:
        _draw_secret(draw, font, scale, goal, left=config.width * 0.69, top=config.height - 130)

    if config.show_status:
        draw.rounded_rectangle(
            scaled_box((18, config.height - 54, config.width - 18, config.height - 12), scale),
            radius=9 * scale,
            fill=PALETTE.panel,
            outline=PALETTE.grid,
        )
        used_guesses = min(max(guess_index, 0), num_guesses)
        won = terminated and used_guesses > 0 and int(board[used_guesses - 1, -1]) == code_size
        status = "won" if won else "exhausted" if terminated else "in progress"
        draw_text(
            draw,
            (left, config.height - 34),
            (
                f"State: {status} | terminated: {str(terminated).lower()} | "
                f"guesses used: {used_guesses}/{num_guesses} | step: {time}"
            ),
            font=font,
            scale=scale,
            fill=PALETTE.goal if won else PALETTE.danger if terminated else PALETTE.ink,
            anchor="lm",
        )
    return finish_canvas(image, config, transparent=transparent)


def _draw_peg(
    draw: ImageDrawType,
    font: ImageFontType,
    scale: int,
    x: float,
    y: float,
    radius: float,
    peg: int,
) -> None:
    """Draw a peg using both colour and symbol."""
    fill = PALETTE.peg_one if peg == 1 else PALETTE.peg_two if peg == 2 else PALETTE.unused
    draw.ellipse(
        scaled_box((x - radius, y - radius, x + radius, y + radius), scale),
        fill=fill,
        outline=PALETTE.ink,
        width=max(2, 3 * scale),
    )
    symbol = "1" if peg == 1 else "2" if peg == 2 else "-"
    text_fill = "#FFFFFF" if peg in (1, 2) else PALETTE.muted
    draw_text(draw, (x, y), symbol, font=font, scale=scale, fill=text_fill, anchor="mm")


def _draw_clues(
    draw: ImageDrawType,
    font: ImageFontType,
    scale: int,
    x: float,
    y: float,
    *,
    exact: int,
    misplaced: int,
) -> None:
    """Draw non-overlapping clue symbols with numeric counts."""
    draw.rectangle(
        scaled_box((x, y - 31, x + 12, y - 19), scale),
        fill=PALETTE.exact,
        outline=PALETTE.ink,
    )
    draw_text(
        draw,
        (x + 20, y - 25),
        f"Exact: {exact}",
        font=font,
        scale=scale,
        anchor="lm",
    )
    centre_x = x + 6
    centre_y = y + 23
    diamond = [
        (round(centre_x * scale), round((centre_y - 7) * scale)),
        (round((centre_x + 7) * scale), round(centre_y * scale)),
        (round(centre_x * scale), round((centre_y + 7) * scale)),
        (round((centre_x - 7) * scale), round(centre_y * scale)),
    ]
    draw.polygon(diamond, fill=PALETTE.misplaced, outline=PALETTE.ink)
    draw_text(
        draw,
        (x + 20, centre_y),
        f"Misplaced: {misplaced}",
        font=font,
        scale=scale,
        anchor="lm",
    )


def _draw_legend(
    draw: ImageDrawType,
    font: ImageFontType,
    scale: int,
    *,
    left: float,
    top: float,
    right: float,
) -> None:
    """Explain rows, pegs, clues, active styling, and secret policy."""
    draw.rounded_rectangle(
        scaled_box((left - 10, top - 10, right, top + 320), scale),
        radius=10 * scale,
        fill=PALETTE.panel,
        outline=PALETTE.grid,
        width=max(1, 2 * scale),
    )
    draw_text(draw, (left, top), "How to read", font=font, scale=scale)
    draw_text(draw, (left, top + 24), "Rows are guesses,", font=font, scale=scale)
    draw_text(draw, (left, top + 42), "oldest first.", font=font, scale=scale)
    _draw_peg(draw, font, scale, left + 12, top + 70, 10, 1)
    draw_text(draw, (left + 31, top + 70), "peg value 1", font=font, scale=scale, anchor="lm")
    _draw_peg(draw, font, scale, left + 12, top + 98, 10, 2)
    draw_text(draw, (left + 31, top + 98), "peg value 2", font=font, scale=scale, anchor="lm")
    draw.rectangle(
        scaled_box((left, top + 120, left + 18, top + 138), scale),
        fill="#FFF4D6",
        outline=PALETTE.agent,
        width=max(1, 2 * scale),
    )
    draw_text(draw, (left + 27, top + 129), "next active row", font=font, scale=scale, anchor="lm")
    draw.ellipse(
        scaled_box((left, top + 150, left + 18, top + 168), scale),
        fill=PALETTE.unused,
        outline=PALETTE.ink,
    )
    draw_text(
        draw,
        (left + 9, top + 159),
        "-",
        font=font,
        scale=scale,
        fill=PALETTE.muted,
        anchor="mm",
    )
    draw_text(draw, (left + 27, top + 159), "unused position", font=font, scale=scale, anchor="lm")
    draw.rectangle(
        scaled_box((left + 3, top + 185, left + 15, top + 197), scale),
        fill=PALETTE.exact,
        outline=PALETTE.ink,
    )
    draw_text(draw, (left + 27, top + 191), "exact position", font=font, scale=scale, anchor="lm")
    centre_x = left + 9
    centre_y = top + 220
    diamond = [
        (round(centre_x * scale), round((centre_y - 7) * scale)),
        (round((centre_x + 7) * scale), round(centre_y * scale)),
        (round(centre_x * scale), round((centre_y + 7) * scale)),
        (round((centre_x - 7) * scale), round(centre_y * scale)),
    ]
    draw.polygon(diamond, fill=PALETTE.misplaced, outline=PALETTE.ink)
    draw_text(draw, (left + 27, top + 212), "correct value,", font=font, scale=scale)
    draw_text(draw, (left + 27, top + 230), "wrong position", font=font, scale=scale)
    draw_text(draw, (left, top + 255), "Numbers are match counts.", font=font, scale=scale)
    draw_text(
        draw,
        (left, top + 281),
        "Secret code is hidden",
        font=font,
        scale=scale,
        fill=PALETTE.muted,
    )
    draw_text(draw, (left, top + 299), "by default.", font=font, scale=scale, fill=PALETTE.muted)


def _draw_secret(
    draw: ImageDrawType,
    font: ImageFontType,
    scale: int,
    goal: np.ndarray,
    *,
    left: float,
    top: float,
) -> None:
    """Draw an explicitly labelled debug-only secret."""
    draw.rounded_rectangle(
        scaled_box((left - 10, top - 10, left + 150, top + 72), scale),
        radius=8 * scale,
        outline=PALETTE.danger,
        width=max(2, 3 * scale),
    )
    draw_text(draw, (left, top), "SECRET (debug)", font=font, scale=scale, fill=PALETTE.danger)
    draw_text(
        draw,
        (left, top + 36),
        " ".join(str(int(peg)) for peg in goal),
        font=font,
        scale=scale,
    )
