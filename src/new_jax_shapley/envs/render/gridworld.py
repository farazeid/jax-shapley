"""Host-side renderer for FastSVERL Gridworld."""

from __future__ import annotations

import jax
import numpy as np

from new_jax_shapley.envs.gridworld import GridworldState
from new_jax_shapley.envs.render.common import (
    PALETTE,
    ImageDrawType,
    ImageFontType,
    RenderConfig,
    cell_box,
    create_canvas,
    draw_text,
    finish_canvas,
    host_array,
    host_scalar,
    resolve_config,
    scaled_box,
)


# This host-only renderer intentionally keeps drawing branches together for visual clarity.
def render_gridworld(  # ruff: ignore[complex-structure]
    state: GridworldState,
    *,
    starts: np.ndarray | jax.Array,
    blocked: np.ndarray | jax.Array,
    config: RenderConfig | None,
    transparent: bool = False,
) -> np.ndarray:
    """Render an unbatched Gridworld state as an RGB array."""
    config = resolve_config(config)
    position = host_array(state.position, "state.position", ndim=1)
    terminated = bool(host_scalar(state.terminated, "state.terminated"))
    time = int(host_scalar(state.time, "state.time"))
    starts = host_array(starts, "starts", ndim=2)
    blocked = host_array(blocked, "blocked", ndim=1)
    if position.shape != (2,):
        raise ValueError(f"Gridworld position must have shape (2,), got {position.shape}")

    image, draw, font, scale = create_canvas(config)
    status_height = 54 if config.show_status else 18
    available_width = config.width - 48
    available_height = config.height - status_height - 30
    cell_size = min(available_width / 4, available_height / 2)
    origin = ((config.width - 4 * cell_size) / 2, 18.0)

    for row in range(2):
        for column in range(4):
            box = cell_box(row, column, origin=origin, cell_size=cell_size)
            coordinate = np.asarray([row, column])
            fill = PALETTE.panel
            if np.array_equal(coordinate, blocked):
                fill = PALETTE.blocked
            elif any(np.array_equal(coordinate, start) for start in starts):
                fill = PALETTE.start
            elif column == 3:
                fill = PALETTE.goal
            draw.rectangle(
                scaled_box(box, scale),
                fill=fill,
                outline=PALETTE.grid,
                width=max(1, 2 * scale),
            )
            if column == 3:
                draw_text(
                    draw,
                    ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
                    "G",
                    font=font,
                    scale=scale,
                    fill="#FFFFFF",
                    anchor="mm",
                )
            if np.array_equal(coordinate, blocked):
                margin = cell_size * 0.25
                draw.line(
                    scaled_box(
                        (box[0] + margin, box[1] + margin, box[2] - margin, box[3] - margin),
                        scale,
                    ),
                    fill="#FFFFFF",
                    width=max(2, 3 * scale),
                )
                draw.line(
                    scaled_box(
                        (box[0] + margin, box[3] - margin, box[2] - margin, box[1] + margin),
                        scale,
                    ),
                    fill="#FFFFFF",
                    width=max(2, 3 * scale),
                )
            if any(np.array_equal(coordinate, start) for start in starts):
                draw_text(
                    draw,
                    (box[2] - 13, box[3] - 13),
                    "S",
                    font=font,
                    scale=scale,
                    fill=PALETTE.ink,
                    anchor="mm",
                )
            if config.show_coordinates:
                draw_text(
                    draw,
                    (box[0] + 5, box[1] + 4),
                    f"{row},{column}",
                    font=font,
                    scale=scale,
                    fill=PALETTE.ink if fill != PALETTE.blocked else "#FFFFFF",
                )

    agent_box = cell_box(
        int(position[0]),
        int(position[1]),
        origin=origin,
        cell_size=cell_size,
        inset=cell_size * 0.24,
    )
    draw.ellipse(
        scaled_box(agent_box, scale),
        fill=PALETTE.agent,
        outline=PALETTE.ink,
        width=max(2, 3 * scale),
    )
    draw_text(
        draw,
        ((agent_box[0] + agent_box[2]) / 2, (agent_box[1] + agent_box[3]) / 2),
        "A",
        font=font,
        scale=scale,
        anchor="mm",
    )

    if config.show_status:
        draw.rounded_rectangle(
            scaled_box(
                (
                    config.width * 0.25,
                    origin[1] + 2 * cell_size + 11,
                    config.width * 0.75,
                    origin[1] + 2 * cell_size + 42,
                ),
                scale,
            ),
            radius=8 * scale,
            fill=PALETTE.panel,
            outline=PALETTE.grid,
        )
        draw_text(
            draw,
            (config.width / 2, origin[1] + 2 * cell_size + 25),
            f"Gridworld · step: {time} · terminated: {str(terminated).lower()}",
            font=font,
            scale=scale,
            anchor="mm",
        )
        if terminated and config.height >= 400:
            draw_text(
                draw,
                (config.width / 2, origin[1] + 2 * cell_size + 47),
                "Raw terminal position is [0,0] by the FastSVERL convention.",
                font=font,
                scale=scale,
                fill=PALETTE.danger,
                anchor="mm",
            )
    _draw_legend(draw, font, scale, config)
    return finish_canvas(image, config, transparent=transparent)


def _draw_legend(
    draw: ImageDrawType,
    font: ImageFontType,
    scale: int,
    config: RenderConfig,
) -> None:
    """Explain every Gridworld symbol inside the returned frame."""
    if config.width < 500:
        draw_text(
            draw,
            (config.width / 2, config.height - 15),
            "Legend: A agent | S start | X blocked | G terminal",
            font=font,
            scale=scale,
            anchor="mm",
        )
        return
    draw.rounded_rectangle(
        scaled_box((24, config.height - 82, config.width - 24, config.height - 6), scale),
        radius=10 * scale,
        fill=PALETTE.panel,
        outline=PALETTE.grid,
    )
    y = config.height - 67
    entries = (
        (0.08, PALETTE.agent, "A", "current agent"),
        (0.31, PALETTE.start, "S", "possible start"),
        (0.55, PALETTE.blocked, "X", "blocked cell"),
        (0.75, PALETTE.goal, "G", "terminal / goal"),
    )
    for fraction, fill, symbol, label in entries:
        x = config.width * fraction
        draw.rectangle(
            scaled_box((x, y, x + 24, y + 24), scale),
            fill=fill,
            outline=PALETTE.ink,
            width=max(1, 2 * scale),
        )
        draw_text(
            draw,
            (x + 12, y + 12),
            symbol,
            font=font,
            scale=scale,
            fill="#FFFFFF" if fill in (PALETTE.blocked, PALETTE.goal) else PALETTE.ink,
            anchor="mm",
        )
        draw_text(
            draw,
            (x + 31, y + 12),
            label,
            font=font,
            scale=scale,
            anchor="lm",
        )
    draw_text(
        draw,
        (config.width / 2, config.height - 19),
        "Coordinates are row,column; row 0 is top and column 0 is left.",
        font=font,
        scale=scale,
        fill=PALETTE.muted,
        anchor="mm",
    )
