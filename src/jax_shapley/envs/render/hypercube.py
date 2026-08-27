"""Host-side renderer for obstacle-free Hypercube environments."""

from __future__ import annotations

import jax
import numpy as np

from jax_shapley.envs.hypercube import HypercubeState
from jax_shapley.envs.render.common import (
    PALETTE,
    ImageDrawType,
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


def render_hypercube(
    state: HypercubeState,
    *,
    side_length: int,
    dimension: int,
    start: np.ndarray | jax.Array,
    goal: np.ndarray | jax.Array,
    config: RenderConfig | None,
    transparent: bool = False,
) -> np.ndarray:
    """Render a Hypercube state as a 2D projection and coordinate tracks."""
    config = resolve_config(config)
    position = host_array(state.position, "state.position", ndim=1)
    terminated = bool(host_scalar(state.terminated, "state.terminated"))
    time = int(host_scalar(state.time, "state.time"))
    start = host_array(start, "start", ndim=1)
    goal = host_array(goal, "goal", ndim=1)
    expected_shape = (dimension,)
    if position.shape != expected_shape:
        raise ValueError(f"Hypercube position must have shape {expected_shape}, got {position.shape}")

    image, draw, font, scale = create_canvas(config)
    margin = 24.0
    panel_width = min(250.0, config.width * 0.38) if dimension > 2 else 0.0
    status_height = 58.0 if config.show_status else 20.0
    grid_width = config.width - 2 * margin - panel_width - (18.0 if panel_width else 0.0)
    grid_height = config.height - 2 * margin - status_height
    cell_size = min(grid_width / side_length, grid_height / side_length)
    origin = (margin, margin)
    hidden_at_goal = dimension == 2 or np.array_equal(position[2:], goal[2:])

    for row in range(side_length):
        for column in range(side_length):
            box = cell_box(row, column, origin=origin, cell_size=cell_size)
            projected = np.asarray([row, column])
            fill = PALETTE.panel
            if np.array_equal(projected, start[:2]):
                fill = PALETTE.start
            draw.rectangle(
                scaled_box(box, scale),
                fill=fill,
                outline=PALETTE.grid,
                width=max(1, 2 * scale),
            )
            if np.array_equal(projected, start[:2]):
                draw_text(
                    draw,
                    (box[2] - 12, box[3] - 12),
                    "S",
                    font=font,
                    scale=scale,
                    fill=PALETTE.ink,
                    anchor="mm",
                )
            if config.show_coordinates:
                draw_text(
                    draw,
                    (box[0] + 4, box[1] + 3),
                    f"{row},{column}",
                    font=font,
                    scale=scale,
                    fill=PALETTE.muted,
                )

    goal_box = cell_box(
        int(goal[0]),
        int(goal[1]),
        origin=origin,
        cell_size=cell_size,
        inset=cell_size * 0.12,
    )
    draw.rectangle(
        scaled_box(goal_box, scale),
        fill=PALETTE.goal if hidden_at_goal else None,
        outline=PALETTE.goal,
        width=max(2, 4 * scale if hidden_at_goal else 2 * scale),
    )
    draw_text(
        draw,
        ((goal_box[0] + goal_box[2]) / 2, (goal_box[1] + goal_box[3]) / 2),
        "G" if hidden_at_goal else "G*",
        font=font,
        scale=scale,
        fill="#FFFFFF" if hidden_at_goal else PALETTE.goal,
        anchor="mm",
    )

    agent_box = cell_box(
        int(position[0]),
        int(position[1]),
        origin=origin,
        cell_size=cell_size,
        inset=cell_size * 0.25,
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
        "A/G" if np.array_equal(position[:2], goal[:2]) else "A",
        font=font,
        scale=scale,
        anchor="mm",
    )

    if dimension > 2:
        panel_left = origin[0] + side_length * cell_size + 18
        draw.rounded_rectangle(
            scaled_box(
                (
                    panel_left - 8,
                    margin - 8,
                    config.width - margin + 8,
                    config.height - 96,
                ),
                scale,
            ),
            radius=10 * scale,
            fill=PALETTE.panel,
            outline=PALETTE.grid,
        )
        _draw_coordinate_panel(
            draw,
            font,
            scale,
            left=panel_left,
            top=margin,
            width=config.width - panel_left - margin,
            position=position,
            goal=goal,
            side_length=side_length,
            hidden_at_goal=hidden_at_goal,
            config=config,
        )

    if config.width >= 500:
        draw.rounded_rectangle(
            scaled_box((16, config.height - 94, config.width - 16, config.height - 5), scale),
            radius=10 * scale,
            fill=PALETTE.panel,
            outline=PALETTE.grid,
        )
    _draw_legend(draw, font, scale, config, dimension)
    if config.show_status:
        draw_text(
            draw,
            (margin, config.height - 20),
            (
                f"Hypercube · {dimension} dimensions · side length {side_length} · "
                f"step: {time} · terminated: {str(terminated).lower()}"
            ),
            font=font,
            scale=scale,
        )
    return finish_canvas(image, config, transparent=transparent)


def _draw_coordinate_panel(
    draw: ImageDrawType,
    font: object,
    scale: int,
    *,
    left: float,
    top: float,
    width: float,
    position: np.ndarray,
    goal: np.ndarray,
    side_length: int,
    hidden_at_goal: bool,
    config: RenderConfig,
) -> None:
    """Draw full-state coordinate tracks beside the x0-x1 projection."""
    draw_text(draw, (left, top), "Full coordinate state", font=font, scale=scale)
    draw_text(
        draw,
        (left, top + 23),
        f"Current: {position.tolist()}",
        font=font,
        scale=scale,
        fill=PALETTE.ink,
    )
    draw_text(
        draw,
        (left, top + 43),
        f"Goal:    {goal.tolist()}",
        font=font,
        scale=scale,
        fill=PALETTE.goal,
    )
    track_left = left + 24
    track_right = left + max(46, width - 12)
    spacing = 48.0 if config.height >= 400 else 27.0
    track_top = top + 82
    for axis, value in enumerate(position):
        y = track_top + axis * spacing
        draw_text(draw, (left, y), f"x{axis}", font=font, scale=scale, anchor="lm")
        draw.line(
            scaled_box((track_left, y, track_right, y), scale),
            fill=PALETTE.grid,
            width=max(1, 2 * scale),
        )
        fraction = float(value) / (side_length - 1)
        goal_fraction = float(goal[axis]) / (side_length - 1)
        marker_x = track_left + fraction * (track_right - track_left)
        goal_x = track_left + goal_fraction * (track_right - track_left)
        radius = 6
        draw.ellipse(
            scaled_box((marker_x - radius, y - radius, marker_x + radius, y + radius), scale),
            fill=PALETTE.agent,
            outline=PALETTE.ink,
            width=max(1, 2 * scale),
        )
        draw.rectangle(
            scaled_box((goal_x - radius, y - radius, goal_x + radius, y + radius), scale),
            outline=PALETTE.goal,
            width=max(1, 2 * scale),
        )
        draw_text(
            draw,
            (track_right, y + 12),
            f"A:{int(value)}  G:{int(goal[axis])}",
            font=font,
            scale=scale,
            fill=PALETTE.muted,
            anchor="ra",
        )
    if not hidden_at_goal and config.height >= 400:
        warning_y = track_top + len(position) * spacing + 5
        draw_text(
            draw,
            (left, warning_y),
            "G*: hidden coordinates differ.",
            font=font,
            scale=scale,
            fill=PALETTE.danger,
        )


def _draw_legend(
    draw: ImageDrawType,
    font: object,
    scale: int,
    config: RenderConfig,
    dimension: int,
) -> None:
    """Explain the projection and every grid and track marker."""
    if config.width < 500:
        draw_text(
            draw,
            (config.width / 2, config.height - 15),
            "Legend: S start | A current | G goal",
            font=font,
            scale=scale,
            anchor="mm",
        )
        return
    draw_text(
        draw,
        (24, config.height - 78),
        "Grid: x0 points down, x1 points right | S start | A current | G* goal projection",
        font=font,
        scale=scale,
    )
    if dimension > 2:
        draw_text(
            draw,
            (24, config.height - 55),
            "Tracks: orange circle = current | green square = goal | nested markers = equal",
            font=font,
            scale=scale,
        )
        draw_text(
            draw,
            (24, config.height - 35),
            "The grid shows only x0,x1; coordinate tracks disambiguate the full state.",
            font=font,
            scale=scale,
            fill=PALETTE.muted,
        )
