"""Shared host-side raster primitives for environment renderers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import numpy as np
from jax import core

RENDER_DEPENDENCY_MESSAGE = (
    "Rendering requires the optional 'render' dependencies. "
    "Install them with: uv sync --extra render"
)


@dataclass(frozen=True)
class RenderConfig:
    """Host-only presentation settings shared by all environment renderers."""

    width: int = 640
    height: int = 480
    supersampling: int = 2
    show_coordinates: bool = True
    show_status: bool = True
    reveal_secret: bool = False


@dataclass(frozen=True)
class Palette:
    """Colour-blind-friendly colours with redundant borders and symbols."""

    background: str = "#F7F7F2"
    panel: str = "#FFFFFF"
    ink: str = "#2D3142"
    muted: str = "#767B91"
    grid: str = "#4F5D75"
    start: str = "#56B4E9"
    goal: str = "#009E73"
    blocked: str = "#4D4D4D"
    agent: str = "#E69F00"
    peg_one: str = "#0072B2"
    peg_two: str = "#D55E00"
    exact: str = "#009E73"
    misplaced: str = "#CC79A7"
    unused: str = "#D8DEE9"
    danger: str = "#D55E00"


PALETTE = Palette()


def resolve_config(config: RenderConfig | None) -> RenderConfig:
    """Return and validate a rendering configuration."""
    config = config or RenderConfig()
    if config.width < 600 or config.height < 420:
        raise ValueError("render width and height must be at least 600 x 420")
    if config.supersampling < 1:
        raise ValueError("render supersampling must be positive")
    return config


def require_pillow() -> tuple[Any, Any, Any]:
    """Import Pillow lazily and provide an actionable missing-extra error."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError as error:
        if error.name == "PIL" or (error.name and error.name.startswith("PIL.")):
            raise ModuleNotFoundError(RENDER_DEPENDENCY_MESSAGE) from error
        raise
    return Image, ImageDraw, ImageFont


def host_array(value: Any, name: str, *, ndim: int) -> np.ndarray:
    """Cross the JAX host boundary and reject tracers and batched state fields."""
    if isinstance(value, core.Tracer):
        raise RuntimeError(
            "Rendering is host-side and cannot run inside jax.jit, jax.vmap, or lax.scan"
        )
    array = np.asarray(jax.device_get(value))
    if array.ndim != ndim:
        raise ValueError(
            f"render() accepts an unbatched state; {name} must have {ndim} dimensions, "
            f"got shape {array.shape}"
        )
    return array


def host_scalar(value: Any, name: str) -> Any:
    """Cross the JAX host boundary for an unbatched scalar."""
    return host_array(value, name, ndim=0).item()


def create_canvas(config: RenderConfig) -> tuple[Any, Any, Any, int]:
    """Create a supersampled Pillow canvas and its default bundled font."""
    Image, ImageDraw, ImageFont = require_pillow()
    scale = config.supersampling
    image = Image.new("RGBA", (config.width * scale, config.height * scale), (247, 247, 242, 0))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(10, 14 * scale))
    return image, draw, font, scale


def finish_canvas(image: Any, config: RenderConfig, *, transparent: bool = False) -> np.ndarray:
    """Downsample a canvas and return transparent RGBA or the public RGB contract."""
    Image, _, _ = require_pillow()
    if config.supersampling != 1:
        image = image.resize((config.width, config.height), Image.Resampling.LANCZOS)
    if transparent:
        output = image
        channels = 4
    else:
        background = Image.new("RGBA", image.size, PALETTE.background)
        output = Image.alpha_composite(background, image).convert("RGB")
        channels = 3
    frame = np.ascontiguousarray(np.asarray(output, dtype=np.uint8))
    expected_shape = (config.height, config.width, channels)
    if frame.shape != expected_shape or frame.dtype != np.uint8:
        raise RuntimeError(
            f"renderer returned {frame.shape} {frame.dtype}; expected {expected_shape} uint8"
        )
    return frame


def scaled_box(box: tuple[float, float, float, float], scale: int) -> tuple[int, int, int, int]:
    """Scale a logical pixel box to supersampled coordinates."""
    return tuple(round(value * scale) for value in box)


def draw_text(
    draw: Any,
    position: tuple[float, float],
    text: str,
    *,
    font: Any,
    scale: int,
    fill: str = PALETTE.ink,
    anchor: str | None = None,
) -> None:
    """Draw text in logical output coordinates."""
    draw.text(
        (round(position[0] * scale), round(position[1] * scale)),
        text,
        font=font,
        fill=fill,
        anchor=anchor,
    )


def cell_box(
    row: int,
    column: int,
    *,
    origin: tuple[float, float],
    cell_size: float,
    inset: float = 0.0,
) -> tuple[float, float, float, float]:
    """Convert a row and column into a logical pixel rectangle."""
    left = origin[0] + column * cell_size + inset
    top = origin[1] + row * cell_size + inset
    right = origin[0] + (column + 1) * cell_size - inset
    bottom = origin[1] + (row + 1) * cell_size - inset
    return left, top, right, bottom
