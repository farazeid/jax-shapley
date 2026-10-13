"""Contract, semantic, privacy, and reference tests for optional RGB rendering."""

from __future__ import annotations

import builtins
import os

# This test launches the current interpreter with fixed source and no shell.
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Protocol, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_shapley.envs import (
    Gridworld,
    GridworldState,
    Hypercube,
    HypercubeState,
    Mastermind222,
    Mastermind222State,
    RenderConfig,
)

pytestmark = pytest.mark.render

REFERENCE_PATH = Path(__file__).parent / "fixtures" / "env_render_reference.npz"
TEST_CONFIG = RenderConfig(width=640, height=480, supersampling=1)


class _RenderableEnvironment(Protocol):
    @property
    def default_params(self) -> object: ...

    def reset(self, key: jax.Array, params: object) -> tuple[jax.Array, object]: ...
    def render(
        self,
        state: object,
        params: object | None = None,
        render_config: RenderConfig | None = None,
    ) -> np.ndarray: ...


def test_rgb_array_contract_and_determinism(environment_and_state: tuple[_RenderableEnvironment, object]) -> None:
    environment, state = environment_and_state
    before = jax.tree.map(lambda value: np.asarray(value).copy(), state)

    first = environment.render(state, environment.default_params, TEST_CONFIG)
    second = environment.render(state, environment.default_params, TEST_CONFIG)

    assert isinstance(first, np.ndarray)
    assert first.shape == (TEST_CONFIG.height, TEST_CONFIG.width, 3)
    assert first.dtype == np.uint8
    assert first.flags.c_contiguous
    assert int(first.min()) >= 0
    assert int(first.max()) <= 255
    np.testing.assert_array_equal(first, second)
    for before_leaf, after_leaf in zip(jax.tree.leaves(before), jax.tree.leaves(state), strict=True):
        np.testing.assert_array_equal(before_leaf, after_leaf)


def test_default_frame_dimensions() -> None:
    environment = Gridworld()
    _, state = environment.reset(jax.random.key(0), environment.default_params)
    assert environment.render(state).shape == (480, 640, 3)


def test_batched_state_is_rejected() -> None:
    environment = Gridworld()
    keys = jax.random.split(jax.random.key(0), 2)
    _, states = jax.vmap(environment.reset, in_axes=(0, None))(keys, environment.default_params)

    with pytest.raises(ValueError, match="unbatched state"):
        environment.render(states, render_config=TEST_CONFIG)


def test_traced_render_is_rejected() -> None:
    environment = Gridworld()
    _, state = environment.reset(jax.random.key(0), environment.default_params)

    with pytest.raises(RuntimeError, match="host-side"):
        jax.jit(lambda current: environment.render(current, render_config=TEST_CONFIG))(state)


def test_base_package_import_does_not_import_renderer_modules() -> None:
    code = (
        "import sys, jax_shapley.envs; "
        'assert not any(name.startswith("jax_shapley.envs.render.") '
        'and not name.endswith(".common") for name in sys.modules)'
    )
    result = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    assert result.returncode == 0, result.stderr


def test_missing_render_extra_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    environment = Gridworld()
    _, state = environment.reset(jax.random.key(0), environment.default_params)
    original_import = builtins.__import__

    def import_without_pillow(
        name: str,
        globals_: Mapping[str, object] | None = None,
        locals_: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> object:
        if name == "PIL" or name.startswith("PIL."):
            raise ModuleNotFoundError("No module named 'PIL'", name="PIL")
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_without_pillow)
    with pytest.raises(ModuleNotFoundError, match="uv sync --extra render"):
        environment.render(state, render_config=TEST_CONFIG)


def test_gridworld_agent_movement_changes_only_dynamic_pixels() -> None:
    environment = Gridworld()
    initial = GridworldState(
        position=jnp.asarray([0, 0], dtype=jnp.int32),
        terminated=jnp.asarray(False),
        time=jnp.asarray(0, dtype=jnp.int32),
    )
    moved = replace(initial, position=jnp.asarray([1, 0], dtype=jnp.int32))
    config = RenderConfig(width=640, height=480, supersampling=1, show_status=False)

    initial_frame = environment.render(initial, render_config=config)
    moved_frame = environment.render(moved, render_config=config)
    changed = np.any(initial_frame != moved_frame, axis=-1)

    assert changed.any()
    assert changed.mean() < 0.12
    np.testing.assert_array_equal(initial_frame[40:300, 470:615], moved_frame[40:300, 470:615])


@pytest.mark.parametrize("dimension", [2, 3, 4, 5])
def test_hypercube_supported_dimensions_render(dimension: int) -> None:
    environment = Hypercube(4, dimension)
    _, state = environment.reset(jax.random.key(0), environment.default_params)
    frame = environment.render(state, render_config=TEST_CONFIG)
    assert frame.shape == (480, 640, 3)


def test_hypercube_hidden_coordinate_changes_panel() -> None:
    environment = Hypercube(4, 5)
    state = HypercubeState(
        position=jnp.asarray([1, 1, 0, 0, 0], dtype=jnp.int32),
        terminated=jnp.asarray(False),
        time=jnp.asarray(3, dtype=jnp.int32),
    )
    changed = replace(state, position=jnp.asarray([1, 1, 2, 0, 0], dtype=jnp.int32))

    first = environment.render(state, render_config=TEST_CONFIG)
    second = environment.render(changed, render_config=TEST_CONFIG)

    assert np.any(first[:, 360:] != second[:, 360:])
    np.testing.assert_array_equal(first[:350, :350], second[:350, :350])


def test_mastermind_secret_is_hidden_by_default() -> None:
    environment = Mastermind222()
    public = mastermind_state(goal=(1, 1), board=np.full((2, 4), -1), guess_index=0)
    different_secret = replace(public, goal=jnp.asarray([2, 2], dtype=jnp.int32))

    first = environment.render(public, render_config=TEST_CONFIG)
    second = environment.render(different_secret, render_config=TEST_CONFIG)
    np.testing.assert_array_equal(first, second)


def test_mastermind_secret_changes_explicit_debug_render() -> None:
    environment = Mastermind222()
    first_state = mastermind_state(goal=(1, 1), board=np.full((2, 4), -1), guess_index=0)
    second_state = replace(first_state, goal=jnp.asarray([2, 2], dtype=jnp.int32))
    config = RenderConfig(width=640, height=480, supersampling=1, reveal_secret=True)

    first = environment.render(first_state, render_config=config)
    second = environment.render(second_state, render_config=config)
    assert np.any(first != second)


def test_mastermind_documented_clue_is_a_real_transition() -> None:
    environment = Mastermind222()
    state = mastermind_state(goal=(1, 2), board=np.full((2, 4), -1), guess_index=0)
    _, next_state, _, _, _ = environment.step_env(
        jax.random.key(0), state, jnp.asarray(1, dtype=jnp.int32), environment.default_params
    )
    np.testing.assert_array_equal(next_state.board[0], np.asarray([2, 2, 1, 0]))


def test_mastermind_board_and_clues_change_public_render() -> None:
    environment = Mastermind222()
    initial = mastermind_state(goal=(1, 2), board=np.full((2, 4), -1), guess_index=0)
    played = mastermind_state(
        goal=(1, 2),
        board=np.asarray([[2, 2, 1, 0], [-1, -1, -1, -1]]),
        guess_index=1,
    )
    assert np.any(
        environment.render(initial, render_config=TEST_CONFIG) != environment.render(played, render_config=TEST_CONFIG)
    )


def test_representative_reference_frames() -> None:
    expected = np.load(REFERENCE_PATH)
    actual = representative_frames()
    assert set(expected.files) == set(actual)
    for name, frame in actual.items():
        np.testing.assert_array_equal(frame, expected[name], err_msg=name)


def test_documentation_webps_have_real_transparency() -> None:
    from PIL import Image

    figures = Path(__file__).parents[1] / "docs" / "figures"
    for name in (
        "envs-render--gridworld.webp",
        "envs-render--hypercube-5d.webp",
        "envs-render--mastermind-222.webp",
    ):
        with Image.open(figures / name) as image:
            assert image.mode == "RGBA"
            alpha = np.asarray(image.getchannel("A"))
            assert int(alpha.min()) == 0
            assert int(alpha.max()) == 255


@pytest.mark.gpu
def test_render_state_originating_on_gpu() -> None:
    try:
        device = jax.devices("gpu")[0]
    except (IndexError, RuntimeError):
        pytest.skip("render GPU-origin test requires an available JAX GPU backend")
    environment = Gridworld()
    _, state = environment.reset(jax.random.key(0), environment.default_params)
    state = jax.device_put(state, device)
    frame = environment.render(state, render_config=TEST_CONFIG)
    assert frame.shape == (480, 640, 3)


@pytest.fixture(
    params=[Gridworld(), Hypercube(2, 2), Hypercube(6, 5), Mastermind222()],
    ids=["gridworld", "hypercube-2x2", "hypercube-6x5", "mastermind-222"],
)
def environment_and_state(request: pytest.FixtureRequest) -> tuple[_RenderableEnvironment, object]:
    environment = cast(_RenderableEnvironment, request.param)
    _, state = environment.reset(jax.random.key(4), environment.default_params)
    return environment, state


def mastermind_state(
    *,
    goal: tuple[int, int],
    board: np.ndarray | None = None,
    guess_index: int = 1,
    terminated: bool = False,
) -> Mastermind222State:
    """Construct a controlled Mastermind rendering state."""
    if board is None:
        board = np.asarray([[2, 2, 1, 0], [-1, -1, -1, -1]])
    return Mastermind222State(
        board=jnp.asarray(board, dtype=jnp.int32),
        goal=jnp.asarray(goal, dtype=jnp.int32),
        guess_index=jnp.asarray(guess_index, dtype=jnp.int32),
        terminated=jnp.asarray(terminated),
        time=jnp.asarray(guess_index, dtype=jnp.int32),
    )


def representative_frames() -> dict[str, np.ndarray]:
    """Build the controlled states covered by exact reference fixtures."""
    gridworld = Gridworld()
    grid_initial = GridworldState(
        position=jnp.asarray([0, 0], dtype=jnp.int32),
        terminated=jnp.asarray(False),
        time=jnp.asarray(0, dtype=jnp.int32),
    )
    grid_terminal = replace(
        grid_initial,
        position=jnp.asarray([0, 0], dtype=jnp.int32),
        terminated=jnp.asarray(True),
        time=jnp.asarray(4, dtype=jnp.int32),
    )
    cube_2 = Hypercube(4, 2)
    cube_5 = Hypercube(4, 5)
    cube_2_state = HypercubeState(
        position=jnp.asarray([2, 1], dtype=jnp.int32),
        terminated=jnp.asarray(False),
        time=jnp.asarray(3, dtype=jnp.int32),
    )
    cube_terminal_state = HypercubeState(
        position=jnp.full(5, 3, dtype=jnp.int32),
        terminated=jnp.asarray(True),
        time=jnp.asarray(15, dtype=jnp.int32),
    )
    cube_5_state = HypercubeState(
        position=jnp.asarray([2, 1, 3, 0, 2], dtype=jnp.int32),
        terminated=jnp.asarray(False),
        time=jnp.asarray(8, dtype=jnp.int32),
    )
    mastermind = Mastermind222()
    empty_board = np.full((2, 4), -1, dtype=np.int32)
    partial_board = np.asarray([[2, 2, 1, 0], [-1, -1, -1, -1]], dtype=np.int32)
    won_board = np.asarray([[2, 2, 1, 0], [0, 1, 2, 2]], dtype=np.int32)
    lost_board = np.asarray([[0, 1, 1, 1], [0, 2, 2, 1]], dtype=np.int32)
    return {
        "gridworld_initial": gridworld.render(grid_initial, render_config=TEST_CONFIG),
        "gridworld_terminal": gridworld.render(grid_terminal, render_config=TEST_CONFIG),
        "hypercube_2d": cube_2.render(cube_2_state, render_config=TEST_CONFIG),
        "hypercube_5d": cube_5.render(cube_5_state, render_config=TEST_CONFIG),
        "hypercube_terminal": cube_5.render(cube_terminal_state, render_config=TEST_CONFIG),
        "mastermind_initial": mastermind.render(
            mastermind_state(goal=(1, 2), board=empty_board, guess_index=0),
            render_config=TEST_CONFIG,
        ),
        "mastermind_partial": mastermind.render(
            mastermind_state(goal=(1, 2), board=partial_board),
            render_config=TEST_CONFIG,
        ),
        "mastermind_won": mastermind.render(
            mastermind_state(goal=(1, 2), board=won_board, guess_index=2, terminated=True),
            render_config=TEST_CONFIG,
        ),
        "mastermind_exhausted": mastermind.render(
            mastermind_state(goal=(1, 2), board=lost_board, guess_index=2, terminated=True),
            render_config=TEST_CONFIG,
        ),
    }
