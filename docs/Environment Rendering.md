# Environment Rendering

Gridworld, Hypercube, and Mastermind-222 expose deterministic host-side RGB rendering for debugging, paper frame grabs, and episode captures. Rendering is deliberately separate from reset, transition, auto-reset, vectorisation, and compiled rollout execution.

## Installation and contract

Install the supported renderer dependencies with `uv sync --extra render`. Gymnax 0.0.9 currently also brings Pillow transitively through Matplotlib, but the explicit extra records the renderer dependency independently of that upstream implementation detail.

`env.render(state, params=None, render_config=None)` returns a C-contiguous `numpy.ndarray` with RGB channel order, `uint8` dtype, and shape `(height, width, 3)`. The default frame is 640 x 480; `RenderConfig` controls its fixed dimensions, supersampling, labels, status, and explicit Mastermind secret disclosure.

The committed WebP environment cards use an internal alpha-aware export of the same canvas: the outer background is genuinely transparent, while board cells, legends, explanatory text, and status remain on opaque panels for readability on light or dark pages. The public environment API remains RGB and therefore does not expose an unstable second array shape.

Rendering performs `jax.device_get` at the host boundary. It accepts one unbatched state, is deterministic, does not mutate state, opens no windows, and raises an error inside `jit`, `vmap`, or `lax.scan`. Render selected states after compiled execution, or synchronise and render individual states from a Python-controlled episode loop; rendered steps are intentionally excluded from throughput benchmarks.

```python
import jax

from jax_shapley.envs import Gridworld, RenderConfig

env = Gridworld()
params = env.default_params
_, state = env.reset(jax.random.key(0), params)
frame = env.render(state, params, RenderConfig(width=640, height=480))
assert frame.shape == (480, 640, 3)
```

## Gridworld

![Rendered Gridworld state](figures/envs-render--gridworld.webp)

Gridworld renders the complete 2 x 4 board with coordinate-labelled cells and an embedded legend for possible start cells `S`, the blocked cell `X`, terminal or goal cells `G`, and the current agent `A`. The status explicitly reports step count and termination. A raw terminal state remains renderable and explains the FastSVERL `[0,0]` terminal-position convention; an auto-reset state renders as the new episode state because the renderer does not reconstruct discarded history.

## Hypercube

![Rendered five-dimensional Hypercube state](figures/envs-render--hypercube-5d.webp)

Two-dimensional Hypercubes render as ordinary grids. Dimensions 3–5 render axes 0 and 1 as an explicitly labelled projection alongside the full current and goal vectors and one coordinate track per dimension. The embedded legend defines `S`, `A`, `G*`, current and goal track markers, nested equal markers, and axis orientation; `G*` explicitly warns when hidden coordinates differ. The footer reports dimension, side length, step count, and termination.

## Mastermind-222

![Rendered Mastermind-222 state](figures/envs-render--mastermind-222.webp)

Mastermind renders each guess as an oldest-first row, numbered blue or orange value pegs, non-overlapping exact and misplaced match counts, an outlined active row, unused positions, and a complete `How to read` legend. Its subtitle explains the 222 configuration, its footer explicitly reports outcome, termination, guesses used, and step count, and its hidden code never affects ordinary output; only `RenderConfig(reveal_secret=True)` adds an explicitly labelled debug secret.

## Validation

Run the optional CPU rendering contracts, semantic checks, privacy checks, and controlled reference frames with `JAX_PLATFORMS=cpu uv run --extra render pytest tests/test_env_rendering.py -m render`. Use `JAX_PLATFORMS=cuda uv run --extra cuda --extra render pytest tests/test_env_rendering.py -m gpu` to check a state produced on GPU when one is available.

## Frame capture

Call `render()` after an already-compiled step, or render selected states retained from a compiled rollout. The README contains the canonical frame-grab and encoding example; GIF, WebP, MP4, and website encoding remain outside the environment API.
