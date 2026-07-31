# jax-shapley

End-to-end JAX/Gymnax ports of the reinforcement-learning environments used by
[FastSVERL](https://github.com/djeb20/fastsverl): Gridworld, all 20 obstacle-free
paper Hypercubes, and Mastermind-222.

## Install

```bash
uv sync --all-groups                 # portable CPU
uv sync --all-groups --extra cuda    # NVIDIA CUDA 13
```

## Use

```python
import jax
from jax_shapley import EpisodeStatisticsWrapper
from jax_shapley.envs import Gridworld, Hypercube, Mastermind222

env = EpisodeStatisticsWrapper(Hypercube(side_length=6, dimension=5))
params = env.default_params
obs, state = env.reset(jax.random.key(0), params)
obs, state, reward, done, info = env.step(
    jax.random.key(1), state, action=0, params=params
)
```

The environments use the Gymnax API. Public `step()` auto-resets terminal instances;
`env.env.step_env()` accesses raw FastSVERL-compatible transitions on the unwrapped
environment. Gridworld preserves the FastSVERL `[0, 0]` terminal-observation quirk.
Hypercube supports side lengths
2–6 and dimensions 2–5 as static JIT specializations. Mastermind-222 preserves
the source action decoding, eight-feature board, clues, rewards, and horizon.

Vectorization uses the standard JAX composition:

```python
batched_reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
batched_step = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))
# Place batched_step in jax.lax.scan for a fully compiled rollout.
```

`EpisodeStatisticsWrapper` accumulates per-environment return, length, and completed-
episode count. Each step adds fixed-shape `info["episode"]` return, length, and completion
fields; `vmap` and `lax.scan` turn them into a masked episode-event stream.

## Rendering

Install the optional renderer interface with `uv sync --extra render`. Rendering is an
explicit host-side operation outside `jit`, `vmap`, and `lax.scan`:

```python
import jax
from PIL import Image

from jax_shapley.envs import Gridworld, RenderConfig

env = Gridworld()
params = env.default_params
_, state = env.reset(jax.random.key(0), params)
frame = env.render(state, params, RenderConfig(width=640, height=480))

assert frame.shape == (480, 640, 3)
assert frame.dtype.name == "uint8"
Image.fromarray(frame).save("gridworld.webp", lossless=True)
```

The public array remains RGB. The committed documentation WebPs use the same renderer with an internal alpha-aware export path so their outer canvas is transparent while boards, legends, and status panels remain opaque and readable.

Call `render()` after each already-compiled step to capture an episode, or retain selected
states from a compiled rollout and render them afterwards. Encoding images or video remains
the caller responsibility. For an `EpisodeStatisticsWrapper` state, render with
`wrapped_env.env.render(state.env_state)`.

See the [environment rendering guide](docs/Environment%20Rendering.md) for environment
cards, higher-dimensional Hypercube projections, and the Mastermind secret policy.

## Validation

Offline fixtures were exhaustively captured from FastSVERL commit
`25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b`; normal tests never download,
install, or import FastSVERL.

```bash
JAX_PLATFORMS=cpu uv run --extra cuda pytest
JAX_PLATFORMS=cuda uv run --extra cuda pytest tests/test_env_transition_parity.py -m gpu
uv run ruff check .
```

GPU-specific commands log that they were skipped and exit successfully when no JAX GPU
backend is available.

CPU and GPU each match all **143,136** source transitions with zero differences.
See the [environment port report](docs/Quantitative%20Comparison%20between%20Gymnasium%20and%20JAX%20environment%20implementations%20of%20FastSVERL.md) for raw
CPU/GPU throughput, speedups, methodology, and reproduction commands.

![Original vs JAX throughput](docs/figures/envs-port--throughput.svg)

Agents, FastSVERL models, replay buffers, training, and paper evaluation
reproduction are intentionally deferred to the later end-to-end JAX work.
