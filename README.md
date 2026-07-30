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
from jax_shapley.envs import Gridworld, Hypercube, Mastermind222

env = Hypercube(side_length=6, dimension=5)
params = env.default_params
obs, state = env.reset(jax.random.key(0), params)
obs, state, reward, done, info = env.step(
    jax.random.key(1), state, action=0, params=params
)
```

The API is Gymnax-native. Public `step()` auto-resets terminal instances;
`step_env()` exposes raw FastSVERL-compatible transitions. Gridworld preserves
FastSVERL's `[0, 0]` terminal-observation quirk. Hypercube supports side lengths
2–6 and dimensions 2–5 as static JIT specializations. Mastermind-222 preserves
the source action decoding, eight-feature board, clues, rewards, and horizon.

Vectorization uses the standard JAX composition:

```python
batched_reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
batched_step = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))
# Place batched_step in jax.lax.scan for a fully compiled rollout.
```

## Validation

Offline fixtures were exhaustively captured from FastSVERL commit
`25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b`; normal tests never download,
install, or import FastSVERL.

```bash
JAX_PLATFORMS=cpu uv run --extra cuda pytest
uv run --extra cuda pytest -m gpu
uv run ruff check .
```

CPU and GPU each match all **143,136** source transitions with zero differences.
See the [environment port report](docs/Quantitative%20Comparison%20between%20Gymnasium%20and%20JAX%20environment%20implementations%20of%20FastSVERL.md) for raw
CPU/GPU throughput, speedups, methodology, and reproduction commands.

![Original vs JAX throughput](docs/figures/envs-port--throughput.svg)

Agents, FastSVERL models, replay buffers, training, and paper evaluation
reproduction are intentionally deferred to the later end-to-end JAX work.
