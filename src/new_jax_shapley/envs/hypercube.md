# Hypercube Environment Specification

`Hypercube` is the end-to-end JAX/Gymnax port of FastSVERL's deterministic `Hypercube-v0` environment for all 20 obstacle-free configurations used in the paper. It is an `n`-sided, `d`-dimensional integer lattice, not merely the corners of a geometric hypercube.

![Rendered five-dimensional Hypercube state](../../../docs/figures/envs-render--hypercube-5d.webp)

| Property | Value |
| --- | --- |
| Public class | `new_jax_shapley.envs.Hypercube` |
| Source environment | FastSVERL `Hypercube-v0` |
| Observation space | `Box(0, n-1, shape=(d,), dtype=int32)` |
| Action space | `Discrete(2d)` |
| Total lattice positions | `n^d` |
| Nonterminal states | `n^d - 1` |
| Deterministic reset and transitions | Yes |
| Obstacles | None |
| Episodically reversible | No; entering the goal ends the episode |
| Time limit | None |
| Supported side lengths | `n in {2, 3, 4, 5, 6}` |
| Supported dimensions | `d in {2, 3, 4, 5}` |
| JAX transformations | `jit`, `vmap`, and `lax.scan` |

## Geometry and objective

A position is a vector `(x0, ..., x(d-1))` in which every coordinate is an integer from 0 through `n-1`. The agent starts at `(0, ..., 0)` and succeeds by reaching `(n-1, ..., n-1)`. Each action changes exactly one coordinate by one.

For `d=2`, the environment is an ordinary `n x n` grid. For `d=3`, it is an `n x n x n` lattice. Dimensions 4 and 5 apply the same rule with additional coordinates.

The shortest start-to-goal trajectory contains `d(n-1)` actions. Many shortest trajectories exist because coordinate increments can be interleaved in different orders.

## Construction and parameters

```python
from new_jax_shapley.envs import Hypercube

env = Hypercube(side_length=6, dimension=5)
params = env.default_params
```

`side_length` and `dimension` are validated at construction and determine array shapes and action-space size. Each dimension is therefore a separate static JIT specialisation; batch together environments with the same construction.

`HypercubeParams` contains only `max_steps_in_episode = 2**31 - 1`, a Gymnax compatibility sentinel. The environment does not use it to truncate episodes and has no artificial timeout.

## Paper configurations

Every combination of the five side lengths and four dimensions is supported. A parity transition count is `(n^d - 1) x 2d` because every action is tested from every nonterminal state.

| `n x d` | Observation features | Actions | Total states | Nonterminal states | Shortest path | Goal reward | Parity transitions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2x2 | 2 | 4 | 4 | 3 | 2 | 3 | 12 |
| 3x2 | 2 | 4 | 9 | 8 | 4 | 5 | 32 |
| 4x2 | 2 | 4 | 16 | 15 | 6 | 7 | 60 |
| 5x2 | 2 | 4 | 25 | 24 | 8 | 9 | 96 |
| 6x2 | 2 | 4 | 36 | 35 | 10 | 11 | 140 |
| 2x3 | 3 | 6 | 8 | 7 | 3 | 5 | 42 |
| 3x3 | 3 | 6 | 27 | 26 | 6 | 8 | 156 |
| 4x3 | 3 | 6 | 64 | 63 | 9 | 11 | 378 |
| 5x3 | 3 | 6 | 125 | 124 | 12 | 14 | 744 |
| 6x3 | 3 | 6 | 216 | 215 | 15 | 17 | 1,290 |
| 2x4 | 4 | 8 | 16 | 15 | 4 | 7 | 120 |
| 3x4 | 4 | 8 | 81 | 80 | 8 | 11 | 640 |
| 4x4 | 4 | 8 | 256 | 255 | 12 | 15 | 2,040 |
| 5x4 | 4 | 8 | 625 | 624 | 16 | 19 | 4,992 |
| 6x4 | 4 | 8 | 1,296 | 1,295 | 20 | 23 | 10,360 |
| 2x5 | 5 | 10 | 32 | 31 | 5 | 9 | 310 |
| 3x5 | 5 | 10 | 243 | 242 | 10 | 14 | 2,420 |
| 4x5 | 5 | 10 | 1,024 | 1,023 | 15 | 19 | 10,230 |
| 5x5 | 5 | 10 | 3,125 | 3,124 | 20 | 24 | 31,240 |
| 6x5 | 5 | 10 | 7,776 | 7,775 | 25 | 29 | 77,750 |

Changing `n` at fixed `d` increases the lattice size and shortest path without changing the number of features or actions. Changing `d` adds an observed coordinate and two actions, multiplies the state count by `n`, and doubles the number of possible feature subsets relevant to Shapley explanations.

## Observation space

The observation is the current coordinate vector as an `int32` array with shape `(d,)`. Every component ranges from 0 through `n-1`.

| Component | Meaning | Values |
| --- | --- | --- |
| `observation[i]` | Position on axis `i` | `{0, ..., n-1}` |

## Environment state

`HypercubeState` is a Flax PyTree with fixed-shape JAX arrays.

| Field | Shape and dtype | Meaning |
| --- | --- | --- |
| `position` | `(d,)`, `int32` | Current coordinate vector |
| `terminated` | scalar, `bool` | Whether the raw transition has reached the goal |
| `time` | scalar, `int32` | Number of transitions taken in the current raw episode |

## Action space

There are two actions per dimension. Even actions increment an axis and odd actions decrement it.

| Action | Axis | Direction |
| ---: | ---: | ---: |
| `2i` | `i` | `+1` |
| `2i + 1` | `i` | `-1` |

For example, when `d=3`, actions 0 and 1 move along axis 0, actions 2 and 3 along axis 1, and actions 4 and 5 along axis 2. Actions must be valid members of `Discrete(2d)`.

## Reset

`reset_env(key, params)` deterministically returns `(0, ..., 0)`, with `terminated=False` and `time=0`. The key is accepted for Gymnax API compatibility but does not affect the result.

## Transition, goal, and rewards

The selected coordinate is incremented or decremented. If the candidate would put any coordinate outside `[0, n-1]`, the entire move is invalid and the old position is retained.

| Condition | Reward | Terminated |
| --- | ---: | --- |
| Ordinary or boundary-invalid step | `-1.0` | `False` |
| Enter `(n-1, ..., n-1)` | `nd - 1` | `True` |

On every raw transition, `time` increases by one. The environment never reports truncation.

## Raw transition and automatic reset

`step_env()` returns the entered goal coordinate as the raw terminal observation and state. Gymnax's public `step()` preserves the terminal reward and returned `done=True`, but automatically replaces the terminal observation and state with a reset at `(0, ..., 0)`.

## Information dictionary

Every raw or public step returns fixed-shape information:

| Key | Dtype | Value |
| --- | --- | --- |
| `discount` | `float32` | `0.0` on termination, otherwise `1.0` |
| `terminated` | `bool` | Raw transition termination flag |
| `truncated` | `bool` | Always `False` |

Wrapping the public environment in `EpisodeStatisticsWrapper` additionally emits masked completed-episode return, length, and completion events under `info["episode"]`, while storing running return, running length, and completed-episode count in the wrapper state.

## Rendering

`env.render(state, params=None, render_config=None)` returns a deterministic, C-contiguous `uint8[height, width, 3]` RGB array. Two-dimensional configurations render as ordinary grids. Dimensions 3–5 render axes 0 and 1 as a labelled projection, alongside full current and goal vectors and a coordinate track for every axis. The legend distinguishes `S`, `A`, projected goal `G*`, current and goal markers, nested equal markers, axis orientation, step count, and termination state.

Rendering is a lazy, optional, host-side operation. It accepts one unbatched state and must run outside `jit`, `vmap`, and `lax.scan`. Install its supported dependencies with `uv sync --extra render`.

## JAX execution

Reset, raw transition, automatic reset, vectorisation, and compiled rollouts contain only JAX operations and fixed-shape PyTrees. Transition dictionaries and state enumeration from the source are intentionally replaced by direct coordinate arithmetic. Same-shaped instances can be batched with `jax.jit(jax.vmap(...))` and rolled out with `jax.lax.scan`.

## Parity and provenance

The offline reference fixture exhaustively covers 143,052 transitions: every nonterminal state/action pair across all 20 paper configurations. CPU and GPU both have zero mismatches in observation, every state field, reward, termination, and truncation against FastSVERL commit `25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b`.

- [Pinned FastSVERL source](https://github.com/djeb20/fastsverl/blob/25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b/fastsverl/fastsverl/envs/hypercube.py)
- Source SHA-256: `af586a54482cd0606a8137ea6df54f0a1c8b594c6e4dfef7fb8c5aacf3361e75`

## Scope and known differences

- The port implements only the paper's obstacle-free configurations; FastSVERL's optional obstacle parameter `p` is out of scope.
- The released FastSVERL behavior is authoritative where it differs from the paper.
- No Python transition dictionaries, state enumeration, or mutable environment state are used in traced execution.
- Different dimensions cannot share one unpadded batch because their observation and state shapes differ.
- Rendering dimensions 3–5 is necessarily a projection and does not imply that hidden coordinates match the goal.
- Agents, replay buffers, training, and paper evaluation reproduction are outside this environment card.
