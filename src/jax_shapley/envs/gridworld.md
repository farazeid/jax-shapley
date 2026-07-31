# Gridworld Environment Specification

`Gridworld` is the end-to-end JAX/Gymnax port of FastSVERL's deterministic `GWB-v0` environment. It is a fixed 2 x 4 grid containing two possible start cells, one blocked cell, two terminal cells, and five nonterminal cells.

![Rendered Gridworld state](../../../docs/figures/envs-render--gridworld.webp)

| Property | Value |
| --- | --- |
| Public class | `jax_shapley.envs.Gridworld` |
| Source environment | FastSVERL `GWB-v0` |
| Observation space | `Box([0, 0], [1, 3], shape=(2,), dtype=int32)` |
| Action space | `Discrete(4)` |
| Nonterminal states | 5 |
| Terminal cells | 2 |
| Deterministic transitions | Yes |
| Stochastic reset | Yes; uniform over two starts |
| Episodically reversible | No; terminal transitions end the episode |
| Time limit | None |
| JAX transformations | `jit`, `vmap`, and `lax.scan` |

## Layout and objective

The state position is `(row, column)`, with row 0 at the top and column 0 at the left. The agent starts at `(0, 0)` or `(1, 0)`, cannot enter the blocked cell `(0, 1)`, and succeeds by entering either terminal cell in column 3.

```text
        column
          0    1    2    3
row 0     S    X    .    G
row 1     S    .    .    G
```

`S` denotes a possible start, `X` the blocked cell, `G` a terminal goal cell, and `.` another nonterminal cell.

## Construction and parameters

```python
from jax_shapley.envs import Gridworld

env = Gridworld()
params = env.default_params
```

`GridworldParams` contains only `max_steps_in_episode = 2**31 - 1`, a Gymnax compatibility sentinel. The environment does not use it to truncate episodes and has no artificial timeout.

## Observation space

The observation is the current position as an `int32` array with shape `(2,)`.

| Index | Meaning | Values |
| ---: | --- | --- |
| 0 | Row | 0 or 1 |
| 1 | Column | 0, 1, 2, or 3 |

The declared bounding box includes coordinates that are not exposed as ordinary reachable observations, including the blocked cell. FastSVERL's terminal convention also means that raw terminal transitions return `[0, 0]`, rather than the entered goal coordinate.

## Environment state

`GridworldState` is a Flax PyTree with fixed-shape JAX arrays.

| Field | Shape and dtype | Meaning |
| --- | --- | --- |
| `position` | `(2,)`, `int32` | Current observation-compatible position |
| `terminated` | scalar, `bool` | Whether the raw transition has terminated |
| `time` | scalar, `int32` | Number of transitions taken in the current raw episode |

## Action space

| Action | Delta | Meaning |
| ---: | --- | --- |
| 0 | `(0, +1)` | Move right |
| 1 | `(+1, 0)` | Move down |
| 2 | `(0, -1)` | Move left |
| 3 | `(-1, 0)` | Move up |

Actions must be valid members of `Discrete(4)`. An action that would leave the grid or enter `(0, 1)` leaves the agent in its current cell and still incurs the ordinary step reward.

## Reset

`reset_env(key, params)` samples uniformly from `(0, 0)` and `(1, 0)`, sets `terminated=False` and `time=0`, and returns the sampled position as the observation. JAX and NumPy random-number streams are not expected to produce seed-for-seed identical starts; reset support is tested instead.

## Transition, goal, and rewards

For valid nonterminal movement, the selected action delta is added to `position`. Invalid movement retains the old position.

| Condition | Reward | Terminated |
| --- | ---: | --- |
| Ordinary or invalid step | `-1.0` | `False` |
| Enter either cell in column 3 | `+9.0` | `True` |

On every raw transition, `time` increases by one. The environment never reports truncation.

## Raw transition and automatic reset

`step_env()` is the FastSVERL parity boundary. When the agent enters a goal, it returns reward `9.0`, `terminated=True`, and the deliberate FastSVERL raw terminal observation and state position `[0, 0]`.

Gymnax's public `step()` preserves the terminal reward and returned `done=True`, but automatically replaces the terminal observation and state with a newly reset episode. Code that needs the source-compatible raw terminal record must call `step_env()`.

## Information dictionary

Every raw or public step returns fixed-shape information:

| Key | Dtype | Value |
| --- | --- | --- |
| `discount` | `float32` | `0.0` on termination, otherwise `1.0` |
| `terminated` | `bool` | Raw transition termination flag |
| `truncated` | `bool` | Always `False` |

Wrapping the public environment in `EpisodeStatisticsWrapper` additionally emits masked completed-episode return, length, and completion events under `info["episode"]`, while storing running return, running length, and completed-episode count in the wrapper state.

## Rendering

`env.render(state, params=None, render_config=None)` returns a deterministic, C-contiguous `uint8[height, width, 3]` RGB array. The figure legend identifies both starts, the blocked cell, terminal cells, the current agent, coordinates, step count, and termination state. A raw terminal render explains the `[0, 0]` convention.

Rendering is a lazy, optional, host-side operation. It accepts one unbatched state and must run outside `jit`, `vmap`, and `lax.scan`. Install its supported dependencies with `uv sync --extra render`.

## JAX execution

Reset, raw transition, automatic reset, vectorisation, and compiled rollouts contain only JAX operations and fixed-shape PyTrees. Hundreds or thousands of same-shaped instances can be composed with `jax.jit(jax.vmap(...))` and `jax.lax.scan`.

## Parity and provenance

The offline reference fixture exhaustively covers all 5 nonterminal states crossed with 4 actions: 20 transitions. CPU and GPU both have zero mismatches in observation, every state field, reward, termination, and truncation against FastSVERL commit `25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b`.

- [Pinned FastSVERL source](https://github.com/djeb20/fastsverl/blob/25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b/fastsverl/fastsverl/envs/gwb.py)
- Source SHA-256: `fa63ba4c4376f0ca3a4e6707c1758739a6fd5084b736ccd230b9188ecaca48bb`

## Scope and known differences

- The released FastSVERL behavior is authoritative where it differs from the paper.
- Seed-for-seed NumPy/JAX trajectory identity is not required.
- The terminal `[0,0]` observation is intentionally preserved even though the conceptual grid has two terminal coordinates.
- Rendering is host-side and excluded from compiled environment throughput.
- Agents, replay buffers, training, and paper evaluation reproduction are outside this environment card.
