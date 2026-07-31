# Mastermind-222 Environment Specification

`Mastermind222` is the end-to-end JAX/Gymnax port of the fixed FastSVERL Mastermind-222 configuration: a hidden code of length 2, at most 2 guesses, and an alphabet of 2 peg values.

![Rendered Mastermind-222 state](../../../docs/figures/envs-render--mastermind-222.webp)

| Property | Value |
| --- | --- |
| Public class | `jax_shapley.envs.Mastermind222` |
| Source environment | FastSVERL `Mastermind-v0`, configured as 222 |
| Code length | 2 |
| Maximum guesses | 2 |
| Peg values | 1 and 2 |
| Hidden codes | 4 |
| Observation space | `Box(-1, 2, shape=(8,), dtype=int32)` |
| Action space | `Discrete(4)` |
| Deterministic transitions | Yes |
| Stochastic reset | Yes; uniform hidden code |
| Reversible | No |
| Time limit | Two guesses, represented as termination rather than truncation |
| JAX transformations | `jit`, `vmap`, and `lax.scan` |

## Objective

The environment samples a hidden two-peg code. On each turn, the agent submits one complete two-peg guess and receives duplicate-aware clues:

- `exact`: pegs with the correct value in the correct position.
- `misplaced`: correct values in the wrong position, excluding already exact matches and without double-counting duplicates.

The agent wins by obtaining two exact matches. An episode terminates immediately on a correct guess or after the second guess.

## Construction and parameters

```python
from jax_shapley.envs import Mastermind222

env = Mastermind222()
params = env.default_params
```

`Mastermind222Params.max_steps_in_episode` is 2. The transition rule itself enforces this horizon by terminating the second guess; `truncated` remains `False`.

## Observation space

The public observation is the `2 x 4` board flattened row-major into an `int32` vector with shape `(8,)`. Each board row has the source-compatible ordering:

```text
[misplaced, first peg, second peg, exact]
```

| Flat indices | Board row | Meaning |
| --- | --- | --- |
| 0–3 | Guess 1 | `[misplaced, peg 0, peg 1, exact]` |
| 4–7 | Guess 2 | `[misplaced, peg 0, peg 1, exact]` |

Every unused cell is `-1`. Used clue cells contain 0, 1, or 2, and used peg cells contain 1 or 2. The declared `Box(-1, 2)` covers the common numeric envelope; positional semantics determine which values are meaningful in each column.

The hidden goal is deliberately absent from the observation.

## Environment state

`Mastermind222State` is a Flax PyTree containing the public board and private Markov information.

| Field | Shape and dtype | Meaning |
| --- | --- | --- |
| `board` | `(2, 4)`, `int32` | Guess and clue history; unused cells are `-1` |
| `goal` | `(2,)`, `int32` | Hidden code, with each peg in `{1, 2}` |
| `guess_index` | scalar, `int32` | Row written by the next raw transition; 0, 1, then 2 after completion |
| `terminated` | scalar, `bool` | Whether the raw episode has ended |
| `time` | scalar, `int32` | Number of guesses taken in the current raw episode |

Policies should consume the observation rather than `state.goal`. The complete state contains the goal because it is required for Markov transition computation.

## Action space and decoding

Actions encode guesses using FastSVERL's least-significant-position-first convention.

| Action | Decoded guess |
| ---: | --- |
| 0 | `[1, 1]` |
| 1 | `[2, 1]` |
| 2 | `[1, 2]` |
| 3 | `[2, 2]` |

For action `a`, peg position `i` is `(a // 2**i) % 2 + 1`. Actions must be valid members of `Discrete(4)`.

## Reset

`reset_env(key, params)` samples uniformly from the four hidden codes, fills the board with `-1`, and sets `guess_index=0`, `terminated=False`, and `time=0`. The observation is eight `-1` values.

JAX and NumPy random-number streams are not expected to produce seed-for-seed identical goals; reset support across all four goals is tested instead.

## Clue calculation

The transition first identifies exact positions. It removes those positions from both sequences, counts each remaining peg value, and sums the per-value minimum counts to obtain `misplaced`. This preserves duplicate-aware Mastermind semantics without Python loops or mutation.

Examples for hidden goal `[1, 2]`:

| Guess | Misplaced | Exact |
| --- | ---: | ---: |
| `[1, 2]` | 0 | 2 |
| `[2, 1]` | 2 | 0 |
| `[1, 1]` | 0 | 1 |
| `[2, 2]` | 0 | 1 |

## Goal, rewards, and termination

The reward is `-1 + 2` for a correct guess and `-1` otherwise.

| Condition | Reward | Terminated |
| --- | ---: | --- |
| Incorrect first guess | `-1.0` | `False` |
| Correct guess on either turn | `+1.0` | `True` |
| Incorrect second guess | `-1.0` | `True` |

Every raw transition writes one complete board row and increments both `guess_index` and `time` by one. The environment never reports truncation; exhausting the two guesses is a terminal condition.

## Raw transition and automatic reset

`step_env()` returns the completed board and raw terminal state when a guess wins or exhausts the second turn. Gymnax's public `step()` preserves the reward and returned `done=True`, but automatically replaces the terminal observation and state with a new empty board and a newly sampled hidden goal.

## Information dictionary

Every raw or public step returns fixed-shape information:

| Key | Dtype | Value |
| --- | --- | --- |
| `discount` | `float32` | `0.0` on termination, otherwise `1.0` |
| `terminated` | `bool` | Raw transition termination flag |
| `truncated` | `bool` | Always `False` |

Wrapping the public environment in `EpisodeStatisticsWrapper` additionally emits masked completed-episode return, length, and completion events under `info["episode"]`, while storing running return, running length, and completed-episode count in the wrapper state.

## Rendering and hidden information

`env.render(state, params=None, render_config=None)` returns a deterministic, C-contiguous `uint8[height, width, 3]` RGB array. It shows guesses oldest-first, peg values, exact and misplaced counts, active and unused rows, outcome, termination, guesses used, step count, and a complete reading legend.

Ordinary rendering is secret-safe: changing only `state.goal` produces an identical image. `RenderConfig(reveal_secret=True)` explicitly opts into an additional panel labelled `SECRET (debug)` and must not be used for agent observations or blinded evaluation.

Rendering is a lazy, optional, host-side operation. It accepts one unbatched state and must run outside `jit`, `vmap`, and `lax.scan`. Install its supported dependencies with `uv sync --extra render`.

## JAX execution

Reset, clue computation, board update, automatic reset, vectorisation, and compiled rollouts contain only JAX operations and fixed-shape PyTrees. The recursive Python state enumeration in the source is intentionally not ported because direct transition computation is sufficient for rollouts.

## Parity and provenance

The offline reference fixture exhaustively covers 64 valid transitions: 4 goals crossed with all 4 first actions, followed by all 4 second actions after each of the 3 incorrect first guesses per goal. CPU and GPU both have zero mismatches in observation, every state field, reward, termination, and truncation against FastSVERL commit `25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b`.

- [Pinned FastSVERL source](https://github.com/djeb20/fastsverl/blob/25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b/fastsverl/fastsverl/envs/mastermind.py)
- Source SHA-256: `16ae8c47583ddf506ab074c965f5c0099c048a23d9090b3704e257e758663b5a`

## Scope and known differences

- Only Mastermind-222 is implemented; larger FastSVERL configurations remain out of scope.
- The released FastSVERL action and board encodings are authoritative where they differ from intuition.
- Seed-for-seed NumPy/JAX goal identity is not required.
- The hidden goal is part of the environment state but never part of the public observation or ordinary render.
- Rendering is host-side and excluded from compiled environment throughput.
- Agents, replay buffers, training, and paper evaluation reproduction are outside this environment card.
