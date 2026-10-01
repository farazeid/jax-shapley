# DQN training

DQN ports FastSVERL commit `25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b` to
JAX and Flax NNX. The behavioral sources are `dqn.py`, `utils.py::Buffer`, and
`training.py::train_dqn/evaluate_dqn`. Baxtub supplies the explicit-state,
small-function organization; FastSVERL supplies the algorithm and settings.

## Shell experiments and CLI

Experiment values live in `experiments/dqn/*.sh`, not YAML or Python preset files.
Python dataclasses define their typed schema and validation. Tyro parses once,
then environment metadata and hidden widths resolve the existing `MLPConfig`.

```bash
uv sync --all-groups
bash experiments/dqn/gridworld.sh --validate-only
bash experiments/dqn/mastermind222.sh --compile-only
bash experiments/dqn/smoke.sh --output /tmp/jax-shapley-smoke
```

Normal recipes run `./check.sh` before starting their jobs. Inspection modes skip
those gates because they launch no training. Every normal CLI invocation traces,
lowers, and compiles initialization, training, and enabled evaluation together from
abstract inputs before creating a run directory or starting writers. Resume validates
and restores the saved arrays before compiling its continuation. Compilation failure
stops execution; there is no eager fallback.

Run an individual experiment directly, or inspect its generated help:

```bash
uv run python -m jax_shapley.train_dqn gridworld --help
uv run python -m jax_shapley.train_dqn mastermind222 \
    --train.num-envs 2 \
    --train.total-timesteps 32 \
    --train.buffer-size 7 \
    --train.batch-size 3 \
    --train.learning-starts 0 \
    --train.train-frequency 1 \
    --hidden-widths 4 \
    --chunk-steps 4 \
    --train.eval-episodes 2
```

Recipes append extra arguments, so repeated scalar flags use the last value.
The Hypercube recipe defaults to all 20 paper configurations and seeds 1–20,
executed sequentially. Its seed and environment selectors reduce the sweep:

```bash
bash experiments/dqn/hypercube.sh \
    --env.side-length 6 --env.dimension 5 --seed 3 --validate-only
```

The Hypercube recipe reproduces only the agent-training part of the upstream
combined agent/characteristic/Shapley experiment. Larger Mastermind environments,
PPO training, explanation models, tracking services, and paper-length performance
validation remain separate work.

## Settings and units

`AdamConfig`, `DQNTrainConfig`, and `DQNRunConfig[EnvConfig]` are frozen, slotted
dataclasses. The three environment subcommands expose only their own parameters.
Network dimensions come from the environment, not user-supplied feature counts.

| Option | Gridworld baseline | Meaning |
| --- | --- | --- |
| `--train.num-envs` | 1 | Parallel environments per vector step |
| `--train.total-timesteps` | 50,000 | Requested transitions across all environments |
| `--train.learning-rate` | 0.00025 | Constant Adam learning rate |
| `--train.gamma` | 1 | Discount, including the paper's undiscounted setting |
| `--train.buffer-size` | 50,000 | Replay transition capacity |
| `--train.batch-size` | 64 | Samples per learning update, with replacement |
| `--train.learning-starts` | 10,000 | Strict warmup threshold in transition units |
| `--train.train-frequency` | 10 | Upstream modulo condition in transition units |
| `--train.target-network-frequency` | 1 | Independent target-update modulo condition |
| `--train.tau` | 0.01 | Soft target update coefficient |
| `--train.start-e`, `--train.end-e` | 1, 0.01 | Epsilon schedule endpoints |
| `--train.exploration-fraction` | 0.5 | Decay duration relative to requested total transitions |
| `--train.adam.beta1`, `--train.adam.beta2` | 0.9, 0.999 | Adam moment decay coefficients |
| `--train.adam.epsilon` | 1e-8 | Epsilon outside the square root; root epsilon is zero |
| `--train.eval-episodes` | 0 | Optional greedy evaluation; zero disables it |
| `--hidden-widths` | 64 64 | ReLU hidden layers; use 128 128 for Hypercube |
| `--seed` | 1 | Explicit JAX PRNG seed |
| `--chunk-steps` | 64 | Vector steps per internal scan block and pending metrics buffer |
| `--checkpoint-mode` | `run` | `none`, `agent`, `agent-replay`, or full `run` snapshot |
| `--output` | `runs` | Parent directory for uniquely named runs |

There is no PPO-style rollout length in DQN. `chunk_steps` groups execution and
bounded event buffering without changing learning timing, random streams, or replay
data. The outer loop and each internal block execute inside the same compiled call.
Inactive tail steps neither update state nor consume keys. Logging and checkpoint
events do not split execution into Python training-loop calls.

Mastermind-222 uses 16 environments, 100,000 transitions, and exploration fraction
0.25. Hypercube uses 16 environments, width 128, end epsilon 0.05, and learning
starts at `total_timesteps // 50`; its shell table preserves upstream's varying
training budgets. Shell recipes explicitly state the learning and Adam settings.

## Behavioral contracts

- Greedy probabilities divide equally among all maximizing actions. Exploration
  mixes those probabilities with a uniform policy. Sampling uses categorical
  probabilities; reported log probabilities match Torch's normalization/clamping.
- `global_step = vector_step * num_envs`. Training runs for
  `total_timesteps // num_envs` vector steps, retaining upstream's floor division.
- Learning requires `global_step > learning_starts` and
  `global_step % train_frequency < num_envs`. A frequency below `num_envs` therefore
  triggers an update every eligible vector step, rather than multiple updates.
- Replay insertion precedes learning. Target updates follow learning and use the
  independent target-frequency condition after the same strict warmup gate.
- Targets are `reward + gamma * max(target_Q(raw_next_obs)) * (1 - terminated)`.
  Truncation permits bootstrapping. Collection retains raw observations while the
  next interaction uses Gymnax's auto-reset observations.
- Replay stores int32 observations/actions, bool termination, and float32 rewards
  and log probabilities. Oversized insertions retain their final capacity items
  and advance the pointer by that retained count, matching upstream.
- Adam has no gradient clipping, weight decay, or learning-rate annealing. The
  target starts as an independent copy of the online network.
- Evaluation resets the environments and samples from greedy tie probabilities.
  It records every completed episode in the final vector step, so its episode
  count may exceed the requested minimum by at most `num_envs - 1`.

Evaluation has the source's unbounded wait for completed episodes. A greedy agent
can fail to terminate in Gridworld/Hypercube; evaluation is disabled by default.

## Execution and host boundaries

Fresh runs use one ahead-of-time compiled invocation for model/optimizer creation,
replay allocation, environment reset, all training blocks, greedy evaluation and
final event emission. Resume uses one compiled continuation and does not recreate
the restored numerical state. The runtime never retraces inside either loop.
Different static configurations require new preflight specializations.

CLI parsing, saved-settings validation, compilation, run-folder creation, CSV/Orbax
writers, final synchronization and summary writing remain on the host. Ordered
`jax.experimental.io_callback` adapters transfer numerical payloads at explicit
boundaries. They stay outside differentiated losses. Compilation invokes no
callbacks and creates no run artifacts, writers or worker threads. There is no eager
fallback. Orbax owns its asynchronous writer; callbacks serialize requests, wait
for the previous save and copy an owned snapshot. Only one save is in flight.

## Selective logging

The pinned trainer selects every completed training episode's return, length and
epsilon, and logs loss/SPS only when an update occurs and
`global_step % 100 < num_envs`. Evaluation records each collected return, including
the final vector-step overshoot. CSV preserves those selected events and their
order; it does not add averaging or drop buffered events. Update rows have loss
and SPS; epsilon belongs to episode rows. TensorBoard/WandB integrations remain
deferred.

Selection is lossy relative to the full training history, as upstream is. Delivery
of the selected records is lossless. Pending storage is bounded by `chunk_steps`
and `num_envs`, independent of total training duration. It flushes on capacity,
selected loss/SPS events, checkpoint boundaries and completion. Evaluation streams
completed returns through a vector-sized buffer rather than retaining all returns.

SPS is the integer `global_step / elapsed_seconds`, measured on the host. Elapsed
training time includes compiled initialization and host callback/checkpoint time.
Resume accumulates committed elapsed time and excludes preflight/downtime.
Clock values cannot match PyTorch timings and are excluded from numerical parity.

## Local artifacts and Orbax recovery

Each run creates `<output>/<environment>-<seed>-<timestamp>/` with:

- `config.json`: resolved typed settings, architecture and pinned reference.
- `metrics.csv`: episode, selected loss/SPS and optional evaluation records.
- `summary.json`: executed transitions, update/evaluation counts, backend and
  elapsed training time through the pre-evaluation boundary. Written only after
  JAX effects and required checkpoint writes finish successfully.
- `checkpoints/`: atomic Orbax composite snapshots, retaining the latest committed
  sequence. Agent, replay, run-state and metadata are separate items.
- `.writer.lock`: a POSIX advisory lock preventing concurrent writers in one folder.

Checkpoint modes replace the previous save booleans:

| Mode | Saved data | Resume CLI |
| --- | --- | --- |
| `none` | No checkpoints; JSON/CSV remain | Unavailable |
| `agent` | Online/target parameters, epsilon and step | Unavailable |
| `agent-replay` | Agent plus complete replay arrays/pointer/size | Unavailable |
| `run` (default) | Agent/replay plus optimizer, key, observations, environment/episode state, counters and execution phase | Supported |

Periodic saves keep upstream's modulo cadence. Tiny runs guard its division by
zero with an interval of at least one transition. A post-training save covers
floor-division boundary misses and commits the state before evaluation. A final
completion save can occur at the same vector step, using a separate monotonically
increasing checkpoint sequence. The previous committed snapshot survives until
its replacement commits. Required callback or asynchronous write failures abort
execution and prevent a success summary.

NPZ checkpoint loading has been removed; offline numerical NPZ fixtures remain.
Loaders validate the schema/reference, saved settings, architecture, original
shapes/dtypes, finite values, phase/counters and replay invariants before use.
Arrays preserve the destination template's sharding. Select a local device with
`jax.default_device(device)` when loading agents/replay or constructing a run
restoration template. Full restoration preserves that template's placement even
when called outside the device context, including replay and PRNG data. Loading
agent or replay items does not read unrelated numerical items.

```python
from pathlib import Path
from jax_shapley.checkpoints import load_agent, load_replay
from jax_shapley.networks import dqn_config

folder = Path("runs/<run-name>")
agent = load_agent(
    folder,
    dqn_config(input_features=8, num_actions=4, hidden_widths=(64, 64)),
    load_replay=False,
)
replay = load_replay(folder)  # Requires agent-replay or run mode.
q_values = agent.model(observations).q_values
```

Full snapshots support downstream portable agent/replay loading and exact
continuation on the same backend. Cross-device loading is supported; CPU/GPU
floating-point trajectories need not be bitwise identical. Use the saved lockfile
and compatible checkpoint schema when loading state.

Snapshots record the key's PRNG implementation (`threefry2x32`, `rbg` or
`unsafe_rbg`) alongside its raw data. Restoration uses that saved implementation,
even when the destination's current JAX default differs. Existing Threefry
checkpoints remain compatible; older RBG snapshots incorrectly labelled as
Threefry are rejected rather than silently reinterpreted.

```bash
uv run python -m jax_shapley.train_dqn resume --run runs/<run-name> --validate-only
uv run python -m jax_shapley.train_dqn resume --run runs/<run-name> --compile-only
uv run python -m jax_shapley.train_dqn resume --run runs/<run-name>
```

Resume uses immutable saved settings, without algorithm overrides or duration
extensions, and writes into the same folder. Each committed snapshot records a CSV
byte boundary and SHA-256 prefix digest. Preflight verifies/restores before taking
the writer lock; the writer rechecks the checkpoint sequence and prefix, rolls back
only its uncommitted tail, and appends. If another writer committed during
preflight, resume stops without changing the log; retry to restore the latest
checkpoint. Changed/missing committed rows are rejected. Inspection modes
are read-only, including file contents and timestamps.

Interrupted evaluation restarts from the committed post-training checkpoint and
its saved key after rolling back uncommitted evaluation rows. A completed
checkpoint finalizes a missing summary without repeating training or evaluation.

## Validation and devices

Run `./check.sh`: ty, Ruff lint, Ruff formatting, and offline pytest. DQN checks
separate configuration, replay, numerical reference parity, collection/scheduling,
NNX backend compilation, CLI/artifacts, and captured shell-command recipes.

`tests/fixtures/fastsverl_dqn.npz` captures source probabilities/log probabilities,
TD targets, losses, gradients, three Adam steps, soft target updates, and epsilon
examples. The collector executes verified, unchanged upstream DQN classes with
controlled numerical dependencies, batches, and weights; it does not initialize
telemetry. Regenerate only in a separate PyTorch environment:

```bash
uv run --script scripts/networks_port/collect_fastsverl_dqn.py \
    /path/to/pinned/fastsverl tests/fixtures/fastsverl_dqn.npz
```

Numerical tolerances are `rtol=3e-5, atol=3e-6` for learning values, gradients, and
updated parameters. Matching seeds do not make Torch/JAX random streams identical;
fixtures control inputs instead. Chunk equivalence checks also compare replay,
parameters, optimizer/environment state, counters and final keys exactly.

`fastsverl_logging.json` captures the unchanged pinned trainer's selected scalar
events using controlled agent/environment inputs in the separate reference
environment. Its collector verifies the source hash:

```bash
uv run --script scripts/networks_port/collect_fastsverl_logging.py \
    --source /path/to/pinned/fastsverl/fastsverl/training.py
```

Compile gates cover fresh/resumed paths, every checkpoint mode, optional evaluation
and callback branches with inert sinks. Integration tests cover training/evaluation
interruptions, exact same-backend numerical continuation, CSV rollback, inspection
immutability, completion recovery, corrupt arrays and asynchronous completion errors.

Apple Silicon uses CPU JAX for iteration and the short smoke recipe. CUDA needs
`uv sync --all-groups --extra cuda`; set `JAX_PLATFORMS=cuda` before invocation to
require that backend. GPU compile tests skip explicitly without an available GPU.
CPU validation establishes no GPU compilation or speed claim. No paper-length
training runs or experiment-scale performance benchmarks are part of this branch's acceptance.

## Small CPU timing check

On this M4 Pro CPU, a Mastermind-222 run with 32 transitions, 15 updates, two
requested evaluation episodes, width 4, replay capacity 7, batch 3 and block size 4
was timed with selective CSV logging and full Orbax writes enabled. One warmup per
path preceded five alternating repetitions; both paths made 12 checkpoint saves
and waited for all writes. Setup/summary JSON writes were excluded equally.

| Path | Numerical entry-point calls | Median execution including writes | Compilation |
| --- | --- | --- | --- |
| Whole run | 1 | 1.567 s | 0.802 s |
| Host chunk orchestration | 13 (initialize + 11 chunks + evaluation) | 1.573 s | 0.594 s |

Checkpoint I/O dominates this tiny example; the difference is negligible.
Compilation was measured separately in one process and is not a cold-process
comparison. The host baseline also performs scalar checkpoint-payload operations;
they are not counted as numerical entry points. This establishes the dispatch
change, not an experiment-scale speedup or GPU performance claim. Raw timings
are in [dqn-whole-run-cpu-timing.json](data/dqn-whole-run-cpu-timing.json).
