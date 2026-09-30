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
lowers, and compiles initialization, training, and enabled evaluation from abstract
state before allocating replay or creating a run directory. Compilation failure
stops execution; there is no eager fallback.

Run an individual experiment directly, or inspect its generated help:

```bash
uv run python -m jax_shapley.train_dqn gridworld --help
uv run python -m jax_shapley.train_dqn mastermind222 \
    --train.num-envs 2 --train.total-timesteps 32 \
    --train.buffer-size 7 --train.batch-size 3 \
    --train.learning-starts 0 --train.train-frequency 1 \
    --hidden-widths 4 --chunk-steps 4 --train.eval-episodes 2
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
| `--chunk-steps` | 64 | Vector steps per compiled host chunk |
| `--output` | `runs` | Parent directory for uniquely named runs |

There is no PPO-style rollout length in DQN. `chunk_steps` groups execution and
host reporting without changing learning timing, random streams, or replay data.
Inactive tail steps neither update state nor consume keys. Host chunks stop at
checkpoint events to preserve the source save cadence.

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

## Local artifacts

Each run creates `<output>/<environment>-<seed>-<timestamp>/` with:

- `config.json`: resolved typed settings, architecture, and pinned reference commit.
- `metrics.csv`: completed episodes, source-cadence losses (modulo 100 transition
  units), and optional evaluation returns.
- `summary.json`: executed transitions, learning/target update counts, evaluation
  count, backend, and elapsed training/orchestration time. This is not a benchmark.
- `model.npz`: online/target float32 parameters, epsilon, and vector-step counter.
- `replay.npz`: typed replay arrays, size, pointer, and matching vector-step counter.

`--no-save-model` and `--no-save-replay` independently disable the optional files.
Saving replaces the latest files at the upstream ten-part cadence and at the end
of a run. Tiny runs use an interval of at least one transition to avoid the
source's host-only division by zero below ten timesteps. A final save also covers
floor-divided runs that miss the last source checkpoint event.

Files use numerical arrays without pickle. Writes replace each archive atomically;
loaders reject invalid shapes/dtypes, nonfinite floating values, invalid counters,
and mixed-step model/replay pairs left by interrupted writes.

```python
from pathlib import Path
from jax_shapley.artifacts import load_checkpoint
from jax_shapley.networks import dqn_config

agent = load_checkpoint(
    Path("runs/<run-name>"),
    dqn_config(input_features=8, num_actions=4, hidden_widths=(64, 64)),
)
q_values = agent.model(observations).q_values
```

Use `load_replay=False` when replay was not saved. Checkpoints support downstream
agent/explanation work; optimizer/environment state and exact interrupted-run
resumption are deliberately not included.

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
parameters, and final keys exactly.

Apple Silicon uses CPU JAX for iteration and the short smoke recipe. CUDA needs
`uv sync --all-groups --extra cuda`; set `JAX_PLATFORMS=cuda` before invocation to
require that backend. GPU compile tests skip explicitly without an available GPU.
CPU validation establishes no GPU compilation or speed claim. No paper-length
training runs or performance benchmarks are part of this branch's acceptance.
