# DQN training

DQN ports FastSVERL commit `25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b` to
JAX and Flax NNX. The behavioral sources are `dqn.py`, `utils.py::Buffer`, and
`training.py::train_dqn/evaluate_dqn`. Baxtub supplies the explicit-state,
small-function organization; FastSVERL supplies the algorithm and settings.

## Source layout and reading order

`src/jax_shapley/algorithms/dqn.py` holds the whole algorithm: `DQNConfig`, the
numerical state (`RunState`, `ReplayState`, `TransitionBatch`), metrics, `QNetwork`
and the functions, in reading order:

- `main(config)` trains (or loads) one agent per seed, tracked by `config.track`,
  and returns the seed-stacked final state, metrics and the explained `Agent`.
- `train` initializes a run and scans `step` over every vector step, then runs
  `evaluate` when `eval_episodes > 0`.
- `step` is one vector step in execution order: action selection, environment
  interaction, replay insertion, online learning and the soft target update.
  The continual-learning experiment reuses it unchanged.
- `log_step` and `log_evaluation` are the host callbacks that log the source's
  selected scalars.
- `initialize`, `trained_agent`, `choose_action`, `log_probabilities`,
  `environment_step` and `dqn_loss` are the lower-level operations.

Each seed is one compiled program: `nnx.jit(partial(train, ...))` compiles it at
the call site. Environments are frozen dataclasses and static compilation inputs.

## Running agents

A DQN agent is the first stage of every experiment (see the README's Experiments
section). Its settings are the `algorithm` field of a config in
`src/jax_shapley/configs/`, and override on the command line:

```bash
uv run python -m jax_shapley.main mastermind222_behaviour --compile-only
uv run python -m jax_shapley.main mastermind222_behaviour --algorithm.total-timesteps 320 --algorithm.learning-starts 160
bash experiments/smoke.sh mastermind222_behaviour
```

`--compile-only` traces, lowers and compiles every stage and exits without
running or tracking anything. Compilation failure stops execution; there is no
eager fallback. `--algorithm.checkpoint GROUP--algorithm--dqn:latest` loads a
previously tracked agent instead of training one; its saved settings must match.

## Settings and units

`DQNConfig` and `AdamConfig` are frozen, slotted dataclasses, validated on
construction. Network dimensions come from the environment.

| Field | Gridworld baseline | Meaning |
| --- | --- | --- |
| `seeds` | (1,) | One agent per seed |
| `num_envs` | 1 | Parallel environments per vector step |
| `total_timesteps` | 50,000 | Requested transitions across all environments |
| `learning_rate` | 0.00025 | Constant Adam learning rate |
| `gamma` | 1 | Discount, including the paper's undiscounted setting |
| `buffer_size` | 50,000 | Replay transition capacity |
| `batch_size` | 64 | Samples per learning update, with replacement |
| `learning_starts` | 10,000 | Strict warmup threshold in transition units |
| `train_frequency` | 10 | Upstream modulo condition in transition units |
| `target_network_frequency` | 1 | Independent target-update modulo condition |
| `tau` | 0.01 | Soft target update coefficient |
| `start_e`, `end_e` | 1, 0.01 | Epsilon schedule endpoints |
| `exploration_fraction` | 0.5 | Decay duration relative to requested total transitions |
| `adam.beta1`, `adam.beta2` | 0.9, 0.999 | Adam moment decay coefficients |
| `adam.epsilon` | 1e-8 | Epsilon outside the square root; root epsilon is zero |
| `eval_episodes` | 0 | Optional greedy evaluation; zero disables it |
| `hidden_widths` | (64, 64) | ReLU hidden layers; (128, 128) for Hypercube |

Mastermind-222 uses 16 environments, 100,000 transitions, and exploration fraction
0.25. Hypercube uses 16 environments, width 128, end epsilon 0.05, and learning
starts at `total_timesteps // 50`; `experiments/hypercube_*__sweep.sh` preserves
upstream's varying training budgets.

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

## Tracking and logging

The pinned trainer selects every completed training episode's return, length and
epsilon, and logs the loss only when an update occurs and
`global_step % 100 < num_envs`. Each seed logs into its own run, named
`NAME--algorithm--dqn--seed-N`, under the source's tags: `agent/episodic_return`,
`agent/episodic_length`, `agent/epsilon` and `agent/DQN_loss`. A tracking backend
holds one value per tag and step, so episodes finishing in the same vector step log
their mean. Evaluation logs each return as `agent/eval_episodic_return`, by episode
number. SPS is not logged: it is not an experiment result.

The callbacks run on the host through `jax.debug.callback` and stay outside
differentiated losses. Compilation invokes no callbacks and opens no runs.

## Checkpoints

After training, the whole seed-stacked final state (networks, optimizer, replay,
environment state, counters and key) is saved with Orbax and pushed as one
artifact, `GROUP--algorithm--dqn`. The checkpoint records the settings it was
trained with, excluding tracking fields; loading it into different settings fails.
Resume is stage-level only: a stage loads a finished stage's checkpoint, and an
interrupted stage reruns.

## Validation and devices

Run `./check.sh`: ty, Ruff lint, Ruff formatting, and offline pytest.

`tests/fixtures/fastsverl_dqn.npz` captures source probabilities/log probabilities,
losses, gradients, three Adam steps, soft target updates, and epsilon examples.
The collector executes verified, unchanged upstream DQN classes with controlled
numerical dependencies, batches, and weights; it does not initialize telemetry.
Regenerate only in a separate PyTorch environment:

```bash
uv run --script scripts/fixtures/collect_fastsverl_dqn.py \
    /path/to/pinned/fastsverl tests/fixtures/fastsverl_dqn.npz
```

Numerical tolerances are `rtol=3e-5, atol=3e-6` for learning values, gradients, and
updated parameters. Matching seeds do not make Torch/JAX random streams identical;
fixtures control inputs instead.

`fastsverl_logging.json` captures the unchanged pinned trainer's selected scalar
events using controlled agent/environment inputs in the separate reference
environment. Its collector verifies the source hash:

```bash
uv run --script scripts/fixtures/collect_fastsverl_logging.py \
    --source /path/to/pinned/fastsverl/fastsverl/training.py
```

Apple Silicon uses CPU JAX for iteration and the smoke script. CUDA needs
`uv sync --all-groups --extra cuda`; set `JAX_PLATFORMS=cuda` before invocation to
require that backend. GPU compile tests skip explicitly without an available GPU.
CPU validation establishes no GPU compilation or speed claim.
