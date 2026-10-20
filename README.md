# jax-shapley

[FastSVERL](https://github.com/djeb20/fastsverl) ([*"Approximating Shapley Explanations in Reinforcement Learning"*](https://openreview.net/forum?id=lJYwfYcoZX)) ported from PyTorch and Gymnasium to end-to-end JAX (Flax NNX).

The environments, DQN agents, and behaviour, prediction and outcome explanations (characteristic and Shapley models) each run as single JIT compiled JAX programs with every seed of a stage in one program. The goal is the paper's results in a fraction of the experiment time.

Reference was FastSVERL commit `25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b`.

<br>
<br>
<br>
<br>
<br>

# Comparison

## Speed

Raw measurements are in `docs/data/`: environment throughput (`envs-port--throughput--*.csv`, including the Colab T4 and TPU v5e-1 runs), per-stage explanation timings against PyTorch (`explanations-port--{jax,upstream}-timing.csv`, `explanations-port--speed.csv`) and whole-experiment timings (`experiment-timings.csv`).

### Environments

Transitions per second with up to 2,048 parallel environments. Hypercube 6x5 on Gymnasium was capped to 16 parallel environments as no more would fit in memory.

On the same CPU (Intel i9-14900K), JAX is 42–120x faster than Gymnasium; on a GPU or TPU it is 265–2,315x faster. T4 and TPU v5e-1 are Google Colab runtimes with weaker CPUs.

| Environment | Gymnasium (CPU) | JAX (CPU) | JAX (TPU v5e-1) | JAX (GPU T4) | JAX (GPU RTX 4080 SUPER) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gridworld | 216,064 | 8,988,350 | 57,352,678 | 107,244,581 | 164,905,947 |
| Hypercube 2x2 | 220,402 | 26,517,196 | 90,418,341 | 106,900,071 | 180,082,076 |
| Hypercube 6x5 | 210,675 | 18,873,044 | 90,377,460 | 107,032,719 | 178,706,862 |
| Mastermind-222 | 66,670 | 6,619,422 | 62,254,573 | 91,638,506 | 154,320,168 |

![Environment throughput against the number of parallel environments](docs/figures/envs-port--throughput.svg)

### Explanation Training

Each stage of the FastSVERL's Gridworld, Mastermind-222, and Hypercube experiments. 20 seeds against the unchanged PyTorch code on the same Apple M4 Pro CPU. Wall time includes start-up and compilation. Per seed and epoch, every stage trains 5.4–10x faster.

| Experiment | Stage | PyTorch | JAX | Speedup |
| --- | --- | ---: | ---: | ---: |
| Gridworld | Characteristic | 244 s | 21 s | 11.7x |
| Gridworld | Shapley | 466 s | 40 s | 11.7x |
| Gridworld Outcome | Outcome Characteristic | 2,559 s | 260 s | 9.8x |
| Mastermind-222 | Characteristic | 271 s | 46 s | 5.9x |
| Mastermind-222 | Shapley | 525 s | 79 s | 6.7x |
| Mastermind-222 Outcome | Outcome Characteristic | 4,489 s | 1,074 s | 4.2x |
| Hypercube (3 cubes) | all stages | 435–458 s | 174–206 s | 2.2–2.5x |

### End-to-end Experiments

Every stage of one experiment, 20 seeds each, wall-clock including compilation; speedups against PyTorch on the same Apple M4 Pro CPU. JAX runs also log every metric with trackio and save each stage's checkpoint, which the PyTorch runs did not. The TPU uses JAX's default float32 matrix-multiply precision. Single runs: the same code varies by up to about 15% between repeats on the M4 Pro.

These networks are small, so the T4 and TPU v5e-1 add kernel-launch and host-callback overhead on Colab's weaker host CPUs, and the M4 Pro CPU is fastest; the accelerators pay off in environment throughput (above).

Why it is faster: environment steps, rollouts, training epochs and evaluation are traced once and compiled (`jit`, `vmap`, `scan`); all seeds of a stage run as one vectorised program; data stays on the device; the host only handles live-logging callbacks and checkpoints.

| Experiment | PyTorch, CPU | JAX, CPU | JAX, T4 GPU | JAX, TPU v5e-1 |
| --- | ---: | ---: | ---: | ---: |
| Gridworld Behaviour | 385 s | 52 s (7.4x) | 187 s | 133 s |
| Mastermind-222 Prediction | 423 s | 62 s (6.8x) | 235 s | 247 s |
| Mastermind-222 Outcome, off-policy | 2,372 s | 308 s (7.7x) | 382 s | 306 s |
| Mastermind-222 Outcome, on-policy | 908 s | 493 s (1.8x) | 1,199 s | 916 s |

## Parity

All 143,136 state transitions (next observations, state, rewards, termination and truncation) match identically between the original Gymnasium and  ported JAX environment.

The raw parity results (`envs-port--parity--{cpu,gpu}.csv`), the statistical comparison's runs and pre-registered margins (`explanations-port--{jax,upstream}-runs.csv`, `explanations-port--margins.csv`, `explanations-port--summary.csv`) and the paper comparison (`paper--comparison.csv`, `paper--claims-results.csv`, criteria in `paper--criteria.md`) are in `docs/data/`.

![Final exact losses, JAX against PyTorch](docs/figures/explanations-port--exact-loss.svg)

<details>
<summary>Deviations from original FastSVERL, and their effect</summary>

| Area | Upstream | Port | Effect |
| --- | --- | --- | --- |
| Random streams | torch and NumPy global generators | explicit JAX keys, split per run and per seed | Effect, runs are not seed-for-seed identical; fixture tests replay recorded draws, and the end-to-end comparison is statistical |
| Precision | float64 value iteration; float32 networks | float32 throughout | Effect, negligible: exact values agree within 1e-5 relative / 1e-4 absolute |
| Exact policy evaluation | value iteration to a 1e-10 change | direct linear solve of `(I − γ C_π) V = R_π` | No effect, same fixed point |
| Non-terminating conditioned policy (γ = 1) | value iteration never terminates (the run hangs) | the run stops with an error naming the number of non-finite exact values, before any training | No effect on results, neither gives one: the run fails fast instead of hanging. Only reachable with a policy that can loop forever, such as a weak greedy Gridworld agent; the paper does not discuss it |
| Exact outcome states | `torch.unique` of the exact buffer | transition-model states, marked valid when present in the exact buffer | No effect, same set of valid states |
| Null value `v(s, ∅)` | network on an all-masked batch | network on one all-masked row, broadcast | No effect, identical values; avoids an XLA CPU miscompilation of broadcast-constant batches |
| Off-policy target after early stopping | best model restored; the target keeps its last weights | the target saved with the best model is restored too | No effect, the target is not used after training |
| Outcome optimiser | Adam over the model only | one Adam over model and target | No effect, the target's gradients are exactly zero, so Adam leaves it unchanged |
| On-policy terminal transitions | stores the reset observation as `n_obs` | stores the pre-reset observation | No effect, masked by `(1 − terminated)` |
| Truncated transitions | `final_observation` | Gymnax pre-reset observation | No effect, same observation |
| Processes | one process per seed | every stage trains all its seeds in one compiled program | No effect, each seed uses the keys of a single run with that seed |
| Checkpoints | model saves every tenth of the run | each stage's final state as one checkpoint artifact; stage-level resume | No effect on results |
| Continual exact-buffer refresh | filled after acting, before the agent's update | filled before the step's DQN update | No effect, same pre-update policy and epsilon |
| Sampled characteristic with no matching pool row | every score ties; `torch.topk`'s tie order picks a row of the freshly permuted pool | the pool row with the top uniform score | No effect in distribution, both pick one uniform pool row; matched rows select identically |

</details>

## Paper Reproduction

The experiments marked run were run with this package and compared against targets extracted from upstream's own plot files, using criteria committed before any comparison: a curve matches at 25% and 100% of training when it is within a factor of 2 of the published mean or within 3 combined standard errors, a Hypercube bar within a factor of 1.5. 274 of 282 compared quantities match.

16 of the paper's 26 claims hold. Of the other 10, 7 need experiments run and 3 fail as described below. Figures of experiments not run show the published results only.

Claims that miss:

- **Figure 1, Mastermind-222.** The paper claims that an outcome characteristic, trained on-policy, ends with lower error (exact loss, lower is better) when the behaviour characteristic it builds on is exact rather than a learnt approximation (the figure's "exact behaviour" and "approx behaviour" lines). In JAX the two end almost level and in the opposite order: 0.0114 with the exact behaviour characteristic against 0.0105 with the approximate one.
- **Figure 8, Gridworld behaviour.** The paper claims that learning a behaviour characteristic from off-policy data with unnormalised importance weights ends with higher error than not reweighting the data at all. In JAX unnormalised weights end slightly lower: 0.0127 against 0.0145.
- **Figure 8, Gridworld prediction.** The same claim for a prediction characteristic. In JAX unnormalised weights again end slightly lower: 0.968 against 0.989.

| Paper | Experiment | Status | Comparison |
| --- | --- | --- | --- |
| Figures 1, 4 | Learn characteristic and Shapley models of the behaviour, prediction and outcome on Gridworld and Mastermind-222, with exact and approximate characteristics | Run | 62 of 68 quantities match |
| Figure 4 | The same behaviour and prediction explanations on the larger Mastermind-333 | Run | 11 of 12 match |
| Figures 2, 5 | How many updates characteristic and Shapley models need to reach a target loss, as Hypercube size and dimension grow the number of states | 31 of 40 cube x kind combinations Run | 61 of 62 bars match |
| Figures 3a, 7–9 | Learn characteristics from off-policy replay data, with and without importance weighting, normalised or not, and with clipping | Run | 58 of 58 match |
| Figures 3b, 10 | Train explanation models alongside the learning agent, at 1, 2, 10 and 50 explanation updates per agent update | Run | 64 of 64 match |
| Figures 3c, 11 | Replace the learnt characteristic with one estimated by sampling, for Shapley models and for outcome characteristics | Mastermind-222 behaviour and prediction run; its outcome panel, Gridworld and Mastermind-333 not run | 18 of 18 match |
| Tables 1–4, Figure 6 and appendix figures | Scale behaviour, prediction and outcome explanations to large Mastermind boards (443, 453, 463) | Not run | — |

Every figure below uses one visual language: JAX solid with filled markers, FastSVERL (published) dashed with open markers; one colour per concept across figures (for example blue for the exact characteristic or exact behaviour, orange for the approximate one, black for on-policy); circles for on-policy and triangles for off-policy data; shaded bands are 95% t intervals of the mean. Published bands are FastSVERL's +/-1 standard error, widened by t for its 20 runs. The outcome figures keep upstream's labels: *approx behaviour* runs are scored against the outcome under the exact behaviour characteristic, so their error includes the behaviour model's; *exact behaviour* runs are scored under the learnt behaviour, isolating the outcome model's error.

### Figures 1 and 4: Gridworld, Mastermind-222, and Mastermind-333 Explanations

#### Gridworld

<table><tr>
<td align="center"><img src="docs/figures/paper/app_gwb_policy_char.svg" height="500" alt="app_gwb_policy_char"><br><sub><code>app_gwb_policy_char</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_gwb_policy_shapley.svg" height="500" alt="app_gwb_policy_shapley"><br><sub><code>app_gwb_policy_shapley</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_gwb_value_char.svg" height="500" alt="app_gwb_value_char"><br><sub><code>app_gwb_value_char</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_gwb_value_shapley.svg" height="500" alt="app_gwb_value_shapley"><br><sub><code>app_gwb_value_shapley</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_gwb_perf_char.svg" height="500" alt="app_gwb_perf_char"><br><sub><code>app_gwb_perf_char</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_gwb_perf_shapley.svg" height="500" alt="app_gwb_perf_shapley"><br><sub><code>app_gwb_perf_shapley</code></sub></td>
</tr></table>

#### Mastermind-222

<table><tr>
<td align="center"><img src="docs/figures/paper/main_mastermind_222_policy_char.svg" height="500" alt="main_mastermind_222_policy_char"><br><sub><code>main_mastermind_222_policy_char</code></sub></td>
<td align="center"><img src="docs/figures/paper/main_mastermind_222_perf_char.svg" height="500" alt="main_mastermind_222_perf_char"><br><sub><code>main_mastermind_222_perf_char</code></sub></td>
<td align="center"><img src="docs/figures/paper/main_mastermind_222_perf_shapley.svg" height="500" alt="main_mastermind_222_perf_shapley"><br><sub><code>main_mastermind_222_perf_shapley</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_mastermind_222_policy_char.svg" height="500" alt="app_mastermind_222_policy_char"><br><sub><code>app_mastermind_222_policy_char</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_mastermind_222_policy_shapley.svg" height="500" alt="app_mastermind_222_policy_shapley"><br><sub><code>app_mastermind_222_policy_shapley</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_mastermind_222_value_char.svg" height="500" alt="app_mastermind_222_value_char"><br><sub><code>app_mastermind_222_value_char</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_mastermind_222_value_shapley.svg" height="500" alt="app_mastermind_222_value_shapley"><br><sub><code>app_mastermind_222_value_shapley</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_mastermind_222_perf_char.svg" height="500" alt="app_mastermind_222_perf_char"><br><sub><code>app_mastermind_222_perf_char</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_mastermind_222_perf_shapley.svg" height="500" alt="app_mastermind_222_perf_shapley"><br><sub><code>app_mastermind_222_perf_shapley</code></sub></td>
</tr></table>

#### Mastermind-333

<table><tr>
<td align="center"><img src="docs/figures/paper/app_mastermind_333_policy_char.svg" height="500" alt="app_mastermind_333_policy_char"><br><sub><code>app_mastermind_333_policy_char</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_mastermind_333_policy_shapley.svg" height="500" alt="app_mastermind_333_policy_shapley"><br><sub><code>app_mastermind_333_policy_shapley</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_mastermind_333_value_char.svg" height="500" alt="app_mastermind_333_value_char"><br><sub><code>app_mastermind_333_value_char</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_mastermind_333_value_shapley.svg" height="500" alt="app_mastermind_333_value_shapley"><br><sub><code>app_mastermind_333_value_shapley</code></sub></td>
</tr></table>

### Figures 2 and 5: Hypercube Scaling

<table><tr>
<td align="center"><img src="docs/figures/paper/hypercube_policy.svg" height="500" alt="hypercube_policy"><br><sub><code>hypercube_policy</code></sub></td>
<td align="center"><img src="docs/figures/paper/hypercube_value.svg" height="500" alt="hypercube_value"><br><sub><code>hypercube_value</code></sub></td>
</tr></table>

### Figures 3a and 7–9: Off-policy characteristics

#### Gridworld

<table><tr>
<td align="center"><img src="docs/figures/paper/app_gwb_offpolicy_weighting_policy.svg" height="500" alt="app_gwb_offpolicy_weighting_policy"><br><sub><code>app_gwb_offpolicy_weighting_policy</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_gwb_offpolicy_weighting_value.svg" height="500" alt="app_gwb_offpolicy_weighting_value"><br><sub><code>app_gwb_offpolicy_weighting_value</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_gwb_offpolicy_clipping_policy.svg" height="500" alt="app_gwb_offpolicy_clipping_policy"><br><sub><code>app_gwb_offpolicy_clipping_policy</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_gwb_offpolicy_clipping_value.svg" height="500" alt="app_gwb_offpolicy_clipping_value"><br><sub><code>app_gwb_offpolicy_clipping_value</code></sub></td>
</tr></table>

#### Mastermind-222

<table><tr>
<td align="center"><img src="docs/figures/paper/main_mastermind_222_offpolicy_policy.svg" height="500" alt="main_mastermind_222_offpolicy_policy"><br><sub><code>main_mastermind_222_offpolicy_policy</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_mastermind_222_offpolicy_policy.svg" height="500" alt="app_mastermind_222_offpolicy_policy"><br><sub><code>app_mastermind_222_offpolicy_policy</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_mastermind_222_offpolicy_value.svg" height="500" alt="app_mastermind_222_offpolicy_value"><br><sub><code>app_mastermind_222_offpolicy_value</code></sub></td>
</tr></table>

### Figures 3b and 10: Continual Learning

#### Gridworld

<table><tr>
<td align="center"><img src="docs/figures/paper/app_gwb_parallel_return.svg" height="500" alt="app_gwb_parallel_return"><br><sub><code>app_gwb_parallel_return</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_gwb_parallel_dqn_loss.svg" height="500" alt="app_gwb_parallel_dqn_loss"><br><sub><code>app_gwb_parallel_dqn_loss</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_gwb_parallel_policy_approx.svg" height="500" alt="app_gwb_parallel_policy_approx"><br><sub><code>app_gwb_parallel_policy_approx</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_gwb_parallel_value_approx.svg" height="500" alt="app_gwb_parallel_value_approx"><br><sub><code>app_gwb_parallel_value_approx</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_gwb_parallel_perf_approx_behaviour.svg" height="500" alt="app_gwb_parallel_perf_approx_behaviour"><br><sub><code>app_gwb_parallel_perf_approx_behaviour</code></sub></td>
</tr></table>

#### Mastermind-222

<table><tr>
<td align="center"><img src="docs/figures/paper/main_mastermind_222_parallel_value_approx.svg" height="500" alt="main_mastermind_222_parallel_value_approx"><br><sub><code>main_mastermind_222_parallel_value_approx</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_mastermind_222_parallel_return.svg" height="500" alt="app_mastermind_222_parallel_return"><br><sub><code>app_mastermind_222_parallel_return</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_mastermind_222_parallel_dqn_loss.svg" height="500" alt="app_mastermind_222_parallel_dqn_loss"><br><sub><code>app_mastermind_222_parallel_dqn_loss</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_mastermind_222_parallel_policy_approx.svg" height="500" alt="app_mastermind_222_parallel_policy_approx"><br><sub><code>app_mastermind_222_parallel_policy_approx</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_mastermind_222_parallel_value_approx.svg" height="500" alt="app_mastermind_222_parallel_value_approx"><br><sub><code>app_mastermind_222_parallel_value_approx</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_mastermind_222_parallel_perf_approx_behaviour.svg" height="500" alt="app_mastermind_222_parallel_perf_approx_behaviour"><br><sub><code>app_mastermind_222_parallel_perf_approx_behaviour</code></sub></td>
</tr></table>

### Figures 3c and 11: sampled characteristics

#### Gridworld

<table><tr>
<td align="center"><img src="docs/figures/paper/app_gwb_policy_sampling.svg" height="500" alt="app_gwb_policy_sampling"><br><sub><code>app_gwb_policy_sampling</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_gwb_value_sampling.svg" height="500" alt="app_gwb_value_sampling"><br><sub><code>app_gwb_value_sampling</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_gwb_perf_sampling.svg" height="500" alt="app_gwb_perf_sampling"><br><sub><code>app_gwb_perf_sampling</code></sub></td>
</tr></table>

#### Mastermind-222

<table><tr>
<td align="center"><img src="docs/figures/paper/main_mastermind_222_policy_sampling.svg" height="500" alt="main_mastermind_222_policy_sampling"><br><sub><code>main_mastermind_222_policy_sampling</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_mastermind_222_policy_sampling.svg" height="500" alt="app_mastermind_222_policy_sampling"><br><sub><code>app_mastermind_222_policy_sampling</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_mastermind_222_value_sampling.svg" height="500" alt="app_mastermind_222_value_sampling"><br><sub><code>app_mastermind_222_value_sampling</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_mastermind_222_perf_sampling.svg" height="500" alt="app_mastermind_222_perf_sampling"><br><sub><code>app_mastermind_222_perf_sampling</code></sub></td>
</tr></table>

#### Mastermind-333

<table><tr>
<td align="center"><img src="docs/figures/paper/app_mastermind_333_policy_sampling.svg" height="500" alt="app_mastermind_333_policy_sampling"><br><sub><code>app_mastermind_333_policy_sampling</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_mastermind_333_value_sampling.svg" height="500" alt="app_mastermind_333_value_sampling"><br><sub><code>app_mastermind_333_value_sampling</code></sub></td>
</tr></table>

### Large Mastermind: Figure 6 and Appendix Figures (Not Run)

<table><tr>
<td align="center"><img src="docs/figures/paper/app_large_policy_char.svg" height="500" alt="app_large_policy_char"><br><sub><code>app_large_policy_char</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_large_policy_shapley.svg" height="500" alt="app_large_policy_shapley"><br><sub><code>app_large_policy_shapley</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_large_value_char.svg" height="500" alt="app_large_value_char"><br><sub><code>app_large_value_char</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_large_value_shapley.svg" height="500" alt="app_large_value_shapley"><br><sub><code>app_large_value_shapley</code></sub></td>
<td align="center"><img src="docs/figures/paper/app_large_perf_offpolicy_shapley.svg" height="500" alt="app_large_perf_offpolicy_shapley"><br><sub><code>app_large_perf_offpolicy_shapley</code></sub></td>
</tr></table>

<br>
<br>
<br>
<br>
<br>

# Running Codebase

## Install

```bash
uv sync --all-groups                 # CPU, including Apple Silicon
uv sync --all-groups --extra cuda    # NVIDIA CUDA 13
uv sync --all-groups --extra render  # environment rendering (Pillow)
```

For a TPU, install the TPU runtime for the locked JAX version into the project environment (`uv pip install --python .venv/bin/python "jax[tpu]==<version>"`), then set `UV_NO_SYNC=1` so `uv run` keeps it.

Experiments track to [trackio](https://github.com/gradio-app/trackio) by default (local, no account) or to wandb (`uv run wandb login`).

## Running experiments

Every experiment is one config, `src/jax_shapley/configs/NAME.py`, run by `experiments/NAME.sh`:

```bash
bash experiments/mastermind222_behaviour.sh        # tracked with trackio

bash experiments/mastermind222_behaviour.sh --track wandb --project my-project

bash experiments/hypercube_behaviour__sweep.sh --side-length 3 --dimension 4 -- --track wandb

bash experiments/smoke.sh mastermind222_behaviour  # every stage, a few updates each

uv run python -m jax_shapley.main mastermind222_behaviour --shapley.1.train.epochs 9
```

- **Compile first.** Every script runs `--compile-only` before it trains: each stage is traced, lowered and compiled at the configured shapes, and nothing runs or is tracked. A compilation failure stops the run; there is no eager fallback.
- **Sweeps** are `configs/NAME__sweep.py`. They leave their swept fields `MISSING`, which `experiments/NAME__sweep.sh` supplies for each sweep point; each point is its own run group.
- **Overrides.** Any config field can be overridden on the command line, for example `--algorithm.total-timesteps`, `--characteristic.train.epochs` or `--shapley.0.seeds 1 2 3`.
- **Runs and artifacts.** Each stage logs its runs as `NAME--STAGE--VARIANT--seed-N` (for example `NAME--shapley--behaviour--with-exact--seed-1`) in one run group, `NAME-TIMESTAMP`, under upstream's metric names, and pushes its whole final state as a checkpoint artifact, `GROUP--STAGE--VARIANT`. Uploads retry transient tracking-service failures.
- **Resume.** A stage given a checkpoint loads it instead of training, for example `--algorithm.checkpoint GROUP--algorithm--dqn:latest`; it must have been saved with the same settings. An interrupted stage reruns.
- **Accelerators.** On a GPU or TPU, keep the CPU platform available for the logging callbacks: `JAX_PLATFORMS=cuda,cpu` or `JAX_PLATFORMS=tpu,cpu`.
- **Paper figures and tables** are regenerated from the tracking backend: `uv run --group benchmark python scripts/paper/figures.py --track wandb --project my-project`, then `scripts/paper/tables.py` and `scripts/paper/compare.py`.

<details>
<summary>Experiment settings (FastSVERL's values)</summary>

| Run | Epochs | Patience | Other |
| --- | --- | --- | --- |
| Gridworld behaviour characteristic | 50 | = epochs | |
| Gridworld prediction characteristic | 30 | = epochs | |
| Gridworld Shapley | 20 | = epochs | |
| Mastermind-222 behaviour characteristic | 50 | 10,000 | |
| Mastermind-222 prediction characteristic | 80 | 10,000 | |
| Mastermind-222 Shapley | 50 | 10,000 | |
| Hypercube (all stages) | 10,000 | = epochs | validation fraction 0.016 (160 rows), exact buffer 100,000, exact condition 0.01 (characteristic) and 0.001 (behaviour Shapley) or 0.01 (prediction Shapley), Shapley without the exact characteristic, `track_batch` |
| Gridworld outcome, off-policy | 128 | = epochs | replay prefix 50,000, validation 80, batch 64, τ 0.001, `track_batch` |
| Gridworld outcome, on-policy | — | — | 100,000 steps over 8 environments, batch 64, τ 0.001 |
| Gridworld outcome Shapley | 20 | = epochs | |
| Mastermind-222 outcome, off-policy | 128 | = epochs | replay prefix 100,000, validation 160, batch 64, τ 0.01, `track_batch` |
| Mastermind-222 outcome, on-policy | — | — | 200,000 steps over 16 environments, batch 64, τ 0.01 |
| Mastermind-222 outcome Shapley | 100 | = epochs | |

Unless stated otherwise: learning rate 2.5e-4, batch 128, a 10,000-observation buffer split 80/20 into train and validation, 16 collection environments, the greedy explained policy, 2x64 ReLU networks (2x128 for Hypercube and larger Mastermind) and γ = 1. DQN agents: 50,000 transitions on Gridworld and 100,000 over 16 environments on Mastermind-222, with exploration fractions 0.5 and 0.25.

</details>

## Code layout

An experiment is a pipeline of stages, **agent → characteristic → [outcome] → Shapley**, or an agent and its explanations trained together (continual learning).

| Module | Contents |
| --- | --- |
| `envs/` | Gridworld, Hypercube and Mastermind as Gymnax environments; the episode-statistics wrapper; rendering |
| `algorithms/dqn.py` | DQN: config, state, one vector step, training, evaluation, the agent stage |
| `algorithms/ppo.py` | PPO actor and critic networks (PPO training is not ported: the paper reports DQN agents only) |
| `characteristic_models/` | `behaviour.py` (the shared characteristic stage and the policy target), `prediction.py` (the value target), `outcome.py` (off- and on-policy outcome characteristics, exact outcome values) |
| `shapley_models/` | `behaviour.py` (the shared Shapley stage), `prediction.py`, `outcome.py` |
| `common/` | networks; coalitions and masking; the shared epoch loop; buffer collection; tabular models and exact policy evaluation; checkpoints |
| `experiment.py`, `main.py`, `continual.py` | experiment configs, the pipeline, and continual learning |
| `configs/` | one config per paper experiment |

## Using the components

The environments follow the Gymnax API, and `step` auto-resets finished episodes:

```python
import jax
from jax_shapley.envs import Hypercube
from jax_shapley.envs.wrappers import EpisodeStatisticsWrapper

env = EpisodeStatisticsWrapper(Hypercube(side_length=6, dimension=5))
params = env.default_params
obs, state = env.reset(jax.random.key(0), params)
obs, state, reward, done, info = env.step(jax.random.key(1), state, action=0, params=params)

batched_step = jax.jit(jax.vmap(env.step, in_axes=(0, 0, 0, None)))  # run inside jax.lax.scan
```

`env.env.step_env()` gives the raw transition before auto-reset. Gridworld keeps FastSVERL's `[0, 0]` terminal observation; Hypercube supports side lengths 2–6 and dimensions 2–5; Mastermind takes any code size, number of guesses and pegs. Observations, actions and counters are int32, rewards float32, flags bool.

Rendering is a host-side call outside `jit`, after `uv sync --extra render`:

```python
from jax_shapley.envs import Gridworld, RenderConfig

env = Gridworld()
_, state = env.reset(jax.random.key(0), env.default_params)
frame = env.render(state, env.default_params, RenderConfig(width=640, height=480))  # uint8 RGB
```

<table><tr>
<td align="center"><img src="docs/figures/envs-render--gridworld.webp" height="500" alt="Gridworld"><br><sub>Gridworld</sub></td>
<td align="center"><img src="docs/figures/envs-render--mastermind-222.webp" height="500" alt="Mastermind-222"><br><sub>Mastermind-222</sub></td>
<td align="center"><img src="docs/figures/envs-render--hypercube-5d.webp" height="500" alt="Hypercube in 5 dimensions"><br><sub>Hypercube in 5 dimensions</sub></td>
</tr></table>

A DQN agent's Q-network:

```python
import jax.numpy as jnp
from jax_shapley.algorithms import dqn
from jax_shapley.envs import Mastermind

config = dqn.DQNConfig(env=Mastermind(2, 2, 2), hidden_widths=(64, 64))
network = dqn.initialize(config, jax.random.key(0)).model
q_values = network(jnp.full((16, 8), -1, dtype=jnp.int32)).q_values
```

Networks use PyTorch's default Linear initialisation, or FastSVERL's orthogonal override where a layer selects it; JAX and PyTorch seeds do not produce identical weights.

## Development

`AGENTS.md` defines the coding and validation conventions.

Every change passes the gates:

```bash
./check.sh   # ty check, ruff check, ruff format --check, pytest
```

Tests are offline and independent of PyTorch. Compile tests trace, lower and compile every stage and every registered experiment; fixture tests check numerical units against upstream; GPU tests skip when no GPU is available.

The fixtures are captured from the pinned FastSVERL checkout in a separate PyTorch environment, for example:

```bash
uv run --script scripts/fixtures/collect_fastsverl_explanations.py /path/to/fastsverl tests/fixtures/fastsverl_explanations.npz
```

The other collectors in `scripts/fixtures/` (transitions, networks, DQN, logging, outcome, pipeline, sampling) follow the same pattern.
