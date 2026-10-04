# Explanations

FastSVERL explains an agent in two learnt stages. A **characteristic model** predicts what the agent would output when only a subset (coalition) of state features is known. A **Shapley model** is then trained against that characteristic, so it outputs per-feature Shapley values in a single forward pass. This guide covers all three explanation types:

| Paper name | Upstream classes | Explained quantity `v_F` |
| --- | --- | --- |
| Behaviour | `PolicyCharacteristicModel`, `PolicyShapley` | Agent policy: `pi_greedy`, or `pi_explore` when `policy="explore"` |
| Prediction | `ValueCharacteristicModel`, `ValueShapley` | Agent value `V(s) = max_a Q(s, a)` |
| Outcome | `PerformanceCharacteristicOffPolicy`, `PerformanceCharacteristicOnPolicy`, `PerformanceShapley` | The agent's return when, at state `e`, it acts on the behaviour characteristic given only `C` (see [Outcome explanations](#outcome-explanations)) |

The behavioural reference is FastSVERL commit `25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b`: `characteristics.py`, `shapley.py`, `utils.py` (`ShapleySampler`), `models.py`, `dqn.py`, `training.py` and the `run_policy_prediction.py`/`run.py` experiment scripts.

## Running explanations

An explanation is three runs. Each one loads the previous run's artifacts, so each stage can be repeated over seeds:

1. A DQN agent, trained with `jax_shapley.train_dqn` (see the [DQN training guide](DQN%20Training.md)).
2. A characteristic run, which explains that agent's behaviour or prediction.
3. A Shapley run, which explains the characteristic.

```bash
uv run python -m jax_shapley.train_explanations characteristic --agent runs/<dqn-run> --kind behaviour
uv run python -m jax_shapley.train_explanations shapley --agent runs/<dqn-run> --characteristic runs/<characteristic-run>
uv run python -m jax_shapley.train_explanations resume --run runs/<explanation-run>
```

Outcome explanations add one stage: an outcome run explains the agent behind a completed *behaviour* characteristic run, and a Shapley run accepts the outcome run as its characteristic.

```bash
uv run python -m jax_shapley.train_explanations outcome --variant off_policy --agent runs/<dqn-run> --characteristic runs/<behaviour-run>
uv run python -m jax_shapley.train_explanations shapley --agent runs/<dqn-run> --characteristic runs/<outcome-run>
```

The explained kind and policy of a Shapley run come from its characteristic. `--buffers characteristic` reuses the characteristic's train and validation buffers (the Hypercube experiments do this); `--buffers fresh` (the default) collects new ones. Exact values always use the characteristic's exact buffer.

Recipes in `experiments/explanations/` hold the paper settings:

```bash
bash experiments/explanations/gridworld.sh characteristic --kind behaviour --agent runs/<dqn-run>
bash experiments/explanations/gridworld.sh shapley --agent runs/<dqn-run> --characteristic runs/<characteristic-run>
bash experiments/explanations/mastermind222.sh characteristic --kind prediction --agent runs/<dqn-run>
bash experiments/explanations/hypercube.sh --kind behaviour --env.side-length 3 --env.dimension 3 --seed 1
bash experiments/explanations/gridworld.sh outcome --variant on_policy --agent runs/<dqn-run> --characteristic runs/<behaviour-run>
bash experiments/explanations/gridworld.sh outcome-shapley --agent runs/<dqn-run> --characteristic runs/<outcome-run>
bash experiments/explanations/smoke.sh
```

The Gridworld and Mastermind recipes run seeds 1–20, or the one given by `--seed`. Every Shapley seed explains the same characteristic, with both exact-loss ground truths (`--with-exact-char` and `--no-with-exact-char`), as upstream does. The Hypercube recipe trains its own agent, characteristic and Shapley model per kind, cube and seed, under `runs/explanations/hypercube/`. `--validate-only` checks every stage's typed settings without reading any artifacts. `--compile-only` compiles a stage against existing artifacts; the chained recipes compile only the agent stage in that mode.

| Run | Epochs | Patience | Other |
| --- | --- | --- | --- |
| Gridworld behaviour characteristic | 50 | = epochs | |
| Gridworld prediction characteristic | 30 | = epochs | |
| Gridworld Shapley | 20 | = epochs | |
| Mastermind-222 behaviour characteristic | 50 | 10,000 | |
| Mastermind-222 prediction characteristic | 80 | 10,000 | |
| Mastermind-222 Shapley | 50 | 10,000 | |
| Hypercube (all stages) | 10,000 | = epochs | validation fraction 0.016 (160 rows), exact buffer 100,000, exact condition 0.01 (characteristic) and 0.001 (behaviour Shapley) or 0.01 (prediction Shapley), `--no-with-exact-char`, `--train.track-batch` |
| Gridworld outcome, off-policy | 128 | = epochs | replay prefix 50,000, validation 80, batch 64, τ 0.001, `--train.track-batch` |
| Gridworld outcome, on-policy | — | — | 100,000 steps over 8 environments, batch 64, τ 0.001 |
| Gridworld outcome Shapley | 20 | = epochs | explains the on-policy outcome characteristic upstream |
| Mastermind-222 outcome, off-policy | 128 | = epochs | replay prefix 100,000, validation 160, batch 64, τ 0.01, `--train.track-batch` |
| Mastermind-222 outcome, on-policy | — | — | 200,000 steps over 16 environments, batch 64, τ 0.01 |
| Mastermind-222 outcome Shapley | 100 | = epochs | explains the off-policy outcome characteristic upstream |

Except where the table says otherwise, runs use learning rate 2.5e-4, batch 128, a 10,000-observation buffer split into train and validation sets, 16 collection environments, and the greedy explained policy. Networks are 2×64 ReLU, or 2×128 for Hypercube. Outcome runs use γ = 1 and run both exact-value groups (`--with-exact-char`, `--no-with-exact-char`).

The [PyTorch/JAX comparison report](Quantitative%20Comparison%20between%20PyTorch%20and%20JAX%20explanations%20of%20FastSVERL.md) validates these recipes end to end against the unchanged upstream code. 29 of 32 statistical comparisons pass, and the 3 exceptions are a protocol confound that the matched protocol resolves. Steady-state training epochs are 2.7–4.7× faster on CPU.

## Seed sweeps

`--seeds 1 2 … 20` trains every seed in **one compiled program**. All per-seed state carries a leading seed axis, built with `nnx.vmap`. Seeds that stop early do nothing while the others continue, and the loop ends when every seed is done. Each seed uses exactly the PRNG keys of a single run with that seed: tests check that a sweep's seeds reproduce single runs exactly in their buffers, and within 1e-5 in their losses and weights.

**Artifacts.** A sweep writes a sweep folder containing `sweep.json` and one **ordinary run folder per seed** (`seed-<s>/`), each holding that seed's single-run settings. A seed folder can be loaded, explained or resumed on its own. `resume --run <sweep folder>` continues every seed together, even when an interruption left the seeds at different checkpoints.

**Pairing.**

- `--agent` may name one DQN run, which every seed explains, or a directory of DQN runs, where each seed explains the run trained with its own seed. The Hypercube recipe uses the directory form.
- A Shapley `--characteristic` may name one characteristic run, which every seed explains (the paper's Gridworld/Mastermind protocol). It may also name a characteristic sweep, where Shapley seed s explains characteristic seed s (the Hypercube protocol, and the comparison's matched protocol).

The recipes train all 20 paper seeds per stage as one sweep; `--seed N` selects a single ordinary run. For 20 Gridworld behaviour characteristics, a sweep takes about 11 s, against about 85 s for 20 separate processes (Apple M4 Pro, CPU).

## Execution, artifacts and resume

Each run is compiled as one numerical invocation before it creates any artifacts. That invocation collects the buffers, computes the exact values and trains epochs in blocks of `--chunk-epochs` (default 1). Buffers are recorded before each step from 16 auto-resetting environments, with actions sampled from the explained policy. Each epoch runs `train_size // batch_size` batches sampled with replacement, then a validation loss over `val_size` resampled rows, then the exact loss (when enabled). Early stopping keeps the best-validation model, and training stops on patience or the exact condition. The run then restores the best model.

A run folder holds `config.json`, `metrics.csv` (`epoch, step, train_loss, validation_loss, exact_loss, best_epoch, seconds`), `summary.json` and an Orbax checkpoint. Checkpoints are written every tenth of the epochs and on completion. Each holds the current and best models, the optimizer, all buffers, the early-stopping state and the PRNG key.

`resume` restores that state, truncates `metrics.csv` to the committed boundary and continues to a result identical to an uninterrupted run. Exact values are recomputed rather than saved. Saved settings record a digest of the explained agent (and characteristic); resuming refuses changed models or settings.

**XLA CPU workaround.** With jax 0.11.0 on CPU, a jitted network applied to a broadcast-constant batch (for example `full_like(observations, MASK)`) can return wrong values from about 64 rows. The null value is therefore computed from one all-masked row and broadcast. `tests/test_jit_equivalence.py` checks compiled results against eager evaluation at production batch sizes.

**Exact-value cost.** Exact characteristic values deduplicate the exact buffer into its unique observations (weighted by frequency) before looping over all `2^F` coalitions, so each coalition processes a few unique states rather than every row. Their tables are sized by the real buffers: a fresh characteristic run collects its buffers in a first compiled program and sizes the tables by the most unique observations in any seed, a resumed run uses its checkpointed buffers, and a Shapley run uses the characteristic's exact buffer. (`--compile-only` keeps the analytic bound, the observation space or the buffer rows.) This makes Mastermind-333 (15 features, 32,768 coalitions) take seconds for a 20-seed sweep instead of about 28 GB per seed. The per-epoch exact loss evaluates only the 512-row chunks holding each coalition's distinct masked observations.

## Outcome explanations

**The characteristic.** `v(e, C)` is the agent's expected return from `e` when, *at `e` only*, it follows the behaviour characteristic's prediction given `C`, and elsewhere its own explained policy. This conditioned policy is `π̂(a | s; e, C)`. A goal-conditioned network on `concat(s, e, C)` learns it with DQN-style updates and a soft-updated target (`OutcomeNetworks`):

- **Off-policy** (`--variant off_policy`, upstream `PerformanceCharacteristicOffPolicy` with `train_models`). Learns `Q(s, a | e, C)` from the agent's replay and returns `v(e, C) = Σ_a π̂(a | e) Q(e, a | e, C)`.
  - The train and validation sets are the first `train_size` and `validation_size` replay transitions, so they overlap.
  - Each batch draws `e` rows independently from the same buffer, a uniform coalition per row, and next actions from `π̂`.
  - The target is updated after every batch.
  - Early stopping, `track_batch` and best-model restore follow the shared epoch loop.
- **On-policy** (`--variant on_policy`, upstream `PerformanceCharacteristicOnPolicy` with `train_V_on_policy`). Learns `V(s | e, C)` while acting with `π̂` in `num_envs` environments, and returns `v(e, C) = V(e | e, C)`.
  - Every decision resamples `e` from the behaviour characteristic's train buffer, plus one uniform coalition shared by every environment.
  - Each step is stored with its `(e, C)`. Every `train_frequency` steps, a uniform batch is learnt from; every `target_network_frequency` steps, the target is updated.
  - The loss and exact loss are logged every 100 global steps, as upstream does; `metrics.csv` leaves the validation column empty.
  - The final model is the last one, and checkpoints are written every tenth of the run.

**Exact values.** For every coalition and every unique state of the behaviour characteristic's exact buffer, the port evaluates `π̂` exactly in the environment's tabular model and returns `Σ_a π̂(a | e) Q(e, a)`. `π̂` uses the exact behaviour characteristic under `with_exact_char`, otherwise the learnt one.

- **Tabular model.** `tabular.py` builds the model by enumerating the JAX environments' reachable hidden states. Mastermind's unseen goal is averaged with equal weight per hidden state, as `get_P` does.
- **Solving.** `policy_q_values` solves `(I − γ C_π) V = R_π` directly. Upstream runs value iteration to a 1e-10 change, which approaches the same fixed point.
- **Evaluated states.** Unique exact-buffer states are non-terminal, so the port evaluates the transition model's states and marks those present in the buffer.

The exact loss is the mean squared error over all valid `(C, e)` pairs.

**Shapley models.** `shapley --characteristic <outcome run>` trains upstream's `PerformanceShapley`:

- Buffers follow the behaviour characteristic's explained policy.
- Exact values use the behaviour characteristic's exact buffer.
- `with_exact_char` explains the exact outcome values, conditioned as the outcome run chose. Otherwise it explains the learnt outcome characteristic.

**Parity.** `scripts/explanations_port/collect_fastsverl_outcome.py` runs the unchanged upstream classes on fixed batches and coalitions, recording the off-policy target's sampled next actions. `tests/fixtures/fastsverl_outcome.npz` holds:

- the transition tables and value-iteration Q tables
- per variant: three updates' losses, step-0 gradients, final and target weights, `get_val` and `null`
- exact tables and exact losses, with and without the exact behaviour characteristic
- `PerformanceShapley` updates, values, exact values and exact losses

`tests/test_outcome.py` matches all of these, and `tests/test_outcome_training.py` checks four properties of the pipelines:

- off-policy buffers are the replay prefixes
- each run's logged exact loss equals the eager exact loss of the restored model
- on-policy replay invariants hold: one coalition per decision, `e` drawn from the pool, and continuity across steps
- sweeps and interrupted runs resume byte-identically

The outcome-specific deviations are in the [deviation ledger](#deviation-ledger).

## Deviation ledger

Every known difference between the port and FastSVERL `25b2cbe`, and why it does not change behaviour (or how it does):

| Area | Upstream | Port | Effect |
| --- | --- | --- | --- |
| Random streams | torch and NumPy global generators | explicit JAX keys, split per run and per seed | Runs are not seed-for-seed identical; parity tests inject recorded draws, and the end-to-end comparison is distributional |
| Precision | float64 value iteration; float32 networks | float32 throughout | Exact values agree within 1e-5 relative / 1e-4 absolute (fixture tests) |
| Exact policy evaluation | value iteration to a 1e-10 change | direct linear solve of `(I − γ C_π) V = R_π` | Same fixed point |
| Non-terminating conditioned policy (γ = 1) | value iteration never terminates (the run hangs) | the solve returns non-finite exact values, which the run logs | Only reachable with a policy that can loop forever; upstream gives no result at all. The comparison harness times out such upstream runs and records them as failures |
| Exact outcome states | `torch.unique` of the exact buffer | transition-model states, marked valid when present in the exact buffer | Same set of valid states; avoids padding to the observation-space bound |
| Null value `v(s, ∅)` | network on an all-masked batch | network on one all-masked row, broadcast | Identical values; works around an XLA CPU miscompilation of broadcast-constant batches (JIT/eager tests guard it) |
| Off-policy target after early stopping | best model restored; the target keeps its last weights | the target saved with the best model is restored too | The target is not used after training |
| Outcome optimiser | Adam over the model only | one Adam over model and target | The target's gradients are exactly zero, so Adam leaves it unchanged (fixture target weights match) |
| On-policy terminal transitions | stores the reset observation as `n_obs` | stores the pre-reset observation | Masked by `(1 − terminated)`; equivalent |
| Truncated transitions | `final_observation` | Gymnax pre-reset observation | Same observation |
| Processes | one process per seed | `--seeds` trains every seed in one compiled program | Each seed uses the keys of a single run with that seed; sweeps equal single runs |
| Checkpoints | model saves every tenth of the run | full-state Orbax checkpoints every tenth of the run, with exact resume | No numerical effect |

## Reference fixture

`tests/fixtures/fastsverl_explanations.npz` is captured by executing the unchanged upstream class ASTs on CPU. Upstream draws randomness from torch and NumPy streams that JAX cannot reproduce, so the collector seeds the weights and fixes every other random input: coalitions, batches, actions and old log-probabilities. Each fixed input is stored next to the outputs it produced. Parity tests load those inputs instead of comparing random streams.

Observations come from the committed upstream transition capture (`fastsverl_transitions.npz`), and the collector checks that file's hash against its metadata. There are three cases:

| Case | Features | Actions | Width | Unique states |
| --- | --- | --- | --- | --- |
| `gridworld` | 2 | 4 | 64 | all 5 captured states |
| `mastermind222` | 8 | 4 | 64 | all 9 captured boards, flattened |
| `hypercube` | 5 | 10 | 128 | every tenth side-3, dimension-5 state |

Each case builds an exact buffer in which states repeat 1–4 times, so the empirical state distribution is non-uniform. Keys are prefixed with the case name:

| Keys | Contents |
| --- | --- |
| `states`, `multiplicities`, `exact_buffer`, `unique`, `counts` | Exact-buffer construction, and the `torch.unique`-sorted states and counts |
| `agent_{0..5}`, `agent_{greedy,explore,value}` | DQN weights (torch `(out, in)` layout) and `v_F` on the unique states; ε = 0.2 |
| `coalitions` | All `2^F` coalitions in `torch.cartesian_prod` order (the first feature varies slowest) |
| `exact_{behaviour,behaviour_explore,prediction}[_null]` | Exact characteristic table `(2^F, U, out)` and the null value |
| `char_{kind}_*` | Sampler probabilities and `tril`; initial weights, targets, coalitions, predictions and losses for each of the 3 Adam steps; step-0 gradients; final weights; exact loss; learnt grand-coalition and null values |
| `char_{kind}_off_{weighted,unweighted,weighted_clipped}_*` | Off-policy actions, old/new log-probs, importance weights, losses, step-0 gradients and final weights, starting from the on-policy initial weights and coalitions |
| `shapley_{kind}_*` | Initial weights, coalition, targets and loss per step; step-0 gradients; final weights; raw, normalised and unique-state Shapley values; exact values and exact losses with and without the exact characteristic |

Each Shapley stage trains against the characteristic in its final, trained state. The adjacent JSON records source and fixture hashes, the torch version, the seed, the learning rate, ε, the off-policy variants and the case layouts.

`tests/test_explanation_fixtures.py` checks the fixture against independent definitions, without reference to any JAX implementation:

- every exact characteristic value is the conditional expectation of `v_F` over the empirical distribution, given the known features
- the exact Shapley values equal the permutation formula
- normalised values are efficient
- the sampler follows the Shapley kernel
- behaviour outputs are probability distributions
- the importance weights match their definitions

## Pipeline parity fixture

`tests/fixtures/fastsverl_pipeline.npz` is captured by running upstream's unchanged `fill_buffer`, `get_data` and `train_models` (`scripts/explanations_port/collect_fastsverl_pipeline.py`). It records every random draw: buffer rows, by reissuing `Buffer.sample`'s own `np.random.choice` call on the same global stream, and coalitions. `tests/test_pipeline_parity.py` replays those draws through the JAX loop's injectable `train_draw`/`validation_draw` and checks:

- **Buffer collection.** 16 vector Hypercube(3, 5) environments and a hand-built, tie-free greedy agent that reaches the goal every 10 steps. The 640 recorded observations, including 48 auto-resets, match exactly.
- **Training loop.**
  - a behaviour characteristic and its Shapley model, with patience 2
  - a prediction characteristic stopped mid-run by the exact-loss condition under `track_batch`

  Every epoch's train, validation and exact loss, every logged step, the stopping epoch and the restored best weights match within 1e-4.
- **Reloaded ε.** It follows upstream's save rule, `linear_schedule((steps − 1) · num_envs)`.

```bash
uv run --script scripts/explanations_port/collect_fastsverl_pipeline.py \
  /path/to/fastsverl tests/fixtures/fastsverl_pipeline.npz
```

## Upstream semantics to preserve

- **Masking.** Unknown features are set to `MASK = -2` directly in the raw observation. Mastermind uses `-1` for empty cells, so a masked cell and an empty cell are adjacent integers.
- **Two coalition distributions.** Characteristic training masks each row with a *uniform* random coalition (`sample_rand`). Shapley training draws *one* Shapley-kernel coalition per batch and reuses it for every row; sizes 1..F−1 are drawn with probability ∝ `1/(k(F−k))`.
- **Shapley loss.** The target is `v(s, C) − v(s, ∅)` from the learnt characteristic. The prediction is the sum over features of `C ⊙ φ(s)`.
- **Normalisation.** `normalise` adds `(v(s, F) − v(s, ∅) − Σφ) / F` to every feature, using the *learnt* characteristic at the grand coalition, so the result is efficient.
- **Exact values.** The exact characteristic weights each state by its frequency in a finite buffer; states seen only once keep the exact loss from reaching zero. The exact Shapley values solve a weighted least-squares problem that gives the empty and grand coalitions zero weight, followed by `normalise_exact`. The fixture confirms this equals the permutation definition of the Shapley value. `with_exact_char` changes only which ground truth the loss is compared against, never training.
- **Off-policy weights.** The ratio is `exp(new − old)`, using the target policy's Torch `Categorical` log-probability, which clamps zero probabilities to float32 ε. With `weighting=True` (the default) it is normalised to sum to 1 over the batch, then clamped to `[1 − clip, 1 + clip]`. With a greedy target policy, a non-greedy action gets a ratio of about `1.2e-7 / p_old`.
- **Training loop quirks**:
  - early stopping keeps a copy of the best validation model and restores it at the end
  - validation resamples the whole validation size with replacement, with fresh coalitions
  - under `track_batch`, the logged step is `(epoch + 1) * n_batches`
  - `exact_condition` compares against the current epoch's exact loss (upstream trains one model per call)

## Modules

| Module | Responsibility |
| --- | --- |
| `coalitions.py` | Shapley-kernel, paired and uniform sampling; masking; enumerating all coalitions in `cartesian_prod` order |
| `characteristics.py` | Typed behaviour/prediction configuration; `v_F` from `greedy_probabilities`, `exploration_probabilities` and max-Q; on- and off-policy losses; exact characteristic |
| `shapley.py` | Shapley loss, normalisation, exact values; the `CharacteristicFunction` protocol |
| `outcome.py` | Outcome networks, conditioned policy, off-/on-policy losses and updates, conditioned replay, exact outcome values |
| `tabular.py` | Observation-level transition models enumerated from the JAX environments; exact policy evaluation |
| `train_explanations.py` | Typed run settings, buffer collection with the trained agent, the compiled epoch/early-stopping loop and the on-policy outcome loop; `main()` handles the host side and `run_characteristic()`, `run_shapley()`, `run_outcome()`, `run_outcome_on_policy()` and `run_outcome_shapley()` the compiled side |
| `explanation_io.py` | Loading and identifying agents and characteristics, settings, CSV metrics, Orbax checkpoints and restore |

The models reuse `MLP` and `LayerConfig` from `networks.py`, including the softmax output layer that the behaviour characteristic needs. Shapley outputs are reshaped with einops `"... (f a) -> ... f a"`.

## Regenerating the fixture

Use a clean checkout of the pinned source. The script installs PyTorch in its own isolated environment and never adds it to the project:

```bash
uv run --script scripts/explanations_port/collect_fastsverl_explanations.py \
  /path/to/fastsverl tests/fixtures/fastsverl_explanations.npz
```

The collector verifies the source commit, the committed bytes of every source file it executes, and the transition-fixture hash. Two runs produce identical files. The outcome fixture is regenerated the same way, after the explanation fixture it reads:

```bash
uv run --script scripts/explanations_port/collect_fastsverl_outcome.py \
  /path/to/fastsverl tests/fixtures/fastsverl_outcome.npz
```
