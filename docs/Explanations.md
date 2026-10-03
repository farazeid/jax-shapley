# Explanations

FastSVERL explains an agent in two learnt stages. A **characteristic model** predicts what the agent would output when only a subset (coalition) of state features is known. A **Shapley model** is then trained against that characteristic, so it outputs per-feature Shapley values in a single forward pass. This guide covers the two explanation types ported so far:

| Paper name | Upstream classes | Explained quantity `v_F` |
| --- | --- | --- |
| Behaviour | `PolicyCharacteristicModel`, `PolicyShapley` | Agent policy: `pi_greedy`, or `pi_explore` when `policy="explore"` |
| Prediction | `ValueCharacteristicModel`, `ValueShapley` | Agent value `V(s) = max_a Q(s, a)` |

Outcome explanations (`Performance*`) depend on the behaviour characteristic and on exact transition models, so they come later.

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

The explained kind and policy of a Shapley run come from its characteristic. `--buffers characteristic` reuses the characteristic's train and validation buffers (the Hypercube experiments do this); `--buffers fresh` (the default) collects new ones. Exact values always use the characteristic's exact buffer.

Recipes in `experiments/explanations/` hold the paper settings:

```bash
bash experiments/explanations/gridworld.sh characteristic --kind behaviour --agent runs/<dqn-run>
bash experiments/explanations/gridworld.sh shapley --agent runs/<dqn-run> --characteristic runs/<characteristic-run>
bash experiments/explanations/mastermind222.sh characteristic --kind prediction --agent runs/<dqn-run>
bash experiments/explanations/hypercube.sh --kind behaviour --env.side-length 3 --env.dimension 3 --seed 1
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

All runs use learning rate 2.5e-4, batch 128, a 10,000-observation buffer split into train and validation sets, 16 collection environments, and the greedy explained policy. Networks are 2×64 ReLU, or 2×128 for Hypercube.

The [PyTorch/JAX comparison report](Quantitative%20Comparison%20between%20PyTorch%20and%20JAX%20explanations%20of%20FastSVERL.md) validates these recipes end to end against the unchanged upstream code. 29 of 32 statistical comparisons pass, and the 3 exceptions are a protocol confound that the matched protocol resolves. Steady-state training epochs are 2.7–4.7× faster on CPU.

## Execution, artifacts and resume

Each run is compiled as one numerical invocation before it creates any artifacts. That invocation collects the buffers, computes the exact values and trains epochs in blocks of `--chunk-epochs` (default 1). Buffers are recorded before each step from 16 auto-resetting environments, with actions sampled from the explained policy. Each epoch runs `train_size // batch_size` batches sampled with replacement, then a validation loss over `val_size` resampled rows, then the exact loss (when enabled). Early stopping keeps the best-validation model, and training stops on patience or the exact condition. The run then restores the best model.

A run folder holds `config.json`, `metrics.csv` (`epoch, step, train_loss, validation_loss, exact_loss, best_epoch, seconds`), `summary.json` and an Orbax checkpoint. Checkpoints are written every tenth of the epochs and on completion. Each holds the current and best models, the optimizer, all buffers, the early-stopping state and the PRNG key.

`resume` restores that state, truncates `metrics.csv` to the committed boundary and continues to a result identical to an uninterrupted run. Exact values are recomputed rather than saved. Saved settings record a digest of the explained agent (and characteristic); resuming refuses changed models or settings.

**XLA CPU workaround.** With jax 0.11.0 on CPU, a jitted network applied to a broadcast-constant batch (for example `full_like(observations, MASK)`) can return wrong values from about 64 rows. The null value is therefore computed from one all-masked row and broadcast. `tests/test_jit_equivalence.py` checks compiled results against eager evaluation at production batch sizes.

**Exact-loss cost.** The exact characteristic table is padded to a static bound on unique observations. Its valid entries are compacted once per run, and each epoch evaluates only the 512-row chunks that hold them.

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
| `shapley.py` | Shapley loss, normalisation, exact values |
| `train_explanations.py` | Typed run settings, buffer collection with the trained agent, and the compiled epoch/early-stopping loop; `main()` handles the host side and `run_characteristic()`/`run_shapley()` the compiled side |
| `explanation_io.py` | Loading and identifying agents and characteristics, settings, CSV metrics, Orbax checkpoints and restore |

The models reuse `MLP` and `LayerConfig` from `networks.py`, including the softmax output layer that the behaviour characteristic needs. Shapley outputs are reshaped with einops `"... (f a) -> ... f a"`.

## Regenerating the fixture

Use a clean checkout of the pinned source. The script installs PyTorch in its own isolated environment and never adds it to the project:

```bash
uv run --script scripts/explanations_port/collect_fastsverl_explanations.py \
  /path/to/fastsverl tests/fixtures/fastsverl_explanations.npz
```

The collector verifies the source commit, the committed bytes of every source file it executes, and the transition-fixture hash. Two runs produce identical files.
