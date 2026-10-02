# Explanations

FastSVERL explains an agent in two learnt stages. A **characteristic model** predicts what the agent would output when only a subset (coalition) of state features is known. A **Shapley model** is then trained against that characteristic, so it outputs per-feature Shapley values in a single forward pass. This guide covers the two explanation types that are next to be ported:

| Paper name | Upstream classes | Explained quantity `v_F` |
| --- | --- | --- |
| Behaviour | `PolicyCharacteristicModel`, `PolicyShapley` | Agent policy: `pi_greedy`, or `pi_explore` when `policy="explore"` |
| Prediction | `ValueCharacteristicModel`, `ValueShapley` | Agent value `V(s) = max_a Q(s, a)` |

Outcome explanations (`Performance*`) depend on the behaviour characteristic and on exact transition models, so they come later.

The behavioural reference is FastSVERL commit `25b2cbe66b0d4a0decc3b63dbc798a79bf2c7d5b`: `characteristics.py`, `shapley.py`, `utils.py` (`ShapleySampler`), `models.py` and `dqn.py`. So far only the reference fixture exists; the JAX modules are not implemented yet.

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

## Upstream semantics to preserve

- **Masking.** Unknown features are set to `MASK = -2` directly in the raw observation. Mastermind uses `-1` for empty cells, so a masked cell and an empty cell are adjacent integers.
- **Two coalition distributions.** Characteristic training masks each row with a *uniform* random coalition (`sample_rand`). Shapley training draws *one* Shapley-kernel coalition per batch and reuses it for every row; sizes 1..F−1 are drawn with probability ∝ `1/(k(F−k))`.
- **Shapley loss.** The target is `v(s, C) − v(s, ∅)` from the learnt characteristic. The prediction is the sum over features of `C ⊙ φ(s)`.
- **Normalisation.** `normalise` adds `(v(s, F) − v(s, ∅) − Σφ) / F` to every feature, using the *learnt* characteristic at the grand coalition, so the result is efficient.
- **Exact values.** The exact characteristic weights each state by its frequency in a finite buffer; states seen only once keep the exact loss from reaching zero. The exact Shapley values solve a weighted least-squares problem that gives the empty and grand coalitions zero weight, followed by `normalise_exact`. The fixture confirms this equals the permutation definition of the Shapley value. `with_exact_char` changes only which ground truth the loss is compared against, never training.
- **Off-policy weights.** The ratio is `exp(new − old)`, using the target policy's Torch `Categorical` log-probability, which clamps zero probabilities to float32 ε. With `weighting=True` (the default) it is normalised to sum to 1 over the batch, then clamped to `[1 − clip, 1 + clip]`. With a greedy target policy, a non-greedy action gets a ratio of about `1.2e-7 / p_old`.
- **Training loop quirks** (for the later `train_models` port):
  - early stopping keeps a deep copy of the best validation model
  - under `track_batch`, the logged step is `(epoch + 1) * n_batches`
  - `exact_condition` compares against the exact loss of the last model in the update list

## Planned modules

| Module | Responsibility |
| --- | --- |
| `coalitions.py` | Shapley-kernel, paired and uniform sampling; masking; enumerating all coalitions in `cartesian_prod` order |
| `characteristics.py` | Typed behaviour/prediction configuration; `v_F` from `greedy_probabilities`, `exploration_probabilities` and max-Q; on- and off-policy losses; exact characteristic |
| `shapley.py` | Shapley loss, normalisation, exact values |
| `train_explanations.py` | Buffer collection with the trained agent, and the epoch/early-stopping loop, using the same `main()`/`run()` shape as `train_dqn.py` |

The models reuse `MLP` and `LayerConfig` from `networks.py`, including the softmax output layer that the behaviour characteristic needs. Shapley outputs are reshaped with einops `"... (f a) -> ... f a"`.

## Regenerating the fixture

Use a clean checkout of the pinned source. The script installs PyTorch in its own isolated environment and never adds it to the project:

```bash
uv run --script scripts/explanations_port/collect_fastsverl_explanations.py \
  /path/to/fastsverl tests/fixtures/fastsverl_explanations.npz
```

The collector verifies the source commit, the committed bytes of every source file it executes, and the transition-fixture hash. Two runs produce identical files.
