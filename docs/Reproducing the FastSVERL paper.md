# Reproducing the FastSVERL paper

This guide maps every figure and table in the FastSVERL paper (arXiv 2511.06094) to the upstream driver that produced it (FastSVERL `25b2cbe`), the JAX recipe that reruns it, and the comparison against the published numbers. It covers the main text and the appendices. PPO agents are not ported because the paper reports DQN agents only.

## Pipeline

1. **Agents.** Agents are trained with `experiments/dqn/*.sh`, the large Mastermind agents with `experiments/dqn/mastermind.sh SIZE`, and the per-seed agents inside the off-policy and Hypercube recipes.
2. **Explanations.** The `experiments/explanations/*.sh` recipes below run explanations; the `runs/paper/run_*.sh` chains run them in dependency order.
3. **Figures and tables.** Both scripts read only from the tracking backend (`--track wandb|trackio`, `--project`, `--entity`), taking each experiment's latest run group from `python -m new_jax_shapley.main NAME --track ...`. Figures and tables whose experiments are not ported to `new_jax_shapley/configs/` yet, or have no tracked runs, are reported missing. `scripts/paper/figures.py` regenerates every published curve with upstream `plot()` statistics: nearest-step merge, interpolation, mean and standard error, `n_batches` scaling, `start_epoch` offsets and SEM correction. It also regenerates the Hypercube bars with upstream `barcharts`. `scripts/paper/tables.py` regenerates Tables 1–4 with upstream `calculate_final_stats` and `plot_explanations.py`. Table 2 restores its models from the experiment's checkpoint artifacts. With wandb, the cloud is the source of truth: everything pulled locally is deleted when the script or experiment ends.
4. **Comparison.** `scripts/paper/compare.py` applies the criteria in `docs/data/paper--criteria.md`, which were committed before any comparison run, to the targets extracted from upstream's own plot files (`scripts/paper/extract_published.py`). It writes `docs/data/paper--comparison.csv`, `docs/data/paper--claims-results.csv` and overlays in `docs/figures/paper/`.

```bash
uv run python -m new_jax_shapley.main mastermind222_behaviour --track wandb
uv run --group benchmark python scripts/paper/figures.py --track wandb
uv run python scripts/paper/tables.py --track wandb
uv run --group benchmark python scripts/paper/compare.py
```

## Figures and tables

| Paper | Upstream driver | JAX recipe | Runs |
| --- | --- | --- | --- |
| Figs 1, 4 (Mastermind-222, Gridworld): characteristic, Shapley, outcome | `fastsverl/{characteristics,shapleys}/dqn_*` | `mastermind222.sh`, `gridworld.sh` | Milestone-2 sweeps, 20 seeds (`docs/data/explanations-port--jax-runs.csv`) |
| Fig 4 (Mastermind-333) | `fastsverl/{characteristics,shapleys}/dqn_{policy,value}_mastermind.py` | `mastermind.sh 333` | 20 seeds; every Shapley seed explains characteristic seed 1 |
| Figs 2, 5: Hypercube updates to target loss | `hypercube/hypercube_{policy,value}.py` | `hypercube.sh` | 20 cubes × 20 seeds, one agent per seed |
| Tables 1, 3, 4; Fig 6: large Mastermind | `fastsverl/{characteristics,shapleys}/dqn_*_mastermind.py` | `mastermind.sh 443\|453\|463` | 10 seeds per stage; one off-policy outcome characteristic over the 10M replay |
| Table 2: Mastermind-463 behaviour explanation | `shapleys/plots/plot_explanations.py` | `scripts/paper/tables.py` | Shapley seed 1 |
| Figs 3a, 7: off-policy characteristics (Mastermind-222) | `offpolicy/*/dqn_{policy,value}_mastermind.py` | `offpolicy.sh mastermind222` | 20 seeds, one agent per seed |
| Figs 8, 9: importance weighting and clipping (Gridworld) | `offpolicy/*/dqn_{policy,value}_gwb.py` | `offpolicy.sh gridworld` | 20 seeds, one agent per seed |
| Figs 3b, 10: continual learning | `parallel/dqn_{policy,value,performance}_*` | `continual.sh` | 20 seeds × ratios 1, 2, 10, 50; w_exact and w_policy groups |
| Figs 3c, 11: sampled characteristics | `sampling/{shapleys,characteristics}/dqn_*` | `sampling.sh` | 20 seeds; model, sampled and exact groups |

## Deviations that affect the comparison

- **Agents.** Agents are retrained in JAX. The large-Mastermind tables, Table 2 and every single-agent figure depend on that one agent. The published standard errors exclude between-agent variance, which is why the criteria use ratio tolerances.
- **Shared agents.** The continual-learning groups share each seed's agent across update ratios, because the explanation models never affect the agent. Upstream reran it per group with the same seed.
- **Reused characteristics.** The sampling recipe reuses one learnt characteristic sweep for both model groups (upstream's same-seed processes train the same characteristic). The outcome groups reuse the behaviour characteristic sweep: upstream's outcome flow trains it with identical settings.
- **Ledger.** The full list is in the deviation ledger in [Explanations](Explanations.md).

## Results (runs stopped 2026-10-04)

The runs were stopped before every experiment finished. Everything below comes from completed runs only. `docs/data/paper--comparison.csv` holds each compared quantity, `docs/data/paper--claims-results.csv` each claim, and `docs/figures/paper/` the published-versus-JAX overlays.

**Coverage.** 42 of 51 published curve figures, 62 of 80 Hypercube bars, Table 1 and Table 2. 301 of 306 compared quantities meet the pre-registered criteria, and 18 of 26 claims hold.

| Experiment | Status | Comparison |
| --- | --- | --- |
| Figs 1, 4: Mastermind-222, Gridworld, Mastermind-333 | Complete | All but four quantities pass (below) |
| Figs 3a, 7–9: off-policy characteristics | Complete | All quantities pass. "IS helps" and "on-policy best" hold; "unnormalised IS worse than none" fails narrowly (below) |
| Figs 3b, 10: continual learning | Complete | All quantities pass; all four "higher ratios mitigate spikes" claims hold |
| Figs 3c, 11: sampled characteristics | Mastermind-222 behaviour and prediction only | All quantities pass; the three Mastermind-222 claims hold. Not run: Gridworld, Mastermind-333, and the sampled-behaviour outcome group |
| Figs 2, 5: Hypercube | 31 of 40 cube × kind combinations; dimension 5 mostly missing | 61 of 62 bars pass. For dimensions 2–4, Spearman(states, updates) is 1.0, 1.0 and 0.9, but the pre-registered claim needs all four dimensions |
| Table 1 (behaviour, Mastermind-443/453/463) | Complete | All 12 quantities pass; updates within 0.7–1.3× and final losses within 0.7–1.04× of the published values |
| Table 2 (Mastermind-463 explanation) | Complete | Nullity claim holds: mean scaled value 0.015 on unused rows versus 0.267 on used rows. On the paper's board the JAX agent guesses ACBB, not CCCB: that board starts from the paper agent's own first guess, so the JAX agent never reaches it |
| Tables 3, 4; Fig 6 (prediction, off-policy outcome) | Not run: stopped during the prediction stage | — |

**Quantities outside tolerance (5 of 306):**

- **Gridworld Shapley with the approximate characteristic, final value:** 1.2e-4 against a published 3.9e-4, a factor of 3.3. Every published run explains one shared characteristic of one agent, so the published standard error (3e-7) excludes agent and characteristic variance.
- **Gridworld prediction Shapley with the exact characteristic, final value:** 1.3e-8 against 1.6e-6. Both have converged; the published value sits just above the 1e-6 floor that would count them as equal.
- **Gridworld off-policy outcome characteristic (exact behaviour) at 25% of training:** a factor of 2.35 (z = 3.4); its final value passes.
- **Mastermind-333 prediction characteristic at 25% of training:** a factor of 2.4; its final value passes.
- **Hypercube 2×4 behaviour Shapley:** 376 updates against 232, a factor of 1.6 (the bar tolerance is 1.5).

**Claims not reproduced.** "Unnormalised IS is worse than no reweighting" (Fig 8, behaviour and prediction) fails narrowly. In JAX, unnormalised IS ends 13% below no reweighting (behaviour) and 2% below (prediction); upstream ends 29% and 7% above. Every value in the figure is within tolerance and the gap is within about one standard error. The importance-weighting code path is identical to upstream's (ratio, identity weighting and weighted MSE).
