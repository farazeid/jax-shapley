# Reproducing the FastSVERL paper

This guide maps every figure and table in the FastSVERL paper (arXiv 2511.06094) to the upstream driver that produced it (FastSVERL `25b2cbe`), the JAX recipe that reruns it, and the comparison against the published numbers. It covers the main text and the appendices. PPO agents are not ported because the paper reports DQN agents only.

## Pipeline

1. **Agents.** Agents are trained with `experiments/dqn/*.sh`, the large Mastermind agents with `experiments/dqn/mastermind.sh SIZE`, and the per-seed agents inside the off-policy and Hypercube recipes.
2. **Explanations.** The `experiments/explanations/*.sh` recipes below run explanations; the `runs/paper/run_*.sh` chains run them in dependency order.
3. **Figures and tables.** `scripts/paper_port/figures.py` regenerates every published curve with upstream `plot()` statistics: nearest-step merge, interpolation, mean and standard error, `n_batches` scaling, `start_epoch` offsets and SEM correction. It also regenerates the Hypercube bars with upstream `barcharts`. `scripts/paper_port/tables.py` regenerates Tables 1–4 with upstream `calculate_final_stats` and `plot_explanations.py`.
4. **Comparison.** `scripts/paper_port/compare.py` applies the criteria in `docs/data/paper--criteria.md`, which were committed before any comparison run, to the targets extracted from upstream's own plot files (`scripts/paper_port/extract_published.py`). It writes `docs/data/paper--comparison.csv`, `docs/data/paper--claims-results.csv` and overlays in `docs/figures/paper/`.

```bash
uv run --group benchmark python scripts/paper_port/figures.py
uv run python scripts/paper_port/tables.py
uv run --group benchmark python scripts/paper_port/compare.py
```

## Figures and tables

| Paper | Upstream driver | JAX recipe | Runs |
| --- | --- | --- | --- |
| Figs 1, 4 (Mastermind-222, Gridworld): characteristic, Shapley, outcome | `fastsverl/{characteristics,shapleys}/dqn_*` | `mastermind222.sh`, `gridworld.sh` | Milestone-2 sweeps, 20 seeds (`docs/data/explanations-port--jax-runs.csv`) |
| Fig 4 (Mastermind-333) | `fastsverl/{characteristics,shapleys}/dqn_{policy,value}_mastermind.py` | `mastermind.sh 333` | 20 seeds; every Shapley seed explains characteristic seed 1 |
| Figs 2, 5: Hypercube updates to target loss | `hypercube/hypercube_{policy,value}.py` | `hypercube.sh` | 20 cubes × 20 seeds, one agent per seed |
| Tables 1, 3, 4; Fig 6: large Mastermind | `fastsverl/{characteristics,shapleys}/dqn_*_mastermind.py` | `mastermind.sh 443\|453\|463` | 10 seeds per stage; one off-policy outcome characteristic over the 10M replay |
| Table 2: Mastermind-463 behaviour explanation | `shapleys/plots/plot_explanations.py` | `scripts/paper_port/tables.py` | Shapley seed 1 |
| Figs 3a, 7: off-policy characteristics (Mastermind-222) | `offpolicy/*/dqn_{policy,value}_mastermind.py` | `offpolicy.sh mastermind222` | 20 seeds, one agent per seed |
| Figs 8, 9: importance weighting and clipping (Gridworld) | `offpolicy/*/dqn_{policy,value}_gwb.py` | `offpolicy.sh gridworld` | 20 seeds, one agent per seed |
| Figs 3b, 10: continual learning | `parallel/dqn_{policy,value,performance}_*` | `continual.sh` | 20 seeds × ratios 1, 2, 10, 50; w_exact and w_policy groups |
| Figs 3c, 11: sampled characteristics | `sampling/{shapleys,characteristics}/dqn_*` | `sampling.sh` | 20 seeds; model, sampled and exact groups |

## Deviations that affect the comparison

- **Agents.** Agents are retrained in JAX. The large-Mastermind tables, Table 2 and every single-agent figure depend on that one agent. The published standard errors exclude between-agent variance, which is why the criteria use ratio tolerances.
- **Shared agents.** The continual-learning groups share each seed's agent across update ratios, because the explanation models never affect the agent. Upstream reran it per group with the same seed.
- **Reused characteristics.** The sampling recipe reuses one learnt characteristic sweep for both model groups (upstream's same-seed processes train the same characteristic). The outcome groups reuse the behaviour characteristic sweep: upstream's outcome flow trains it with identical settings.
- **Ledger.** The full list is in the deviation ledger in [Explanations](Explanations.md).

## Results

Results are added when the runs complete.
