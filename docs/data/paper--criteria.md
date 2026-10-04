# Pre-registered criteria: JAX runs against the published FastSVERL results

Committed before any JAX run used in the comparison exists. The published targets are `paper--published-{curves,bars,tables}.csv`, extracted from upstream's own plot outputs and tables at FastSVERL `25b2cbe` (`scripts/paper_port/extract_published.py`). `scripts/paper_port/compare.py` applies these rules unchanged.

## Why tolerances rather than equivalence tests

Most published results come from one agent (and for Shapley runs, one characteristic) shared by every run. Their standard errors therefore exclude between-agent variance, which in the 20-seed upstream comparison moved single-agent results by up to about 1.7× between frameworks. A z-test against those standard errors alone would fail for reasons unrelated to the port. Each quantity instead passes when **any** of these holds:

1. **Ratio:** `|log10(JAX / published)| ≤ log10(2)` (within a factor of 2).
2. **Statistical:** `|JAX − published| / sqrt(se_published² + se_JAX²) ≤ 3`. Here `se_published` is the half-width of the published shaded band and `se_JAX` the JAX standard error over runs.
3. **Converged to zero:** both values are below `1e-6`.

Hypercube runs train an agent per seed, so their standard errors include agent variance. Bars use the tighter ratio `log10(1.5)` in rule 1.

## Quantities compared

| Target | Quantity | Rule |
| --- | --- | --- |
| Curves (Figs 1, 3, 4, 7–11) | Mean at 25% and at 100% of each published series' x range, after mapping JAX runs to the published x units (updates, environment steps, `start_epoch` offsets) | 1, 2 or 3 at both checkpoints |
| Bars (Figs 2, 5) | Mean updates to the target loss for every (cube, model) | 1 (factor 1.5), 2 or 3 |
| Tables 1, 3, 4 | Updates to converge and final loss, per domain and model | 1 or 2 |

## Claims

Each paper figure's stated finding is also checked on the JAX runs alone (`paper--claims.csv`). "Final" is the mean over runs at the end of the published x range. A claim passes when its relation holds.

## Table 2 (added 2026-10-04, before any trained Mastermind-463 Shapley model existed)

Table 2 explains one board reached by the paper's agent. Its opening guess (AACA) is the paper agent's own, so another agent need not reach that board or act well there. Values on that board are reported but not compared. The checked finding is the one that does not depend on the board: unused guess slots get a neutral influence (`table2_nullity`, over every greedy state of the JAX agent). The JAX agent's greedy guess on the paper's board is reported against the paper's CCCB.
