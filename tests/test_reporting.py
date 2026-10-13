"""Paper table statistics and explanation boards against the source's formulas and published layout."""

import numpy as np
import pytest

from scripts.paper.tables import (
    SOURCE_BATCHES,
    GroupStats,
    board_values,
    format_scientific,
    latex_table,
    standard_error,
)

# experiments/fastsverl/shapleys/plots/main_large_policy_shapley.tex at the reference commit.
PUBLISHED_TABLE_1 = r"""\begin{tabular}{lll}
\toprule
Group & Updates & Final Loss \\
\midrule
Mastermind-443 Characteristic & $(1.097 \pm 0.112) \times 10^{6}$ & $(3.825 \pm 0.022) \times 10^{-3}$ \\
Mastermind-443 Shapley & $(7.305 \pm 0.681) \times 10^{5}$ & $(1.292 \pm 0.036) \times 10^{-3}$ \\
\bottomrule
\end{tabular}
"""


def test_format_scientific_matches_the_source_layout() -> None:
    assert format_scientific(1_097_000.0, 112_000.0) == r"$(1.097 \pm 0.112) \times 10^{6}$"
    assert format_scientific(0.003825, 0.000022) == r"$(3.825 \pm 0.022) \times 10^{-3}$"
    assert format_scientific(0.0, 0.5) == r"$(0.000 \pm 0.500) \times 10^{0}$"
    assert format_scientific(-250.0, 10.0) == r"$(-2.500 \pm 0.100) \times 10^{2}$"


def test_latex_table_reproduces_the_published_layout() -> None:
    groups = [
        GroupStats("Mastermind-443 Characteristic", 1_097_000.0, 112_000.0, 0.003825, 0.000022, 10),
        GroupStats("Mastermind-443 Shapley", 730_500.0, 68_100.0, 0.001292, 0.000036, 10),
    ]
    assert latex_table(groups) == PUBLISHED_TABLE_1


def test_group_statistics_follow_calculate_final_stats() -> None:
    # Source: updates = last logged epoch * n_batches, with n_batches = (0.8 * 10,000) // 128.
    assert pytest.approx(62.0) == SOURCE_BATCHES
    updates = np.asarray([100.0, 200.0, 300.0]) * SOURCE_BATCHES
    # The source: np.std(values, ddof=1) / np.sqrt(n), so (6200, 12400, 18600) gives 6200 / sqrt(3).
    assert standard_error(updates) == pytest.approx(6200 / np.sqrt(3))
    assert standard_error(np.asarray([0.25, 0.5, 0.75])) == pytest.approx(0.25 / np.sqrt(3))


def test_board_values_flip_rows_and_scale_by_the_largest_magnitude() -> None:
    contributions = np.asarray([0.0, 1.0, -4.0, 2.0, 0.5, 0.0], np.float32)
    board = board_values(contributions, guesses=2)
    np.testing.assert_allclose(board, [[0.5, 0.125, 0.0], [0.0, 0.25, -1.0]])
    np.testing.assert_array_equal(board_values(np.zeros(6, np.float32), guesses=3), np.zeros((3, 2)))
