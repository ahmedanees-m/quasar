"""Gate G-R.1 as a test, under the `gate` marker."""

from __future__ import annotations

import pytest

from experiments.wp_r.g_r_1_oracle_vs_ed import THRESHOLD, run

pytestmark = pytest.mark.gate


def test_g_r_1_oracle_matches_exact_diagonalisation() -> None:
    """G-R.1: analytic oracle against exact diagonalisation, threshold 1e-9."""
    passed, measured, cases = run()
    assert cases, "the gate ran no cases"
    assert passed, (
        f"G-R.1 FAILED: max abs error {measured['max_abs_error']:.3e} "
        f"against threshold {THRESHOLD:.0e}. Worst case: {measured['worst_case']}"
    )
