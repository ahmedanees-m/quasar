"""Gate G-R.1 as an executable acceptance criterion.

The pre-registered thresholds are not aspirational documents. Each one is a test, and a
failing gate test is a red build. Runs under the `gate` marker, which CI executes nightly
and at release rather than on every push.
"""

from __future__ import annotations

import pytest

from experiments.wp_r_rebuild.g_r_1_oracle_vs_ed import THRESHOLD, run

pytestmark = pytest.mark.gate


def test_g_r_1_oracle_matches_exact_diagonalisation() -> None:
    """G-R.1: analytic oracle against brute-force exact diagonalisation, threshold 1e-9."""
    passed, measured, cases = run()
    assert cases, "the gate ran no cases, which is a failure and not a pass"
    assert passed, (
        f"G-R.1 FAILED: max abs error {measured['max_abs_error']:.3e} "
        f"against threshold {THRESHOLD:.0e}. Worst case: {measured['worst_case']}"
    )
