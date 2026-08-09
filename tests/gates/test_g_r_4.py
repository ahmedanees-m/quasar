"""Gate G-R.4 as an executable acceptance criterion."""

from __future__ import annotations

import pytest

from experiments.wp_r_rebuild.g_r_4_error_threshold import THRESHOLD, run

pytestmark = pytest.mark.gate


def test_g_r_4_error_threshold_matches_the_analytic_prediction() -> None:
    """G-R.4: surplus from the compiled Hamiltonian against the analytic class reduction."""
    passed, measured, cases = run()
    assert cases, "the gate ran no cases, which is a failure and not a pass"
    assert passed, (
        f"G-R.4 FAILED: max |delta m| {measured['max_abs_delta_m']:.3e} against threshold "
        f"{THRESHOLD:.0e}. Worst case: {measured['worst_case']}"
    )


def test_the_mu_linear_assembly_still_matches_the_compiler() -> None:
    """The sweep assembles H(mu) from two compilations instead of one per point.

    That shortcut is what makes the gate affordable, and it is only valid while it agrees
    with the compiler exactly. If it ever drifts, the gate above would still pass while
    measuring something other than the compiled operator.
    """
    _, _, cases = run()
    worst = max(case["assembly_error"] for case in cases)
    assert worst < 1e-12, f"mu-linear assembly diverged from the compiler by {worst:.2e}"
