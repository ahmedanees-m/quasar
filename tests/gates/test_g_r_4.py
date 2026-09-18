"""Gate G-R.4 as a test, under the `gate` marker."""

from __future__ import annotations

import pytest

from experiments.wp_r.g_r_4_error_threshold import THRESHOLD, run

pytestmark = pytest.mark.gate


@pytest.fixture(scope="module")
def gate_result():
    """One run of the gate, shared by every test in this file."""
    return run()


def test_g_r_4_error_threshold_matches_the_analytic_prediction(gate_result) -> None:
    """G-R.4: surplus from the compiled Hamiltonian against the analytic class reduction."""
    passed, measured, cases = gate_result
    assert cases, "the gate ran no cases"
    assert passed, (
        f"G-R.4 FAILED: max |delta m| {measured['max_abs_delta_m']:.3e} against threshold "
        f"{THRESHOLD:.0e}. Worst case: {measured['worst_case']}"
    )


def test_the_mu_linear_assembly_matches_the_compiler(gate_result) -> None:
    """The sweep assembles H(mu) from two compilations; it must equal the compiled operator."""
    _, _, cases = gate_result
    worst = max(case["assembly_error"] for case in cases)
    assert worst < 1e-12, f"mu-linear assembly diverged from the compiler by {worst:.2e}"
