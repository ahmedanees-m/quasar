"""Gate G-R.6 as a test, under the `gate` marker."""

from __future__ import annotations

import pytest

from experiments.wp_r.g_r_6_varqite import COSINE_THRESHOLD, reps_for, run

pytestmark = pytest.mark.gate


@pytest.fixture(scope="module")
def gate_result():
    """One run of the gate, shared by every test in this file."""
    return run()


def test_g_r_6_varqite_reaches_the_reference_at_constant_depth(gate_result) -> None:
    """G-R.6: cosine >= 0.999, and depth identical at tau = 2.5 and tau = 20."""
    passed, measured, cases = gate_result
    assert cases, "the gate ran no configurations"
    assert measured["all_depths_unchanged"], "circuit depth changed with imaginary time"
    assert measured["min_cosine"] >= COSINE_THRESHOLD, (
        f"G-R.6 FAILED on accuracy: min cosine {measured['min_cosine']:.7f} against threshold "
        f"{COSINE_THRESHOLD}. Worst case: {measured['worst_case']}"
    )
    assert passed


def test_the_ansatz_rule() -> None:
    """reps = L + 2."""
    assert [reps_for(n) for n in (3, 4, 5, 6)] == [5, 6, 7, 8]


def test_the_hardware_route_reproduces_the_mclachlan_quantities(gate_result) -> None:
    """The McLachlan quantities from circuit measurements match the state-vector derivatives."""
    _, measured, _ = gate_result
    assert measured["hardware_route_max_force_error"] < 1e-10
    assert measured["hardware_route_max_tensor_error"] < 1e-10


def test_energy_rises_shrink_with_the_step(gate_result) -> None:
    """Energy rises come from explicit Euler overshoot and shrink with the step size.

    The continuous McLachlan flow cannot raise the energy: ``dE/dtau`` equals
    ``-(1/2) grad(E)^T (A + delta I)^-1 grad(E)``, non-positive because A is a Gram matrix. At
    L = 6 the largest rise is 4.35e-2, 1.03e-3 and 2.13e-4 for dtau of 0.05, 0.02 and 0.01.
    """
    _, measured, _ = gate_result
    assert measured["energy_rise_shrinks_with_step"], (
        "the energy rise did not shrink as the step size shrank: "
        f"{measured['descent_refinement']}"
    )
    finest = measured["descent_refinement"][-1]
    assert finest["largest_rise_relative_to_span"] < 1e-2, (
        f"at the finest step the energy still rose by "
        f"{finest['largest_rise_relative_to_span']:.2e} of the total descent"
    )
