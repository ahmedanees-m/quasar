"""Gate G-R.6 as an executable acceptance criterion."""

from __future__ import annotations

import pytest

from experiments.wp_r_rebuild.g_r_6_varqite import COSINE_THRESHOLD, reps_for, run

pytestmark = pytest.mark.gate


@pytest.fixture(scope="module")
def gate_result():
    """One run of the gate, shared by every test in this file.

    Each test used to call ``run`` itself, so the gate executed once per test. It is
    deterministic, the eigensolver start vector having been pinned, so the repeats
    produced identical numbers at full price. This gate is the slowest in the project and
    was running three times: on its own it held the suite for the better part of an hour,
    which is how a suite stops being one anybody waits for.
    """
    return run()


def test_g_r_6_varqite_reaches_the_reference_at_constant_depth(gate_result) -> None:
    """G-R.6: cosine >= 0.999, and depth identical at tau = 2.5 and tau = 20."""
    passed, measured, cases = gate_result
    assert cases, "the gate ran no configurations, which is a failure and not a pass"
    assert measured["all_depths_unchanged"], (
        "circuit depth changed with imaginary time, which is the one property that makes "
        "varQITE a near-term method"
    )
    assert measured["min_cosine"] >= COSINE_THRESHOLD, (
        f"G-R.6 FAILED on accuracy: min cosine {measured['min_cosine']:.7f} against threshold "
        f"{COSINE_THRESHOLD}. Worst case: {measured['worst_case']}"
    )
    assert passed


def test_the_registered_ansatz_rule_is_the_one_used() -> None:
    """reps = L + 2, from the pre-run scan disclosed in revision 6. If this drifts, the
    gate is measuring a different method than the one registered."""
    assert [reps_for(n) for n in (3, 4, 5, 6)] == [5, 6, 7, 8]


def test_the_hardware_route_still_reproduces_the_mclachlan_quantities(gate_result) -> None:
    """Not part of the registered threshold, and the most important check in the gate.

    varQITE is computed here by differentiating a state vector. That is only a legitimate
    stand-in for a near-term method because the same quantities come from circuit
    measurements. If this drifts, the method stops being hardware-faithful while every
    accuracy number stays exactly as good.
    """
    _, measured, _ = gate_result
    assert measured["hardware_route_max_force_error"] < 1e-10
    assert measured["hardware_route_max_tensor_error"] < 1e-10


def test_energy_rises_are_discretisation_and_not_a_defect(gate_result) -> None:
    """This test originally demanded strict monotonicity, and that was wrong.

    The continuous McLachlan flow genuinely cannot raise the energy: ``dE/dtau`` equals
    ``-(1/2) grad(E)^T (A + delta I)^-1 grad(E)``, non-positive because A is a Gram matrix.
    But the integrator is explicit Euler, which overshoots at finite step, so a strict
    requirement was asserting something about the discretisation rather than about the
    method, and it failed on half the configurations.

    The falsifiable statement is the one that separates the two cases: an overshoot shrinks
    as the step shrinks, and a defect does not. Measured at L = 6, the largest rise went
    4.35e-2, 1.03e-3, 2.13e-4 for dtau of 0.05, 0.02, 0.01, roughly quadratic in the step,
    and always in the first step or two where the flow is stiffest.
    """
    _, measured, _ = gate_result
    assert measured["energy_rise_shrinks_with_step"], (
        "the energy rise did not shrink as the step size shrank, which is the signature of "
        "a defect rather than of Euler overshoot: "
        f"{measured['descent_refinement']}"
    )
    finest = measured["descent_refinement"][-1]
    assert finest["largest_rise_relative_to_span"] < 1e-2, (
        f"at the finest step the energy still rose by "
        f"{finest['largest_rise_relative_to_span']:.2e} of the total descent"
    )
