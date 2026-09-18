"""Gate G-R.7 as a test, under the `gate` marker."""

from __future__ import annotations

import pytest

from experiments.wp_r.g_r_7_motta import (
    COSINE_THRESHOLD,
    ENERGY_RISE_TOLERANCE,
    MAX_WEIGHT,
    run,
)

pytestmark = pytest.mark.gate


@pytest.fixture(scope="module")
def gate_result():
    """One run of the gate, shared by every test in this file."""
    return run()


def test_g_r_7_motta_reaches_the_reference_with_a_descending_energy(gate_result) -> None:
    """G-R.7: cosine >= 0.95, and no energy increase beyond 1e-10 on any step."""
    passed, measured, cases = gate_result
    assert cases, "the gate ran no configurations"
    assert measured["all_energies_descend"], (
        f"G-R.7 FAILED on descent: {measured['total_rises_beyond_tolerance']} steps rose "
        f"beyond {ENERGY_RISE_TOLERANCE:.0e}, largest {measured['largest_energy_rise']:.3e}"
    )
    assert measured["min_cosine"] >= COSINE_THRESHOLD, (
        f"G-R.7 FAILED on accuracy: min cosine {measured['min_cosine']:.7f} against threshold "
        f"{COSINE_THRESHOLD}. Worst case: {measured['worst_case']}"
    )
    assert passed


def test_even_y_strings_give_a_zero_right_hand_side(gate_result) -> None:
    """Even-Y strings contribute nothing to Motta's right-hand side for a real state."""
    _, measured, _ = gate_result
    assert (
        measured["parity_even_y_max_abs"] == 0.0
    ), "an even-Y string contributed to the right-hand side"
    assert measured["parity_odd_y_min_norm"] > 1e-6


def test_the_support_cutoff() -> None:
    """max_weight = 2. Weight 1 reaches cosine 0.8298, below the threshold."""
    assert MAX_WEIGHT == 2
