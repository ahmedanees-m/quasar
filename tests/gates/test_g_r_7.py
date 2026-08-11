"""Gate G-R.7 as an executable acceptance criterion."""

from __future__ import annotations

import pytest

from experiments.wp_r_rebuild.g_r_7_motta import (
    COSINE_THRESHOLD,
    ENERGY_RISE_TOLERANCE,
    MAX_WEIGHT,
    run,
)

pytestmark = pytest.mark.gate


@pytest.fixture(scope="module")
def gate_result():
    """One run of the gate, shared by every test in this file.

    Each test used to call ``run`` itself, so the gate executed once per test. It is
    deterministic, ADR-0016 having pinned the eigensolver's start vector, so the repeats
    produced identical numbers at full price.
    """
    return run()


def test_g_r_7_motta_reaches_the_reference_with_a_descending_energy(gate_result) -> None:
    """G-R.7: cosine >= 0.95, and no energy increase beyond 1e-10 on any step."""
    passed, measured, cases = gate_result
    assert cases, "the gate ran no configurations, which is a failure and not a pass"
    assert measured["all_energies_descend"], (
        f"G-R.7 FAILED on descent: {measured['total_rises_beyond_tolerance']} steps rose "
        f"beyond {ENERGY_RISE_TOLERANCE:.0e}, largest {measured['largest_energy_rise']:.3e}. "
        f"An ascending energy is the failure the planning documents record for this method"
    )
    assert measured["min_cosine"] >= COSINE_THRESHOLD, (
        f"G-R.7 FAILED on accuracy: min cosine {measured['min_cosine']:.7f} against threshold "
        f"{COSINE_THRESHOLD}. Worst case: {measured['worst_case']}"
    )
    assert passed


def test_the_wrong_parity_basis_is_still_dead(gate_result) -> None:
    """The reason the generator basis is what it is, checked in every gate run.

    If someone widened the basis to include Y-free strings, believing more generators must
    help, the method would silently stop working in exactly the way the planning documents
    describe. This keeps the evidence for that in the artefact rather than only in a
    docstring.
    """
    _, measured, _ = gate_result
    assert measured["parity_even_y_max_abs"] == 0.0, (
        "an even-Y string contributed something, which would mean the parity argument for "
        "the generator basis is wrong"
    )
    assert measured["parity_odd_y_min_norm"] > 1e-6


def test_the_registered_support_cutoff_is_the_one_used() -> None:
    """max_weight = 2, from the pre-run scan in Amendment 7. Weight 1 fails the accuracy
    threshold outright at 0.8298, so this is not a free parameter."""
    assert MAX_WEIGHT == 2
