"""Gate G-R.5 as an executable acceptance criterion."""

from __future__ import annotations

import pytest

from experiments.wp_r_rebuild.g_r_5_rugged import SEEDS, THRESHOLD, run

pytestmark = pytest.mark.gate


@pytest.fixture(scope="module")
def gate_result():
    """One run of the gate, shared by every test in this file.

    Each test used to call ``run`` itself, so the gate executed once per test. It is
    deterministic, ADR-0016 having pinned the eigensolver's start vector, so the repeats
    produced identical numbers at full price.
    """
    return run()


def test_g_r_5_rugged_landscapes_match_exact_diagonalisation(gate_result) -> None:
    """G-R.5: compiled Hamiltonian against exact diagonalisation on NK landscapes."""
    passed, measured, cases = gate_result
    assert cases, "the gate ran no instances, which is a failure and not a pass"
    assert len(SEEDS) == 10, "the registered seed set is 0 through 9"
    assert passed, (
        f"G-R.5 FAILED: min cosine {measured['min_cosine']:.9f} against threshold "
        f"{THRESHOLD}, with {measured['n_below_threshold']} instances below it. "
        f"Worst case: {measured['worst_case']}"
    )


def test_ruggedness_rises_with_connectivity(gate_result) -> None:
    """Not part of the registered threshold, but a broken landscape generator would make the
    gate above meaningless: it would be comparing two routes on landscapes that are not
    actually rugged. WP3 gate G-3 owns this properly."""
    _, measured, _ = gate_result
    assert measured["local_optima_monotone_in_k"], (
        "local optima should rise with K on the seed mean; the ruggedness axis is not "
        "varying ruggedness"
    )
    assert measured[
        "autocorrelation_monotone_in_k"
    ], "fitness autocorrelation should fall with K on the seed mean"
