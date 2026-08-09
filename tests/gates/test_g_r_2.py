"""Gate G-R.2 as an executable acceptance criterion."""

from __future__ import annotations

import pytest

from experiments.wp_r_rebuild.g_r_2_hamiltonian_vs_oracle import (
    REQUIRED_CONFIGURATIONS,
    THRESHOLD,
    run,
)

pytestmark = pytest.mark.gate


def test_g_r_2_hamiltonian_ground_state_matches_the_oracle() -> None:
    """G-R.2: compiled Hamiltonian against the analytic quasispecies, cosine >= 0.999999."""
    passed, measured, cases = run()
    assert len(cases) == REQUIRED_CONFIGURATIONS, (
        f"G-R.2 registers {REQUIRED_CONFIGURATIONS} configurations, the run produced "
        f"{len(cases)}"
    )
    assert passed, (
        f"G-R.2 FAILED: min cosine {measured['min_cosine']:.9f} against threshold "
        f"{THRESHOLD}. Worst case: {measured['worst_case']}"
    )
