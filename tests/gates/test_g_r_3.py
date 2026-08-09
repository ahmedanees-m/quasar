"""Gate G-R.3 as an executable acceptance criterion."""

from __future__ import annotations

import pytest

from experiments.wp_r_rebuild.g_r_3_trotter_scaling import (
    COSINE_THRESHOLD,
    EXPONENT_BOUNDS,
    R_SQUARED_THRESHOLD,
    run,
)

pytestmark = pytest.mark.gate


def test_g_r_3_trotter_converges_with_second_order_error() -> None:
    """G-R.3: convergence at cosine >= 0.999 and a fitted exponent in [1.8, 2.2]."""
    passed, measured, cases = run()
    assert cases, "the gate ran no configurations, which is a failure and not a pass"
    assert measured["all_converged"], (
        f"G-R.3 FAILED on convergence: min cosine {measured['min_cosine']:.7f} against "
        f"threshold {COSINE_THRESHOLD}"
    )
    assert measured["fits_above_r_squared_threshold"], (
        f"G-R.3 FAILED on fit quality: min R squared {measured['min_r_squared']:.5f} against "
        f"threshold {R_SQUARED_THRESHOLD}. A poor fit means the error is not a clean power "
        f"law, so the exponent below is not meaningful either"
    )
    assert measured["exponents_within_bounds"], (
        f"G-R.3 FAILED on exponent: range {measured['min_fitted_exponent']:.3f} to "
        f"{measured['max_fitted_exponent']:.3f} against bounds {EXPONENT_BOUNDS}"
    )
    assert passed
