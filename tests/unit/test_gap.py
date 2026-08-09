"""Spectral gap: the closed forms, the sector question, and the precision floor.

Each test here corresponds to a statement WP1 makes in prose. The plan's instruction for
T1.1 is "derive what is actually true for each landscape family" rather than assert it, and
a derivation nobody checked is an assertion with extra steps.
"""

from __future__ import annotations

import numpy as np
import pytest

from quasarstack.classical.landscapes import additive_fitness
from quasarstack.spectral.gap import (
    additive_gap,
    class_gap,
    class_gap_extended,
    class_tridiagonal,
    dense_gap,
    locate_gap_minimum,
    pure_mutation_gap,
    symmetric_sector_holds_lambda2,
)

pytestmark = pytest.mark.fast


def single_peak_classes(n_sites: int, height: float) -> np.ndarray:
    classes = np.zeros(n_sites + 1)
    classes[0] = height
    return classes


@pytest.mark.parametrize("n_sites", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("mu", [0.05, 0.2, 0.5, 1.0])
def test_additive_gap_matches_brute_force(n_sites: int, mu: float) -> None:
    """``Delta = 2 min_i sqrt(a_i^2 + mu^2)``, against a dense diagonalisation."""
    a = np.random.default_rng(n_sites * 100 + int(mu * 100)).uniform(0.2, 2.0, size=n_sites)
    assert dense_gap(additive_fitness(a), mu) == pytest.approx(additive_gap(a, mu), abs=1e-12)


def test_additive_gap_does_not_close_with_system_size() -> None:
    """The additive family is easy for every method at every size, which is why it is a
    ruler and not a target. Same per-site fitness at every L gives the same gap."""
    gaps = {L: dense_gap(additive_fitness(np.ones(L)), 0.2) for L in (2, 4, 6, 8, 10)}
    assert len(set(round(g, 12) for g in gaps.values())) == 1, gaps
    assert gaps[2] == pytest.approx(additive_gap(np.ones(2), 0.2), abs=1e-12)


def test_lambda_two_of_the_additive_generator_is_l_fold_degenerate() -> None:
    """Second eigenvalue flips the cheapest single site, and with equal sites there are L
    equally cheap ways to do it. The degeneracy is the reason the gap is L-independent."""
    n_sites = 8
    fitness = additive_fitness(np.ones(n_sites))
    from quasarstack.analytic.exact_diag import mutation_selection_generator

    values = np.linalg.eigvalsh(mutation_selection_generator(fitness, 0.2).toarray())[::-1]
    assert int(np.sum(np.isclose(values, values[1], atol=1e-9))) == n_sites


@pytest.mark.parametrize("n_sites", [8, 16, 32])
def test_gap_saturates_at_twice_mu_above_the_error_threshold(n_sites: int) -> None:
    """Once selection has lost, the generator is the mutation operator and its gap is 2 mu.
    Approached as L grows, so the tolerance loosens at small L on purpose."""
    classes = single_peak_classes(n_sites, 1.0)
    measured = class_gap(classes, 0.5)
    assert measured == pytest.approx(pure_mutation_gap(0.5), rel=4.0 / n_sites)


@pytest.mark.parametrize("mu", [0.05, 0.1, 0.2, 0.4])
@pytest.mark.parametrize("n_sites", [4, 6, 8])
def test_class_reduction_finds_the_true_second_eigenvalue_for_the_single_peak(
    n_sites: int, mu: float
) -> None:
    """The reduction diagonalises one sector out of ``2**L``. For the single peak
    ``lambda_2`` turns out to live in it, so the cheap gap is the real gap. Checked rather
    than assumed, because it is not true by construction and need not hold elsewhere."""
    report = symmetric_sector_holds_lambda2(single_peak_classes(n_sites, 1.0), mu)
    assert report["lambda2_is_symmetric"], report


def test_symmetric_sector_gap_can_only_exceed_the_full_gap() -> None:
    """A sector is a subspace, so restricting to it can only remove eigenvalues from
    between lambda_1 and lambda_2. This is the inequality that makes a disagreement
    interpretable rather than just an error."""
    for n_sites in (4, 6, 8):
        for mu in (0.05, 0.15, 0.3):
            report = symmetric_sector_holds_lambda2(single_peak_classes(n_sites, 1.0), mu)
            assert report["gap_symmetric_sector"] >= report["gap_full"] - 1e-12


def test_class_tridiagonal_agrees_with_the_quasispecies_construction() -> None:
    """Same matrix as `crow_kimura.class_quasispecies` builds internally. If these ever
    drift apart, the gap map and the quasispecies would describe different operators."""
    from quasarstack.analytic.crow_kimura import class_quasispecies

    classes = single_peak_classes(6, 1.5)
    diagonal, offdiagonal = class_tridiagonal(classes, 0.2)
    from scipy.linalg import eigh_tridiagonal

    top = float(eigh_tridiagonal(diagonal, offdiagonal)[0][-1])
    assert top == pytest.approx(class_quasispecies(classes, 0.2)[2], abs=1e-12)


def test_extended_precision_agrees_with_float64_where_float64_is_trustworthy() -> None:
    """Away from the avoided crossing the two must agree. If they do not, the Sturm
    bisection is wrong, not the eigensolver."""
    classes = single_peak_classes(16, 1.0)
    for mu in ("0.05", "0.2", "0.5"):
        assert float(class_gap_extended(classes, mu, dps=50)) == pytest.approx(
            class_gap(classes, float(mu)), rel=1e-10
        )


@pytest.mark.slow
def test_the_threshold_sits_at_mu_times_l_equal_to_the_peak_height() -> None:
    """``mu* L -> height`` with a 1/L correction, and the collapse across heights is exact.

    Two heights are used because a law stated for one height is a coincidence. The ratio
    ``mu* L / height`` must agree between them to the precision of the search, not merely
    be close.
    """
    ratios = {}
    for height in (1.0, 2.5):
        for n_sites in (16, 64):
            result = locate_gap_minimum(
                lambda size, h=height: single_peak_classes(size, h),
                n_sites,
                f"{0.3 * height / n_sites}",
                f"{3.0 * height / n_sites}",
                dps=40,
                iterations=90,
            )
            ratios[(height, n_sites)] = float(result["mu_star_times_L"]) / height

    for n_sites in (16, 64):
        assert ratios[(1.0, n_sites)] == pytest.approx(ratios[(2.5, n_sites)], rel=1e-6)
    # The finite-size correction is 1/L, so doubling L twice quarters the excess over 1.
    assert (ratios[(1.0, 64)] - 1.0) < 0.4 * (ratios[(1.0, 16)] - 1.0)


def test_dense_gap_refuses_a_size_it_cannot_hold() -> None:
    with pytest.raises(ValueError, match="dense_limit"):
        dense_gap(np.zeros(1 << 14), 0.2)


def test_negative_mutation_rate_is_rejected() -> None:
    for call in (
        lambda: additive_gap(np.ones(3), -0.1),
        lambda: pure_mutation_gap(-0.1),
        lambda: class_tridiagonal(np.zeros(4), -0.1),
    ):
        with pytest.raises(ValueError, match="non-negative"):
            call()
