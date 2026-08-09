"""The order parameter of the error threshold, and how its location is extracted.

The error catastrophe is a localisation-delocalisation transition, and the quantity that
sees it is the surplus, or magnetisation,

    m = sum_sigma p(sigma) * (1 - 2 d(sigma) / L)

where ``d`` is the Hamming distance from the master sequence. It runs from ``m = 1`` when the
population sits entirely on the master to ``m = 0`` when the population is spread uniformly
over sequence space and the genetic information is gone.

Locating the threshold at finite L
----------------------------------

At finite L there is no singularity, so "the threshold" is a definition rather than a
discovery, and the definition has to be stated. The one used here is the peak of the
susceptibility ``chi = -dm/dmu``: the mutation rate at which the population's identity is
being lost fastest. It is standard, it is what sharpens toward the true transition as L
grows, and it is computed the same way for every landscape so that comparisons between them
are meaningful.

The width reported alongside it is the full width at half maximum of that peak. Reporting
the width is not decoration. At small L the peak is broad enough that quoting a single
threshold value without it would suggest a precision the finite system does not have.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quasarstack.analytic.exact_diag import infer_n_sites


def magnetisation(probs: NDArray[np.float64], n_sites: int | None = None) -> float:
    """Surplus of the distribution: +1 fully on the master sequence, 0 fully delocalised.

    Accepts a distribution over all ``2**L`` genotypes.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if n_sites is None:
        n_sites = infer_n_sites(probs)
    index = np.arange(probs.size, dtype=np.uint64)
    distance = np.bitwise_count(index).astype(np.float64)
    return float(np.sum(probs * (1.0 - 2.0 * distance / n_sites)))


def localisation(probs: NDArray[np.float64], reference: int, n_sites: int | None = None) -> float:
    """Surplus measured from an arbitrary reference genotype rather than from all-wild-type.

    `magnetisation` hardcodes the master sequence as genotype 0, which is right for the
    single peak and for any additive landscape, and wrong for a rugged one. Measurement:
    Rough Mount Fuji keeps its optimum at genotype 0 in 97% of instances at roughness 0.3,
    where it has 1.4 local optima, and in 25% at roughness 1.0 at L = 12, where it has 121.
    Every family that is meaningfully rugged has its optimum at a random genotype. So a
    ruggedness sweep that keeps measuring from genotype 0 stops measuring localisation and
    starts measuring how far the optimum has wandered, which is a different quantity that
    happens to look like a threshold crossing.

    Passing the instance's own fittest genotype as ``reference`` makes the order parameter
    mean the same thing at every ruggedness: how concentrated the quasispecies is on the
    fittest sequence. It reduces to `magnetisation` exactly when that sequence is genotype 0.
    See ADR-0017.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if n_sites is None:
        n_sites = infer_n_sites(probs)
    if not 0 <= reference < probs.size:
        raise ValueError(f"reference must index a genotype, got {reference} for {probs.size}")
    index = np.arange(probs.size, dtype=np.uint64)
    distance = np.bitwise_count(index ^ np.uint64(reference)).astype(np.float64)
    return float(np.sum(probs * (1.0 - 2.0 * distance / n_sites)))


def magnetisation_from_classes(class_probs: NDArray[np.float64]) -> float:
    """Surplus from an ``L + 1`` Hamming-class distribution.

    The same quantity as :func:`magnetisation`, computed in linear rather than exponential
    time. Used wherever the landscape is permutation symmetric, and cross-checked against
    the genotype-level version in the tests.
    """
    class_probs = np.asarray(class_probs, dtype=np.float64)
    n_sites = class_probs.size - 1
    distance = np.arange(n_sites + 1, dtype=np.float64)
    return float(np.sum(class_probs * (1.0 - 2.0 * distance / n_sites)))


def susceptibility(
    mus: NDArray[np.float64], magnetisations: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``chi = -dm/dmu``, by central differences on the sweep grid."""
    mus = np.asarray(mus, dtype=np.float64)
    magnetisations = np.asarray(magnetisations, dtype=np.float64)
    if mus.shape != magnetisations.shape:
        raise ValueError(f"shape mismatch: {mus.shape} against {magnetisations.shape}")
    if mus.size < 3:
        raise ValueError("need at least three sweep points to differentiate")
    return -np.gradient(magnetisations, mus)


def crossover_point(
    mus: NDArray[np.float64], magnetisations: NDArray[np.float64], fraction: float = 0.5
) -> float:
    """Mutation rate at which the surplus first falls to ``fraction`` of its initial value.

    Linear interpolation between the two bracketing grid points. Returns NaN when the
    surplus never falls that far inside the sweep.

    This exists because the susceptibility peak, the natural definition for a genuine
    transition, is not defined for every landscape at accessible system size. A landscape
    additive in the surplus decays monotonically with its steepest slope at zero mutation
    rate, so the peak sits on the boundary of any sweep and carries no information. The
    half-surplus crossover is defined for any monotone decay, and it is comparable across
    landscape families, which is what a statement about epistasis shifting the threshold
    requires.
    """
    mus = np.asarray(mus, dtype=np.float64)
    m = np.asarray(magnetisations, dtype=np.float64)
    target = fraction * m[0]

    below = np.flatnonzero(m <= target)
    if below.size == 0:
        return float("nan")
    index = int(below[0])
    if index == 0:
        return float(mus[0])

    m_hi, m_lo = m[index - 1], m[index]
    if m_hi == m_lo:
        return float(mus[index])
    weight = (m_hi - target) / (m_hi - m_lo)
    return float(mus[index - 1] + weight * (mus[index] - mus[index - 1]))


def locate_threshold(
    mus: NDArray[np.float64], magnetisations: NDArray[np.float64]
) -> dict[str, float]:
    """Threshold location and transition width, from the susceptibility peak.

    Returns
    -------
    dict
        ``mu_c`` the peak location, ``chi_max`` its height, ``width`` the full width at half
        maximum of the peak, and ``m_at_mu_c`` the surplus there.

    Notes
    -----
    The peak is located on the sweep grid, so ``mu_c`` is resolved only to the grid spacing.
    That resolution is recorded with the result rather than hidden by interpolation, because
    at these system sizes the width is the honest uncertainty and it is far larger than the
    spacing.
    """
    mus = np.asarray(mus, dtype=np.float64)
    chi = susceptibility(mus, magnetisations)

    peak = int(np.argmax(chi))
    chi_max = float(chi[peak])
    half = 0.5 * chi_max

    above = np.flatnonzero(chi >= half)
    width = float(mus[above[-1]] - mus[above[0]]) if above.size > 1 else 0.0

    return {
        "mu_c": float(mus[peak]),
        "chi_max": chi_max,
        "width": width,
        "m_at_mu_c": float(np.asarray(magnetisations)[peak]),
        "grid_spacing": float(np.min(np.diff(mus))),
        "mu_half": crossover_point(mus, magnetisations, 0.5),
        # A peak on the first grid point is not a peak. It means the surplus is steepest at
        # the smallest mutation rate swept, so there is no interior transition to locate and
        # mu_c should be read as "none found", with mu_half used instead.
        "peak_is_interior": bool(0 < peak < len(mus) - 1),
    }
