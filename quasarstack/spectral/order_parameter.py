"""The order parameter of the error threshold and the location of the transition.

The error catastrophe is a localisation-delocalisation transition, measured by the surplus, or
magnetisation,

    m = sum_sigma p(sigma) * (1 - 2 d(sigma) / L)

where ``d`` is the Hamming distance from the master sequence. It runs from ``m = 1`` with the
population on the master sequence to ``m = 0`` with the population uniform over sequence space.

Locating the threshold at finite L
----------------------------------

At finite L there is no singularity. The threshold is defined here as the peak of the
susceptibility ``chi = -dm/dmu``, which sharpens toward the transition as L grows and is computed
the same way for every landscape. The full width at half maximum of that peak is reported with
it.
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
    """Surplus measured from an arbitrary reference genotype.

    `magnetisation` measures from genotype 0, which is the optimum for the single peak and additive
    landscapes but not for rugged ones: Rough Mount Fuji at L = 12 keeps its optimum at genotype 0 in
    97% of instances at roughness 0.3 and in 25% at roughness 1.0. Passing the instance's fittest
    genotype as ``reference`` measures concentration on the fittest sequence at every ruggedness, and
    reduces to `magnetisation` when that sequence is genotype 0.
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

    The same quantity as :func:`magnetisation`, in linear rather than exponential time, for
    permutation-symmetric landscapes.
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

    Linear interpolation between the bracketing grid points; NaN when the surplus does not fall that
    far within the sweep.

    A landscape additive in the surplus decays monotonically with its steepest slope at zero mutation
    rate, so its susceptibility peak lies on the boundary of any sweep. The half-surplus crossover is
    defined for any monotone decay and is comparable across landscape families.
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
    The peak is located on the sweep grid, so ``mu_c`` is resolved to the grid spacing; at these
    system sizes the width is much larger than the spacing.
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
        # A peak on the first grid point means the surplus is steepest at the smallest mutation rate
        # swept, so there is no interior transition; mu_half is used instead.
        "peak_is_interior": bool(0 < peak < len(mus) - 1),
    }
