"""Fitness landscapes, in the spin convention.

Only the families that work package WP-R needs are here: additive with optional pairwise
epistasis, and permutation-symmetric (class-dependent) fitness, of which the single peak is
a special case. WP3 extends this module with NK, spin glass, Rough Mount Fuji,
House of Cards and Block families, and gate G-3 judges those.

**Convention, binding project-wide.** Fitness is written in the spin convention

    f(sigma) = sum_i a_i z_i + sum_{i<j} b_ij z_i z_j

where ``z_i = +1`` when site i is wild type and ``z_i = -1`` when it is mutated. This is
the eigenvalue of the Pauli Z operator under the project's encoding, in which the qubit
state ``|0>`` is wild type and ``|1>`` is mutated.

The projector form ``a_i (I + Z_i) / 2`` is *not* used. Mixing the two is silent: the
resulting quasispecies is a plausible-looking distribution that is simply wrong. See
`DECISIONS.md` ADR-0002.

Genotype indexing follows `quasarstack.io.conventions`: the fitness vector returned by
these functions has length ``2**L``, and entry j is the fitness of the genotype whose site
i is mutated exactly when bit i of j is set.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# A fitness vector is a dense array of length 2**L. At L = 24 that is 134 MB, which is well
# past anything this project sweeps, so the guard catches an argument mistake rather than
# imposing a real limit.
MAX_SITES = 24


def _check_sites(n_sites: int) -> None:
    if n_sites < 1:
        raise ValueError(f"n_sites must be at least 1, got {n_sites}")
    if n_sites > MAX_SITES:
        raise ValueError(
            f"n_sites = {n_sites} would need a fitness vector of 2**{n_sites} entries; "
            f"the guard is set at {MAX_SITES}"
        )


def spin_matrix(n_sites: int) -> NDArray[np.int8]:
    """Return the ``(n_sites, 2**n_sites)`` array of z values.

    Entry ``[i, j]`` is +1 when site i of genotype j is wild type and -1 when it is
    mutated. Held as int8 so that the array stays small enough to reuse across the pairwise
    loop instead of being recomputed per term.
    """
    _check_sites(n_sites)
    index = np.arange(1 << n_sites, dtype=np.int64)
    z = np.empty((n_sites, 1 << n_sites), dtype=np.int8)
    for site in range(n_sites):
        z[site] = 1 - 2 * ((index >> site) & 1).astype(np.int8)
    return z


def additive_fitness(
    a: NDArray[np.float64], b: NDArray[np.float64] | None = None
) -> NDArray[np.float64]:
    """Fitness vector for an additive landscape with optional pairwise epistasis.

    Parameters
    ----------
    a
        Length-L array of per-site coefficients. Positive ``a_i`` makes wild type fitter at
        site i.
    b
        Optional ``(L, L)`` array of pairwise couplings. Only the strict upper triangle is
        read, so ``b[i, j]`` for ``i < j`` is the coupling and everything else is ignored.
        Passing a symmetric matrix therefore does not double count.

    Returns
    -------
    ndarray
        Length ``2**L`` fitness vector.
    """
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"a must be one-dimensional, got shape {a.shape}")
    n_sites = a.size
    z = spin_matrix(n_sites)

    fitness = np.zeros(1 << n_sites, dtype=np.float64)
    for site in range(n_sites):
        fitness += a[site] * z[site]

    if b is not None:
        b = np.asarray(b, dtype=np.float64)
        if b.shape != (n_sites, n_sites):
            raise ValueError(f"b must have shape ({n_sites}, {n_sites}), got {b.shape}")
        for i in range(n_sites):
            for j in range(i + 1, n_sites):
                if b[i, j] != 0.0:
                    fitness += b[i, j] * (z[i] * z[j])

    return fitness


def class_fitness(f_by_class: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fitness vector for a permutation-symmetric landscape.

    Parameters
    ----------
    f_by_class
        Length ``L + 1`` array; entry d is the fitness shared by every genotype with exactly
        d mutated sites.

    Returns
    -------
    ndarray
        Length ``2**L`` fitness vector.
    """
    f_by_class = np.asarray(f_by_class, dtype=np.float64)
    if f_by_class.ndim != 1 or f_by_class.size < 2:
        raise ValueError(
            f"f_by_class must be one-dimensional of length L+1, got {f_by_class.shape}"
        )
    n_sites = f_by_class.size - 1
    _check_sites(n_sites)
    index = np.arange(1 << n_sites, dtype=np.uint64)
    weights = np.bitwise_count(index).astype(np.int64)
    return f_by_class[weights]


def single_peak_classes(n_sites: int, height: float) -> NDArray[np.float64]:
    """Class fitnesses for the sharp-peak landscape.

    The master sequence, meaning zero mutated sites, has fitness ``height``; every other
    genotype has fitness zero. This is the control landscape: the analytic oracle and the
    Dixit-Srivastava-Vishnoi baseline both apply here, and no quantum method is expected to
    offer anything.
    """
    _check_sites(n_sites)
    f = np.zeros(n_sites + 1, dtype=np.float64)
    f[0] = float(height)
    return f


def uniform_additive_classes(n_sites: int, a: float) -> NDArray[np.float64]:
    """Class fitnesses equivalent to an additive landscape with every ``a_i`` equal to ``a``.

    With uniform coefficients the additive landscape is permutation symmetric, because
    ``sum_i z_i = L - 2d`` depends only on the number of mutated sites d. That makes it the
    one family reachable by both analytic routes, which is what lets gate G-R.1 cross-check
    the closed-form product solution against the Hamming-class reduction as well as against
    exact diagonalisation.
    """
    _check_sites(n_sites)
    d = np.arange(n_sites + 1, dtype=np.float64)
    return a * (n_sites - 2.0 * d)
