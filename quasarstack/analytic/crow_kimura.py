"""The analytic ruler: closed-form quasispecies solutions of the Crow-Kimura model.

Nothing in this module ever forms the ``2**L`` generator. That is the point. Gate G-R.1
compares what is computed here against brute-force exact diagonalisation in
`quasarstack.analytic.exact_diag`, and the comparison only means something because the two
are independent constructions rather than the same computation written twice.

The model
---------

In the Crow-Kimura parallel mutation-selection model the unnormalised genotype frequencies
obey a linear equation ``dp/dt = W p`` with generator

    W = diag(f) + mu * sum_i (X_i - I)

where ``f`` is the Malthusian fitness, ``mu`` is the per-site mutation rate, and ``X_i``
flips site i. The quasispecies is the Perron eigenvector of W, and the equilibrium mean
fitness is its eigenvalue. W is symmetric here because mutation is symmetric between wild
type and mutant, so ``-W`` is stoquastic and its ground state is sign definite. That is what
lets L1 and L2 normalisation pick the same ray. See docs/notes.md.

Two exactly solvable families
-----------------------------

**Additive fitness.** With ``f(sigma) = sum_i a_i z_i`` and no coupling, W is a sum of
single-site operators acting on different sites, which therefore commute. The spectrum is
the sum of single-site spectra and the Perron eigenvector is a product state. Each site
contributes

    w_i = [[a_i - mu,      mu   ],
           [   mu, -a_i - mu]]

with largest eigenvalue ``-mu + sqrt(a_i**2 + mu**2)`` and eigenvector proportional to
``(1, r_i)`` where ``r_i = (sqrt(a_i**2 + mu**2) - a_i) / mu``. This is a closed form: no
eigensolver is called at any point, at any L.

**Permutation-symmetric fitness.** When ``f`` depends only on the number of mutated sites d,
the Perron eigenvector is permutation symmetric too, so the per-genotype amplitude ``q_d``
is shared across a Hamming class. Counting mutations out of a class-d genotype gives ``L-d``
routes up and ``d`` routes down, so

    dq_d/dt = (f_d - mu L) q_d + mu (L - d) q_{d+1} + mu d q_{d-1}

an ``(L+1)``-dimensional tridiagonal problem. It is not symmetric, but conjugating by
``diag(sqrt(binom(L, d)))`` makes it so, with off-diagonal ``mu sqrt((d+1)(L-d))``. The cost
is therefore linear in L rather than exponential.

The two families overlap when every ``a_i`` is equal, since ``sum_i z_i = L - 2d`` is then a
function of d alone. Gate G-R.1 uses that overlap to check the closed form against the class
reduction as well as against exact diagonalisation, which turns the gate into a three-way
agreement rather than a pairwise one.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh_tridiagonal
from scipy.special import gammaln


def _site_ratio(a: float, mu: float) -> tuple[float, float]:
    """Return ``(v0, v1)``, the unnormalised top eigenvector of one site's generator.

    Written in whichever algebraically equivalent form avoids cancellation. The direct
    expression ``(sqrt(a**2 + mu**2) - a) / mu`` subtracts two nearly equal numbers when
    ``a >> mu``, which is exactly the strong-selection regime the error-threshold sweep
    spends most of its time in, so that branch is rationalised to ``mu / (sqrt(...) + a)``
    instead. For ``a < 0`` both terms of the original numerator are positive and the direct
    form is the stable one.
    """
    root = float(np.hypot(a, mu))
    if mu == 0.0:
        if a > 0.0:
            return 1.0, 0.0
        if a < 0.0:
            return 0.0, 1.0
        raise ValueError(
            "a = 0 and mu = 0 leaves the site degenerate; the quasispecies is undefined"
        )
    if a >= 0.0:
        return 1.0, mu / (root + a)
    return 1.0, (root - a) / mu


def additive_mean_fitness(a: NDArray[np.float64], mu: float) -> float:
    """Equilibrium mean fitness of an additive landscape.

    The Perron eigenvalue of the full generator, obtained as the sum of the single-site
    largest eigenvalues ``-mu + sqrt(a_i**2 + mu**2)``.
    """
    a = np.asarray(a, dtype=np.float64)
    _check_mu(mu)
    return float(np.sum(-mu + np.hypot(a, mu)))


def additive_quasispecies(a: NDArray[np.float64], mu: float) -> NDArray[np.float64]:
    """Closed-form quasispecies distribution for an additive landscape.

    Parameters
    ----------
    a
        Length-L array of per-site fitness coefficients in the spin convention. Positive
        ``a_i`` favours wild type at site i.
    mu
        Per-site mutation rate, non-negative.

    Returns
    -------
    ndarray
        Length ``2**L`` L1-normalised, non-negative distribution, indexed as described in
        `quasarstack.io.conventions`.

    Notes
    -----
    No eigensolver is used. The product is accumulated site by site, which keeps the layout
    consistent with the project's index convention: site i carries weight ``2**i``, so
    appending a site doubles the array with the wild-type half first.
    """
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 1 or a.size < 1:
        raise ValueError(f"a must be a non-empty one-dimensional array, got shape {a.shape}")
    _check_mu(mu)

    probs = np.ones(1, dtype=np.float64)
    for coefficient in a:
        v0, v1 = _site_ratio(float(coefficient), mu)
        probs = np.concatenate([probs * v0, probs * v1])

    total = float(probs.sum())
    if total <= 0.0:
        raise ValueError("degenerate additive landscape: the product state has zero total weight")
    normalised: NDArray[np.float64] = probs / total
    return normalised


def class_distribution(
    f_by_class: NDArray[np.float64], mu: float
) -> tuple[NDArray[np.float64], float]:
    """Class probabilities and mean fitness, without ever forming the ``2**L`` vector.

    The point of the Hamming-class reduction is that a permutation-symmetric landscape needs
    only ``L + 1`` numbers. `class_quasispecies` then expands those back to one entry per
    genotype, which is what the gates comparing against exact diagonalisation need, and which
    costs ``2**L`` words. Callers that only want the class distribution or the mean fitness
    should not pay that: at ``L = 32`` the expansion alone asks for 32 GiB and the reduction
    it is built on is a 33-dimensional tridiagonal solve.

    Returns ``(class_probs, mean_fitness)``, both as in `class_quasispecies`.
    """
    class_probs, _, _, mean_fitness = _class_solve(f_by_class, mu)
    return class_probs, mean_fitness


def _class_solve(
    f_by_class: NDArray[np.float64], mu: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
    """Shared core: ``(class_probs, per_genotype_amplitude, class_total, mean_fitness)``."""
    f_by_class = np.asarray(f_by_class, dtype=np.float64)
    if f_by_class.ndim != 1 or f_by_class.size < 2:
        raise ValueError(
            f"f_by_class must be one-dimensional of length L+1, got {f_by_class.shape}"
        )
    _check_mu(mu)

    n_sites = f_by_class.size - 1
    d = np.arange(n_sites + 1, dtype=np.float64)

    diagonal = f_by_class - mu * n_sites
    offdiagonal = mu * np.sqrt((d[:-1] + 1.0) * (n_sites - d[:-1]))

    eigenvalues, eigenvectors = eigh_tridiagonal(diagonal, offdiagonal)
    mean_fitness = float(eigenvalues[-1])
    # Perron: the top eigenvector of a symmetric tridiagonal with non-negative off-diagonals
    # is sign definite, so the modulus recovers it rather than destroying information.
    top = np.abs(eigenvectors[:, -1])

    # log binom(L, d), used to undo the symmetrising conjugation without overflowing.
    log_binom = gammaln(n_sites + 1.0) - gammaln(d + 1.0) - gammaln(n_sites - d + 1.0)
    half = np.exp(0.5 * log_binom)

    class_weight = half * top  # = binom(L, d) * q_d, the unnormalised class total
    class_total = float(class_weight.sum())
    if class_total <= 0.0:
        raise ValueError("degenerate class landscape: the Perron vector has zero total weight")

    return class_weight / class_total, top / half, class_total, mean_fitness


def class_quasispecies(
    f_by_class: NDArray[np.float64], mu: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Quasispecies of a permutation-symmetric landscape, by Hamming-class reduction.

    Parameters
    ----------
    f_by_class
        Length ``L + 1`` array of class fitnesses; entry d applies to every genotype with d
        mutated sites.
    mu
        Per-site mutation rate, non-negative.

    Returns
    -------
    genotype_probs
        Length ``2**L`` L1-normalised distribution over genotypes.
    class_probs
        Length ``L + 1`` L1-normalised distribution over Hamming classes, the quantity the
        storage policy keeps for every swept cell.
    mean_fitness
        The Perron eigenvalue, which is the equilibrium mean fitness.

    Notes
    -----
    Solves an ``(L+1)``-dimensional symmetric tridiagonal eigenproblem, never the ``2**L``
    one. Binomial coefficients are carried through their logarithms so that the conjugating
    factors stay finite at large L, where ``binom(L, L/2)`` overflows well before the
    probabilities themselves do.

    The expansion back to one entry per genotype is the only ``2**L`` step and it is what
    the gates comparing against exact diagonalisation need. Use `class_distribution` when
    the class probabilities are enough; beyond L of about 25 the expansion is the thing that
    stops this working, not the eigenproblem.
    """
    class_probs, per_genotype, class_total, mean_fitness = _class_solve(f_by_class, mu)
    n_sites = class_probs.size - 1

    index = np.arange(1 << n_sites, dtype=np.uint64)
    weights = np.bitwise_count(index).astype(np.int64)
    genotype_probs = per_genotype[weights] / class_total

    return genotype_probs, class_probs, mean_fitness


def single_peak_quasispecies(
    n_sites: int, height: float, mu: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Quasispecies of the sharp-peak landscape, the control case.

    Fitness ``height`` on the master sequence and zero everywhere else. Returns the same
    triple as :func:`class_quasispecies`.
    """
    from quasarstack.classical.landscapes import single_peak_classes

    return class_quasispecies(single_peak_classes(n_sites, height), mu)


def _check_mu(mu: float) -> None:
    if not np.isfinite(mu) or mu < 0.0:
        raise ValueError(f"mu must be finite and non-negative, got {mu}")
