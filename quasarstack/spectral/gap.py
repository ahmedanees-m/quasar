"""Spectral gap of the mutation-selection generator. WP1 task T1.2.

The gap ``Delta = lambda_1 - lambda_2`` governs how fast *any* eigenvector-extraction method
converges, quantum or classical. Imaginary-time evolution suppresses the leading contaminant
as ``exp(-Delta tau)``; power iteration contracts by ``lambda_2 / lambda_1`` per step; QSVT
needs a polynomial whose degree scales with ``1 / Delta``. So the gap map is the object that
decides where a quantum method could possibly help, and it is worth computing carefully.

Three closed forms this module reproduces
-----------------------------------------

**Additive landscapes have an exact gap and it never closes.** The generator is a sum of
commuting single-site terms with eigenvalues ``-mu +/- sqrt(a_i^2 + mu^2)``, so the whole
spectrum is every sum of those choices. The largest is the Perron eigenvalue and the second
largest flips the cheapest single site, giving

    Delta = 2 min_i sqrt(a_i^2 + mu^2)

independent of L, with lambda_2 L-fold degenerate. The additive family is therefore *easy*
for every method at every size. It is a ruler, not a target, and no advantage claim can be
built on it.

**Above the error threshold the gap saturates at the pure-mutation value.** Once selection
has lost, the generator is dominated by ``mu sum_i (X_i - I)``, whose spectrum is
``mu(L - 2k) - mu L`` and whose gap is exactly ``2 mu``. Approached from below as L grows.

**The threshold itself sits at ``mu L = height``** for the single peak, with a ``1 / L``
finite-size correction. The collapse across peak heights is exact.

Two numerical traps, both real
------------------------------

**The class reduction sees only one sector.** A permutation-symmetric landscape reduces to an
``(L+1)``-dimensional tridiagonal problem, but that is the symmetric sector alone, ``L + 1``
of the ``2**L`` eigenvalues. The Perron vector lives there; ``lambda_2`` need not. Which gap
matters depends on what the initial state overlaps, so `symmetric_sector_holds_lambda2`
checks rather than assumes.

**At the threshold the minimum is an avoided crossing, and float64 cannot find it.** The gap
as a function of ``mu`` near the threshold behaves like ``sqrt(Delta_min^2 + c^2 (mu - mu*)^2)``.
Rounding ``mu*`` in the sixth decimal changed a measured minimum by ten percent at L = 32, and
past L is roughly 64 the minimum falls below what a float64 eigensolver can resolve. Worse,
two different LAPACK routines agree with each other to 1e-16 while both are wrong, because
they share the failure. `class_gap_extended` and `locate_gap_minimum` therefore work at arbitrary
precision through a Sturm bisection, which needs only ``O(L)`` arithmetic per evaluation and
never forms the matrix.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh_tridiagonal

from quasarstack.analytic.exact_diag import mutation_selection_generator

__all__ = [
    "additive_gap",
    "class_gap",
    "class_gap_extended",
    "class_tridiagonal",
    "dense_gap",
    "eigenvector_condition_number",
    "locate_gap_minimum",
    "pure_mutation_gap",
    "sparse_gap",
    "spectral_gap",
    "symmetric_sector_holds_lambda2",
]


def additive_gap(a: NDArray[np.float64], mu: float) -> float:
    """Closed-form spectral gap of an additive landscape, ``2 min_i sqrt(a_i^2 + mu^2)``.

    Exact, and independent of L. See the module docstring for the derivation.
    """
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"a must be a non-empty one-dimensional array, got {a.shape}")
    if mu < 0.0:
        raise ValueError(f"mu must be non-negative, got {mu}")
    return float(2.0 * np.min(np.hypot(a, mu)))


def pure_mutation_gap(mu: float) -> float:
    """Gap of the mutation operator alone, ``2 mu``. The value the gap saturates at once
    selection has lost, which is to say above the error threshold."""
    if mu < 0.0:
        raise ValueError(f"mu must be non-negative, got {mu}")
    return float(2.0 * mu)


def class_tridiagonal(
    f_by_class: NDArray[np.float64], mu: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Symmetrised Hamming-class reduction of the generator.

    Returns ``(diagonal, offdiagonal)`` of the ``(L+1)``-dimensional symmetric tridiagonal
    matrix representing the permutation-symmetric sector. Identical to the construction
    inside `quasarstack.analytic.crow_kimura.class_quasispecies`, exposed separately here
    because the gap needs the whole spectrum rather than only the top eigenvector.
    """
    f_by_class = np.asarray(f_by_class, dtype=np.float64)
    if f_by_class.ndim != 1 or f_by_class.size < 2:
        raise ValueError(
            f"f_by_class must be one-dimensional of length L+1, got {f_by_class.shape}"
        )
    if mu < 0.0:
        raise ValueError(f"mu must be non-negative, got {mu}")
    n_sites = f_by_class.size - 1
    d = np.arange(n_sites + 1, dtype=np.float64)
    diagonal = f_by_class - mu * n_sites
    offdiagonal = mu * np.sqrt((d[:-1] + 1.0) * (n_sites - d[:-1]))
    return diagonal, offdiagonal


def class_gap(f_by_class: NDArray[np.float64], mu: float) -> float:
    """Gap within the permutation-symmetric sector, in float64.

    Fast and adequate away from the error threshold. Near the threshold, and beyond
    L of roughly 64 anywhere, use `class_gap_extended`; the module docstring says why.
    """
    diagonal, offdiagonal = class_tridiagonal(f_by_class, mu)
    n = diagonal.size
    values = eigh_tridiagonal(diagonal, offdiagonal, select="i", select_range=(n - 2, n - 1))[0]
    return float(values[1] - values[0])


def dense_gap(fitness: NDArray[np.float64], mu: float, dense_limit: int = 12) -> float:
    """Gap of the full ``2**L`` generator, for landscapes with no symmetry to exploit.

    Guarded by ``dense_limit`` for the same reason `exact_diag.perron_vector` is: a dense
    solve at L = 14 wants 2 GB and the image pins BLAS to one thread.
    """
    fitness = np.asarray(fitness, dtype=np.float64)
    n_sites = int(round(float(np.log2(fitness.size))))
    if n_sites > dense_limit:
        raise ValueError(
            f"L = {n_sites} exceeds dense_limit = {dense_limit}; a dense gap at this size "
            f"needs {(1 << n_sites) ** 2 * 8 / 2**30:.1f} GiB. Use the class reduction if "
            f"the landscape is permutation symmetric, or raise dense_limit deliberately."
        )
    matrix = mutation_selection_generator(fitness, mu).toarray()
    values = np.linalg.eigvalsh(matrix)
    return float(values[-1] - values[-2])


def sparse_gap(fitness: NDArray[np.float64], mu: float, tol: float = 0.0) -> float:
    """Gap from the top two eigenvalues only, via Lanczos on the sparse generator.

    The generator has ``L + 1`` non-zeros per row, so asking for two extreme eigenvalues is
    far cheaper than a full decomposition: the image pins BLAS to one thread and a dense
    4096 by 4096 solve there takes minutes, which the gap map cannot afford thousands of.

    ``which="LA"`` rather than ``"LM"`` because the generator is indefinite and the largest
    *magnitude* eigenvalue is generally the most negative one, not the Perron eigenvalue.
    """
    from scipy.sparse.linalg import eigsh

    fitness = np.asarray(fitness, dtype=np.float64)
    operator = mutation_selection_generator(fitness, mu)
    if fitness.size <= 4:  # Lanczos needs k < n - 1; tiny cases go dense.
        values = np.linalg.eigvalsh(operator.toarray())
        return float(values[-1] - values[-2])
    values = eigsh(operator, k=2, which="LA", tol=tol, return_eigenvectors=False)
    return float(abs(values[1] - values[0]))


def spectral_gap(fitness: NDArray[np.float64], mu: float, sparse_above: int = 8) -> float:
    """Gap of the full generator, choosing dense or Lanczos by size.

    Dense below the crossover because it is exact and unconditionally convergent; Lanczos
    above it because dense stops being affordable. The crossover is a performance choice
    and the two agree to solver tolerance either side of it, which
    `tests/unit/test_gap.py` checks rather than assumes.
    """
    fitness = np.asarray(fitness, dtype=np.float64)
    n_sites = int(round(float(np.log2(fitness.size))))
    if n_sites <= sparse_above:
        return dense_gap(fitness, mu)
    return sparse_gap(fitness, mu)


def eigenvector_condition_number(gap: float) -> float:
    """How much a perturbation of the generator moves the Perron vector. WP1 task T1.3.

    First-order perturbation theory gives ``||delta v|| <= ||E|| / Delta`` for a symmetric
    operator, so ``1 / Delta`` is the condition number of the eigenvector problem. This is
    the quantity that degrades at the error threshold, and it degrades exponentially there.
    Returns infinity at zero gap rather than raising, because a closed gap is a physical
    statement about the cell and the caller should be able to record it.
    """
    return float("inf") if gap <= 0.0 else float(1.0 / gap)


def symmetric_sector_holds_lambda2(
    f_by_class: NDArray[np.float64], mu: float, dense_limit: int = 12
) -> dict[str, Any]:
    """Does the class reduction see the true second eigenvalue, or only the symmetric one?

    The Perron vector is always in the symmetric sector. ``lambda_2`` of the full generator
    need not be, and if it is not then the class reduction reports a gap larger than the one
    a general initial state actually experiences. Measured rather than assumed.
    """
    f_by_class = np.asarray(f_by_class, dtype=np.float64)
    n_sites = f_by_class.size - 1
    weights = np.bitwise_count(np.arange(1 << n_sites, dtype=np.uint64)).astype(np.int64)
    full = dense_gap(f_by_class[weights], mu, dense_limit=dense_limit)
    symmetric = class_gap(f_by_class, mu)
    return {
        "L": n_sites,
        "mu": float(mu),
        "gap_full": full,
        "gap_symmetric_sector": symmetric,
        "difference": abs(full - symmetric),
        # The symmetric sector is a subspace, so its gap can only be larger or equal.
        "lambda2_is_symmetric": bool(abs(full - symmetric) <= 1e-9 * max(1.0, abs(full))),
    }


def _sturm_count_below(diagonal: list, offdiagonal_squared: list, x, tiny) -> int:
    """Number of eigenvalues strictly below ``x``, by the Sturm sequence.

    For a symmetric tridiagonal the sequence ``d_1 = a_1 - x``,
    ``d_i = (a_i - x) - b_{i-1}^2 / d_{i-1}`` has exactly as many negative entries as there
    are eigenvalues below ``x``. Only ``b^2`` is needed, never ``b``, and the recurrence is
    ``O(L)``. A zero pivot is nudged rather than divided by; the standard trick, and the
    nudge is far below the working precision.
    """
    count = 0
    d = diagonal[0] - x
    if d < 0:
        count += 1
    for i in range(1, len(diagonal)):
        if d == 0:
            d = tiny
        d = (diagonal[i] - x) - offdiagonal_squared[i - 1] / d
        if d < 0:
            count += 1
    return count


def class_gap_extended(f_by_class, mu, dps: int = 60):
    """Gap within the symmetric sector at ``dps`` decimal digits, by Sturm bisection.

    Returns an ``mpmath.mpf``. Use where float64 cannot be trusted: near the error threshold,
    or beyond L of roughly 64 anywhere. ``mu`` and the class fitnesses are accepted as
    anything ``mpmath.mpf`` can take, including strings, which is the way to avoid
    contaminating a high-precision calculation with a float64 literal.
    """
    try:
        from mpmath import mp, mpf
    except ImportError as exc:  # pragma: no cover - the image ships mpmath via sympy
        raise ImportError(
            "class_gap_extended needs mpmath, which the pinned image provides through "
            "sympy. Install it, or use class_gap and accept the float64 resolution floor."
        ) from exc

    previous = mp.dps
    try:
        mp.dps = dps
        mu = mpf(mu)
        classes = [mpf(v) for v in f_by_class]
        n_sites = len(classes) - 1

        diagonal = [f - mu * n_sites for f in classes]
        off_squared = [mu**2 * mpf(d + 1) * (n_sites - d) for d in range(n_sites)]
        tiny = mpf(10) ** (-dps - 10)

        # Gershgorin: every eigenvalue lies within one row's radius of its diagonal entry.
        radii = [mp.sqrt(off_squared[0]) if n_sites else mpf(0)]
        for i in range(1, n_sites):
            radii.append(mp.sqrt(off_squared[i - 1]) + mp.sqrt(off_squared[i]))
        radii.append(mp.sqrt(off_squared[-1]) if n_sites else mpf(0))
        lo = min(diagonal[i] - radii[i] for i in range(n_sites + 1)) - 1
        hi = max(diagonal[i] + radii[i] for i in range(n_sites + 1)) + 1

        target = (hi - lo) * mpf(10) ** (-dps + 5)
        found = []
        for k in (n_sites, n_sites - 1):  # 0-indexed ascending: the top two
            left, right = lo, hi
            while right - left > target:
                middle = (left + right) / 2
                if _sturm_count_below(diagonal, off_squared, middle, tiny) <= k:
                    left = middle
                else:
                    right = middle
            found.append((left + right) / 2)
        return found[0] - found[1]
    finally:
        mp.dps = previous


def locate_gap_minimum(
    f_by_class_of,
    n_sites: int,
    mu_low,
    mu_high,
    dps: int = 60,
    iterations: int = 220,
) -> dict[str, Any]:
    """Find ``mu`` minimising the symmetric-sector gap, at arbitrary precision.

    Golden-section search, because the gap is unimodal in ``mu`` around the threshold but
    its minimum is an avoided crossing far too sharp for a float64 optimiser: at L = 32 a
    change in ``mu`` of one part in a million moves the measured minimum by ten percent.

    Parameters
    ----------
    f_by_class_of
        Callable taking ``n_sites`` and returning the length ``L+1`` class fitness array.
        A callable rather than an array so the caller cannot accidentally hold the
        landscape fixed while varying L.
    n_sites
        L.
    mu_low, mu_high
        Bracket. For a single peak of the given height the minimum sits near
        ``height / L``, so a bracket of ``(0.3 * height / L, 3 * height / L)`` is safe.
    iterations
        Golden-section steps. Each shrinks the bracket by a factor of 0.618, so 220 steps
        take a bracket of order ``1 / L`` down to about 1e-46.
    """
    from mpmath import mp, mpf

    previous = mp.dps
    try:
        mp.dps = dps
        classes = f_by_class_of(n_sites)
        invphi = (mp.sqrt(5) - 1) / 2

        a, b = mpf(mu_low), mpf(mu_high)
        c, d = b - invphi * (b - a), a + invphi * (b - a)
        fc = class_gap_extended(classes, c, dps=dps)
        fd = class_gap_extended(classes, d, dps=dps)
        for _ in range(iterations):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - invphi * (b - a)
                fc = class_gap_extended(classes, c, dps=dps)
            else:
                a, c, fc = c, d, fd
                d = a + invphi * (b - a)
                fd = class_gap_extended(classes, d, dps=dps)
        mu_star = (a + b) / 2
        gap = class_gap_extended(classes, mu_star, dps=dps)
        return {
            "L": n_sites,
            "mu_star": mp.nstr(mu_star, 20),
            "mu_star_times_L": mp.nstr(mu_star * n_sites, 20),
            "min_gap": mp.nstr(gap, 20),
            "min_gap_float": float(gap),
            "bracket_width": mp.nstr(b - a, 6),
            "dps": dps,
        }
    finally:
        mp.dps = previous
