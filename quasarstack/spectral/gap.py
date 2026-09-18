"""Spectral gap of the mutation-selection generator.

The gap ``Delta = lambda_1 - lambda_2`` governs how fast any eigenvector-extraction method
converges. Imaginary-time evolution suppresses the leading contaminant as ``exp(-Delta tau)``,
power iteration contracts by ``lambda_2 / lambda_1`` per step, and QSVT needs a polynomial whose
degree scales with ``1 / Delta``.

Closed forms
------------

**Additive landscapes.** The generator is a sum of commuting single-site terms with eigenvalues
``-mu +/- sqrt(a_i^2 + mu^2)``, so the spectrum is every sum of those choices. The second largest
eigenvalue flips the cheapest single site, giving

    Delta = 2 min_i sqrt(a_i^2 + mu^2)

independent of L, with lambda_2 L-fold degenerate.

**Above the error threshold** the generator is dominated by ``mu sum_i (X_i - I)``, with spectrum
``mu(L - 2k) - mu L`` and gap ``2 mu``, approached from below as L grows.

**The threshold** sits at ``mu L = height`` for the single peak, with a ``1 / L`` finite-size
correction, and the collapse across peak heights is exact.

Numerical points
----------------

**The class reduction covers one sector.** A permutation-symmetric landscape reduces to an
``(L+1)``-dimensional tridiagonal problem for the symmetric sector only. The Perron vector lies
there; ``lambda_2`` need not, so `symmetric_sector_holds_lambda2` checks it.

**The minimum at the threshold is an avoided crossing.** Near the threshold the gap behaves like
``sqrt(Delta_min^2 + c^2 (mu - mu*)^2)``. Rounding ``mu*`` in the sixth decimal changes the
minimum by ten percent at L = 32, and beyond L of about 64 the minimum is below float64
resolution. `class_gap_extended` and `locate_gap_minimum` work at arbitrary precision through
Sturm bisection, which needs ``O(L)`` arithmetic per evaluation.
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

    Exact and independent of L.
    """
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"a must be a non-empty one-dimensional array, got {a.shape}")
    if mu < 0.0:
        raise ValueError(f"mu must be non-negative, got {mu}")
    return float(2.0 * np.min(np.hypot(a, mu)))


def pure_mutation_gap(mu: float) -> float:
    """Gap of the mutation operator alone, ``2 mu``, the value above the error threshold."""
    if mu < 0.0:
        raise ValueError(f"mu must be non-negative, got {mu}")
    return float(2.0 * mu)


def class_tridiagonal(
    f_by_class: NDArray[np.float64], mu: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Symmetrised Hamming-class reduction of the generator.

    Returns ``(diagonal, offdiagonal)`` of the ``(L+1)``-dimensional symmetric tridiagonal matrix for
    the permutation-symmetric sector, the same construction as in
    `quasarstack.analytic.crow_kimura.class_quasispecies`.
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

    Adequate away from the error threshold. Near the threshold, or beyond L of about 64, use
    `class_gap_extended`.
    """
    diagonal, offdiagonal = class_tridiagonal(f_by_class, mu)
    n = diagonal.size
    values = eigh_tridiagonal(diagonal, offdiagonal, select="i", select_range=(n - 2, n - 1))[0]
    return float(values[1] - values[0])


def dense_gap(fitness: NDArray[np.float64], mu: float, dense_limit: int = 12) -> float:
    """Gap of the full ``2**L`` generator, for landscapes without symmetry.

    Limited by ``dense_limit``: a dense solve at L = 14 needs 2 GB.
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
    """Gap from the top two eigenvalues, via Lanczos on the sparse generator.

    The generator has ``L + 1`` non-zeros per row, so two extreme eigenvalues cost far less than a
    full decomposition; with single-threaded BLAS a dense 4096 by 4096 solve takes minutes.

    ``which="LA"`` because the generator is indefinite and its largest-magnitude eigenvalue is
    generally the most negative one.
    """
    from scipy.sparse.linalg import eigsh

    from quasarstack.numerics import deterministic_start

    fitness = np.asarray(fitness, dtype=np.float64)
    operator = mutation_selection_generator(fitness, mu)
    if fitness.size <= 4:  # Lanczos needs k < n - 1; tiny cases go dense.
        values = np.linalg.eigvalsh(operator.toarray())
        return float(values[-1] - values[-2])
    values = eigsh(
        operator,
        k=2,
        which="LA",
        tol=tol,
        return_eigenvectors=False,
        v0=deterministic_start(operator.shape[0]),
    )
    return float(abs(values[1] - values[0]))


def spectral_gap(fitness: NDArray[np.float64], mu: float, sparse_above: int = 8) -> float:
    """Gap of the full generator, dense or Lanczos by size.

    Dense below the crossover, Lanczos above it; the two agree to solver tolerance on either side.
    """
    fitness = np.asarray(fitness, dtype=np.float64)
    n_sites = int(round(float(np.log2(fitness.size))))
    if n_sites <= sparse_above:
        return dense_gap(fitness, mu)
    return sparse_gap(fitness, mu)


def eigenvector_condition_number(gap: float) -> float:
    """Sensitivity of the Perron vector to perturbations of the generator.

    First-order perturbation theory gives ``||delta v|| <= ||E|| / Delta`` for a symmetric operator, so
    ``1 / Delta`` is the condition number of the eigenvector problem. Returns infinity at zero gap.
    """
    return float("inf") if gap <= 0.0 else float(1.0 / gap)


def symmetric_sector_holds_lambda2(
    f_by_class: NDArray[np.float64], mu: float, dense_limit: int = 12
) -> dict[str, Any]:
    """Whether the class reduction contains the true second eigenvalue.

    The Perron vector is always in the symmetric sector; ``lambda_2`` of the full generator need not
    be, in which case the class reduction overstates the gap seen by a general initial state.
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


def _sturm_count_below(
    diagonal: list[Any], offdiagonal_squared: list[Any], x: Any, tiny: Any
) -> int:
    """Number of eigenvalues strictly below ``x``, by the Sturm sequence.

    For a symmetric tridiagonal the sequence ``d_1 = a_1 - x``,
    ``d_i = (a_i - x) - b_{i-1}^2 / d_{i-1}`` has as many negative entries as there are eigenvalues
    below ``x``. Only ``b^2`` is needed and the recurrence is ``O(L)``. A zero pivot is replaced by a
    value far below working precision.
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


def class_gap_extended(f_by_class: Any, mu: Any, dps: int = 60) -> Any:
    """Gap within the symmetric sector at ``dps`` decimal digits, by Sturm bisection.

    Returns a ``decimal.Decimal``, for use near the error threshold or beyond L of about 64. ``mu`` and
    the class fitnesses are accepted as anything ``Decimal`` takes; pass strings, since
    ``Decimal(0.1)`` captures the float64 approximation to 0.1.

    Uses the standard library ``decimal`` module, so no extended-precision dependency is needed; the
    recurrence uses only arithmetic and comparison, with one square root for the Gershgorin bound.
    """
    from decimal import Decimal, localcontext

    with localcontext() as context:
        # A margin over the requested digits so the working precision does not itself
        # become the error floor during the bisection.
        context.prec = dps + 10

        mu = Decimal(mu)
        classes = [Decimal(v) for v in f_by_class]
        n_sites = len(classes) - 1

        diagonal = [f - mu * n_sites for f in classes]
        off_squared = [mu**2 * Decimal(d + 1) * Decimal(n_sites - d) for d in range(n_sites)]
        tiny = Decimal(10) ** (-dps - 10)

        # Gershgorin: every eigenvalue lies within one row's radius of its diagonal entry.
        radii = [off_squared[0].sqrt() if n_sites else Decimal(0)]
        for i in range(1, n_sites):
            radii.append(off_squared[i - 1].sqrt() + off_squared[i].sqrt())
        radii.append(off_squared[-1].sqrt() if n_sites else Decimal(0))
        lo = min(diagonal[i] - radii[i] for i in range(n_sites + 1)) - 1
        hi = max(diagonal[i] + radii[i] for i in range(n_sites + 1)) + 1

        target = (hi - lo) * Decimal(10) ** (-dps + 5)
        two = Decimal(2)
        found = []
        for k in (n_sites, n_sites - 1):  # 0-indexed ascending: the top two
            left, right = lo, hi
            while right - left > target:
                middle = (left + right) / two
                if _sturm_count_below(diagonal, off_squared, middle, tiny) <= k:
                    left = middle
                else:
                    right = middle
            found.append((left + right) / two)
        return found[0] - found[1]


def locate_gap_minimum(
    f_by_class_of: Any,
    n_sites: int,
    mu_low: Any,
    mu_high: Any,
    dps: int = 60,
    iterations: int = 220,
) -> dict[str, Any]:
    """Find ``mu`` minimising the symmetric-sector gap, at arbitrary precision.

    Golden-section search: the gap is unimodal in ``mu`` around the threshold, but its minimum is an
    avoided crossing too sharp for a float64 optimiser; at L = 32 a change in ``mu`` of one part in a
    million moves the minimum by ten percent.

    Parameters
    ----------
    f_by_class_of
        Callable taking ``n_sites`` and returning the length ``L+1`` class fitness array.
    n_sites
        L.
    mu_low, mu_high
        Bracket. For a single peak of the given height the minimum sits near ``height / L``, so
        ``(0.3 * height / L, 3 * height / L)`` is safe.
    iterations
        Golden-section steps. Each shrinks the bracket by a factor of 0.618, so 220 steps take a
        bracket of order ``1 / L`` to about 1e-46.
    """
    from decimal import Decimal, localcontext

    with localcontext() as context:
        context.prec = dps + 10
        classes = f_by_class_of(n_sites)
        two = Decimal(2)
        invphi = (Decimal(5).sqrt() - 1) / two

        a, b = Decimal(mu_low), Decimal(mu_high)
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
        mu_star = (a + b) / two
        gap = class_gap_extended(classes, mu_star, dps=dps)
        return {
            "L": n_sites,
            "mu_star": f"{mu_star:.20e}",
            "mu_star_times_L": f"{mu_star * n_sites:.20e}",
            "min_gap": f"{gap:.20e}",
            "min_gap_float": float(gap),
            "bracket_width": f"{b - a:.6e}",
            "dps": dps,
        }
