"""Tensor-train primitives for the matrix-product baseline. WP6.

Written in numpy rather than through quimb, for two reasons. The pinned image has quimb but
the authoring machine does not, so a quimb implementation could not be validated anywhere
before it ran on the compute VM, and this project has already lost one six-minute gate run to
an unverified assumption about what the image contains.
And the operator here is a diagonal function of the bits plus a transverse field, which needs
so little of a general tensor-network library that the dependency would carry more risk than
it removes.

Conventions
-----------

A vector of length ``2**L`` becomes ``L`` cores, core ``i`` of shape ``(r[i], 2, r[i+1])``
with ``r[0] = r[L] = 1``. Core ``i`` carries **site i**, matching the little-endian indexing
in `quasarstack.io.conventions`: site i is bit i is qubit i. That costs one transpose on the
way in and one on the way out, because ``numpy.reshape`` puts the *most* significant bit on
axis 0, and doing it here rather than at each call site is what stops an endianness bug from
appearing later as an unexplained factor.

What the two hard operations are for
------------------------------------

**Rounding** is the truncation that makes the method approximate, and it is where the whole
bond-dimension question lives. It is done by sweeping right-to-left to bring the train into
left-canonical form, then left-to-right truncating singular values. Canonicalising first is
not optional: truncating a non-canonical train discards singular values of the local core
rather than of the global state, which silently keeps the wrong directions.

**Hadamard product** is how a diagonal operator is applied. Multiplying two vectors
elementwise corresponds to taking the Kronecker product of the cores site by site, which
multiplies the bond dimensions. That is why the operator's own bond dimension matters: it is
the factor by which the state's bond dimension grows per step before rounding pulls it back.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "from_tt",
    "hadamard",
    "inner",
    "norm",
    "normalise",
    "to_tt",
    "tt_round",
    "apply_single_site",
    "max_bond",
]


def to_tt(
    vector: NDArray[np.float64], max_bond: int | None = None, tolerance: float = 1e-14
) -> tuple[list[NDArray[np.float64]], float]:
    """TT-SVD. Returns ``(cores, discarded_weight)``.

    ``discarded_weight`` is the total squared singular weight thrown away, relative to the
    squared norm, which is the quantity criterion 4 asks to be tracked at every step.
    """
    vector = np.asarray(vector, dtype=np.float64)
    size = vector.size
    n_sites = size.bit_length() - 1
    if 1 << n_sites != size:
        raise ValueError(f"length must be a power of two, got {size}")

    # reshape puts the most significant bit on axis 0; reverse so axis i is site i.
    tensor = np.transpose(vector.reshape((2,) * n_sites), axes=list(range(n_sites))[::-1])

    squared_norm = float(np.sum(tensor**2))
    cores: list[NDArray[np.float64]] = []
    discarded = 0.0
    left = 1
    remainder = tensor.reshape(1, -1)

    for _ in range(n_sites - 1):
        remainder = remainder.reshape(left * 2, -1)
        u, singular, vt = _svd(remainder)
        keep = _rank_to_keep(singular, max_bond, tolerance)
        discarded += float(np.sum(singular[keep:] ** 2))
        u, singular, vt = u[:, :keep], singular[:keep], vt[:keep]
        cores.append(u.reshape(left, 2, keep))
        remainder = singular[:, None] * vt
        left = keep

    cores.append(remainder.reshape(left, 2, 1))
    return cores, discarded / squared_norm if squared_norm > 0 else 0.0


def _svd(matrix: NDArray[np.float64]) -> tuple[NDArray, NDArray, NDArray]:
    """Thin SVD with a fallback driver, because the fast one fails on rank-deficient input.

    numpy calls LAPACK ``gesdd``, a divide-and-conquer routine that is the right default and
    that occasionally refuses to converge on matrices which are numerically rank deficient.
    G-6 hit exactly that. It rounds at bond dimension up to 128 while the states it evolves
    have true rank 2 to 16, so most of each matrix is floating-point noise; the gate died
    with ``SVD did not converge`` after running for the better part of a day and wrote no
    record at all. ``gesvd`` is the older Golub-Reinsch routine, slower and numerically more
    forgiving, and it is only ever reached on the rare failure.

    The finiteness check runs first so that a NaN arriving from somewhere upstream reports
    itself as that, rather than as a convergence failure two layers down. The two have very
    different causes and the LAPACK message does not distinguish them.
    """
    if not np.all(np.isfinite(matrix)):
        raise ValueError(
            f"SVD input is not finite: {int(np.sum(np.isnan(matrix)))} NaN and "
            f"{int(np.sum(np.isinf(matrix)))} infinite entries in a "
            f"{matrix.shape[0]}x{matrix.shape[1]} matrix. This is an upstream problem, "
            f"not a convergence failure."
        )
    try:
        return np.linalg.svd(matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd as scipy_svd

        left, singular, right = scipy_svd(matrix, full_matrices=False, lapack_driver="gesvd")
        return np.asarray(left), np.asarray(singular), np.asarray(right)


def _rank_to_keep(singular: NDArray[np.float64], max_bond: int | None, tolerance: float) -> int:
    """How many singular values survive: the tolerance cut, then the hard bond cap."""
    if singular.size == 0:
        return 1
    significant = int(np.sum(singular > tolerance * singular[0])) if singular[0] > 0 else 1
    keep = max(significant, 1)
    if max_bond is not None:
        keep = min(keep, max_bond)
    return keep


def from_tt(cores: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    """Contract back to a dense vector. For validation and for small-L comparisons only."""
    result = cores[0]
    for core in cores[1:]:
        result = np.tensordot(result, core, axes=([-1], [0]))
    n_sites = len(cores)
    tensor = result.reshape((2,) * n_sites)
    # Undo the transpose from to_tt, then flatten back to little-endian order.
    return np.transpose(tensor, axes=list(range(n_sites))[::-1]).reshape(-1)


def tt_round(
    cores: list[NDArray[np.float64]], max_bond: int, tolerance: float = 1e-14
) -> tuple[list[NDArray[np.float64]], float]:
    """Truncate to ``max_bond``, returning ``(cores, discarded_weight)``.

    Right-to-left QR to reach left-canonical form, then left-to-right SVD truncation. The
    canonicalisation is what makes the discarded singular values global rather than local.
    """
    cores = [core.copy() for core in cores]
    n_sites = len(cores)

    # Right-to-left: make every core except the first right-orthogonal.
    for site in range(n_sites - 1, 0, -1):
        left, physical, right = cores[site].shape
        matrix = cores[site].reshape(left, physical * right)
        q, r = np.linalg.qr(matrix.T)
        rank = q.shape[1]
        cores[site] = q.T.reshape(rank, physical, right)
        cores[site - 1] = np.tensordot(cores[site - 1], r.T, axes=([-1], [0]))

    squared_norm = float(np.sum(cores[0] ** 2))
    discarded = 0.0

    # Left-to-right: truncate.
    for site in range(n_sites - 1):
        left, physical, right = cores[site].shape
        matrix = cores[site].reshape(left * physical, right)
        u, singular, vt = _svd(matrix)
        keep = _rank_to_keep(singular, max_bond, tolerance)
        discarded += float(np.sum(singular[keep:] ** 2))
        u, singular, vt = u[:, :keep], singular[:keep], vt[:keep]
        cores[site] = u.reshape(left, physical, keep)
        cores[site + 1] = np.tensordot(singular[:, None] * vt, cores[site + 1], axes=([-1], [0]))

    return cores, discarded / squared_norm if squared_norm > 0 else 0.0


def hadamard(
    left: list[NDArray[np.float64]], right: list[NDArray[np.float64]]
) -> list[NDArray[np.float64]]:
    """Elementwise product of two trains, by Kronecker product of the cores site by site.

    Bond dimensions multiply, so this is always followed by `tt_round`. The growth factor is
    the *operator's* bond dimension, which is why `mpo_analysis` measures it.
    """
    if len(left) != len(right):
        raise ValueError(f"trains have different lengths, {len(left)} and {len(right)}")
    product = []
    for a, b in zip(left, right, strict=True):
        la, physical, ra = a.shape
        lb, _, rb = b.shape
        # einsum over the shared physical index, then fuse the two bond pairs.
        core = np.einsum("ipj,kpl->ikpjl", a, b).reshape(la * lb, physical, ra * rb)
        product.append(core)
    return product


def apply_single_site(
    cores: list[NDArray[np.float64]], matrices: list[NDArray[np.float64]]
) -> list[NDArray[np.float64]]:
    """Apply a 2 by 2 operator on each site. No bond growth, exact."""
    if len(cores) != len(matrices):
        raise ValueError("one matrix per site is required")
    return [
        np.einsum("pq,iqj->ipj", np.asarray(m, dtype=np.float64), core)
        for core, m in zip(cores, matrices, strict=True)
    ]


def inner(left: list[NDArray[np.float64]], right: list[NDArray[np.float64]]) -> float:
    """Overlap of two trains, contracted bond by bond rather than through dense vectors."""
    result = np.ones((1, 1))
    for a, b in zip(left, right, strict=True):
        result = np.einsum("ik,ipj,kpl->jl", result, a, b)
    return float(result.reshape(()))


def norm(cores: list[NDArray[np.float64]]) -> float:
    return float(np.sqrt(max(inner(cores, cores), 0.0)))


def normalise(cores: list[NDArray[np.float64]]) -> list[NDArray[np.float64]]:
    """Scale to unit L2 norm, spreading the factor over the cores.

    Spread rather than applied to one core because imaginary-time evolution multiplies the
    norm by a factor per step that is exponentially large or small in the number of sites,
    and concentrating it in a single core overflows long before the state itself does.
    """
    scale = norm(cores)
    if scale == 0.0:
        raise ValueError("cannot normalise a zero train")
    per_core = scale ** (1.0 / len(cores))
    return [core / per_core for core in cores]


def max_bond(cores: list[NDArray[np.float64]]) -> int:
    return max(core.shape[0] for core in cores[1:]) if len(cores) > 1 else 1
