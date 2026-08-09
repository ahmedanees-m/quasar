"""Brute-force reference: build the full generator and diagonalise it.

This is the independent construction that the closed forms in
`quasarstack.analytic.crow_kimura` are checked against. It knows nothing about Hamming
classes, product states, or permutation symmetry: it assembles

    W = diag(f) + mu * sum_i (X_i - I)

as a ``2**L`` by ``2**L`` sparse matrix from an arbitrary fitness vector and takes its
Perron eigenvector. Being slow and structure-blind is the whole value of it.

W is symmetric, because a point mutation is as likely to restore a site as to break it in
this model, so the eigenproblem is a symmetric one and the Perron vector is sign definite.

Solver policy, from `GATES.md` section 1 and `DECISIONS.md` ADR-0004: dense below
L = 12 where it is both fast and maximally accurate, sparse ``eigsh`` at and above it.
Dense is forbidden outright above L = 12, where the matrix would need 2.1 GB and then 34 GB.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from quasarstack.io.conventions import assert_dense_allowed

DENSE_LIMIT = 12


def infer_n_sites(fitness: NDArray[np.float64]) -> int:
    """Recover L from the length of a fitness vector, rejecting non-power-of-two lengths."""
    size = int(np.asarray(fitness).size)
    if size < 2 or (size & (size - 1)) != 0:
        raise ValueError(f"fitness vector length must be a power of two and at least 2, got {size}")
    return size.bit_length() - 1


def mutation_selection_generator(fitness: NDArray[np.float64], mu: float) -> sp.csr_matrix:
    """Assemble the Crow-Kimura generator as a sparse matrix.

    Parameters
    ----------
    fitness
        Length ``2**L`` Malthusian fitness vector, indexed per
        `quasarstack.io.conventions`.
    mu
        Per-site mutation rate, non-negative.

    Returns
    -------
    scipy.sparse.csr_matrix
        The ``2**L`` by ``2**L`` generator, with ``L + 1`` non-zeros per row: the diagonal
        ``f(sigma) - mu L`` and one entry ``mu`` for each single-site flip.
    """
    fitness = np.asarray(fitness, dtype=np.float64)
    n_sites = infer_n_sites(fitness)
    if not np.isfinite(mu) or mu < 0.0:
        raise ValueError(f"mu must be finite and non-negative, got {mu}")

    dim = 1 << n_sites
    index = np.arange(dim, dtype=np.int64)

    rows = [index]
    cols = [index]
    data = [fitness - mu * n_sites]
    for site in range(n_sites):
        rows.append(index)
        cols.append(index ^ (1 << site))
        data.append(np.full(dim, mu, dtype=np.float64))

    return sp.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dim, dim),
    ).tocsr()


def perron_vector(
    fitness: NDArray[np.float64], mu: float, dense_limit: int = DENSE_LIMIT
) -> tuple[NDArray[np.float64], float, float]:
    """Perron eigenvector of the generator, by brute force.

    Parameters
    ----------
    fitness
        Length ``2**L`` fitness vector.
    mu
        Per-site mutation rate.
    dense_limit
        Largest L solved densely. Above it, sparse ``eigsh`` is used.

    Returns
    -------
    probs
        Length ``2**L`` L1-normalised, non-negative quasispecies distribution.
    mean_fitness
        The Perron eigenvalue, which is the equilibrium mean fitness.
    gap
        ``lambda_1 - lambda_2``, the spectral gap. Reported rather than discarded because
        the Perron eigenvector is only well conditioned while this is comfortably non-zero,
        and mapping how it closes near the error threshold is the substance of WP1.
    """
    fitness = np.asarray(fitness, dtype=np.float64)
    n_sites = infer_n_sites(fitness)
    generator = mutation_selection_generator(fitness, mu)

    if n_sites <= dense_limit:
        assert_dense_allowed(n_sites, limit=DENSE_LIMIT)
        eigenvalues, eigenvectors = eigh(generator.toarray())
        top = eigenvectors[:, -1]
        mean_fitness = float(eigenvalues[-1])
        gap = float(eigenvalues[-1] - eigenvalues[-2])
    else:
        eigenvalues, eigenvectors = eigsh(generator, k=2, which="LA")
        order = np.argsort(eigenvalues)[::-1]
        top = eigenvectors[:, order[0]]
        mean_fitness = float(eigenvalues[order[0]])
        gap = float(eigenvalues[order[0]] - eigenvalues[order[1]])

    probs = np.abs(top)
    total = probs.sum()
    if total <= 0.0:
        raise ValueError("the Perron vector came back with zero total weight")
    return probs / total, mean_fitness, gap
