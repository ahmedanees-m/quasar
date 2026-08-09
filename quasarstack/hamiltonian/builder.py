"""Biology to qubits: compiling a fitness landscape into a Pauli operator.

What is built
-------------

The mutation-selection generator is

    W = sum_i a_i Z_i + sum_{i<j} b_ij Z_i Z_j + mu sum_i (X_i - I)

whose Perron eigenvector is the quasispecies. Circuits find *ground* states, not top
eigenvectors, so what this module returns is the negated operator

    H = -W = -sum_i a_i Z_i - sum_{i<j} b_ij Z_i Z_j - mu sum_i X_i + mu L I

whose ground state is the same vector. H is stoquastic: every off-diagonal entry is
``-mu <= 0``, so by Perron-Frobenius its ground state is sign definite, which is what lets
L1 and L2 normalisation select the same ray. See `DECISIONS.md` ADR-0003. The identity term
is carried rather than dropped, so that the operator matches the generator entry for entry
and not merely up to a shift.

Two routes to the same operator
-------------------------------

**Structured.** ``additive_hamiltonian`` writes the diagonal directly as ``a_i Z_i`` and
``b_ij Z_i Z_j``. Sparse by construction: ``L + |b| + L + 1`` terms.

**Walsh-Hadamard.** ``diagonal_hamiltonian`` takes an arbitrary fitness vector and recovers
its exact Pauli decomposition, since any diagonal operator expands as

    f = sum_S c_S prod_{i in S} Z_i,    c_S = 2^-L sum_sigma f(sigma) prod_{i in S} z_i

and that inner sum is a Walsh-Hadamard transform. This route handles landscapes with no
structure to exploit, which is every family WP3 adds, and its term count is the honest cost
measure for that landscape. The single-peak projector is the extreme case: it needs all 2^L
terms, and that is the comparison gate G-R.10 measures.

For an additive landscape both routes must produce the same operator, and a test asserts it.

Endianness
----------

Qiskit Pauli strings are little-endian: the rightmost character is qubit 0. Building those
strings by hand is the classic silent error, so every term here is constructed through
``SparsePauliOp.from_sparse_list``, which takes qubit indices explicitly and never asks the
caller to reason about string position. Qubit i carries site i, matching
`quasarstack.io.conventions`, and a test pins that correspondence.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from qiskit.quantum_info import SparsePauliOp

from quasarstack.io.conventions import assert_dense_allowed

# Pauli coefficients below this are treated as absent. Set well under the 1e-9 gate
# tolerances so that dropping a term can never be what makes a gate pass.
PAULI_TOLERANCE = 1e-12


def _check_mu(mu: float) -> None:
    if not np.isfinite(mu) or mu < 0.0:
        raise ValueError(f"mu must be finite and non-negative, got {mu}")


def mutation_terms(n_sites: int, mu: float) -> list[tuple[str, list[int], complex]]:
    """The ``-mu sum_i X_i + mu L I`` part of H, as sparse-list entries."""
    _check_mu(mu)
    terms: list[tuple[str, list[int], complex]] = [
        ("X", [site], complex(-mu)) for site in range(n_sites)
    ]
    terms.append(("I", [0], complex(mu * n_sites)))
    return terms


def additive_hamiltonian(
    a: NDArray[np.float64], mu: float, b: NDArray[np.float64] | None = None
) -> SparsePauliOp:
    """Compile an additive (optionally epistatic) landscape into the stoquastic H.

    Parameters
    ----------
    a
        Length-L per-site coefficients, spin convention.
    b
        Optional ``(L, L)`` couplings; only the strict upper triangle is read.
    mu
        Per-site mutation rate.

    Returns
    -------
    SparsePauliOp
        ``H = -W``, whose ground state is the quasispecies.
    """
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 1 or a.size < 1:
        raise ValueError(f"a must be a non-empty one-dimensional array, got shape {a.shape}")
    n_sites = a.size
    _check_mu(mu)

    terms: list[tuple[str, list[int], complex]] = []
    for site in range(n_sites):
        if abs(a[site]) > PAULI_TOLERANCE:
            terms.append(("Z", [site], complex(-a[site])))

    if b is not None:
        b = np.asarray(b, dtype=np.float64)
        if b.shape != (n_sites, n_sites):
            raise ValueError(f"b must have shape ({n_sites}, {n_sites}), got {b.shape}")
        for i in range(n_sites):
            for j in range(i + 1, n_sites):
                if abs(b[i, j]) > PAULI_TOLERANCE:
                    terms.append(("ZZ", [i, j], complex(-b[i, j])))

    terms.extend(mutation_terms(n_sites, mu))
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n_sites).simplify()


def walsh_hadamard(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """In-place-style fast Walsh-Hadamard transform of a length-2^L vector.

    Returns the unnormalised transform, whose entry S is
    ``sum_sigma values[sigma] * (-1)**popcount(S & sigma)``. That sign is exactly
    ``prod_{i in S} z_i``, which is why the transform is the Pauli decomposition.
    """
    out = np.array(values, dtype=np.float64, copy=True)
    size = out.size
    if size < 2 or (size & (size - 1)) != 0:
        raise ValueError(f"length must be a power of two and at least 2, got {size}")
    step = 1
    while step < size:
        for start in range(0, size, step * 2):
            left = out[start : start + step].copy()
            right = out[start + step : start + 2 * step].copy()
            out[start : start + step] = left + right
            out[start + step : start + 2 * step] = left - right
        step *= 2
    return out


def diagonal_pauli_terms(
    fitness: NDArray[np.float64], tolerance: float = PAULI_TOLERANCE
) -> list[tuple[str, list[int], complex]]:
    """Exact Pauli decomposition of a diagonal operator, as sparse-list entries.

    The returned terms represent ``-diag(fitness)``, matching the sign convention of H.
    """
    fitness = np.asarray(fitness, dtype=np.float64)
    size = fitness.size
    if size < 2 or (size & (size - 1)) != 0:
        raise ValueError(f"fitness length must be a power of two and at least 2, got {size}")
    n_sites = size.bit_length() - 1

    coefficients = walsh_hadamard(fitness) / size
    terms: list[tuple[str, list[int], complex]] = []
    for subset in range(size):
        coefficient = coefficients[subset]
        if abs(coefficient) <= tolerance:
            continue
        qubits = [site for site in range(n_sites) if subset >> site & 1]
        if not qubits:
            terms.append(("I", [0], complex(-coefficient)))
        else:
            terms.append(("Z" * len(qubits), qubits, complex(-coefficient)))
    return terms


def diagonal_hamiltonian(fitness: NDArray[np.float64], mu: float) -> SparsePauliOp:
    """Compile an arbitrary fitness landscape into the stoquastic H.

    Works for any landscape, structured or not, at the cost of a term count that reflects
    how unstructured the landscape actually is.
    """
    fitness = np.asarray(fitness, dtype=np.float64)
    n_sites = fitness.size.bit_length() - 1
    _check_mu(mu)
    terms = diagonal_pauli_terms(fitness)
    terms.extend(mutation_terms(n_sites, mu))
    return SparsePauliOp.from_sparse_list(terms, num_qubits=n_sites).simplify()


def pauli_term_count(operator: SparsePauliOp) -> int:
    """Number of Pauli terms with a non-negligible coefficient.

    The honest cost metric for a landscape family, and the quantity gate G-R.10 compares
    between the sparse additive form and the single-peak projector.
    """
    return int(np.sum(np.abs(operator.coeffs) > PAULI_TOLERANCE))


def ground_state(
    operator: SparsePauliOp, dense_limit: int = 12
) -> tuple[NDArray[np.float64], float]:
    """Ground state of a stoquastic H, returned as an L1-normalised distribution.

    Parameters
    ----------
    operator
        The Hamiltonian, expected to be one of the stoquastic operators this module builds.
    dense_limit
        Largest qubit count solved densely; above it, sparse ``eigsh`` on the smallest
        algebraic eigenvalue.

    Returns
    -------
    probs
        Length ``2**L`` L1-normalised, non-negative distribution.
    energy
        The ground-state energy, which is minus the equilibrium mean fitness.
    """
    from scipy.linalg import eigh
    from scipy.sparse.linalg import eigsh

    n_sites = operator.num_qubits
    if n_sites <= dense_limit:
        assert_dense_allowed(n_sites, limit=dense_limit)
        matrix = np.asarray(operator.to_matrix(), dtype=np.complex128)
        if np.max(np.abs(matrix.imag)) > 1e-12:
            raise ValueError("the Hamiltonian is not real; the stoquastic construction is broken")
        eigenvalues, eigenvectors = eigh(matrix.real)
        vector = eigenvectors[:, 0]
        energy = float(eigenvalues[0])
    else:
        sparse = operator.to_matrix(sparse=True).real
        eigenvalues, eigenvectors = eigsh(sparse, k=1, which="SA")
        vector = eigenvectors[:, 0]
        energy = float(eigenvalues[0])

    probs = np.abs(np.asarray(vector, dtype=np.float64))
    total = float(probs.sum())
    if total <= 0.0:
        raise ValueError("the ground state came back with zero total weight")
    return probs / total, energy
