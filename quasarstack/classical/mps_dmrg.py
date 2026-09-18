"""Baseline C: the quasispecies by DMRG on matrix-product states.

The generator ``W = diag(f) + mu sum_i (X_i - I)`` is assembled as a matrix-product operator
and its dominant eigenvector is found with two-site DMRG from quimb.

The operator is exact. The diagonal part is compressed from the fitness table by TT-SVD, so
its bond dimension across each cut is the rank of the fitness values arranged as a matrix
across that cut. The transverse field has bond dimension two. The two are joined in block
form, so the bond dimension of the generator is the sum.

Site ``j`` of the chain carries locus ``L - 1 - j``. That is the axis order of
``fitness.reshape((2,) * L)``, so a contracted state is indexed like every other vector in
the package, with genotype ``k`` at position ``k``.

Landscapes that are unchanged when every locus is complemented, such as the spin glass without
a field, have a Perron vector in the symmetric sector and a partner in the antisymmetric
sector whose eigenvalue approaches it exponentially in ``L`` at low mutation rate. For these
``shift * P`` is added, with ``P = prod_i X_i``. ``P`` commutes with ``W``, so eigenvectors
are unchanged, the Perron eigenvalue rises by ``shift`` and the partner falls by ``shift``.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import quimb.tensor as qtn
from numpy.typing import NDArray

__all__ = ["dominant_state", "generator_mpo", "is_complement_symmetric"]

IDENTITY = np.eye(2)
PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]])

# Lanczos settings for each two-site update. With quimb's defaults (tolerance 1e-3, four Krylov
# vectors) the eigenvalue fluctuates near 1e-10 between sweeps, and the local solve does not
# converge on landscapes with many equal fitness values, where the top local eigenvalues cluster.
LOCAL_EIG_TOL = 1e-10
LOCAL_EIG_NCV = 10
INITIAL_BOND = 16


def is_complement_symmetric(fitness: NDArray[np.float64]) -> bool:
    """True if ``f(x) == f(not x)`` for every genotype.

    Complementing every bit of index ``k`` gives ``2**L - 1 - k``, which reverses the table.
    """
    fitness = np.asarray(fitness, dtype=np.float64)
    scale = max(float(np.max(np.abs(fitness))), 1.0)
    return bool(np.allclose(fitness, fitness[::-1], rtol=0.0, atol=1e-12 * scale))


def _fitness_cores(fitness: NDArray[np.float64], cutoff: float) -> list[NDArray[np.float64]]:
    """TT-SVD of the fitness table in C order. Core ``j`` has shape ``(r_j, 2, r_{j+1})``."""
    n_sites = fitness.size.bit_length() - 1
    cores: list[NDArray[np.float64]] = []
    left = 1
    remainder = fitness.reshape(1, -1)
    for _ in range(n_sites - 1):
        remainder = remainder.reshape(left * 2, -1)
        u, singular, vt = np.linalg.svd(remainder, full_matrices=False)
        keep = int(np.sum(singular > cutoff * singular[0])) if singular[0] > 0.0 else 1
        keep = max(keep, 1)
        cores.append(u[:, :keep].reshape(left, 2, keep))
        remainder = singular[:keep, None] * vt[:keep]
        left = keep
    cores.append(remainder.reshape(left, 2, 1))
    return cores


def _transverse_block(mu: float, first: bool, last: bool) -> NDArray[np.float64]:
    """``mu sum_i (X_i - I)`` in the standard two-state form, shape ``(l, r, 2, 2)``."""
    field = mu * (PAULI_X - IDENTITY)
    if first:
        block = np.zeros((1, 2, 2, 2))
        block[0, 0], block[0, 1] = IDENTITY, field
    elif last:
        block = np.zeros((2, 1, 2, 2))
        block[0, 0], block[1, 0] = field, IDENTITY
    else:
        block = np.zeros((2, 2, 2, 2))
        block[0, 0], block[0, 1], block[1, 1] = IDENTITY, field, IDENTITY
    return block


def generator_mpo(
    fitness: NDArray[np.float64], mu: float, shift: float = 0.0, cutoff: float = 1e-13
) -> tuple[qtn.MatrixProductOperator, int]:
    """The generator, plus ``shift * prod_i X_i`` if ``shift`` is non-zero, as an MPO.

    Returns the operator and the largest bond dimension of its diagonal part.
    """
    fitness = np.asarray(fitness, dtype=np.float64)
    size = fitness.size
    n_sites = size.bit_length() - 1
    if 1 << n_sites != size or n_sites < 2:
        raise ValueError(f"fitness length must be a power of two, at least 4, got {size}")

    cores = _fitness_cores(fitness, cutoff)
    arrays = []
    for site, core in enumerate(cores):
        first, last = site == 0, site == n_sites - 1
        blocks = [
            np.einsum("apb,pq->abpq", core, IDENTITY),
            _transverse_block(mu, first, last),
        ]
        if shift:
            parity = PAULI_X * (shift if first else 1.0)
            blocks.append(parity.reshape(1, 1, 2, 2))
        arrays.append(_join(blocks, first, last))
    operator_bond = max(core.shape[2] for core in cores[:-1])
    return qtn.MatrixProductOperator(arrays, shape="lrud"), operator_bond


def _join(blocks: list[NDArray[np.float64]], first: bool, last: bool) -> NDArray[np.float64]:
    """Direct sum of MPO site tensors ``(l, r, 2, 2)``, end sites with the open bond dropped."""
    if first:
        return np.asarray(np.concatenate(blocks, axis=1)[0], dtype=np.float64)
    if last:
        return np.asarray(np.concatenate(blocks, axis=0)[:, 0], dtype=np.float64)
    rows = sum(block.shape[0] for block in blocks)
    cols = sum(block.shape[1] for block in blocks)
    joined = np.zeros((rows, cols, 2, 2))
    row = col = 0
    for block in blocks:
        joined[row : row + block.shape[0], col : col + block.shape[1]] = block
        row += block.shape[0]
        col += block.shape[1]
    return joined


def dominant_state(
    fitness: NDArray[np.float64],
    mu: float,
    max_bond: int,
    *,
    budget: float | None = None,
    tol: float = 1e-12,
    cutoff: float = 1e-12,
    max_sweeps: int = 100,
    seed: int = 0,
) -> dict[str, Any]:
    """Dominant eigenvector of the generator by two-site DMRG.

    Sweeps start from a seeded random state of bond dimension ``min(INITIAL_BOND, max_bond)``,
    alternate in direction, and stop when the eigenvalue has changed by less than ``tol`` relative
    to its magnitude over two consecutive sweeps, after ``max_sweeps``, or once ``budget`` seconds
    have passed. The clock is read between sweeps. ``cutoff`` bounds the squared singular weight
    discarded at each update; ``max_bond`` caps the bond dimension.

    The initial state is random because two-site updates grow entanglement only between
    neighbouring sites: from a product state the sweep can settle on a stationary state of low bond
    dimension when loci interact at a distance.

    The distribution is the modulus of the eigenvector normalised to unit sum, the same convention
    as the exact reference.
    """
    if max_bond < 1:
        raise ValueError(f"max_bond must be at least 1, got {max_bond}")
    began = time.monotonic()
    fitness = np.asarray(fitness, dtype=np.float64)
    n_sites = fitness.size.bit_length() - 1

    symmetric = is_complement_symmetric(fitness)
    shift = max(float(np.ptp(fitness)), 1.0) if symmetric else 0.0
    operator, operator_bond = generator_mpo(fitness, mu, shift=shift)

    initial = qtn.MPS_rand_state(
        n_sites, bond_dim=min(INITIAL_BOND, max_bond), seed=seed, dtype="float64"
    )
    solver = qtn.DMRG2(operator, bond_dims=[max_bond], cutoffs=cutoff, which="LA", p0=initial)
    solver.opts["local_eig_tol"] = LOCAL_EIG_TOL
    solver.opts["local_eig_ncv"] = LOCAL_EIG_NCV

    energies: list[float] = []
    bonds: list[int] = []
    converged = False
    exhausted = False
    for sweep in range(max_sweeps):
        if budget is not None and sweep > 0 and time.monotonic() - began >= budget:
            exhausted = True
            break
        solver.solve(max_sweeps=1, sweep_sequence="RL"[sweep % 2], tol=0.0)
        energies.append(float(np.real(solver.energy)) - shift)
        bonds.append(int(solver.state.max_bond()))
        if len(energies) >= 3:
            scale = tol * max(abs(energies[-1]), 1.0)
            if max(abs(energies[-1] - energies[-2]), abs(energies[-2] - energies[-3])) <= scale:
                converged = True
                break

    amplitudes = np.asarray(solver.state.to_dense(), dtype=np.complex128).ravel()
    probabilities = np.abs(amplitudes)
    total = float(probabilities.sum())
    if total <= 0.0:
        raise ValueError("DMRG returned a zero state")

    return {
        "distribution": probabilities / total,
        "eigenvalue": energies[-1],
        "converged": converged,
        "budget_exhausted": exhausted,
        "sweeps": len(energies),
        "seconds": time.monotonic() - began,
        "max_bond": max_bond,
        "bond_reached": bonds[-1],
        "operator_bond_dimension": operator_bond,
        "complement_symmetric": symmetric,
        "energy_by_sweep": energies,
        "bond_by_sweep": bonds,
        "cutoff": cutoff,
    }
