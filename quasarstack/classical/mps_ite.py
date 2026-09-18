"""Matrix-product imaginary-time evolution, used as a cross-check for Baseline C.

Baseline C is DMRG (`quasarstack.classical.mps_dmrg`). This module evolves a tensor train under
``exp(tau W)`` with ``W = diag(f) + mu sum_i (X_i - I)``, so the state converges to the Perron
eigenvector. The two parts split:

- ``exp(dtau diag(f))`` is diagonal, so applying it is a Hadamard product with the vector
  ``exp(dtau f)``, which multiplies bond dimensions and is followed by rounding.
- ``exp(dtau mu (X_i - I))`` is a product of single-site two-by-two matrices, applied exactly
  with no bond growth.

Second-order Trotter: half a diagonal step, a full transverse step, half a diagonal step.

Bond dimension of the step
--------------------------

`mpo_analysis` measures the bond dimension of ``diag(f)``, which sets the cost of representing
the operator. The cost of a step is set by the bond dimension of ``exp(dtau f)``, a different
function. For the single peak, ``f`` is a delta of rank 1 while ``exp(dtau f) = 1 +
(e^{dtau h} - 1) delta`` has rank 2; an additive ``f`` goes from rank 2 to rank 1, because the
exponential of a sum of per-site terms is a product of per-site factors. Both are reported.

Overflow
--------

``exp(dtau f)`` overflows for ordinary fitness at large ``L``, and imaginary time multiplies the
norm by a factor exponential in the number of sites. The fitness is shifted by its maximum
before exponentiating, a uniform rescaling, and `normalise` spreads the normalisation factor
across all cores.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quasarstack.classical.tensor_train import (
    apply_single_site,
    from_tt,
    hadamard,
    inner,
    max_bond,
    normalise,
    to_tt,
    tt_round,
)

__all__ = ["evolve", "step_operator_bond_dimension"]


def _transverse_matrix(mu: float, dtau: float) -> NDArray[np.float64]:
    """``exp(dtau mu (X - I))`` as a two-by-two matrix.

    ``X`` has eigenvalues +/-1, so the exponential is
    ``e^{-dtau mu} (cosh(dtau mu) I + sinh(dtau mu) X)``, written out exactly.
    """
    theta = mu * dtau
    decay = np.exp(-theta)
    kernel = np.array([[np.cosh(theta), np.sinh(theta)], [np.sinh(theta), np.cosh(theta)]])
    return np.asarray(decay * kernel, dtype=np.float64)


def step_operator_bond_dimension(
    fitness: NDArray[np.float64], dtau: float, tolerance: float = 1e-12
) -> dict[str, int]:
    """Bond dimension of ``f`` and of ``exp(dtau f)``.

    The second sets the per-step cost; see the module docstring.
    """
    fitness = np.asarray(fitness, dtype=np.float64)
    shifted = np.exp(dtau * (fitness - fitness.max()))
    return {
        "fitness_bond_dimension": max_bond(to_tt(fitness, tolerance=tolerance)[0]),
        "step_operator_bond_dimension": max_bond(to_tt(shifted, tolerance=tolerance)[0]),
    }


def evolve(
    fitness: NDArray[np.float64],
    mu: float,
    max_bond_dimension: int,
    dtau: float = 0.05,
    max_steps: int = 4000,
    convergence: float = 1e-9,
    initial: NDArray[np.float64] | None = None,
) -> dict[str, object]:
    """Imaginary-time evolution to the quasispecies, in tensor-train form.

    Returns the final state as a dense probability vector with the discarded weight at every step.

    Convergence is judged on the state, by the overlap between successive steps, rather than on the
    energy, which is flat near the minimum. The test is on infidelity per unit imaginary time:
    near the fixed point the state approaches it as ``exp(-gap tau)``, so the infidelity between
    successive steps goes as ``dtau**2``, and a fixed per-step threshold would stop earlier in tau
    for smaller steps. Dividing by ``dtau**2`` makes the stopping point independent of the step.
    With this criterion the single peak at L = 8 stops at tau of 26.9, 26.9, 26.9 and 26.8 for
    ``dtau`` of 0.1, 0.05, 0.02 and 0.01, with total variation settling at 5.1e-5, the remaining
    ``exp(-gap tau)`` component at finite tau.
    """
    fitness = np.asarray(fitness, dtype=np.float64)
    size = fitness.size
    n_sites = size.bit_length() - 1
    if 1 << n_sites != size:
        raise ValueError(f"fitness length must be a power of two, got {size}")
    if max_bond_dimension < 1:
        raise ValueError(f"max_bond_dimension must be at least 1, got {max_bond_dimension}")

    # Shift by the maximum so the exponential cannot overflow; a uniform shift of the
    # generator moves every eigenvalue equally and leaves the eigenvector untouched.
    half_step = np.exp(0.5 * dtau * (fitness - fitness.max()))
    diagonal_cores, _ = to_tt(half_step)
    transverse = [_transverse_matrix(mu, dtau)] * n_sites

    if initial is None:
        state = np.full(size, 1.0 / np.sqrt(size))
    else:
        state = np.asarray(initial, dtype=np.float64) / np.linalg.norm(initial)
    cores, _ = to_tt(state, max_bond=max_bond_dimension)
    cores = normalise(cores)

    truncation: list[float] = []
    overlaps: list[float] = []
    converged = False
    steps_taken = 0

    for step in range(max_steps):
        previous = cores
        discarded = 0.0

        cores, dropped = tt_round(hadamard(cores, diagonal_cores), max_bond_dimension)
        discarded += dropped
        cores = apply_single_site(cores, transverse)
        cores, dropped = tt_round(hadamard(cores, diagonal_cores), max_bond_dimension)
        discarded += dropped
        cores = normalise(cores)

        truncation.append(discarded)
        overlap = abs(inner(previous, cores)) / max(
            np.sqrt(abs(inner(previous, previous)) * abs(inner(cores, cores))), 1e-300
        )
        overlaps.append(overlap)
        steps_taken = step + 1
        # Per unit imaginary time, not per step: see the docstring.
        if (1.0 - overlap) / (dtau * dtau) < convergence:
            converged = True
            break

    amplitudes = from_tt(cores)
    # The train holds amplitudes; the Perron vector is sign definite, so the modulus gives the
    # probability distribution (docs/theory.md, section 8).
    probabilities = np.abs(amplitudes)
    total = probabilities.sum()
    if total <= 0.0:
        raise ValueError("evolution produced a zero state")

    return {
        "distribution": probabilities / total,
        "converged": converged,
        "steps": steps_taken,
        "tau": steps_taken * dtau,
        "max_bond_dimension": max_bond_dimension,
        "final_bond_dimension": max_bond(cores),
        "dtau": dtau,
        "total_discarded_weight": float(np.sum(truncation)),
        "max_discarded_weight_in_one_step": float(np.max(truncation)) if truncation else 0.0,
        "final_discarded_weight": truncation[-1] if truncation else 0.0,
        "truncation_history": truncation,
        "final_step_overlap": overlaps[-1] if overlaps else 0.0,
    }
