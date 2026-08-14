"""Matrix-product imaginary-time evolution: Baseline C. WP6 tasks T6.1 and T6.2.

Evolves a tensor train under ``exp(tau W)`` with ``W = diag(f) + mu sum_i (X_i - I)``, the
mutation-selection generator, so the state converges to its Perron eigenvector, which is the
quasispecies. The two parts split cleanly and that is what makes the method cheap here:

- ``exp(dtau diag(f))`` is **diagonal**, so applying it is an elementwise product with the
  vector ``exp(dtau f)``. In tensor-train form that is a Hadamard product, which multiplies
  bond dimensions and is followed by a rounding step.
- ``exp(dtau mu (X_i - I))`` is a **product of single-site** two-by-two matrices, applied
  exactly with no bond growth at all.

Second-order Trotter: half a diagonal step, a full transverse step, half a diagonal step.

The rank that matters is the exponential's, not the Hamiltonian's
------------------------------------------------------------------

`mpo_analysis` measures the bond dimension of ``diag(f)``, which is the right quantity for
the cost of representing the operator. The cost of a *step* is set by the bond dimension of
``exp(dtau f)``, and the two are not the same function. The single peak is the clearest case:
``f`` is a delta and has rank 1 at every cut, while ``exp(dtau f) = 1 + (e^{dtau h} - 1) delta``
is a constant plus a delta and has rank 2. An additive ``f`` goes the other way, from rank 2
to rank 1, because the exponential of a sum of per-site terms is a product of per-site
factors.

Both are reported. Conflating them would misstate the per-step cost in either direction
depending on the family.

Overflow, which is not a detail at these sizes
-----------------------------------------------

``exp(dtau f)`` overflows for perfectly ordinary fitness once ``L`` is large, and imaginary
time multiplies the norm by a factor that is exponential in the number of sites per unit
time. Two things prevent that. The fitness is shifted by its maximum before exponentiating,
which is a uniform rescaling of the state and changes nothing physical. And `normalise`
spreads the normalisation factor across all cores rather than putting it in one, so no single
core carries a number the others do not.
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
    ``e^{-dtau mu} (cosh(dtau mu) I + sinh(dtau mu) X)``, written out rather than obtained
    from a matrix exponential so it stays exact and cheap inside the step loop.
    """
    theta = mu * dtau
    decay = np.exp(-theta)
    kernel = np.array([[np.cosh(theta), np.sinh(theta)], [np.sinh(theta), np.cosh(theta)]])
    return np.asarray(decay * kernel, dtype=np.float64)


def step_operator_bond_dimension(
    fitness: NDArray[np.float64], dtau: float, tolerance: float = 1e-12
) -> dict[str, int]:
    """Bond dimension of ``f`` and of ``exp(dtau f)``, which are different functions.

    The second is what sets the per-step cost. See the module docstring for why they differ
    and in which direction for which family.
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

    Returns the final state as a dense probability vector along with the truncation history
    that `G-6` criterion 4 requires: the discarded weight at **every** step, not only at the
    end, since a method that quietly throws away weight throughout and looks converged at the
    last step is exactly the failure this baseline could hide.

    Convergence is judged on the state, by the overlap between successive steps, rather than
    on the energy. Energy is flat near the minimum, so it stops moving before the state does
    and would report convergence early.

    **The test is on infidelity per unit imaginary time, not per step**, and the distinction
    is not cosmetic. Near the fixed point the state approaches it as ``exp(-gap tau)``, so the
    change over one step goes as ``dtau`` and the infidelity between successive steps goes as
    ``dtau**2``. A fixed per-step threshold therefore stops *earlier in tau* the smaller the
    step is, which is the opposite of what a step refinement is for.

    Measured before the fix, single peak at L = 8: ``dtau`` of 0.1, 0.05 and 0.02 stopped at
    tau of 22.5, 19.8 and 16.2 and gave total variation 1.1e-4, 3.1e-4 and 8.0e-4. The error
    rose as the step shrank, with truncation at 1e-29 so it was not truncation. That reads as
    a broken Trotter scheme and is nothing of the kind: the runs were simply less converged.
    Dividing by ``dtau**2`` makes the stopping point a property of the physics rather than of
    the discretisation. The project has made this mistake once before, on varQITE's tolerance,
    and the lesson is the same one: judge on a rate, and get the power right.

    After the fix the same sweep stops at tau of 26.9, 26.9, 26.9 and 26.8 for ``dtau`` of
    0.1, 0.05, 0.02 and 0.01, and the total variation settles to a floor of 5.1e-5 rather
    than growing. That floor is the remaining ``exp(-gap tau)`` contaminant at finite tau,
    not Trotter error, which is why it does not vanish as the step shrinks.
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
    # The train holds amplitudes; the biology wants a probability distribution, and the
    # Perron vector is sign definite so the modulus recovers it rather than destroying
    # information. See docs/theory.md section 8 on the amplitude-probability distinction.
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
