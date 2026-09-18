"""Baseline B: landscapes whose quasispecies is computable in polynomial time.

Given a fitness vector, `applicability` decides whether the quasispecies can be computed
without forming the ``2^L`` generator, and `solve` computes it. Two classes qualify:

- **Additive.** ``f = sum_i a_i z_i`` makes the generator a sum of commuting single-site terms,
  so the quasispecies is a product state with a per-site closed form. Cost ``O(L)``.
- **Permutation symmetric.** Fitness depending on Hamming weight alone reduces the problem to
  an ``(L+1)``-dimensional tridiagonal eigenproblem (Swetina and Schuster, 1982). Cost ``O(L)``
  after the reduction.

Dixit, Srivastava and Vishnoi (arXiv:1203.1287) cite Swetina and Schuster for the class
reduction; their own algorithm solves the steady state of a finite-population chain at a cost
not polynomial in L. The additive branch also covers distinct per-site coefficients, which lie
outside class invariance, so this class is a strict superset of the class-invariant one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from quasarstack.analytic.crow_kimura import additive_quasispecies, class_quasispecies

__all__ = [
    "ADDITIVE_TOLERANCE",
    "applicability",
    "coverage_map",
    "solve",
]

# Residual below which a landscape counts as additive or symmetric: well above float noise and
# well below any meaningful epistasis.
ADDITIVE_TOLERANCE = 1e-9


def _spin_design(n_sites: int) -> NDArray[np.float64]:
    index = np.arange(1 << n_sites)[:, None]
    return np.column_stack([np.ones(1 << n_sites), 1.0 - 2.0 * ((index >> np.arange(n_sites)) & 1)])


def applicability(
    fitness: NDArray[np.float64], tolerance: float = ADDITIVE_TOLERANCE
) -> dict[str, Any]:
    """Decide whether this landscape is in the polynomial-time class.

    Returns the decision, the class name and the residuals of both tests.
    """
    fitness = np.asarray(fitness, dtype=np.float64)
    size = fitness.size
    n_sites = size.bit_length() - 1
    if 1 << n_sites != size:
        raise ValueError(f"fitness length must be a power of two, got {size}")

    scale = max(float(np.abs(fitness).max()), 1.0)

    # Additive: fit the L + 1 spin coefficients and see what is left over.
    design = _spin_design(n_sites)
    coefficients, *_ = np.linalg.lstsq(design, fitness, rcond=None)
    additive_residual = float(np.max(np.abs(fitness - design @ coefficients))) / scale

    # Permutation symmetric: does fitness depend on Hamming weight alone?
    weights = np.bitwise_count(np.arange(size, dtype=np.uint64)).astype(np.int64)
    by_class = np.zeros(n_sites + 1)
    symmetric_residual = 0.0
    for weight in range(n_sites + 1):
        values = fitness[weights == weight]
        by_class[weight] = float(values.mean())
        symmetric_residual = max(
            symmetric_residual, float(np.max(np.abs(values - by_class[weight])))
        )
    symmetric_residual /= scale

    is_additive = additive_residual <= tolerance
    is_symmetric = symmetric_residual <= tolerance

    return {
        "L": n_sites,
        "applies": bool(is_additive or is_symmetric),
        "class": "additive" if is_additive else ("permutation_symmetric" if is_symmetric else None),
        "is_additive": bool(is_additive),
        "is_permutation_symmetric": bool(is_symmetric),
        "additive_residual": additive_residual,
        "symmetric_residual": symmetric_residual,
        "tolerance": tolerance,
        "cost": "O(L)" if (is_additive or is_symmetric) else "exponential, out of class",
        # Kept so `solve` does not repeat the fit, and so the record shows the fitted coefficients.
        "_coefficients": coefficients.tolist(),
        "_class_fitness": by_class.tolist(),
    }


def solve(
    fitness: NDArray[np.float64], mu: float, tolerance: float = ADDITIVE_TOLERANCE
) -> dict[str, Any]:
    """Compute the quasispecies in polynomial time, or refuse with a reason.

    Out-of-class instances are refused rather than solved by exact diagonalisation.
    """
    fitness = np.asarray(fitness, dtype=np.float64)
    verdict = applicability(fitness, tolerance)
    if not verdict["applies"]:
        raise ValueError(
            f"landscape is outside the polynomial-time class: additive residual "
            f"{verdict['additive_residual']:.3e} and symmetric residual "
            f"{verdict['symmetric_residual']:.3e} both exceed {tolerance:.1e}. This baseline "
            f"does not apply here and must not be reported as covering the cell."
        )

    if verdict["is_additive"]:
        coefficients = np.array(verdict["_coefficients"])
        # The constant term shifts every eigenvalue equally and leaves the eigenvector
        # untouched, so the distribution comes from the site coefficients alone.
        distribution = additive_quasispecies(coefficients[1:], mu)
        return {**verdict, "distribution": distribution, "route": "additive_product_state"}

    genotype_probs, class_probs, mean_fitness = class_quasispecies(
        np.array(verdict["_class_fitness"]), mu
    )
    return {
        **verdict,
        "distribution": genotype_probs,
        "class_probabilities": class_probs.tolist(),
        "mean_fitness": mean_fitness,
        "route": "hamming_class_reduction",
    }


def coverage_map(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Which grid cells this baseline covers.

    Each cell must carry a ``fitness`` array and the labels that identify it.
    """
    decided = []
    for cell in cells:
        verdict = applicability(np.asarray(cell["fitness"], dtype=np.float64))
        decided.append(
            {
                **{k: v for k, v in cell.items() if k != "fitness"},
                "covered": verdict["applies"],
                "class": verdict["class"],
                "additive_residual": verdict["additive_residual"],
                "symmetric_residual": verdict["symmetric_residual"],
            }
        )
    covered = sum(row["covered"] for row in decided)
    return {
        "n_cells": len(decided),
        "n_covered": covered,
        "fraction_covered": covered / len(decided) if decided else 0.0,
        "cells": decided,
    }
