"""How hard is each landscape for a matrix-product operator? WP6 task T6.3.

`G-6` criterion 3 asks for the MPO bond dimension per family, for at least two site orderings
on the non-local families, and for the structural disadvantage to be **stated rather than
exploited** where MPS is at one. This module answers all three exactly, and it does so without
running any tensor-network evolution at all.

The observation that makes it cheap and exact
----------------------------------------------

The mutation-selection operator is `diag(f) + mu sum_i (X_i - I)`. The transverse part is a
sum of single-site terms, so its MPO bond dimension is 2 for any landscape and any ordering;
it contributes nothing to the difficulty. Everything is in `diag(f)`.

And for a diagonal operator the bond dimension across a cut has a closed form. Writing

    diag(f)  =  sum_x f(x) |x><x|

and splitting the sites into a left block `l` and a right block `r`, the MPO bond dimension
across that cut is exactly the **rank of the matrix `F[l, r] = f(l, r)`**, the fitness vector
reshaped into a `2^k` by `2^(L-k)` array. No SVD sweep over a tensor train is needed, only one
SVD per cut, and at `L = 14` the largest is 128 by 128.

So this is not an estimate of the bond dimension. It is the bond dimension, and it is an
exact lower bound on what any MPS carrying the operator must hold.

Why the site ordering is a real variable and not a tuning knob
---------------------------------------------------------------

An NK landscape with adjacent neighbourhoods is local on the chain, so a cut severs only the
`K` terms that straddle it and the rank stays small. The same landscape with a random site
ordering severs many more. Reporting only the good ordering would be exploiting the structural
advantage; reporting only the bad one would be manufacturing a disadvantage. Both are
measured, the better is used for the baseline as the gate directs, and the gap between them
is recorded because it is the honest size of the ordering effect.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "RANK_TOLERANCE",
    "compare_orderings",
    "mpo_bond_dimensions",
    "permute_sites",
]

# Singular values below this fraction of the largest are treated as zero. Loose enough that
# float noise does not inflate the rank, tight enough that a genuinely small but non-zero
# Schmidt coefficient still counts: at these sizes the two are twelve orders apart.
RANK_TOLERANCE = 1e-10


def permute_sites(fitness: NDArray[np.float64], order: list[int]) -> NDArray[np.float64]:
    """Relabel the sites, returning the fitness vector in the new site order.

    ``order[j]`` is the original site that becomes site ``j``. Little-endian throughout,
    matching `quasarstack.io.conventions`.
    """
    fitness = np.asarray(fitness, dtype=np.float64)
    n_sites = fitness.size.bit_length() - 1
    if sorted(order) != list(range(n_sites)):
        raise ValueError(f"order must be a permutation of 0..{n_sites - 1}, got {order}")

    index = np.arange(fitness.size, dtype=np.int64)
    source = np.zeros_like(index)
    for new_site, old_site in enumerate(order):
        source |= ((index >> new_site) & 1) << old_site
    return fitness[source]


def mpo_bond_dimensions(
    fitness: NDArray[np.float64], tolerance: float = RANK_TOLERANCE
) -> dict[str, Any]:
    """Exact MPO bond dimension of ``diag(f)`` at every cut of the chain.

    Returns the per-cut dimensions and the maximum, which is the one that sets the cost.
    The theoretical ceiling at cut ``k`` is ``2**min(k, L-k)``; the ratio to it is reported
    because a family sitting at the ceiling is one where MPS has nothing to exploit and the
    baseline should say so.
    """
    fitness = np.asarray(fitness, dtype=np.float64)
    size = fitness.size
    n_sites = size.bit_length() - 1
    if 1 << n_sites != size:
        raise ValueError(f"fitness length must be a power of two, got {size}")

    scale = float(np.abs(fitness).max())
    per_cut = []
    for cut in range(1, n_sites):
        # Little-endian: the low `cut` bits are the left block, so the reshape puts the
        # right block on the slow axis and the left block on the fast one.
        matrix = fitness.reshape(1 << (n_sites - cut), 1 << cut)
        singular = np.linalg.svd(matrix, compute_uv=False)
        rank = int(np.sum(singular > tolerance * max(scale, 1.0)))
        ceiling = 1 << min(cut, n_sites - cut)
        per_cut.append(
            {
                "cut": cut,
                "bond_dimension": rank,
                "ceiling": ceiling,
                "fraction_of_ceiling": rank / ceiling,
            }
        )

    largest = max(per_cut, key=lambda row: row["bond_dimension"]) if per_cut else None

    # The middle cut is the one that decides difficulty, because its ceiling is the largest.
    # Judging saturation on the maximum over cuts instead would call *every* family
    # saturated: cut 1 has a ceiling of 2, and even a purely additive landscape has rank 2
    # there. Additive sits at 2 out of 64 at the middle cut, which is the true picture.
    middle = per_cut[len(per_cut) // 2] if per_cut else None

    return {
        "L": n_sites,
        "max_bond_dimension": largest["bond_dimension"] if largest else 1,
        "middle_cut": middle["cut"] if middle else 0,
        "middle_cut_bond_dimension": middle["bond_dimension"] if middle else 1,
        "middle_cut_ceiling": middle["ceiling"] if middle else 1,
        "middle_cut_fraction_of_ceiling": middle["fraction_of_ceiling"] if middle else 0.0,
        "per_cut": per_cut,
        # At the ceiling there is no low-rank structure to find and no amount of chi helps.
        "saturates_the_ceiling": bool(middle and middle["fraction_of_ceiling"] >= 1.0),
    }


def compare_orderings(
    fitness: NDArray[np.float64],
    orderings: dict[str, list[int]],
    tolerance: float = RANK_TOLERANCE,
) -> dict[str, Any]:
    """Bond dimension under each site ordering, and how much the choice is worth.

    `G-6` criterion 3 requires at least two orderings on the non-local families and says the
    better one is used. This returns all of them, names the best, and reports the ratio
    between best and worst so the size of the effect is on the record rather than absorbed
    into a favourable default.
    """
    results = {
        name: mpo_bond_dimensions(permute_sites(fitness, order), tolerance)
        for name, order in orderings.items()
    }
    dimensions = {name: result["max_bond_dimension"] for name, result in results.items()}
    best = min(dimensions, key=lambda name: dimensions[name])
    worst = max(dimensions, key=lambda name: dimensions[name])

    return {
        "by_ordering": results,
        "max_bond_dimension_by_ordering": dimensions,
        "best_ordering": best,
        "worst_ordering": worst,
        "best_over_worst_ratio": dimensions[worst] / max(dimensions[best], 1),
        "ordering_matters": bool(dimensions[worst] > dimensions[best]),
    }
