"""Bond dimension of the mutation-selection operator as a matrix-product operator.

The operator is ``diag(f) + mu sum_i (X_i - I)``. The transverse part is a sum of single-site
terms with MPO bond dimension 2 for any landscape and ordering, so the bond dimension is set by
``diag(f)``.

For a diagonal operator the bond dimension across a cut has a closed form. Writing
``diag(f) = sum_x f(x) |x><x|`` and splitting the sites into a left block ``l`` and a right block
``r``, the bond dimension across the cut is the rank of ``F[l, r] = f(l, r)``, the fitness vector
reshaped into a ``2^k`` by ``2^(L-k)`` array. One SVD per cut suffices; at ``L = 14`` the largest
is 128 by 128. This is exact, and it is a lower bound on the bond dimension of any MPO carrying
the operator.

The site ordering matters for non-local families. An NK landscape with adjacent neighbourhoods
is local on the chain, so a cut severs only the ``K`` terms that straddle it; a random ordering
severs more. Both orderings are measured and their ratio is reported.
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

# Singular values below this fraction of the largest are treated as zero, well above float noise.
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

    Returns the per-cut dimensions and their maximum. The ceiling at cut ``k`` is
    ``2**min(k, L-k)``, and the ratio to it is reported.
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

    # Saturation is judged at the middle cut, which has the largest ceiling; at cut 1 even an
    # additive landscape reaches its ceiling of 2.
    middle = per_cut[len(per_cut) // 2] if per_cut else None

    return {
        "L": n_sites,
        "max_bond_dimension": largest["bond_dimension"] if largest else 1,
        "middle_cut": middle["cut"] if middle else 0,
        "middle_cut_bond_dimension": middle["bond_dimension"] if middle else 1,
        "middle_cut_ceiling": middle["ceiling"] if middle else 1,
        "middle_cut_fraction_of_ceiling": middle["fraction_of_ceiling"] if middle else 0.0,
        "per_cut": per_cut,
        # At the ceiling there is no low-rank structure to exploit.
        "saturates_the_ceiling": bool(middle and middle["fraction_of_ceiling"] >= 1.0),
    }


def compare_orderings(
    fitness: NDArray[np.float64],
    orderings: dict[str, list[int]],
    tolerance: float = RANK_TOLERANCE,
) -> dict[str, Any]:
    """Bond dimension under each site ordering, and the ratio between best and worst."""
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
