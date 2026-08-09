"""MPO bond dimension: is the rank formula right, and does it say what it seems to say?

For a diagonal operator the bond dimension across a cut is exactly the rank of the fitness
vector matricised across that cut. These tests pin that against cases whose answer is known
independently, because the whole WP6 structural argument rests on it.
"""

from __future__ import annotations

import numpy as np
import pytest

from quasarstack.classical.landscapes import (
    additive_fitness,
    class_fitness,
    house_of_cards_fitness,
    nk_fitness,
    single_peak_classes,
    spin_glass_fitness,
)
from quasarstack.classical.mpo_analysis import (
    compare_orderings,
    mpo_bond_dimensions,
    permute_sites,
)

pytestmark = pytest.mark.fast


def test_a_single_peak_is_rank_one_everywhere() -> None:
    """f is a delta, so the matricised f is an outer product of two deltas at every cut.

    Worth pinning because it is the sharpest illustration of the WP6 point: the landscape
    that costs 4108 Pauli terms as a projector is the *cheapest possible* operator for a
    matrix-product representation.
    """
    report = mpo_bond_dimensions(class_fitness(single_peak_classes(10, 1.0)))
    assert report["max_bond_dimension"] == 1
    assert all(row["bond_dimension"] == 1 for row in report["per_cut"])


def test_an_additive_landscape_has_bond_dimension_two() -> None:
    """f = sum_i a_i z_i splits across any cut into (left sum) + (right sum), which is a
    sum of two separable terms, so the rank is 2 and cannot be 1 unless one side is empty."""
    a = np.random.default_rng(0).uniform(0.3, 1.5, size=10)
    report = mpo_bond_dimensions(additive_fitness(a))
    interior = [row for row in report["per_cut"] if 1 < row["cut"] < 9]
    assert all(row["bond_dimension"] == 2 for row in interior), report["per_cut"]


def test_house_of_cards_saturates_the_middle_cut() -> None:
    """Independent draws per genotype means the matricised fitness is a random matrix with
    no structure, so its rank is full and no bond dimension helps."""
    report = mpo_bond_dimensions(house_of_cards_fitness(10, seed=0))
    assert report["saturates_the_ceiling"]
    assert report["middle_cut_bond_dimension"] == report["middle_cut_ceiling"]


def test_saturation_is_judged_at_the_middle_cut_not_the_maximum_over_cuts() -> None:
    """A maximum over cuts calls everything saturated: cut 1 has a ceiling of 2 and even an
    additive landscape reaches it. Judging at the middle cut is what makes the flag mean
    'no exploitable structure' rather than 'has at least two distinct values'."""
    a = np.random.default_rng(0).uniform(0.3, 1.5, size=10)
    report = mpo_bond_dimensions(additive_fitness(a))
    assert not report["saturates_the_ceiling"]
    assert max(row["fraction_of_ceiling"] for row in report["per_cut"]) == 1.0
    assert report["middle_cut_fraction_of_ceiling"] < 0.1


def test_bond_dimension_grows_with_nk_connectivity() -> None:
    """K sets how far epistasis reaches along the chain, so it sets how many terms a cut
    severs, so it sets the rank. The ordering of these is the structural claim WP6 makes."""
    dimensions = [
        mpo_bond_dimensions(nk_fitness(10, k, seed=0))["middle_cut_bond_dimension"]
        for k in (0, 1, 2, 4)
    ]
    assert dimensions == sorted(dimensions), dimensions
    assert dimensions[0] < dimensions[-1]


def test_permutation_is_a_relabelling_and_nothing_else() -> None:
    """A site permutation must not change the multiset of fitness values, only where they
    sit. If it did, the ordering comparison would be comparing different landscapes."""
    fitness = nk_fitness(8, 2, seed=1)
    permuted = permute_sites(fitness, [3, 0, 7, 1, 6, 2, 5, 4])
    assert np.allclose(np.sort(fitness), np.sort(permuted))
    assert not np.allclose(fitness, permuted)


def test_permuting_back_recovers_the_original() -> None:
    fitness = nk_fitness(8, 2, seed=1)
    order = [3, 0, 7, 1, 6, 2, 5, 4]
    inverse = [order.index(site) for site in range(8)]
    assert np.array_equal(permute_sites(permute_sites(fitness, order), inverse), fitness)


def test_site_ordering_changes_the_bond_dimension_of_a_chain_local_landscape() -> None:
    """The measurement G-6 criterion 3 asks for. NK with adjacent neighbourhoods is local on
    the chain, so interleaving the sites severs more terms and costs bond dimension. The
    effect has to be real for 'the better ordering is used' to mean anything."""
    n_sites = 10
    result = compare_orderings(
        nk_fitness(n_sites, 2, seed=0),
        {
            "identity": list(range(n_sites)),
            "even_then_odd": list(range(0, n_sites, 2)) + list(range(1, n_sites, 2)),
        },
    )
    assert result["ordering_matters"]
    assert result["best_ordering"] == "identity"
    assert result["best_over_worst_ratio"] > 1.5


def test_an_invalid_ordering_is_rejected() -> None:
    with pytest.raises(ValueError, match="permutation"):
        permute_sites(np.zeros(16), [0, 1, 2, 2])


def test_spin_glass_stays_low_rank_despite_being_rugged() -> None:
    """The finding that matters for WP7: the spin glass is rugged, is cheap in Pauli space,
    and is *also* cheap for a matrix-product operator. A family that is easy for the tensor
    network is a poor place to look for quantum advantage."""
    report = mpo_bond_dimensions(spin_glass_fitness(10, seed=0))
    assert report["middle_cut_fraction_of_ceiling"] < 0.5
    assert not report["saturates_the_ceiling"]
