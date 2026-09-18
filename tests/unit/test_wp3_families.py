"""Spin glass, house of cards, Rough Mount Fuji and block landscapes.

Each family is tested for where its optimum sits as well as for ruggedness.
"""

from __future__ import annotations

import numpy as np
import pytest

from quasarstack.classical.landscapes import (
    block_fitness,
    house_of_cards_fitness,
    rough_mount_fuji_fitness,
    ruggedness_statistics,
    spin_glass_fitness,
)
from quasarstack.hamiltonian.builder import diagonal_hamiltonian, pauli_term_count

pytestmark = pytest.mark.fast

BUILDERS = {
    "spin_glass": lambda n, s: spin_glass_fitness(n, s),
    "house_of_cards": lambda n, s: house_of_cards_fitness(n, s),
    "rough_mount_fuji": lambda n, s: rough_mount_fuji_fitness(n, s, 1.0, 0.5),
    "block": lambda n, s: block_fitness(n, 2, s),
}


@pytest.mark.parametrize("name", list(BUILDERS))
def test_family_reproduces_from_its_seed_despite_global_rng_traffic(name: str) -> None:
    build = BUILDERS[name]
    first = build(8, 3)
    np.random.seed(99)  # noqa: NPY002
    np.random.random(5000)  # noqa: NPY002
    assert first.tobytes() == build(8, 3).tobytes()


@pytest.mark.parametrize("name", list(BUILDERS))
def test_different_seeds_give_different_landscapes(name: str) -> None:
    build = BUILDERS[name]
    assert build(8, 0).tobytes() != build(8, 1).tobytes()


def test_spin_glass_pauli_count_is_quadratic_not_exponential() -> None:
    """The spin glass is rugged and cheap to compile.

    Its fitness has the ``b_ij Z_i Z_j`` form the compiler builds natively, so the expansion is
    ``L(L-1)/2`` weight-two terms plus ``L`` weight-one terms plus a constant, independent of the
    seed. House of cards is equally rugged and fully dense.

    """
    for n_sites in (4, 6, 8):
        expected = n_sites * (n_sites - 1) // 2 + n_sites + 1
        for seed in (0, 1, 2):
            operator = diagonal_hamiltonian(spin_glass_fitness(n_sites, seed), 0.2)
            # The mutation part adds L transverse terms on top of the diagonal expansion.
            assert pauli_term_count(operator) <= expected + n_sites


def test_house_of_cards_is_dense_in_pauli_space() -> None:
    """House of cards has no structure in Pauli space."""
    n_sites = 6
    operator = diagonal_hamiltonian(house_of_cards_fitness(n_sites, 0), 0.2)
    assert pauli_term_count(operator) >= (1 << n_sites)


@pytest.mark.parametrize("roughness", [0.0, 0.1, 0.3])
def test_rough_mount_fuji_keeps_the_master_sequence_while_it_is_barely_rugged(
    roughness: float,
) -> None:
    """Rough Mount Fuji keeps the master sequence only at low roughness.

    At L = 12 the optimum stays at genotype 0 in 97 to 100 percent of instances up to roughness
    0.3, 62 percent at 0.5 and 25 percent at 1.0, and retention worsens with L. At roughness 0.3
    the landscape has 1.4 local optima. The order parameter is therefore measured from each
    instance's own optimum.

    """
    weights = [
        ruggedness_statistics(rough_mount_fuji_fitness(8, seed, 1.0, roughness))[
            "optimum_hamming_weight"
        ]
        for seed in range(20)
    ]
    assert float(np.mean(weights)) < 0.2, weights


def test_rugged_families_move_their_optimum() -> None:
    """Every family with more than a few local optima has its optimum away from genotype 0."""
    from quasarstack.classical.landscapes import nk_fitness

    candidates = {
        "rmf_1.0": lambda s: rough_mount_fuji_fitness(10, s, 1.0, 1.0),
        "nk_k2": lambda s: nk_fitness(10, 2, s),
        "spin_glass": lambda s: spin_glass_fitness(10, s),
        "house_of_cards": lambda s: house_of_cards_fitness(10, s),
    }
    for name, build in candidates.items():
        stats = [ruggedness_statistics(build(s)) for s in range(10)]
        optima = float(np.mean([s["n_local_optima"] for s in stats]))
        anchored = float(np.mean([s["optimum_hamming_weight"] == 0 for s in stats]))
        assert optima > 3.0, f"{name} is not rugged enough to be an interesting case"
        assert anchored < 0.6, f"{name} unexpectedly keeps the master sequence: {anchored}"


def test_rough_mount_fuji_ruggedness_rises_with_roughness() -> None:
    """Ruggedness rises with roughness."""
    optima = [
        float(
            np.mean(
                [
                    ruggedness_statistics(rough_mount_fuji_fitness(8, seed, 1.0, r))[
                        "n_local_optima"
                    ]
                    for seed in range(10)
                ]
            )
        )
        for r in (0.0, 0.3, 1.0, 3.0)
    ]
    assert all(b >= a for a, b in zip(optima[:-1], optima[1:], strict=True)), optima
    assert optima[-1] > 10 * optima[0]


def test_nk_moves_its_optimum() -> None:
    """NK's optimum sits at a random genotype near Hamming weight L/2."""
    from quasarstack.classical.landscapes import nk_fitness

    weights = [
        ruggedness_statistics(nk_fitness(8, 2, seed))["optimum_hamming_weight"]
        for seed in range(10)
    ]
    assert float(np.mean(weights)) > 2.0, weights


def test_block_size_one_is_additive_and_block_size_l_is_house_of_cards() -> None:
    """The two ends of the block model are the two families it interpolates between."""
    assert ruggedness_statistics(block_fitness(8, 1, 0))["n_local_optima"] == 1
    full = ruggedness_statistics(block_fitness(8, 8, 0))["n_local_optima"]
    hoc = ruggedness_statistics(house_of_cards_fitness(8, 0))["n_local_optima"]
    assert abs(full - hoc) <= 0.5 * max(full, hoc)


@pytest.mark.parametrize("name", ["spin_glass", "house_of_cards", "block"])
def test_standardised_families_have_unit_spread(name: str) -> None:
    """Standardisation keeps a ruggedness sweep from also varying selection strength."""
    values = BUILDERS[name](8, 0)
    assert float(values.std()) == pytest.approx(1.0, abs=1e-12)
    assert float(values.mean()) == pytest.approx(0.0, abs=1e-12)


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ValueError, match="block_size"):
        block_fitness(8, 9, 0)
    with pytest.raises(ValueError, match="non-negative"):
        rough_mount_fuji_fitness(8, 0, -1.0, 0.5)
