"""NK landscapes and the ruggedness statistics that describe them.

Two things are being defended. That K varies ruggedness and nothing else, which is what the
standardisation is for and what ADR-0011 was written about. And that the statistics
reporting where the optimum sits actually work, since that report is now a requirement on
every ruggedness axis the project uses.
"""

from __future__ import annotations

import numpy as np
import pytest

from quasarstack.classical.landscapes import (
    additive_fitness,
    class_fitness,
    nk_fitness,
    ruggedness_statistics,
    single_peak_classes,
)
from quasarstack.hamiltonian.builder import diagonal_hamiltonian, pauli_term_count

pytestmark = pytest.mark.fast


def test_landscape_reproduces_exactly_from_its_seed() -> None:
    first = nk_fitness(8, 3, seed=7)
    second = nk_fitness(8, 3, seed=7)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, nk_fitness(8, 3, seed=8))


def test_standardisation_fixes_the_selection_strength() -> None:
    """K must vary ruggedness alone. Raw NK spread shrinks as 1/sqrt(L) and grows with K, so
    an unstandardised sweep would vary selection strength at the same time, and any result
    would be a mixture of the two. This is the ADR-0011 lesson applied in advance."""
    for n_sites in (6, 8):
        for k in (0, 2, 5):
            fitness = nk_fitness(n_sites, k, seed=1, amplitude=2.5)
            assert fitness.mean() == pytest.approx(0.0, abs=1e-12)
            assert fitness.std() == pytest.approx(2.5, rel=1e-12)


@pytest.mark.parametrize("n_sites", [4, 6, 8])
def test_k_zero_is_additive(n_sites: int) -> None:
    """With no epistatic partners the landscape is a sum of single-site terms, so its exact
    Pauli decomposition must have only weight-one support: L longitudinal terms, L transverse
    terms and an identity."""
    fitness = nk_fitness(n_sites, 0, seed=3)
    assert pauli_term_count(diagonal_hamiltonian(fitness, 0.2)) == 2 * n_sites + 1


def test_pauli_support_grows_with_connectivity() -> None:
    counts = [
        pauli_term_count(diagonal_hamiltonian(nk_fitness(8, k, seed=0), 0.2)) for k in (0, 2, 4, 7)
    ]
    assert counts == sorted(counts)
    # K = L - 1 saturates: every Z subset, plus one transverse term per site.
    assert counts[-1] == 2**8 + 8


def test_ruggedness_rises_with_connectivity() -> None:
    """WP3 task T3.3, checked on the seed mean because a single instance is noisy."""
    optima, autocorrelation = [], []
    for k in (0, 1, 2, 4, 7):
        stats = [ruggedness_statistics(nk_fitness(8, k, seed=s)) for s in range(10)]
        optima.append(np.mean([s["n_local_optima"] for s in stats]))
        autocorrelation.append(np.mean([s["autocorrelation"] for s in stats]))
    assert optima == sorted(optima), f"local optima should rise with K, got {optima}"
    assert autocorrelation == sorted(
        autocorrelation, reverse=True
    ), f"autocorrelation should fall with K, got {autocorrelation}"


def test_k_zero_has_a_single_local_optimum() -> None:
    """An additive landscape is single-peaked by construction: every mutation has a fixed
    sign of effect, so there is exactly one optimum."""
    for seed in range(5):
        assert ruggedness_statistics(nk_fitness(8, 0, seed=seed))["n_local_optima"] == 1


def test_nk_landscapes_have_no_master_sequence() -> None:
    """The ADR-0011 report, and the reason it matters here.

    An NK optimum sits at a random genotype, near Hamming weight L/2, not at all-wild-type.
    So this family has no master sequence, and statements about the error threshold, which is
    defined by delocalisation away from one, do not carry over unchanged.
    """
    weights = [
        ruggedness_statistics(nk_fitness(10, 4, seed=s))["optimum_hamming_weight"]
        for s in range(20)
    ]
    assert np.mean(weights) == pytest.approx(5.0, abs=1.5)
    assert sum(1 for w in weights if w == 0) <= 1, "the optimum should not sit on the master"


def test_statistics_on_a_landscape_whose_answer_is_known() -> None:
    """The sharp peak: exactly one local optimum, sitting on the master sequence."""
    stats = ruggedness_statistics(class_fitness(single_peak_classes(6, 1.0)))
    assert stats["optimum_index"] == 0
    assert stats["optimum_hamming_weight"] == 0


def test_additive_landscape_optimum_is_where_the_signs_say() -> None:
    a = np.array([1.0, -2.0, 0.5])
    stats = ruggedness_statistics(additive_fitness(a))
    assert stats["n_local_optima"] == 1
    assert stats["optimum_hamming_weight"] == 1  # only site 1 is fitter mutated


def test_connectivity_bounds_are_enforced() -> None:
    with pytest.raises(ValueError, match="k must be between"):
        nk_fitness(5, 5, seed=0)
    with pytest.raises(ValueError, match="k must be between"):
        nk_fitness(5, -1, seed=0)


def test_unknown_neighbourhood_is_rejected() -> None:
    with pytest.raises(ValueError, match="neighbourhood must be"):
        nk_fitness(5, 2, seed=0, neighbourhood="triangular")


def test_random_neighbourhoods_are_also_reproducible() -> None:
    first = nk_fitness(8, 3, seed=2, neighbourhood="random")
    second = nk_fitness(8, 3, seed=2, neighbourhood="random")
    assert np.array_equal(first, second)
    assert not np.array_equal(first, nk_fitness(8, 3, seed=2, neighbourhood="adjacent"))
