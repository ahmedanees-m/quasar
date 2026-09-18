"""The Wright-Fisher baseline in genotype-count space.

The count-space form relies on individuals being exchangeable. It is checked against an
individual-based reference implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from quasarstack.analytic.crow_kimura import single_peak_quasispecies
from quasarstack.classical.wright_fisher import (
    mutation_step,
    sample_stationary,
    selection_step,
    simulate,
)

pytestmark = pytest.mark.fast


def individual_based_mutation(
    counts: np.ndarray, n_sites: int, probability: float, rng: np.random.Generator
) -> np.ndarray:
    """The O(N L) individual-based implementation, used as a reference."""
    individuals = np.repeat(np.arange(counts.size), counts)
    flips = rng.random((individuals.size, n_sites)) < probability
    for site in range(n_sites):
        individuals = individuals ^ (flips[:, site].astype(np.int64) << site)
    return np.bincount(individuals, minlength=counts.size)


def test_count_space_mutation_has_the_same_law_as_the_individual_based_one() -> None:
    """Same distribution, not the same draw. Compared through means over many repeats, with
    a tolerance set by the sampling error.
    """
    n_sites, population, probability, repeats = 4, 4000, 0.08, 400
    start = np.zeros(1 << n_sites, dtype=np.int64)
    start[0] = population

    fast = np.zeros(1 << n_sites)
    slow = np.zeros(1 << n_sites)
    rng_fast = np.random.default_rng(0)
    rng_slow = np.random.default_rng(1)
    for _ in range(repeats):
        fast += mutation_step(start, n_sites, probability, rng_fast)
        slow += individual_based_mutation(start, n_sites, probability, rng_slow)
    fast /= fast.sum()
    slow /= slow.sum()

    assert 0.5 * float(np.abs(fast - slow).sum()) < 0.01


def test_mutation_conserves_the_population() -> None:
    rng = np.random.default_rng(0)
    counts = rng.multinomial(10_000, np.full(16, 1 / 16))
    assert int(mutation_step(counts, 4, 0.1, rng).sum()) == 10_000


def test_selection_conserves_the_population_and_favours_fitness() -> None:
    rng = np.random.default_rng(0)
    counts = np.array([500, 500], dtype=np.int64)
    weights = np.array([1.5, 1.0])
    after = selection_step(counts, weights, rng)
    assert int(after.sum()) == 1000
    assert after[0] > after[1]


def test_mutation_with_probability_one_half_reaches_the_uniform_distribution() -> None:
    """A limit with a known answer: at u = 1/2 every site is randomised, so one step from
    any start gives the uniform distribution over genotypes."""
    n_sites = 5
    counts = np.zeros(1 << n_sites, dtype=np.int64)
    counts[7] = 200_000
    after = mutation_step(counts, n_sites, 0.5, np.random.default_rng(0))
    fractions = after / after.sum()
    assert np.max(np.abs(fractions - 1 / (1 << n_sites))) < 0.002


def test_zero_mutation_leaves_the_population_where_it_is() -> None:
    counts = np.zeros(8, dtype=np.int64)
    counts[3] = 1000
    assert np.array_equal(mutation_step(counts, 3, 0.0, np.random.default_rng(0)), counts)


def test_simulation_reproduces_from_its_seed() -> None:
    fitness = np.zeros(64)
    fitness[0] = 1.0
    first = simulate(fitness, 0.1, 10_000, 300, seed=5)["distribution"]
    np.random.seed(1234)  # noqa: NPY002
    np.random.random(1000)  # noqa: NPY002
    second = simulate(fitness, 0.1, 10_000, 300, seed=5)["distribution"]
    assert np.array_equal(first, second)


@pytest.mark.slow
def test_converges_toward_the_analytic_quasispecies_as_population_grows() -> None:
    """Convergence toward the analytic quasispecies at small N. G-4 tests N = 1e6."""
    n_sites, mu = 6, 0.12
    reference, _, _ = single_peak_quasispecies(n_sites, 1.0, mu)
    fitness = np.zeros(1 << n_sites)
    fitness[0] = 1.0

    variations = []
    for population in (10**3, 10**4, 10**5):
        result = sample_stationary(fitness, mu, population, 2000, [0, 1, 2], dt=0.01)
        distribution = np.asarray(result["distribution"])
        variations.append(0.5 * float(np.abs(distribution - reference).sum()))

    assert variations[-1] < variations[0], variations
    assert variations[-1] < 0.05, variations


def test_a_time_step_that_would_make_selection_weights_negative_is_refused() -> None:
    """With 1 + f dt below zero the multinomial weights would be negative, so the time step
    is rejected.
    """
    fitness = np.array([1.0, -30.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="selection weight non-positive"):
        simulate(fitness, 0.1, 1000, 10, seed=0, dt=0.1)


def test_a_time_step_that_breaks_the_continuous_limit_is_refused() -> None:
    with pytest.raises(ValueError, match="exceeds 0.5"):
        simulate(np.zeros(4), 20.0, 1000, 10, seed=0, dt=0.1)
