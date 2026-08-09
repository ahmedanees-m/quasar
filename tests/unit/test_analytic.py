"""The analytic ruler, and the brute-force reference it is measured against.

Gate G-R.1 is the full sweep. These are the fast checks that localise a failure: closed
forms against values worked out by hand, limits that have to hold for any correct
implementation, and small-L agreement between the two independent routes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import comb

from quasarstack.analytic.crow_kimura import (
    additive_mean_fitness,
    additive_quasispecies,
    class_quasispecies,
    single_peak_quasispecies,
)
from quasarstack.analytic.exact_diag import (
    infer_n_sites,
    mutation_selection_generator,
    perron_vector,
)
from quasarstack.classical.landscapes import (
    additive_fitness,
    class_fitness,
    single_peak_classes,
    uniform_additive_classes,
)
from quasarstack.io.conventions import genotype_to_index

pytestmark = pytest.mark.fast


# --- the closed form, against algebra done by hand -------------------------------------


def test_single_site_closed_form_matches_hand_algebra() -> None:
    """At L = 1 the whole problem is a 2x2 matrix that can be solved on paper.

    w = [[a - mu, mu], [mu, -a - mu]], top eigenvalue -mu + sqrt(a^2 + mu^2), eigenvector
    proportional to (1, (sqrt(a^2 + mu^2) - a) / mu).
    """
    a, mu = 1.3, 0.4
    root = math.sqrt(a * a + mu * mu)
    ratio = (root - a) / mu
    expected = np.array([1.0, ratio]) / (1.0 + ratio)

    assert np.allclose(additive_quasispecies(np.array([a]), mu), expected, atol=1e-15)
    assert additive_mean_fitness(np.array([a]), mu) == pytest.approx(-mu + root, abs=1e-15)


def test_closed_form_is_stable_under_strong_selection() -> None:
    """a >> mu is where the naive expression loses precision to cancellation.

    The mutant weight should track mu / (2a) rather than collapsing to zero or to noise.
    """
    a, mu = 1e6, 1e-3
    probs = additive_quasispecies(np.array([a]), mu)
    assert probs[1] == pytest.approx(mu / (2.0 * a), rel=1e-9)
    assert probs[0] == pytest.approx(1.0, abs=1e-8)


def test_quasispecies_is_a_probability_distribution() -> None:
    rng = np.random.default_rng(3)
    a = rng.uniform(0.2, 2.0, size=6)
    probs = additive_quasispecies(a, 0.3)
    assert probs.shape == (64,)
    assert probs.sum() == pytest.approx(1.0, abs=1e-14)
    assert (probs >= 0).all()


def test_zero_mutation_puts_everything_on_the_fittest_genotype() -> None:
    """With mu = 0 selection is unopposed and the quasispecies collapses to one sequence."""
    a = np.array([1.0, -2.0, 0.5])  # site 1 is fitter mutated, the others wild type
    probs = additive_quasispecies(a, 0.0)
    assert probs[genotype_to_index("010")] == pytest.approx(1.0)
    assert probs.sum() == pytest.approx(1.0)


def test_increasing_mutation_moves_mass_off_the_master_sequence() -> None:
    a = np.full(6, 1.0)
    masses = [additive_quasispecies(a, mu)[0] for mu in (0.05, 0.2, 0.5, 1.0, 2.0)]
    assert all(x > y for x, y in zip(masses, masses[1:], strict=False))


# --- the class reduction ----------------------------------------------------------------


@pytest.mark.parametrize("n_sites", [2, 4, 7])
def test_flat_landscape_gives_the_binomial_distribution(n_sites: int) -> None:
    """With no selection, mutation alone equilibrates to uniform over genotypes.

    Uniform over 2^L genotypes means the Hamming classes carry binomial weight, which is a
    check on the binomial bookkeeping in the conjugating transform.
    """
    _, class_probs, mean_fitness = class_quasispecies(np.zeros(n_sites + 1), 0.4)
    expected = comb(n_sites, np.arange(n_sites + 1)) / 2.0**n_sites
    assert np.allclose(class_probs, expected, atol=1e-12)
    assert mean_fitness == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("n_sites", [2, 3, 5, 8])
@pytest.mark.parametrize("a_value", [0.5, 2.0])
@pytest.mark.parametrize("mu", [0.1, 0.6])
def test_the_two_analytic_routes_agree_where_they_overlap(
    n_sites: int, a_value: float, mu: float
) -> None:
    """Uniform additive fitness is solvable both ways, so the two must land on the same
    distribution. Neither route can be checked by the other anywhere else."""
    closed_form = additive_quasispecies(np.full(n_sites, a_value), mu)
    by_class, _, _ = class_quasispecies(uniform_additive_classes(n_sites, a_value), mu)
    assert np.allclose(closed_form, by_class, atol=1e-13)


def test_class_probabilities_sum_to_one_on_a_sharp_peak() -> None:
    genotype_probs, class_probs, _ = single_peak_quasispecies(6, 3.0, 0.25)
    assert class_probs.sum() == pytest.approx(1.0, abs=1e-14)
    assert genotype_probs.sum() == pytest.approx(1.0, abs=1e-14)
    assert genotype_probs[0] == pytest.approx(class_probs[0], abs=1e-14)


# --- the brute-force reference -----------------------------------------------------------


def test_generator_structure_at_l2() -> None:
    """L + 1 non-zeros per row: the diagonal and one entry per single-site flip."""
    fitness = additive_fitness(np.array([1.0, 0.5]))
    mu = 0.3
    w = mutation_selection_generator(fitness, mu).toarray()

    assert np.allclose(np.diag(w), fitness - mu * 2)
    assert np.allclose(w, w.T)
    for row in range(4):
        assert np.count_nonzero(w[row]) == 3
    # genotype "00" connects to "10" and "01", not to "11"
    assert w[genotype_to_index("00"), genotype_to_index("10")] == pytest.approx(mu)
    assert w[genotype_to_index("00"), genotype_to_index("01")] == pytest.approx(mu)
    assert w[genotype_to_index("00"), genotype_to_index("11")] == 0.0


def test_generator_rows_sum_to_the_fitness_alone() -> None:
    """Mutation conserves probability, so it contributes nothing to a row sum.

    What is left is the fitness, which is exactly the sense in which the generator is
    non-conservative: a pure mutation operator would give zero row sums.
    """
    fitness = additive_fitness(np.array([1.0, -0.5, 0.25]))
    w = mutation_selection_generator(fitness, 0.7)
    assert np.allclose(np.asarray(w.sum(axis=1)).ravel(), fitness)


def test_infer_n_sites_rejects_non_power_of_two() -> None:
    assert infer_n_sites(np.zeros(8)) == 3
    with pytest.raises(ValueError, match="power of two"):
        infer_n_sites(np.zeros(6))


def test_exact_diagonalisation_of_a_flat_landscape_is_uniform() -> None:
    probs, mean_fitness, gap = perron_vector(np.zeros(16), 0.5)
    assert np.allclose(probs, 1.0 / 16, atol=1e-13)
    assert mean_fitness == pytest.approx(0.0, abs=1e-13)
    assert gap > 0.0


@pytest.mark.parametrize("n_sites", [2, 3, 4, 5])
@pytest.mark.parametrize("mu", [0.1, 0.5, 1.0])
def test_oracle_agrees_with_exact_diagonalisation_at_small_size(n_sites: int, mu: float) -> None:
    """A fast slice of gate G-R.1, so a regression surfaces on push rather than nightly."""
    rng = np.random.default_rng(1)
    a = rng.uniform(0.25, 2.0, size=n_sites)
    oracle = additive_quasispecies(a, mu)
    reference, ref_lambda, _ = perron_vector(additive_fitness(a), mu)
    assert np.max(np.abs(oracle - reference)) < 1e-12
    assert additive_mean_fitness(a, mu) == pytest.approx(ref_lambda, abs=1e-12)

    peak = single_peak_classes(n_sites, 2.0)
    by_class, _, class_lambda = class_quasispecies(peak, mu)
    ref_peak, peak_lambda, _ = perron_vector(class_fitness(peak), mu)
    assert np.max(np.abs(by_class - ref_peak)) < 1e-12
    assert class_lambda == pytest.approx(peak_lambda, abs=1e-12)
