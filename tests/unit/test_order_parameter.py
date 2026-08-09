"""The order parameter of the error threshold, and how its location is extracted.

Locating a threshold at finite L is a definition, not a discovery, so what is pinned here is
that the definition behaves as claimed on curves whose answer is known by construction, and
that it says so honestly when there is no interior transition to find.
"""

from __future__ import annotations

import numpy as np
import pytest

from quasarstack.analytic.crow_kimura import class_quasispecies
from quasarstack.classical.landscapes import (
    pairwise_uniform_classes,
    single_peak_classes,
    uniform_additive_classes,
)
from quasarstack.io.conventions import genotype_to_index
from quasarstack.spectral.order_parameter import (
    crossover_point,
    locate_threshold,
    magnetisation,
    magnetisation_from_classes,
    susceptibility,
)

pytestmark = pytest.mark.fast


def test_surplus_is_one_on_the_master_and_minus_one_on_the_all_mutant() -> None:
    n_sites = 4
    probs = np.zeros(1 << n_sites)
    probs[genotype_to_index("0000")] = 1.0
    assert magnetisation(probs, n_sites) == pytest.approx(1.0)

    probs = np.zeros(1 << n_sites)
    probs[genotype_to_index("1111")] = 1.0
    assert magnetisation(probs, n_sites) == pytest.approx(-1.0)


def test_uniform_population_has_zero_surplus() -> None:
    """Full delocalisation: the genetic information is gone."""
    assert magnetisation(np.full(64, 1 / 64), 6) == pytest.approx(0.0, abs=1e-14)


def test_half_mutated_genotype_has_zero_surplus() -> None:
    probs = np.zeros(16)
    probs[genotype_to_index("1010")] = 1.0
    assert magnetisation(probs, 4) == pytest.approx(0.0)


@pytest.mark.parametrize("n_sites", [3, 5, 7])
@pytest.mark.parametrize("mu", [0.1, 0.7])
def test_the_two_surplus_routes_agree(n_sites: int, mu: float) -> None:
    """Genotype-level and Hamming-class-level computations of the same quantity."""
    f_by_class = single_peak_classes(n_sites, 1.5)
    genotype_probs, class_probs, _ = class_quasispecies(f_by_class, mu)
    assert magnetisation(genotype_probs, n_sites) == pytest.approx(
        magnetisation_from_classes(class_probs), abs=1e-12
    )


def test_susceptibility_is_positive_where_the_surplus_falls() -> None:
    mus = np.linspace(0.1, 2.0, 40)
    m = 1.0 / (1.0 + mus)
    assert (susceptibility(mus, m) > 0).all()


def test_susceptibility_needs_a_grid() -> None:
    with pytest.raises(ValueError, match="at least three"):
        susceptibility(np.array([0.1, 0.2]), np.array([1.0, 0.9]))


def test_threshold_locator_finds_a_planted_peak() -> None:
    """A logistic drop centred at 0.8 should be located there, with a sensible width."""
    mus = np.linspace(0.01, 2.0, 400)
    centre = 0.8
    m = 1.0 / (1.0 + np.exp((mus - centre) / 0.05))
    found = locate_threshold(mus, m)
    assert found["mu_c"] == pytest.approx(centre, abs=0.02)
    assert found["peak_is_interior"]
    assert 0.05 < found["width"] < 0.5
    assert found["mu_half"] == pytest.approx(centre, abs=0.02)


def test_locator_reports_when_no_interior_peak_exists() -> None:
    """A landscape additive in the surplus decays monotonically with its steepest slope at
    zero mutation rate, so there is no threshold to locate and the flag must say so rather
    than returning the boundary as though it were an answer."""
    mus = np.round(np.arange(0.01, 2.001, 0.01), 10)
    f_by_class = uniform_additive_classes(6, 0.25)
    m = np.array([magnetisation_from_classes(class_quasispecies(f_by_class, mu)[1]) for mu in mus])
    found = locate_threshold(mus, m)
    assert not found["peak_is_interior"]
    # the crossover is still well defined and usable
    assert 0.0 < found["mu_half"] < 2.0


def test_sharp_peak_does_have_an_interior_threshold() -> None:
    """The contrast to the test above: the canonical error-threshold landscape."""
    mus = np.round(np.arange(0.01, 2.001, 0.01), 10)
    f_by_class = single_peak_classes(8, 1.0)
    m = np.array([magnetisation_from_classes(class_quasispecies(f_by_class, mu)[1]) for mu in mus])
    found = locate_threshold(mus, m)
    assert found["peak_is_interior"]
    assert 0.05 < found["mu_c"] < 0.3


def test_crossover_returns_nan_when_the_surplus_never_falls_far_enough() -> None:
    mus = np.linspace(0.01, 0.05, 20)
    m = np.linspace(1.0, 0.99, 20)
    assert np.isnan(crossover_point(mus, m, 0.5))


def test_synergistic_coupling_holds_the_population_together_to_higher_mutation() -> None:
    """The direction the planning documents predict, checked at one size.

    Gate G-R.4 reports this across sizes and does not require it. The test fixes L = 8 and
    the mean-field coupling scale, because at fixed coupling the total interaction grows as
    L squared and the apparent direction reverses between sizes, which is a normalisation
    artefact rather than physics.
    """
    mus = np.round(np.arange(0.01, 3.001, 0.01), 10)
    n_sites = 8

    def crossover(coupling: float) -> float:
        f_by_class = pairwise_uniform_classes(n_sites, 0.5, coupling / (n_sites - 1))
        m = np.array(
            [magnetisation_from_classes(class_quasispecies(f_by_class, mu)[1]) for mu in mus]
        )
        return crossover_point(mus, m, 0.5)

    assert crossover(2.0) > crossover(1.0) > crossover(0.0)
