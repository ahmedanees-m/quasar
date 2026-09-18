"""Conservation and reversibility of the mutation-selection generator."""

from __future__ import annotations

import numpy as np
import pytest

from quasarstack.analytic.exact_diag import mutation_selection_generator
from quasarstack.classical.landscapes import additive_fitness
from quasarstack.spectral.perron import (
    mutation_generator,
    reversibility_report,
    selection_generator,
    symmetrising_measure,
)

pytestmark = pytest.mark.fast


def test_symmetric_mutation_gives_a_symmetric_generator() -> None:
    """Symmetric per-site mutation gives a symmetric generator, reversible with respect to
    the uniform measure.
    """
    generator = mutation_generator(4, 0.3) + selection_generator(
        additive_fitness(np.array([1.0, 0.5, -0.2, 0.8]))
    )
    report = reversibility_report(generator)
    assert report["is_symmetric"]
    assert report["is_reversible"]
    assert report["reversibility_defect"] == 0.0
    assert report["stationary_measure_is_uniform"]


def test_the_generator_is_not_conservative() -> None:
    """Columns sum to the fitness, not to zero, so it is not a Markov generator.

    Non-conservation and nonreversibility are distinct properties.

    """
    fitness = additive_fitness(np.array([1.0, 0.5, -0.2]))
    generator = mutation_generator(3, 0.3) + selection_generator(fitness)
    report = reversibility_report(generator)
    assert not report["is_conservative"]
    assert report["max_abs_column_sum"] > 0.1
    # but reversible
    assert report["is_reversible"]


def test_asymmetric_per_site_mutation_is_still_reversible() -> None:
    """Different forward and backward rates make the matrix non-symmetric but not
    nonreversible. Independent per-site flips form a product of two-state birth-death
    processes, and those are reversible with respect to a product measure whatever the
    rates."""
    generator = mutation_generator(4, 0.3, mu_backward=0.03)
    report = reversibility_report(generator)
    assert not report["is_symmetric"]
    assert report["is_reversible"]
    assert report["reversibility_defect"] < 1e-12
    assert not report["stationary_measure_is_uniform"]


def test_direction_specific_context_dependence_breaks_reversibility() -> None:
    """Context dependence applied to one direction breaks reversibility.

    CpG and APOBEC effects raise C to T without raising T to C.

    """
    generator = mutation_generator(4, 0.3, context_strength=1.5)
    report = reversibility_report(generator)
    assert not report["is_reversible"]
    assert report["reversibility_defect"] > 0.1


def test_two_sided_context_dependence_stays_reversible() -> None:
    """Context dependence applied to both directions keeps reversibility.

    The rate separates into a direction-dependent part times a context-dependent part, and the
    product cancels from Kolmogorov's cycle condition.

    """
    n_sites, mu, strength = 4, 0.3, 1.5
    dim = 1 << n_sites
    generator = np.zeros((dim, dim))
    for source in range(dim):
        for site in range(n_sites):
            rate = mu * (1.0 + strength) if source >> ((site - 1) % n_sites) & 1 else mu
            generator[source ^ (1 << site), source] += rate
            generator[source, source] -= rate
    assert reversibility_report(generator)["is_reversible"]


def test_recovered_measure_satisfies_detailed_balance() -> None:
    generator = mutation_generator(4, 0.4, mu_backward=0.05)
    measure, defect = symmetrising_measure(generator)
    assert measure is not None
    assert defect < 1e-12
    assert measure.sum() == pytest.approx(1.0)
    off = generator - np.diag(np.diag(generator))
    flux = measure[None, :] * off
    assert np.max(np.abs(flux - flux.T)) < 1e-12


def test_perron_module_agrees_with_the_exact_diag_generator() -> None:
    """The two constructions of the generator match."""
    a = np.array([1.0, 0.5, -0.2])
    mu = 0.3
    fitness = additive_fitness(a)
    here = mutation_generator(3, mu) + selection_generator(fitness)
    there = mutation_selection_generator(fitness, mu).toarray()
    assert np.max(np.abs(here - there)) < 1e-12


@pytest.mark.parametrize(
    "landscape",
    ["flat", "additive", "epistatic", "single_peak", "nk_rugged", "random", "wildly_scaled"],
)
@pytest.mark.parametrize("mutation", ["symmetric", "asymmetric", "context_dependent"])
def test_selection_cannot_change_reversibility(landscape: str, mutation: str) -> None:
    """Claim S7 of `docs/theory.md`: selection cannot change reversibility.

    Detailed balance constrains only off-diagonal entries, and selection is diagonal, so no
    landscape changes the reversibility defect. The context-dependent case is included because
    it is nonreversible; selection leaves its nonzero defect unchanged as well.

    """
    n_sites = 5
    size = 1 << n_sites
    rng = np.random.default_rng(hash((landscape, mutation)) % (2**32))

    if mutation == "symmetric":
        mutation_part = mutation_generator(n_sites, 0.15)
    elif mutation == "asymmetric":
        mutation_part = mutation_generator(n_sites, 0.15, 0.06)
    else:
        mutation_part = mutation_generator(n_sites, 0.15, 0.06, context_strength=0.8)

    fitness = {
        "flat": np.zeros(size),
        "additive": np.linspace(-1.0, 1.0, size),
        "epistatic": additive_fitness(
            rng.uniform(-2.0, 2.0, size=n_sites), rng.normal(size=(n_sites, n_sites))
        ),
        "single_peak": np.eye(1, size, 0).ravel() * 3.0,
        "nk_rugged": rng.normal(size=size),
        "random": rng.uniform(-5.0, 5.0, size=size),
        # Fitness spanning fifteen orders of magnitude.
        "wildly_scaled": rng.uniform(-1e7, 1e7, size=size) * 10.0 ** rng.integers(-8, 8, size),
    }[landscape]

    before = reversibility_report(mutation_part)
    after = reversibility_report(mutation_part + selection_generator(fitness))

    assert after["is_reversible"] == before["is_reversible"]
    assert after["reversibility_defect"] == pytest.approx(before["reversibility_defect"], abs=1e-12)
    # Adding fitness removes conservation and leaves reversibility unchanged.
    if landscape != "flat":
        assert not after["is_conservative"]
