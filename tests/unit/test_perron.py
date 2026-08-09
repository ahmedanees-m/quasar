"""Reversibility of the mutation-selection generator.

These tests pin the finding that decides how Route B can be built. The execution plan rests
Route B on a result for nonreversible Markov chains, so whether our generator is
nonreversible is not a detail, it is the premise.
"""

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
    """What QUASAR implements. Symmetric, therefore reversible with respect to uniform."""
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

    This is the property execution plan v4 correctly identifies. It is also a different
    property from nonreversibility, and the next test shows the two do not travel together.
    """
    fitness = additive_fitness(np.array([1.0, 0.5, -0.2]))
    generator = mutation_generator(3, 0.3) + selection_generator(fitness)
    report = reversibility_report(generator)
    assert not report["is_conservative"]
    assert report["max_abs_column_sum"] > 0.1
    # and yet
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
    """The only route into the nonreversible class found so far.

    The context factor must apply to one direction only. Biologically that is the faithful
    choice, since CpG and APOBEC effects raise C to T without raising T to C.
    """
    generator = mutation_generator(4, 0.3, context_strength=1.5)
    report = reversibility_report(generator)
    assert not report["is_reversible"]
    assert report["reversibility_defect"] > 0.1


def test_two_sided_context_dependence_stays_reversible() -> None:
    """The control that explains why the one-sidedness matters.

    A context factor on both directions makes the rate separate into a direction-dependent
    part times a context-dependent part, and that product cancels out of Kolmogorov's cycle
    condition.
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


@pytest.mark.parametrize("context", [0.0, 1.5])
def test_selection_cannot_change_reversibility(context: float) -> None:
    """Detailed balance constrains off-diagonal entries only, and selection is diagonal.

    So no fitness landscape, however rugged, can make the generator nonreversible. That
    matters for scoping: ruggedness is the project's main axis, and it is irrelevant to this
    property.
    """
    mutation = mutation_generator(4, 0.3, context_strength=context)
    bare = reversibility_report(mutation)
    rng = np.random.default_rng(0)
    for _ in range(3):
        rugged = additive_fitness(rng.uniform(-2.0, 2.0, size=4), rng.normal(size=(4, 4)))
        with_selection = reversibility_report(mutation + selection_generator(rugged))
        assert with_selection["is_reversible"] == bare["is_reversible"]
        assert with_selection["reversibility_defect"] == pytest.approx(
            bare["reversibility_defect"], abs=1e-12
        )


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
    """The two constructions of the same operator must match, so the reversibility finding
    is about the operator the rest of the project actually uses."""
    a = np.array([1.0, 0.5, -0.2])
    mu = 0.3
    fitness = additive_fitness(a)
    here = mutation_generator(3, mu) + selection_generator(fitness)
    there = mutation_selection_generator(fitness, mu).toarray()
    assert np.max(np.abs(here - there)) < 1e-12
