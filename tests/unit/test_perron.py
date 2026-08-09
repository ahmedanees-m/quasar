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


@pytest.mark.parametrize(
    "landscape",
    ["flat", "additive", "epistatic", "single_peak", "nk_rugged", "random", "wildly_scaled"],
)
@pytest.mark.parametrize("mutation", ["symmetric", "asymmetric", "context_dependent"])
def test_selection_cannot_change_reversibility(landscape: str, mutation: str) -> None:
    """Claim S7 of `docs/theory.md`, which was asserted before this test existed.

    Detailed balance constrains only off-diagonal entries, and selection is diagonal.
    So no landscape, however rugged, however epistatic, however badly scaled, can move
    the reversibility defect of the generator. This matters because it closes off the
    tempting repair to ADR-0010: if ruggedness could buy nonreversibility, Route B's
    stated foundation could be recovered by choosing a harder landscape family. It
    cannot. The property is fixed entirely by the mutation model.

    The context-dependent case is included because it is the one mutation model that *is*
    nonreversible; the claim is that selection leaves its defect alone too, not merely
    that it leaves zero alone.
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
        # Deliberately absurd dynamic range: if a diagonal could perturb the answer at all,
        # entries spanning fifteen orders of magnitude would be where it showed.
        "wildly_scaled": rng.uniform(-1e7, 1e7, size=size) * 10.0 ** rng.integers(-8, 8, size),
    }[landscape]

    before = reversibility_report(mutation_part)
    after = reversibility_report(mutation_part + selection_generator(fitness))

    assert after["is_reversible"] == before["is_reversible"]
    assert after["reversibility_defect"] == pytest.approx(before["reversibility_defect"], abs=1e-12)
    # And the two properties really are independent: adding fitness destroys conservation
    # while leaving reversibility untouched, which is the distinction ADR-0010 turns on.
    if landscape != "flat":
        assert not after["is_conservative"]
