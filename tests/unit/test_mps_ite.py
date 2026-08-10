"""Tensor-train primitives and matrix-product imaginary time. WP6 Baseline C.

The primitives are checked against dense linear algebra, exactly, because every WP6 number
rests on them and a subtly wrong Kronecker ordering would produce plausible output. The
evolution is checked against exact diagonalisation.
"""

from __future__ import annotations

import numpy as np
import pytest

from quasarstack.analytic.exact_diag import perron_vector
from quasarstack.classical.landscapes import (
    additive_fitness,
    class_fitness,
    nk_fitness,
    single_peak_classes,
)
from quasarstack.classical.mps_ite import evolve, step_operator_bond_dimension
from quasarstack.classical.tensor_train import (
    apply_single_site,
    from_tt,
    hadamard,
    norm,
    to_tt,
    tt_round,
)

pytestmark = pytest.mark.fast


@pytest.mark.parametrize("n_sites", [2, 3, 4, 6, 8])
def test_decomposition_round_trips_exactly(n_sites: int) -> None:
    vector = np.random.default_rng(n_sites).normal(size=1 << n_sites)
    cores, discarded = to_tt(vector)
    assert discarded == pytest.approx(0.0, abs=1e-15)
    assert np.max(np.abs(from_tt(cores) - vector)) < 1e-12


def test_site_ordering_is_little_endian() -> None:
    """Core i carries site i, bit i, qubit i. `numpy.reshape` puts the *most* significant
    bit on axis 0, so a missing transpose would reverse the sites and show up only as an
    unexplained factor much later. Checked against an explicit Kronecker product built in
    little-endian order."""
    n_sites = 5
    rng = np.random.default_rng(0)
    vector = rng.normal(size=1 << n_sites)
    matrices = [rng.normal(size=(2, 2)) for _ in range(n_sites)]

    cores, _ = to_tt(vector)
    got = from_tt(apply_single_site(cores, matrices))

    dense = matrices[n_sites - 1]
    for site in range(n_sites - 2, -1, -1):
        dense = np.kron(dense, matrices[site])
    assert np.max(np.abs(got - dense @ vector)) < 1e-12


def test_hadamard_is_the_elementwise_product() -> None:
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=64), rng.normal(size=64)
    product = hadamard(to_tt(a)[0], to_tt(b)[0])
    assert np.max(np.abs(from_tt(product) - a * b)) < 1e-12


def test_rounding_without_truncation_changes_nothing() -> None:
    fitness = nk_fitness(8, 2, seed=0)
    cores, _ = to_tt(fitness)
    rounded, discarded = tt_round(cores, max_bond=1 << 4)
    assert discarded == pytest.approx(0.0, abs=1e-15)
    assert np.max(np.abs(from_tt(rounded) - fitness)) < 1e-12


def test_discarded_weight_matches_the_error_it_predicts() -> None:
    """The reported truncation weight has to mean something. For an orthogonal truncation
    the discarded squared weight is the squared relative error, and criterion 4 reports it
    at every step, so a number that did not track the actual error would be worse than none.
    """
    fitness = nk_fitness(8, 2, seed=0)
    cores, _ = to_tt(fitness)
    rounded, discarded = tt_round(cores, max_bond=4)
    relative = np.linalg.norm(from_tt(rounded) - fitness) / np.linalg.norm(fitness)
    assert discarded == pytest.approx(relative**2, rel=0.05)


def test_norm_agrees_with_the_dense_norm() -> None:
    vector = np.random.default_rng(3).normal(size=256)
    assert norm(to_tt(vector)[0]) == pytest.approx(float(np.linalg.norm(vector)), rel=1e-12)


@pytest.mark.parametrize("name", ["additive", "single_peak", "nk_k2"])
def test_evolution_reproduces_exact_diagonalisation(name: str) -> None:
    """Criterion 1 of G-6, at a size the unit suite can afford."""
    n_sites, mu = 8, 0.2
    rng = np.random.default_rng(0)
    fitness = {
        "additive": lambda: additive_fitness(rng.uniform(0.3, 1.5, size=n_sites)),
        "single_peak": lambda: class_fitness(single_peak_classes(n_sites, 1.0)),
        "nk_k2": lambda: nk_fitness(n_sites, 2, seed=0),
    }[name]()

    reference = np.abs(perron_vector(fitness, mu)[0])
    reference = reference / reference.sum()

    result = evolve(fitness, mu, max_bond_dimension=16, dtau=0.05, max_steps=1500)
    distribution = np.asarray(result["distribution"])
    cosine = float(
        distribution @ reference / (np.linalg.norm(distribution) * np.linalg.norm(reference))
    )
    assert cosine >= 0.999, (name, cosine)
    assert result["converged"], name


def test_the_exponential_has_a_different_rank_from_the_hamiltonian() -> None:
    """A refinement that matters for the cost model, and which points both ways.

    `mpo_analysis` measures the bond dimension of diag(f), which is right for representing
    the operator. The per-step cost is set by the bond dimension of exp(dtau f), and these
    differ: an additive f goes 2 to 1, because the exponential of a sum is a product, while
    a single peak goes 1 to 2, because exp of a delta is a constant plus a delta.
    """
    n_sites = 8
    additive = step_operator_bond_dimension(
        additive_fitness(np.random.default_rng(0).uniform(0.3, 1.5, size=n_sites)), 0.05
    )
    peak = step_operator_bond_dimension(class_fitness(single_peak_classes(n_sites, 1.0)), 0.05)
    assert additive["step_operator_bond_dimension"] < additive["fitness_bond_dimension"]
    assert peak["step_operator_bond_dimension"] > peak["fitness_bond_dimension"]


def test_truncation_history_covers_every_step() -> None:
    """Criterion 4: recorded at every step, not only at the end. A method that discards
    weight throughout and looks clean at the last step is what this prevents."""
    result = evolve(nk_fitness(6, 2, seed=0), 0.2, max_bond_dimension=2, dtau=0.05, max_steps=50)
    assert len(result["truncation_history"]) == result["steps"]
    assert result["total_discarded_weight"] > 0.0


def test_a_zero_bond_dimension_is_refused() -> None:
    with pytest.raises(ValueError, match="max_bond_dimension"):
        evolve(np.zeros(16), 0.2, max_bond_dimension=0)
