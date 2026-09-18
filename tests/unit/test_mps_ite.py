"""Tensor-train primitives and matrix-product imaginary-time evolution.

The primitives are checked exactly against dense linear algebra, and the evolution against exact
diagonalisation.
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
    _svd,
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
    """Core i carries site i, bit i, qubit i. `numpy.reshape` puts the most significant bit
    on axis 0, so the sites are reversed after reshaping. Checked against an explicit Kronecker
    product in little-endian order.
    """
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
    """For an orthogonal truncation the discarded squared weight equals the squared relative
    error.
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
    """Criterion 1 of G-6 at a small size."""
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
    """The rank of exp(dtau f) differs from the rank of diag(f).

    An additive f goes from 2 to 1, since the exponential of a sum is a product, and a single
    peak goes from 1 to 2, since the exponential of a delta is a constant plus a delta.

    """
    n_sites = 8
    additive = step_operator_bond_dimension(
        additive_fitness(np.random.default_rng(0).uniform(0.3, 1.5, size=n_sites)), 0.05
    )
    peak = step_operator_bond_dimension(class_fitness(single_peak_classes(n_sites, 1.0)), 0.05)
    assert additive["step_operator_bond_dimension"] < additive["fitness_bond_dimension"]
    assert peak["step_operator_bond_dimension"] > peak["fitness_bond_dimension"]


def test_truncation_history_covers_every_step() -> None:
    """Discarded weight is recorded at every step."""
    result = evolve(nk_fitness(6, 2, seed=0), 0.2, max_bond_dimension=2, dtau=0.05, max_steps=50)
    assert len(result["truncation_history"]) == result["steps"]
    assert result["total_discarded_weight"] > 0.0


def test_a_zero_bond_dimension_is_refused() -> None:
    with pytest.raises(ValueError, match="max_bond_dimension"):
        evolve(np.zeros(16), 0.2, max_bond_dimension=0)


class TestSvdFallback:
    """SVD with a fallback driver."""

    def test_rank_deficient_matrix_decomposes(self) -> None:
        """A matrix whose rank is far below its size."""
        rng = np.random.default_rng(0)
        left = rng.normal(size=(128, 3))
        matrix = left @ rng.normal(size=(3, 128))
        u, singular, vt = _svd(matrix)
        assert np.allclose(u * singular @ vt, matrix, atol=1e-10)
        assert np.sum(singular > 1e-9 * singular[0]) == 3

    def test_falls_back_when_the_fast_driver_gives_up(self, monkeypatch) -> None:
        """A failure of gesdd falls back to gesvd."""
        rng = np.random.default_rng(1)
        matrix = rng.normal(size=(40, 24))

        def refuse(*args, **kwargs):
            raise np.linalg.LinAlgError("SVD did not converge")

        monkeypatch.setattr(np.linalg, "svd", refuse)
        u, singular, vt = _svd(matrix)
        assert np.allclose(u * singular @ vt, matrix, atol=1e-10)

    def test_non_finite_input_says_so(self) -> None:
        """Non-finite input raises a clear error."""
        matrix = np.ones((4, 4))
        matrix[2, 1] = np.nan
        with pytest.raises(ValueError, match="not finite"):
            _svd(matrix)
