"""Motta QITE and the parity of its generator basis.

For a real state, Motta's right-hand side vanishes identically on Pauli strings with an even
number of Y factors. These tests check the odd-Y basis and the vanishing contribution of the
even-Y strings.
"""

from __future__ import annotations

import numpy as np
import pytest

from quasarstack.analytic.crow_kimura import additive_quasispecies
from quasarstack.hamiltonian.builder import additive_hamiltonian
from quasarstack.ite.qite_motta import (
    build_generators,
    evolve,
    odd_y_strings,
    solve_generator,
    zero_rhs_demonstration,
)
from quasarstack.scoring.metrics import cosine_similarity

pytestmark = pytest.mark.fast


def _matrix(a: np.ndarray, mu: float) -> np.ndarray:
    return np.asarray(additive_hamiltonian(a, mu).to_matrix()).real


def _uniform(n_sites: int) -> np.ndarray:
    dimension = 1 << n_sites
    return np.full(dimension, 1.0 / np.sqrt(dimension))


def test_every_generator_string_has_an_odd_number_of_ys() -> None:
    for n_sites in (2, 3, 4):
        for weight in range(1, n_sites + 1):
            for label in odd_y_strings(n_sites, weight):
                assert label.count("Y") % 2 == 1, label
                assert len(label) == n_sites


def test_generator_count_matches_the_combinatorics() -> None:
    """sum_k C(L, k) (3^k - 1) / 2 generators."""
    from math import comb

    for n_sites in (3, 4, 6):
        for max_weight in range(1, min(3, n_sites) + 1):
            expected = sum(comb(n_sites, k) * (3**k - 1) // 2 for k in range(1, max_weight + 1))
            assert len(odd_y_strings(n_sites, max_weight)) == expected


def test_generators_are_real_and_antisymmetric() -> None:
    """A real antisymmetric generator exponentiates to a real orthogonal matrix, which keeps a
    real state real.
    """
    generators = build_generators(3, 2)
    for matrix in generators.matrices:
        dense = matrix.toarray()
        assert np.max(np.abs(dense.imag)) == 0.0
        assert np.allclose(dense, -dense.T, atol=1e-14)


def test_even_y_strings_give_a_zero_right_hand_side() -> None:
    """Even-Y strings give a zero right-hand side.

    In Motta's form the right-hand side is ``Re(-i <psi| sigma_I |Delta>)``. For a real state and
    a real residual the bracket is real whenever ``sigma_I`` is real, so ``-i`` times it is purely
    imaginary and its real part is zero. The Y-free strings are the natural choice for a
    Hamiltonian built from X and Z, and they contribute nothing. The vanishing holds in the
    complex form, not in the real-arithmetic form the module solves.

    """
    a = np.array([1.0, 0.7, 1.2])
    report = zero_rhs_demonstration(_uniform(3), _matrix(a, 0.3), 3, 2, dtau=0.05)
    assert report["n_even_y_generators"] > 0, "the comparison needs a non-empty wrong basis"
    assert (
        report["even_y_rhs_max_abs"] == 0.0
    ), "every even-Y string must contribute exactly zero, not merely a small amount"
    assert report["odd_y_rhs_norm"] > 1e-3, "the correct basis must carry signal"


def test_a_step_keeps_the_state_real_and_normalised() -> None:
    a = np.array([1.0, 0.7, 1.2])
    matrix = _matrix(a, 0.3)
    generators = build_generators(3, 2)
    from quasarstack.ite.qite_motta import step

    state, rhs, condition = step(_uniform(3), matrix, generators, dtau=0.05)
    assert np.max(np.abs(np.imag(state))) == 0.0
    assert np.linalg.norm(state) == pytest.approx(1.0, abs=1e-12)
    assert rhs > 0.0


def test_the_solve_carries_signal_at_the_start() -> None:
    a = np.array([1.0, 0.7, 1.2])
    _, rhs, _ = solve_generator(_uniform(3), _matrix(a, 0.3), build_generators(3, 2), dtau=0.05)
    assert rhs > 1e-3


def test_energy_descends_and_it_reaches_the_quasispecies() -> None:
    a = np.array([1.0, 0.7, 1.2])
    mu = 0.3
    evolution = evolve(_matrix(a, mu), n_sites=3, tau=10.0, dtau=0.05, max_weight=2)
    energies = evolution.energies
    assert all(
        later <= earlier + 1e-10 for earlier, later in zip(energies, energies[1:], strict=False)
    ), "the energy ascended"
    assert cosine_similarity(evolution.probs, additive_quasispecies(a, mu)) > 0.95


def test_full_weight_reproduces_imaginary_time_closely() -> None:
    """With full generator support nothing is truncated, so the unitary step tracks the
    exact non-unitary step closely.
    """
    a = np.array([1.0, 0.7])
    mu = 0.3
    evolution = evolve(_matrix(a, mu), n_sites=2, tau=20.0, dtau=0.05, max_weight=2)
    assert cosine_similarity(evolution.probs, additive_quasispecies(a, mu)) > 0.999


def test_step_count_must_divide_the_budget() -> None:
    with pytest.raises(ValueError, match="whole number"):
        evolve(_matrix(np.ones(2), 0.3), n_sites=2, tau=1.0, dtau=0.3)


def test_weight_bounds_are_enforced() -> None:
    with pytest.raises(ValueError, match="max_weight must be"):
        odd_y_strings(4, 5)
    with pytest.raises(ValueError, match="max_weight must be"):
        odd_y_strings(4, 0)
