"""Baseline C: the generator as an MPO, and DMRG against exact diagonalisation."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("quimb")

from quasarstack.analytic.exact_diag import mutation_selection_generator, perron_vector
from quasarstack.classical.landscapes import (
    class_fitness,
    house_of_cards_fitness,
    nk_fitness,
    single_peak_classes,
    spin_glass_fitness,
)
from quasarstack.classical.mps_dmrg import (
    dominant_state,
    generator_mpo,
    is_complement_symmetric,
)

pytestmark = pytest.mark.fast


def dense(fitness: np.ndarray, mu: float) -> np.ndarray:
    return np.asarray(mutation_selection_generator(fitness, mu).todense())


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.mark.parametrize("n_sites", [2, 3, 5, 6])
def test_mpo_contracts_to_the_generator(n_sites: int) -> None:
    rng = np.random.default_rng(n_sites)
    fitness = rng.normal(size=1 << n_sites)
    mu = 0.37
    operator, _ = generator_mpo(fitness, mu)
    got = np.asarray(operator.to_dense()).real
    assert np.max(np.abs(got - dense(fitness, mu))) < 1e-12


def test_parity_term_is_the_global_flip() -> None:
    n_sites = 5
    fitness = np.random.default_rng(1).normal(size=1 << n_sites)
    operator, _ = generator_mpo(fitness, 0.2, shift=0.8)
    flip = np.eye(1 << n_sites)[::-1]
    expected = dense(fitness, 0.2) + 0.8 * flip
    assert np.max(np.abs(np.asarray(operator.to_dense()).real - expected)) < 1e-12


def test_operator_bond_dimension_is_the_cut_rank() -> None:
    _, bond = generator_mpo(class_fitness(single_peak_classes(8, 1.0)), 0.1)
    assert bond == 1
    _, bond = generator_mpo(house_of_cards_fitness(8, seed=0), 0.1)
    assert bond == 16


def test_complement_symmetry_is_detected() -> None:
    assert is_complement_symmetric(spin_glass_fitness(8, seed=0))
    assert not is_complement_symmetric(nk_fitness(8, 2, seed=0))


@pytest.mark.parametrize(
    "fitness",
    [
        class_fitness(single_peak_classes(8, 1.0)),
        nk_fitness(8, 2, seed=0),
        house_of_cards_fitness(8, seed=1),
        spin_glass_fitness(8, seed=0),
    ],
    ids=["single_peak", "nk_k2", "house_of_cards", "spin_glass"],
)
@pytest.mark.parametrize("mu", [0.02, 0.2])
def test_matches_exact_diagonalisation(fitness: np.ndarray, mu: float) -> None:
    reference, eigenvalue = perron_vector(fitness, mu)[:2]
    result = dominant_state(fitness, mu, max_bond=16)
    assert result["converged"]
    assert cosine(result["distribution"], reference) > 1.0 - 1e-10
    assert result["eigenvalue"] == pytest.approx(float(eigenvalue), abs=1e-9)
    assert result["distribution"].sum() == pytest.approx(1.0)


def test_capped_bond_dimension_is_respected() -> None:
    result = dominant_state(house_of_cards_fitness(8, seed=0), 0.2, max_bond=2)
    assert max(result["bond_by_sweep"]) <= 2


def test_budget_stops_between_sweeps() -> None:
    result = dominant_state(nk_fitness(8, 4, seed=0), 0.2, max_bond=16, budget=0.0)
    assert result["budget_exhausted"]
    assert result["sweeps"] == 1


def test_repeatable_for_a_fixed_seed() -> None:
    fitness = nk_fitness(8, 1, seed=0)
    first = dominant_state(fitness, 0.1, max_bond=16, seed=3)
    second = dominant_state(fitness, 0.1, max_bond=16, seed=3)
    assert np.array_equal(first["distribution"], second["distribution"])
    assert first["energy_by_sweep"] == second["energy_by_sweep"]
