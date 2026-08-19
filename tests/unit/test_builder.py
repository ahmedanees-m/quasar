"""The biology-to-qubit compiler.

Two things are being defended here. The first is the sign and shift convention: the
compiler emits ``-W``, identity term included, so that its matrix cancels the generator
exactly rather than agreeing only up to a constant. The second is endianness, which is the
error most likely to survive every spectral check, because permuting the computational
basis leaves the spectrum untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

from quasarstack.analytic.crow_kimura import additive_quasispecies
from quasarstack.analytic.exact_diag import mutation_selection_generator
from quasarstack.classical.landscapes import additive_fitness, class_fitness, single_peak_classes
from quasarstack.hamiltonian.builder import (
    additive_hamiltonian,
    diagonal_hamiltonian,
    diagonal_pauli_terms,
    ground_state,
    pauli_term_count,
    walsh_hadamard,
)
from quasarstack.io.conventions import genotype_to_index
from quasarstack.scoring.metrics import cosine_similarity

pytestmark = pytest.mark.fast


def test_qubit_i_carries_site_i() -> None:
    """The endianness test.

    A single strongly positive coefficient on site 0 must push mass onto the genotype whose
    *site 0* is wild type. If Pauli strings were being built rightmost-first by hand, this
    would land on the last site instead and every spectral check would still pass.
    """
    a = np.array([5.0, 0.0, 0.0])
    hamiltonian = additive_hamiltonian(a, mu=0.2)
    diagonal = np.diag(np.asarray(hamiltonian.to_matrix()).real)

    # H = -W, so the lowest diagonal entry is the fittest genotype.
    fittest = int(np.argmin(diagonal))
    assert fittest == genotype_to_index("000")
    # Flipping site 0 must cost more than flipping site 2, which carries no fitness.
    assert diagonal[genotype_to_index("100")] > diagonal[genotype_to_index("001")]


def test_compiled_operator_is_minus_the_generator() -> None:
    """Entry by entry, not merely up to a shift. The identity term is what makes this hold."""
    rng = np.random.default_rng(0)
    for n_sites in (2, 3, 4):
        a = rng.uniform(0.25, 2.0, size=n_sites)
        mu = 0.35
        compiled = np.asarray(additive_hamiltonian(a, mu).to_matrix()).real
        generator = mutation_selection_generator(additive_fitness(a), mu).toarray()
        assert np.max(np.abs(compiled + generator)) < 1e-12


def test_epistatic_coupling_reaches_the_operator() -> None:
    a = np.zeros(2)
    b = np.array([[0.0, 0.9], [0.0, 0.0]])
    compiled = np.asarray(additive_hamiltonian(a, 0.1, b).to_matrix()).real
    generator = mutation_selection_generator(additive_fitness(a, b), 0.1).toarray()
    assert np.max(np.abs(compiled + generator)) < 1e-12


def test_hamiltonian_is_stoquastic() -> None:
    """Every off-diagonal entry non-positive. This is what makes the ground state sign
    definite, which is the whole justification for the L1 decode. If it ever fails, the
    normalisation argument in docs/notes.md fails with it."""
    rng = np.random.default_rng(2)
    a = rng.uniform(-1.0, 2.0, size=4)
    matrix = np.asarray(additive_hamiltonian(a, 0.4).to_matrix()).real
    off_diagonal = matrix - np.diag(np.diag(matrix))
    assert off_diagonal.max() <= 1e-15


def test_walsh_hadamard_matches_the_naive_transform() -> None:
    rng = np.random.default_rng(4)
    values = rng.normal(size=8)
    naive = np.array(
        [
            sum(values[sigma] * (-1) ** bin(subset & sigma).count("1") for sigma in range(8))
            for subset in range(8)
        ]
    )
    assert np.allclose(walsh_hadamard(values), naive, atol=1e-12)


def test_walsh_hadamard_is_its_own_inverse_up_to_scale() -> None:
    rng = np.random.default_rng(5)
    values = rng.normal(size=16)
    assert np.allclose(walsh_hadamard(walsh_hadamard(values)) / 16, values, atol=1e-12)


@pytest.mark.parametrize("n_sites", [2, 3, 4, 5])
def test_the_two_compiler_routes_agree_on_additive_landscapes(n_sites: int) -> None:
    """The structured build and the Walsh-Hadamard build are different code paths to the
    same operator, so disagreement localises a bug to one of them."""
    rng = np.random.default_rng(6)
    a = rng.uniform(0.25, 2.0, size=n_sites)
    mu = 0.3
    structured = np.asarray(additive_hamiltonian(a, mu).to_matrix()).real
    transformed = np.asarray(diagonal_hamiltonian(additive_fitness(a), mu).to_matrix()).real
    assert np.max(np.abs(structured - transformed)) < 1e-12


def test_additive_term_count_is_linear_in_size() -> None:
    """L longitudinal terms, L transverse terms, one identity."""
    for n_sites in (3, 6, 9):
        a = np.full(n_sites, 1.0)
        assert pauli_term_count(additive_hamiltonian(a, 0.2)) == 2 * n_sites + 1


def test_single_peak_projector_needs_every_z_subset() -> None:
    """The sharp peak has no structure to exploit, so its decomposition is dense: all 2^L
    Z-subsets, of which the identity merges with the mutation identity, plus L transverse
    terms. This is the numerator of the ratio gate G-R.10 measures."""
    for n_sites in (3, 4, 5):
        fitness = class_fitness(single_peak_classes(n_sites, 2.0))
        assert pauli_term_count(diagonal_hamiltonian(fitness, 0.2)) == 2**n_sites + n_sites


def test_diagonal_terms_reconstruct_the_diagonal() -> None:
    rng = np.random.default_rng(7)
    fitness = rng.normal(size=16)
    terms = diagonal_pauli_terms(fitness)
    rebuilt = np.zeros(16)
    for label, qubits, coefficient in terms:
        if label == "I":
            rebuilt += coefficient.real
            continue
        sign = np.ones(16)
        for qubit in qubits:
            sign *= np.array([1 - 2 * (j >> qubit & 1) for j in range(16)])
        rebuilt += coefficient.real * sign
    # diagonal_pauli_terms encodes -diag(fitness), matching the sign convention of H.
    assert np.allclose(rebuilt, -fitness, atol=1e-12)


@pytest.mark.parametrize("n_sites", [2, 4, 6])
@pytest.mark.parametrize("mu", [0.15, 0.8])
def test_ground_state_is_the_analytic_quasispecies(n_sites: int, mu: float) -> None:
    """A fast slice of gate G-R.2."""
    rng = np.random.default_rng(8)
    a = rng.uniform(0.25, 2.0, size=n_sites)
    probs, energy = ground_state(additive_hamiltonian(a, mu))
    oracle = additive_quasispecies(a, mu)
    assert cosine_similarity(probs, oracle) > 1 - 1e-12
    assert probs.sum() == pytest.approx(1.0, abs=1e-14)
    assert (probs >= 0).all()
    # H = -W, so the ground-state energy is minus the equilibrium mean fitness.
    assert -energy == pytest.approx(float(np.sum(-mu + np.hypot(a, mu))), abs=1e-10)
