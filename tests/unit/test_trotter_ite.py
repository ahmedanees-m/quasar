"""Trotterised imaginary-time evolution: modules M, S and E.

The propagator is non-unitary by necessity, so the usual reassurance that a circuit
preserves norm does not apply and the checks have to be different. What is pinned here is
that each module is the operator it claims to be, that the splitting is genuinely second
order rather than merely converging, and that the structural circuit is not mistaken for the
propagator.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from quasarstack.analytic.crow_kimura import additive_quasispecies
from quasarstack.analytic.exact_diag import mutation_selection_generator
from quasarstack.circuit.trotter_ite import (
    apply_mutation,
    apply_selection,
    evolve,
    evolve_exact,
    trotter_circuit,
    trotter_step,
    uniform_state,
)
from quasarstack.classical.landscapes import additive_fitness
from quasarstack.io.conventions import genotype_to_index
from quasarstack.scoring.metrics import cosine_similarity

pytestmark = pytest.mark.fast


def test_uniform_state_is_normalised_and_flat() -> None:
    state = uniform_state(4)
    assert state.shape == (16,)
    assert np.linalg.norm(state) == pytest.approx(1.0)
    assert np.allclose(state, state[0])


@pytest.mark.parametrize("n_sites", [1, 2, 3, 4])
def test_module_m_is_the_transverse_field_exponential(n_sites: int) -> None:
    """Compare against the explicitly exponentiated sum of X operators."""
    mu, dtau = 0.4, 0.3
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    total = np.zeros((1 << n_sites, 1 << n_sites))
    for site in range(n_sites):
        operator = np.array([[1.0]])
        # site i occupies bit i, so it is the i-th factor from the right
        for position in range(n_sites):
            factor = pauli_x if position == site else np.eye(2)
            operator = np.kron(factor, operator) if position >= 0 else operator
        # rebuild with correct ordering: bit 0 is the fastest-varying index
        operator = np.array([[1.0]])
        for position in reversed(range(n_sites)):
            factor = pauli_x if position == site else np.eye(2)
            operator = np.kron(operator, factor)
        total += operator

    rng = np.random.default_rng(0)
    state = rng.normal(size=1 << n_sites)
    expected = expm(mu * dtau * total) @ state
    assert np.allclose(apply_mutation(state, mu, dtau), expected, atol=1e-12)


def test_module_s_is_a_diagonal_scaling_up_to_a_positive_constant() -> None:
    """apply_selection shifts the exponent by its maximum to stay finite, which multiplies
    the result by a positive scalar. The ray is what matters, so the check is on direction."""
    fitness = np.array([2.0, -1.0, 0.5, 0.0])
    state = np.array([1.0, 1.0, 1.0, 1.0])
    got = apply_selection(state, fitness, 0.7)
    expected = np.exp(fitness * 0.7) * state
    assert cosine_similarity(got, expected) == pytest.approx(1.0, abs=1e-14)


def test_a_trotter_step_leaves_the_state_normalised() -> None:
    fitness = additive_fitness(np.array([1.0, 0.5, -0.3]))
    state = trotter_step(uniform_state(3), fitness, 0.3, 0.1)
    assert np.linalg.norm(state) == pytest.approx(1.0)


def test_step_count_must_divide_the_total_time() -> None:
    """A step size that does not divide tau would silently compare different total times
    across the scaling sweep, which is exactly what the exponent fit must not do."""
    fitness = additive_fitness(np.array([1.0, 0.5]))
    with pytest.raises(ValueError, match="whole number"):
        evolve(fitness, 0.3, tau=1.0, dtau=0.3)


def test_exact_propagator_matches_a_direct_matrix_exponential() -> None:
    fitness = additive_fitness(np.array([1.0, -0.5, 0.75]))
    mu, tau = 0.3, 1.5
    generator = mutation_selection_generator(fitness, mu).toarray()
    direct = expm(generator * tau) @ uniform_state(3)
    direct = np.abs(direct) / np.abs(direct).sum()
    assert np.allclose(evolve_exact(fitness, mu, tau), direct, atol=1e-12)


def test_splitting_error_falls_by_four_when_the_step_is_halved() -> None:
    """Second order, checked directly rather than only through the fitted exponent.

    A fit can look clean while sitting on the wrong power, so the ratio is worth asserting
    on its own terms.
    """
    fitness = additive_fitness(np.array([1.2, 0.6, -0.4, 0.9]))
    mu, tau = 0.3, 2.0
    reference = evolve_exact(fitness, mu, tau)

    coarse, _ = evolve(fitness, mu, tau, 0.05)
    fine, _ = evolve(fitness, mu, tau, 0.025)
    coarse_error = np.max(np.abs(coarse - reference))
    fine_error = np.max(np.abs(fine - reference))
    assert coarse_error / fine_error == pytest.approx(4.0, rel=0.05)


def test_evolution_reaches_the_analytic_quasispecies() -> None:
    a = np.array([1.0, 0.7, 1.4])
    mu = 0.3
    probs, n_steps = evolve(additive_fitness(a), mu, tau=40.0, dtau=0.01)
    assert n_steps == 4000
    assert cosine_similarity(probs, additive_quasispecies(a, mu)) > 1 - 1e-6


def test_without_mutation_selection_collapses_onto_the_fittest_genotype() -> None:
    """Module S alone, the limit the explainer describes.

    The runner-up here is one flip away at the weakest site, so it keeps weight
    ``exp(-2 * 0.5 * tau)`` and the collapse is only exact as tau grows. tau = 40 puts that
    residual at about 1e-17, which is what lets the tolerance be tight enough to mean
    something. At tau = 20 it is 2e-9, and a test that passed at 1e-8 there would be
    asserting the tolerance rather than the physics.
    """
    a = np.array([1.0, -2.0, 0.5])
    probs, _ = evolve(additive_fitness(a), mu=0.0, tau=40.0, dtau=0.1)
    assert probs[genotype_to_index("010")] == pytest.approx(1.0, abs=1e-12)


def test_without_selection_mutation_diffuses_to_uniform() -> None:
    """Module M alone, the other limit."""
    probs, _ = evolve(np.zeros(16), mu=0.4, tau=20.0, dtau=0.1)
    assert np.allclose(probs, 1.0 / 16, atol=1e-9)


def test_structural_circuit_is_the_unitary_analogue_and_is_labelled_as_such() -> None:
    """It exists for resource reporting. It must not be mistaken for the propagator, so the
    test asserts what it is: a unitary circuit with the same interaction pattern."""
    n_sites = 4
    a = np.full(n_sites, 1.0)
    b = np.zeros((n_sites, n_sites))
    b[0, 1] = 0.5
    b[2, 3] = 0.5
    circuit = trotter_circuit(n_sites, a, 0.3, 0.1, b)

    two_qubit = [inst for inst in circuit.data if len(inst.qubits) == 2]
    assert len(two_qubit) == 4, "one Rzz per coupling, per half-layer"
    assert circuit.num_qubits == n_sites
    assert "trotter_ite" in trotter_circuit.__module__

    from qiskit.quantum_info import Operator

    matrix = Operator(circuit).data
    assert np.allclose(matrix @ matrix.conj().T, np.eye(1 << n_sites), atol=1e-10)


def test_structural_circuit_depth_grows_linearly_in_size() -> None:
    """The additive case has no two-qubit gates, so depth should not grow with L at all
    beyond the fixed layer structure. Recorded because the planning documents make a depth
    claim and it should be measured rather than repeated."""
    depths = [trotter_circuit(n, np.full(n, 1.0), 0.3, 0.1).depth() for n in (2, 4, 8, 12)]
    assert len(set(depths)) == 1, f"expected constant depth without couplings, got {depths}"
