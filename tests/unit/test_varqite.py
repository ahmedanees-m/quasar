"""varQITE: the ansatz, McLachlan's equations, and the claim that hardware could run it.

The load-bearing test here is `test_the_hardware_route_reproduces_both_quantities`. The
module computes A and C by differentiating a state vector, which no quantum computer can do.
That is only legitimate because the same two objects are obtainable from circuit
measurements, and the test is what makes that a fact about this code rather than a
statement about the literature.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from quasarstack.analytic.crow_kimura import additive_quasispecies
from quasarstack.classical.landscapes import additive_fitness
from quasarstack.hamiltonian.builder import additive_hamiltonian
from quasarstack.ite.varqite import (
    Ansatz,
    evolve,
    geometric_tensor_and_force,
    verify_hardware_route,
)
from quasarstack.scoring.metrics import cosine_similarity

pytestmark = pytest.mark.fast


def _matrix(a: np.ndarray, mu: float) -> np.ndarray:
    return np.asarray(additive_hamiltonian(a, mu).to_matrix()).real


@pytest.mark.parametrize(("n_sites", "reps"), [(1, 1), (2, 1), (3, 2), (4, 2)])
def test_the_numpy_state_matches_the_qiskit_circuit(n_sites: int, reps: int) -> None:
    """The fast path and the circuit must be the same object.

    Everything downstream computes on the numpy state while the resource numbers come from
    the circuit. If the two ever disagreed, the accuracy and the depth would be describing
    different things.
    """
    ansatz = Ansatz(n_sites, reps=reps)
    params = np.random.default_rng(0).normal(size=ansatz.n_parameters)
    from_circuit = np.asarray(Statevector.from_instruction(ansatz.circuit(params)).data)
    assert np.max(np.abs(from_circuit.imag)) < 1e-14, "the ansatz must keep the state real"
    assert np.allclose(ansatz.state(params), from_circuit.real, atol=1e-13)


def test_initial_parameters_give_the_uniform_superposition() -> None:
    ansatz = Ansatz(4, reps=2)
    state = ansatz.state(ansatz.initial_parameters())
    assert np.allclose(state, 1.0 / 4.0, atol=1e-14)


def test_the_state_stays_normalised() -> None:
    ansatz = Ansatz(3, reps=3)
    params = np.random.default_rng(1).normal(size=ansatz.n_parameters)
    assert np.linalg.norm(ansatz.state(params)) == pytest.approx(1.0, abs=1e-13)


def test_derivatives_match_finite_differences() -> None:
    """The analytic derivative against a numerical one. If the generator insertion were
    wrong, McLachlan's equations would integrate a different flow and still look plausible."""
    ansatz = Ansatz(3, reps=2)
    rng = np.random.default_rng(2)
    params = rng.normal(size=ansatz.n_parameters)
    eps = 1e-6
    for index in range(ansatz.n_parameters):
        plus, minus = params.copy(), params.copy()
        plus[index] += eps
        minus[index] -= eps
        numerical = (ansatz.state(plus) - ansatz.state(minus)) / (2 * eps)
        assert np.allclose(ansatz.derivative(params, index), numerical, atol=1e-7)


def test_derivative_overlap_with_the_state_vanishes() -> None:
    """The identity that collapses McLachlan's equations to their simple form. The module
    asserts it at run time; this checks the assertion is satisfiable rather than latent."""
    ansatz = Ansatz(4, reps=2)
    params = np.random.default_rng(3).normal(size=ansatz.n_parameters)
    state = ansatz.state(params)
    for index in range(ansatz.n_parameters):
        assert abs(float(ansatz.derivative(params, index) @ state)) < 1e-12


def test_the_geometric_tensor_is_symmetric_and_positive_semidefinite() -> None:
    ansatz = Ansatz(3, reps=2)
    params = np.random.default_rng(4).normal(size=ansatz.n_parameters)
    tensor, _, _ = geometric_tensor_and_force(ansatz, params, _matrix(np.ones(3), 0.3))
    assert np.allclose(tensor, tensor.T, atol=1e-14)
    assert np.linalg.eigvalsh(tensor).min() > -1e-12


@pytest.mark.parametrize(("n_sites", "reps"), [(2, 1), (3, 2)])
def test_the_hardware_route_reproduces_both_quantities(n_sites: int, reps: int) -> None:
    """The test that licenses calling this method hardware-faithful.

    C is recomputed from the parameter-shift rule on the energy and A from the
    fidelity-shift rule. Neither touches a derivative state, so both are available from
    circuit measurements alone.
    """
    rng = np.random.default_rng(5)
    a = rng.uniform(0.25, 2.0, size=n_sites)
    ansatz = Ansatz(n_sites, reps=reps)
    params = rng.normal(size=ansatz.n_parameters)
    report = verify_hardware_route(ansatz, params, _matrix(a, 0.3))
    assert report["force_max_abs_error"] < 1e-10
    assert report["tensor_max_abs_error"] < 1e-10


def test_energy_descends_monotonically() -> None:
    """Imaginary time minimises the energy. An ascending energy was one of the three bugs
    the planning documents record, in the sibling Motta method."""
    a = np.array([1.0, 0.6, 1.3])
    evolution = evolve(Ansatz(3, reps=3), _matrix(a, 0.3), tau=10.0, dtau=0.05)
    energies = evolution.energies
    assert all(
        later <= earlier + 1e-9 for earlier, later in zip(energies, energies[1:], strict=False)
    )


def test_it_reaches_the_analytic_quasispecies() -> None:
    a = np.array([1.0, 0.6, 1.3])
    mu = 0.3
    evolution = evolve(Ansatz(3, reps=3), _matrix(a, mu), tau=40.0, dtau=0.05)
    assert cosine_similarity(evolution.probs, additive_quasispecies(a, mu)) > 0.999
    assert evolution.probs.sum() == pytest.approx(1.0, abs=1e-14)
    assert (evolution.probs >= 0).all()


def test_depth_does_not_grow_with_imaginary_time() -> None:
    """The property that makes varQITE a near-term method, and the registered criterion of
    gate G-R.6. Only the parameters move; the circuit is the same circuit."""
    a = np.array([1.0, 0.6, 1.3])
    matrix = _matrix(a, 0.3)
    ansatz = Ansatz(3, reps=3)
    shapes = []
    for tau in (2.5, 20.0):
        evolution = evolve(ansatz, matrix, tau=tau, dtau=0.05)
        circuit = ansatz.circuit(evolution.params)
        shapes.append((circuit.depth(), circuit.count_ops().get("cx", 0)))
    assert shapes[0] == shapes[1], f"depth changed with imaginary time: {shapes}"


def test_convergence_is_judged_on_the_state_not_the_parameters() -> None:
    """The ansatz has gauge directions, so parameters keep drifting after the state settles.
    A converged run must therefore be allowed to report a parameter change far larger than
    its tolerance without that meaning anything is wrong."""
    a = np.array([1.0, 0.6, 1.3])
    evolution = evolve(Ansatz(3, reps=3), _matrix(a, 0.3), tau=60.0, dtau=0.05, tolerance=1e-6)
    assert evolution.converged
    assert evolution.final_state_rate < 1e-6
    assert evolution.tau_used < 60.0, "early stopping should have triggered"


def test_the_imaginary_time_reached_does_not_depend_on_the_step_size() -> None:
    """The stopping rule is a rate, so halving the step must not halve the budget reported.

    A per-step criterion trips sooner at a smaller step purely because each step moves less,
    which would make `tau_used` an artefact of the integrator. That number is what ADR-0013
    asks WP7 to compare across methods, so it has to mean the same thing at every step size.
    """
    a = np.array([1.0, 0.6, 1.3])
    matrix = _matrix(a, 0.3)
    reached = [
        evolve(Ansatz(3, reps=3), matrix, tau=60.0, dtau=dtau, tolerance=1e-6).tau_used
        for dtau in (0.05, 0.025, 0.01)
    ]
    assert (
        max(reached) / min(reached) < 1.5
    ), f"the imaginary time reached varied with the step size: {reached}"


def test_step_count_must_divide_the_budget() -> None:
    with pytest.raises(ValueError, match="whole number"):
        evolve(Ansatz(2, reps=1), _matrix(np.ones(2), 0.3), tau=1.0, dtau=0.3)


def test_zero_mutation_puts_the_population_on_the_fittest_genotype() -> None:
    a = np.array([1.0, -2.0, 0.5])
    evolution = evolve(Ansatz(3, reps=3), _matrix(a, 0.0), tau=40.0, dtau=0.05)
    fittest = int(np.argmax(additive_fitness(a)))
    assert evolution.probs[fittest] > 0.999
