"""Route A, primary: variational imaginary-time evolution by McLachlan's principle.

Why this method exists in the project
-------------------------------------

Imaginary-time evolution is non-unitary, so it cannot be run as a circuit directly. varQITE
sidesteps that by never changing the circuit: a fixed ansatz is held, and only its parameters
move, steered along the imaginary-time trajectory. **The circuit depth is therefore constant
in imaginary time**, which is the single property that makes the method usable on near-term
hardware, and it is what gate G-R.6 measures.

The equations
-------------

For a normalised state, imaginary-time evolution is

    d|psi>/dtau = -(H - <H>) |psi>

McLachlan's variational principle projects that onto the tangent space of the ansatz,
minimising the residual, and yields a linear system

    A theta_dot = C

The ansatz here uses Ry rotations and CNOTs from the all-zero state, so **every amplitude
stays real**. The derivative of Ry inserts ``-i Y / 2``, which is itself a real matrix, so
the derivative states are real too. That collapses the general complex expressions to

    A_ij = <d_i psi | d_j psi>          the quantum geometric tensor
    C_i  = -<d_i psi | H | psi>

using ``<d_i psi | psi> = 0``, which follows from differentiating the normalisation and holds
exactly for real states. The code asserts that overlap rather than assuming it.

Both quantities are measurable on hardware
------------------------------------------

This matters, because a method that only works given a state vector is not a near-term
method and should not be presented as one.

``C`` is the ordinary energy gradient in disguise. Differentiating ``<H>`` gives
``d_i <H> = 2 Re <d_i psi|H|psi>``, so ``C_i = -(1/2) d_i <H>``, which the parameter-shift
rule returns from two circuit evaluations per parameter. ``A`` is the Fubini-Study metric,
obtainable from four fidelity evaluations per pair by the same shift rule.

Inside this module both are computed by differentiating the state vector, because at the
sizes these gates run that is exact and vastly cheaper than 4P^2 circuit evaluations per
step. `verify_hardware_route` computes the same two objects through the parameter-shift and
fidelity-shift formulas and compares, so the claim that they are measurable is checked
rather than asserted. Gate G-R.6 records that comparison.

The energy must descend, and where it does not
----------------------------------------------

The continuous flow cannot raise the energy, and this is derivable rather than hoped for.
Along the trajectory,

    dE/dtau = grad(E) . theta_dot = grad(E) . (A + delta I)^-1 C

and since ``C = -(1/2) grad(E)``, that is ``-(1/2) grad(E)^T (A + delta I)^-1 grad(E)``,
which is non-positive because ``A`` is a Gram matrix, hence positive semidefinite, and the
ridge makes the sum positive definite. The regularisation therefore cannot break descent; it
only shortens the step in the directions the ansatz barely moves.

What *can* raise the energy is the explicit Euler integrator overshooting at finite step
size, and gate G-R.6 measures whether that is what is happening rather than assuming it. An
energy that ascends and does not shrink with the step size would be a bug of the kind the
planning documents record for the sibling Motta method; one that shrinks with the step size
is discretisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

# Ridge term on the geometric tensor. A is routinely singular, because an ansatz almost
# always has directions that do not change the state, and an exactly singular solve would
# make the trajectory explode rather than ignore those directions. Declared here so the
# regularisation is a stated choice and appears in every result record.
DEFAULT_RIDGE = 1e-6


@dataclass(frozen=True)
class Ansatz:
    """A fixed real-amplitude circuit: Ry layers separated by a CNOT chain.

    Real amplitudes are not incidental. The quasispecies is the ground state of a stoquastic
    operator and is sign-definite, so a real ansatz can represent it exactly, and keeping the
    state real is what reduces McLachlan's equations to their simplest form. See
    `DECISIONS.md` ADR-0003.
    """

    n_sites: int
    reps: int = 2
    _gates: list[tuple[str, tuple[int, ...]]] = field(default_factory=list, compare=False)

    def __post_init__(self) -> None:
        gates: list[tuple[str, tuple[int, ...]]] = []
        for _ in range(self.reps):
            for qubit in range(self.n_sites):
                gates.append(("ry", (qubit,)))
            for qubit in range(self.n_sites - 1):
                gates.append(("cx", (qubit, qubit + 1)))
        for qubit in range(self.n_sites):
            gates.append(("ry", (qubit,)))
        object.__setattr__(self, "_gates", gates)

    @property
    def n_parameters(self) -> int:
        return sum(1 for name, _ in self._gates if name == "ry")

    def initial_parameters(self) -> NDArray[np.float64]:
        """Parameters giving the uniform superposition over all genotypes.

        The maximally uninformative starting population, and the one with guaranteed non-zero
        overlap with the Perron vector, which is strictly positive. The first Ry layer is set
        to a quarter turn and everything after it to zero, so the CNOTs act trivially.
        """
        params = np.zeros(self.n_parameters, dtype=np.float64)
        params[: self.n_sites] = np.pi / 2
        return params

    def state(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        """The real state vector this circuit prepares."""
        return self._run(params, derivative_index=None)

    def derivative(self, params: NDArray[np.float64], index: int) -> NDArray[np.float64]:
        """d|psi>/d(theta_index), by inserting the rotation's generator in place."""
        return self._run(params, derivative_index=index)

    def circuit(self, params: NDArray[np.float64] | None = None) -> QuantumCircuit:
        """The Qiskit circuit, for transpilation and resource reporting."""
        values = ParameterVector("t", self.n_parameters)
        circuit = QuantumCircuit(self.n_sites, name=f"varqite_r{self.reps}")
        cursor = 0
        for name, qubits in self._gates:
            if name == "ry":
                circuit.ry(values[cursor], qubits[0])
                cursor += 1
            else:
                circuit.cx(*qubits)
        if params is not None:
            circuit = circuit.assign_parameters(dict(zip(values, params, strict=True)))
        return circuit

    def _run(
        self, params: NDArray[np.float64], derivative_index: int | None
    ) -> NDArray[np.float64]:
        state = np.zeros(1 << self.n_sites, dtype=np.float64)
        state[0] = 1.0
        cursor = 0
        for name, qubits in self._gates:
            if name == "cx":
                state = _apply_cx(state, qubits[0], qubits[1], self.n_sites)
                continue
            angle = float(params[cursor])
            state = _apply_ry(state, qubits[0], angle, self.n_sites)
            if cursor == derivative_index:
                # d/dtheta Ry(theta) = Ry(theta) . (-i Y / 2), and -i Y / 2 is real.
                state = _apply_generator(state, qubits[0], self.n_sites)
            cursor += 1
        return state


def _reshape(state: NDArray[np.float64], qubit: int, n_sites: int) -> NDArray[np.float64]:
    """View the state with the given qubit on the middle axis. Site i occupies bit i."""
    return state.reshape(1 << (n_sites - qubit - 1), 2, 1 << qubit)


def _apply_ry(
    state: NDArray[np.float64], qubit: int, angle: float, n_sites: int
) -> NDArray[np.float64]:
    cos, sin = float(np.cos(angle / 2)), float(np.sin(angle / 2))
    view = _reshape(state, qubit, n_sites)
    low, high = view[:, 0, :], view[:, 1, :]
    return np.concatenate(
        [(cos * low - sin * high)[:, None, :], (sin * low + cos * high)[:, None, :]], axis=1
    ).reshape(-1)


def _apply_generator(state: NDArray[np.float64], qubit: int, n_sites: int) -> NDArray[np.float64]:
    """Apply ``-i Y / 2``, which is the real matrix [[0, -1/2], [1/2, 0]]."""
    view = _reshape(state, qubit, n_sites)
    low, high = view[:, 0, :], view[:, 1, :]
    return np.concatenate([(-0.5 * high)[:, None, :], (0.5 * low)[:, None, :]], axis=1).reshape(-1)


def _apply_cx(
    state: NDArray[np.float64], control: int, target: int, n_sites: int
) -> NDArray[np.float64]:
    index = np.arange(state.size)
    flipped = np.where((index >> control) & 1, index ^ (1 << target), index)
    return state[flipped]


def geometric_tensor_and_force(
    ansatz: Ansatz, params: NDArray[np.float64], hamiltonian: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Return ``A``, ``C`` and the current energy.

    ``hamiltonian`` is the dense real matrix of the stoquastic operator whose ground state is
    the quasispecies.
    """
    state = ansatz.state(params)
    derivatives = np.array([ansatz.derivative(params, i) for i in range(ansatz.n_parameters)])

    overlaps = derivatives @ state
    if np.max(np.abs(overlaps)) > 1e-9:
        raise AssertionError(
            f"<d_i psi|psi> should vanish for a normalised real state, got "
            f"{np.max(np.abs(overlaps)):.2e}; the ansatz or the derivative is wrong"
        )

    tensor = derivatives @ derivatives.T
    h_state = hamiltonian @ state
    force = -(derivatives @ h_state)
    energy = float(state @ h_state)
    return tensor, force, energy


def force_components(
    ansatz: Ansatz,
    params: NDArray[np.float64],
    hamiltonian: NDArray[np.float64],
    indices: list[int] | None = None,
) -> NDArray[np.float64]:
    """The McLachlan force for selected parameters only.

    ``C_i = -<d_i psi|H|psi>``, which is minus half the energy gradient. Computing only the
    components asked for matters for the barren-plateau diagnostic, which needs one component
    over many random parameter draws: going through the full geometric tensor would build
    every derivative state and the whole Gram matrix to use one number, which at L = 8 with
    eighty parameters is eighty times the work.
    """
    h_state = hamiltonian @ ansatz.state(params)
    wanted = range(ansatz.n_parameters) if indices is None else indices
    return np.array([-(ansatz.derivative(params, i) @ h_state) for i in wanted])


def step(
    ansatz: Ansatz,
    params: NDArray[np.float64],
    hamiltonian: NDArray[np.float64],
    dtau: float,
    ridge: float = DEFAULT_RIDGE,
) -> tuple[NDArray[np.float64], float]:
    """One explicit Euler step of the McLachlan flow. Returns new parameters and the energy."""
    tensor, force, energy = geometric_tensor_and_force(ansatz, params, hamiltonian)
    update = np.linalg.solve(tensor + ridge * np.eye(tensor.shape[0]), force)
    return params + dtau * update, energy


@dataclass
class Evolution:
    """The outcome of a varQITE run, including what it cost to get there."""

    probs: NDArray[np.float64]
    params: NDArray[np.float64]
    energies: list[float]
    tau_used: float
    steps: int
    converged: bool
    final_state_rate: float
    final_state_change: float
    final_parameter_change: float


def evolve(
    ansatz: Ansatz,
    hamiltonian: NDArray[np.float64],
    tau: float,
    dtau: float,
    ridge: float = DEFAULT_RIDGE,
    params: NDArray[np.float64] | None = None,
    tolerance: float = 1e-10,
) -> Evolution:
    """Run varQITE until it stops moving, or until ``tau``, whichever comes first.

    Parameters
    ----------
    tau
        Ceiling on imaginary time. Reaching it without converging is reported, not hidden.
    tolerance
        Stop once the **state stops moving**, judged as a rate:
        ``||psi_new - psi_prev|| / dtau`` below this.

        A rate, not a per-step change, and the distinction is not pedantic. A per-step
        criterion trips sooner at a smaller step purely because each step moves less, so the
        imaginary time reached would depend on the step size. Measured on Motta-QITE, whose
        stopping rule was the same: accuracy *fell* from 0.9999997 to 0.9999731 as dtau went
        from 0.1 to 0.01, because the finer run stopped earlier in tau rather than because it
        was worse. Since `tau_used` is the budget-needed-for-accuracy number ADR-0013 asks
        WP7 to compare across methods, a step-size-dependent one would be actively
        misleading.

    Notes
    -----
    Convergence is judged on the state, not on the parameters, and the difference is not
    cosmetic. The ansatz has gauge directions, combinations of parameters that move without
    changing the state at all, so the parameter update norm keeps fluctuating long after the
    state has settled. Measured on a rugged L = 4 instance: at step 400 the state was already
    at cosine 0.99991 against the reference while the largest parameter update was still
    5.1e-2, and it was still 6.2e-2 at step 200 when the state was at 0.99955. A
    parameter-space criterion would either never trigger or trigger at an arbitrary moment
    determined by where the gauge drift happened to be.

    Early stopping is not only a speed measure. ADR-0013 records that a fixed imaginary-time
    budget quietly disadvantages the method on small-gap instances, because the time needed
    scales as one over the spectral gap. Reporting `tau_used` alongside the accuracy is the
    "budget needed for accuracy" half of the fairness protocol recommended there, and it is
    the number WP7 will want per cell.
    """
    ratio = tau / dtau
    max_steps = int(round(ratio))
    if abs(ratio - max_steps) > 1e-9:
        raise ValueError(f"tau / dtau must be a whole number, got {ratio}")

    current = ansatz.initial_parameters() if params is None else np.array(params, float)
    state = ansatz.state(current)
    energies: list[float] = []
    state_rate = float("inf")
    state_change = float("inf")
    parameter_change = float("inf")
    steps = 0

    for _ in range(max_steps):
        previous_params, previous_state = current, state
        current, energy = step(ansatz, current, hamiltonian, dtau, ridge)
        state = ansatz.state(current)
        energies.append(energy)
        steps += 1
        state_rate = float(np.linalg.norm(state - previous_state)) / dtau
        state_change = 1.0 - abs(float(previous_state @ state))
        parameter_change = float(np.max(np.abs(current - previous_params)))
        if state_rate < tolerance:
            break

    probs = np.abs(state)
    total = float(probs.sum())
    if total <= 0.0:
        raise ValueError("varQITE produced a state with zero total weight")

    return Evolution(
        probs=probs / total,
        params=current,
        energies=energies,
        tau_used=steps * dtau,
        steps=steps,
        converged=state_rate < tolerance,
        final_state_rate=state_rate,
        final_state_change=state_change,
        final_parameter_change=parameter_change,
    )


def verify_hardware_route(
    ansatz: Ansatz,
    params: NDArray[np.float64],
    hamiltonian: NDArray[np.float64],
) -> dict[str, float]:
    """Recompute ``A`` and ``C`` the way hardware would, and compare.

    ``C`` from the parameter-shift rule on the energy, two evaluations per parameter.
    ``A`` from the fidelity-shift rule, four evaluations per pair. Neither touches a
    derivative state, so both are available from circuit measurements alone.

    This is what turns "the method is hardware-faithful" from a claim into a measurement.
    """
    tensor, force, _ = geometric_tensor_and_force(ansatz, params, hamiltonian)
    n = ansatz.n_parameters
    half = np.pi / 2

    def energy_at(theta: NDArray[np.float64]) -> float:
        state = ansatz.state(theta)
        return float(state @ hamiltonian @ state)

    def fidelity(theta_a: NDArray[np.float64], theta_b: NDArray[np.float64]) -> float:
        return float((ansatz.state(theta_a) @ ansatz.state(theta_b)) ** 2)

    shift_force = np.zeros(n)
    for i in range(n):
        plus, minus = params.copy(), params.copy()
        plus[i] += half
        minus[i] -= half
        # C_i = -(1/2) d_i <H>, and the shift rule gives d_i <H> as half the difference.
        shift_force[i] = -0.5 * (energy_at(plus) - energy_at(minus)) / 2.0

    shift_tensor = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            terms = 0.0
            for si in (+1, -1):
                for sj in (+1, -1):
                    theta = params.copy()
                    theta[i] += si * half
                    theta[j] += sj * half
                    terms += -si * sj * fidelity(params, theta)
            shift_tensor[i, j] = shift_tensor[j, i] = terms / 8.0

    return {
        "force_max_abs_error": float(np.max(np.abs(shift_force - force))),
        "tensor_max_abs_error": float(np.max(np.abs(shift_tensor - tensor))),
        "force_scale": float(np.max(np.abs(force))),
        "tensor_scale": float(np.max(np.abs(tensor))),
    }
