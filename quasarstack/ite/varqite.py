"""Route A: variational imaginary-time evolution by McLachlan's principle.

Imaginary-time evolution is non-unitary. varQITE keeps a fixed ansatz and moves only its
parameters along the imaginary-time trajectory, so the circuit depth is constant in imaginary
time.

The equations
-------------

For a normalised state, imaginary-time evolution is

    d|psi>/dtau = -(H - <H>) |psi>

McLachlan's variational principle projects this onto the tangent space of the ansatz and gives
the linear system

    A theta_dot = C

The ansatz uses Ry rotations and CNOTs from the all-zero state, so every amplitude is real. The
derivative of Ry inserts ``-i Y / 2``, a real matrix, so the derivative states are real and

    A_ij = <d_i psi | d_j psi>          the quantum geometric tensor
    C_i  = -<d_i psi | H | psi>

using ``<d_i psi | psi> = 0``, which holds for real normalised states and is asserted.

Hardware measurement
--------------------

``C_i = -(1/2) d_i <H>``, obtained from the parameter-shift rule with two circuit evaluations
per parameter. ``A`` is the Fubini-Study metric, obtained from four fidelity evaluations per
pair. This module computes both by differentiating the state vector, which is exact and cheaper
at these sizes; `verify_hardware_route` computes them through the shift rules and compares.

Energy descent
--------------

Along the continuous flow,

    dE/dtau = grad(E) . theta_dot = -(1/2) grad(E)^T (A + delta I)^-1 grad(E) <= 0

since ``A`` is a Gram matrix and the ridge makes ``A + delta I`` positive definite. An energy
increase can only come from the explicit Euler integrator at finite step size, and shrinks with
the step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

# Ridge term on the geometric tensor. A is routinely singular, since an ansatz has directions that
# do not change the state, and an unregularised solve would diverge along them.
DEFAULT_RIDGE = 1e-6


@dataclass(frozen=True)
class Ansatz:
    """A fixed real-amplitude circuit: Ry layers separated by a CNOT chain.

    The quasispecies is the ground state of a stoquastic operator and is sign definite, so a real
    ansatz can represent it, and real amplitudes give McLachlan's equations their simplest form.
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

        This state has non-zero overlap with the strictly positive Perron vector. The first Ry layer is a
        quarter turn and later layers are zero, so the CNOTs act trivially.
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

    ``C_i = -<d_i psi|H|psi>``, minus half the energy gradient. Computing selected components avoids
    building every derivative state and the full Gram matrix when only one component is needed, as in
    the gradient-variance diagnostic.
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
    """Run varQITE until the state stops moving, or until ``tau``.

    Parameters
    ----------
    tau
        Ceiling on imaginary time. Reaching it without converging is reported.
    tolerance
        Stop once ``||psi_new - psi_prev|| / dtau`` falls below this. A rate rather than a per-step
        change, so that the imaginary time reached does not depend on the step size.

    Notes
    -----
    Convergence is judged on the state, not the parameters: the ansatz has gauge directions in
    which parameters move without changing the state, so parameter updates keep fluctuating after
    the state has settled. On a rugged L = 4 instance the state reaches cosine 0.99991 at step 400
    while the largest parameter update is still 5.1e-2.

    `tau_used` is reported with the accuracy, since the imaginary time needed scales as one over
    the spectral gap.
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
    """Recompute ``A`` and ``C`` through the shift rules and compare.

    ``C`` comes from the parameter-shift rule on the energy, two evaluations per parameter, and
    ``A`` from the fidelity-shift rule, four evaluations per pair. Neither uses a derivative state,
    so both are available from circuit measurements alone.
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
