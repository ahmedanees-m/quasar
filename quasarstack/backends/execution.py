"""Transpile, sample, mitigate, decode.

**Endianness.** Qiskit returns counts keyed by little-endian bitstrings; reading one directly
as a genotype reverses every sequence. Every conversion goes through
`quasarstack.io.conventions`.

**Readout mitigation.** Inverting the assignment matrix produces negative probabilities when
shot noise is comparable to the correction, and clipping biases the result. The correction here
is a constrained least-squares projection onto the probability simplex.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from quasarstack.backends.hardware import DeviceModel, build_noise_model, coupling_map
from quasarstack.io.conventions import genotype_to_index, qiskit_bitstring_to_genotype


def transpile_for(
    circuit: QuantumCircuit, device: DeviceModel, optimisation_level: int = 1, seed: int = 0
) -> QuantumCircuit:
    """Transpile to the device basis and connectivity."""
    return transpile(
        circuit,
        basis_gates=device.basis_gates,
        coupling_map=coupling_map(device, circuit.num_qubits),
        optimization_level=optimisation_level,
        seed_transpiler=seed,
    )


def resource_report(circuit: QuantumCircuit) -> dict[str, int]:
    """Depth and two-qubit gate count.

    Two-qubit gates dominate the error budget on both device classes.
    """
    counts = circuit.count_ops()
    two_qubit = sum(n for gate, n in counts.items() if gate in {"cz", "cx", "rxx", "swap", "ecr"})
    return {
        "depth": int(circuit.depth()),
        "two_qubit_gates": int(two_qubit),
        "total_gates": int(sum(counts.values())),
    }


def sample_distribution(
    circuit: QuantumCircuit,
    device: DeviceModel,
    shots: int,
    seed: int,
    noiseless: bool = False,
) -> NDArray[np.float64]:
    """Run the circuit and decode counts into a genotype distribution.

    Entry j of the returned array is the genotype whose site i is mutated exactly when bit i of j
    is set.
    """
    measured = circuit.copy()
    measured.measure_all()

    n_qubits = circuit.num_qubits
    simulator = AerSimulator(
        noise_model=None if noiseless else build_noise_model(device, n_qubits),
        seed_simulator=seed,
    )
    prepared = transpile(measured, simulator, optimization_level=0)
    counts = simulator.run(prepared, shots=shots).result().get_counts()

    probs = np.zeros(1 << n_qubits, dtype=np.float64)
    for bitstring, count in counts.items():
        genotype = qiskit_bitstring_to_genotype(bitstring)
        probs[genotype_to_index(genotype)] += count
    normalised: NDArray[np.float64] = probs / probs.sum()
    return normalised


def assignment_matrix(
    device: DeviceModel, n_qubits: int, shots: int, seed: int
) -> NDArray[np.float64]:
    """Estimate the readout assignment matrix from calibration circuits.

    Each computational basis state is prepared and measured, so the estimate carries the shot
    noise of a real calibration rather than the exact device parameters.
    """
    dimension = 1 << n_qubits
    matrix = np.zeros((dimension, dimension), dtype=np.float64)
    for prepared_index in range(dimension):
        circuit = QuantumCircuit(n_qubits)
        for qubit in range(n_qubits):
            if prepared_index >> qubit & 1:
                circuit.x(qubit)
        observed = sample_distribution(circuit, device, shots=shots, seed=seed + prepared_index)
        matrix[:, prepared_index] = observed
    return matrix


def mitigate_readout(
    observed: NDArray[np.float64], assignment: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Undo readout error by projecting onto the closest distribution.

    Solves ``min_p ||A p - observed||`` subject to ``p >= 0`` and ``sum p = 1``.
    """
    from scipy.optimize import nnls

    dimension = observed.size
    # Append the normalisation as a weighted row, so the non-negative solve enforces it too.
    weight = float(np.abs(assignment).max()) * 10.0
    stacked = np.vstack([assignment, np.full((1, dimension), weight)])
    target = np.concatenate([observed, [weight]])

    solution, _ = nnls(stacked, target)
    total = float(solution.sum())
    if total <= 0.0:
        raise ValueError("readout mitigation collapsed the distribution to zero")
    corrected: NDArray[np.float64] = solution / total
    return corrected


def run_pipeline(
    circuit: QuantumCircuit,
    device: DeviceModel,
    shots: int,
    seed: int,
    calibration_shots: int | None = None,
) -> dict[str, object]:
    """Transpile, sample noiselessly and noisily, mitigate, and report resources.

    Returns raw and mitigated distributions side by side.
    """
    transpiled = transpile_for(circuit, device)
    resources = resource_report(transpiled)

    noiseless = sample_distribution(transpiled, device, shots, seed, noiseless=True)
    raw = sample_distribution(transpiled, device, shots, seed + 1)

    calibration = assignment_matrix(
        device, circuit.num_qubits, calibration_shots or shots, seed + 1000
    )
    mitigated = mitigate_readout(raw, calibration)

    return {
        "device": device.name,
        "simulated": True,
        "resources": resources,
        "shots": shots,
        "noiseless": noiseless,
        "raw": raw,
        "mitigated": mitigated,
        "assignment_matrix_diagonal_min": float(np.min(np.diag(calibration))),
    }
