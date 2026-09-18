"""Simulated device noise models.

Results produced through this module are simulations; the hardware run uses
`experiments/wp8_live_qpu/qpu_sweep.py` and reports job identifiers.

Two device classes
------------------

**Superconducting, IBM Heron class.** Fast gates and short coherence. Two-qubit error
dominates, connectivity is limited so the transpiler inserts swaps, and readout error is about
one percent.

**Trapped ion.** Slow gates and long coherence, all-to-all connectivity, and readout an order
of magnitude better. The limiting factor is gate duration against circuit depth.

Parameters are representative of each device class, from published typical figures, not a
calibration snapshot of a named device.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    depolarizing_error,
    thermal_relaxation_error,
)


@dataclass(frozen=True)
class DeviceModel:
    """Representative parameters for a device class. Times in seconds."""

    name: str
    basis_gates: list[str]
    single_qubit_error: float
    two_qubit_error: float
    single_qubit_duration: float
    two_qubit_duration: float
    t1: float
    t2: float
    readout_error_0_as_1: float
    readout_error_1_as_0: float
    coupling: str
    provenance: str = field(
        default="representative of the device class, not a calibration snapshot"
    )

    def readout_matrix(self) -> list[list[float]]:
        """Single-qubit assignment matrix, rows are prepared states."""
        return [
            [1.0 - self.readout_error_0_as_1, self.readout_error_0_as_1],
            [self.readout_error_1_as_0, 1.0 - self.readout_error_1_as_0],
        ]


HERON_LIKE = DeviceModel(
    name="superconducting_heron_like",
    basis_gates=["rz", "sx", "x", "cz"],
    single_qubit_error=2.0e-4,
    two_qubit_error=3.0e-3,
    single_qubit_duration=32e-9,
    two_qubit_duration=68e-9,
    t1=250e-6,
    t2=150e-6,
    readout_error_0_as_1=1.0e-2,
    readout_error_1_as_0=1.5e-2,
    coupling="linear nearest neighbour, so the transpiler inserts swaps",
)

TRAPPED_ION_LIKE = DeviceModel(
    name="trapped_ion_like",
    basis_gates=["rz", "rx", "ry", "rxx"],
    single_qubit_error=1.0e-4,
    two_qubit_error=2.0e-3,
    single_qubit_duration=10e-6,
    two_qubit_duration=200e-6,
    t1=10.0,
    t2=1.0,
    readout_error_0_as_1=2.0e-3,
    readout_error_1_as_0=2.0e-3,
    coupling="all to all, so no swaps are needed",
)


def build_noise_model(device: DeviceModel, n_qubits: int) -> NoiseModel:
    """Assemble an Aer noise model: depolarising, thermal relaxation and readout error.

    Depolarising and thermal relaxation are composed: depolarising alone ignores circuit duration,
    and relaxation alone ignores gate error.
    """
    model = NoiseModel(basis_gates=device.basis_gates)

    single_qubit_gates = [g for g in device.basis_gates if g not in {"cz", "cx", "rxx"}]
    two_qubit_gates = [g for g in device.basis_gates if g in {"cz", "cx", "rxx"}]

    one = depolarizing_error(device.single_qubit_error, 1).compose(
        thermal_relaxation_error(device.t1, device.t2, device.single_qubit_duration)
    )
    model.add_all_qubit_quantum_error(one, single_qubit_gates)

    if two_qubit_gates:
        relaxation = thermal_relaxation_error(
            device.t1, device.t2, device.two_qubit_duration
        ).expand(thermal_relaxation_error(device.t1, device.t2, device.two_qubit_duration))
        two = depolarizing_error(device.two_qubit_error, 2).compose(relaxation)
        model.add_all_qubit_quantum_error(two, two_qubit_gates)

    readout = ReadoutError(device.readout_matrix())
    for qubit in range(n_qubits):
        model.add_readout_error(readout, [qubit])

    return model


def coupling_map(device: DeviceModel, n_qubits: int) -> list[list[int]] | None:
    """Linear chain for the superconducting class, unrestricted for trapped ions.

    Restricted connectivity makes the transpiler insert swaps, which add two-qubit gates.
    """
    if device.coupling.startswith("all to all"):
        return None
    return [[i, i + 1] for i in range(n_qubits - 1)] + [[i + 1, i] for i in range(n_qubits - 1)]
