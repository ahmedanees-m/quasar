"""Noise models, the sampling pipeline, and the decode that G-R.8 turned up.

Two things here would be silent if wrong. Endianness survives a round trip through
transpilation and sampling, which no accuracy number would reveal because a permuted
distribution is still a distribution. And the square-root decode inverts the encoding, which
matters because the undecoded measurement scores 0.987 on cosine while being the wrong
object entirely.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit

from quasarstack.backends.execution import (
    assignment_matrix,
    mitigate_readout,
    resource_report,
    sample_distribution,
    transpile_for,
)
from quasarstack.backends.hardware import (
    HERON_LIKE,
    TRAPPED_ION_LIKE,
    build_noise_model,
    coupling_map,
)
from quasarstack.io.conventions import decode_from_measurement, genotype_to_index
from quasarstack.scoring.metrics import cosine_similarity, total_variation

pytestmark = pytest.mark.fast


def test_both_device_models_build_a_noise_model() -> None:
    for device in (HERON_LIKE, TRAPPED_ION_LIKE):
        model = build_noise_model(device, 3)
        assert model.noise_instructions, f"{device.name} produced an empty noise model"


def test_connectivity_differs_between_the_device_classes() -> None:
    """Not a detail: restricted connectivity forces swaps, and swaps are two-qubit gates."""
    assert coupling_map(TRAPPED_ION_LIKE, 4) is None
    chain = coupling_map(HERON_LIKE, 4)
    assert chain is not None and [0, 1] in chain


def test_the_superconducting_model_pays_for_connectivity_it_does_not_have() -> None:
    """Restricted connectivity costs two-qubit gates, and that cost is real.

    A star, where one qubit talks to every other, cannot be embedded in a chain without
    routing. A single long-range gate is not enough to show this: with only two active
    qubits the transpiler simply places them next to each other, which is why the first
    version of this test compared 1 against 1.
    """
    n_sites = 5
    circuit = QuantumCircuit(n_sites)
    circuit.h(0)
    for target in range(1, n_sites):
        circuit.cx(0, target)

    chain = resource_report(transpile_for(circuit, HERON_LIKE))
    ion = resource_report(transpile_for(circuit, TRAPPED_ION_LIKE))
    assert (
        chain["two_qubit_gates"] > ion["two_qubit_gates"]
    ), f"a star on a chain should need routing: chain {chain}, ion {ion}"


@pytest.mark.parametrize("genotype", ["100", "001", "110"])
def test_endianness_survives_the_whole_pipeline(genotype: str) -> None:
    """The test that no accuracy number would catch.

    A basis state is prepared, transpiled, sampled and decoded. If the bitstring convention
    were wrong anywhere along that path, the result would be a permuted distribution:
    non-negative, normalised, and pointing at the wrong genotype.
    """
    n_sites = len(genotype)
    circuit = QuantumCircuit(n_sites)
    for site, value in enumerate(genotype):
        if value == "1":
            circuit.x(site)

    probs = sample_distribution(
        transpile_for(circuit, HERON_LIKE), HERON_LIKE, shots=2000, seed=0, noiseless=True
    )
    assert int(np.argmax(probs)) == genotype_to_index(genotype)
    assert probs[genotype_to_index(genotype)] == pytest.approx(1.0, abs=1e-12)


def test_the_measured_assignment_matrix_is_nearly_the_identity() -> None:
    """Readout error is small, so calibration should be diagonally dominant. It is measured
    from circuits rather than copied from the device parameters, so it carries shot noise."""
    matrix = assignment_matrix(HERON_LIKE, 2, shots=4000, seed=0)
    assert matrix.shape == (4, 4)
    assert np.allclose(matrix.sum(axis=0), 1.0, atol=1e-12)
    assert np.min(np.diag(matrix)) > 0.9


def test_mitigation_returns_a_genuine_distribution() -> None:
    """A plain inverse routinely returns negative entries once shot noise is comparable to
    the correction. Clipping them away biases the result, so the solve is constrained."""
    assignment = np.array([[0.98, 0.03], [0.02, 0.97]])
    observed = np.array([0.6, 0.4])
    corrected = mitigate_readout(observed, assignment)
    assert (corrected >= 0).all()
    assert corrected.sum() == pytest.approx(1.0)


def test_mitigation_recovers_a_known_distortion() -> None:
    assignment = np.array([[0.9, 0.1], [0.1, 0.9]])
    truth = np.array([0.7, 0.3])
    corrected = mitigate_readout(assignment @ truth, assignment)
    assert np.allclose(corrected, truth, atol=1e-8)


def test_the_decode_inverts_squaring() -> None:
    """The finding behind G-R.8: the circuit holds the distribution in its amplitudes, so a
    measurement returns it squared."""
    rng = np.random.default_rng(0)
    quasispecies = rng.random(16)
    quasispecies /= quasispecies.sum()

    measured = quasispecies**2 / (quasispecies**2).sum()
    assert np.allclose(decode_from_measurement(measured), quasispecies, atol=1e-12)


def test_the_undecoded_measurement_looks_fine_on_cosine_and_is_not() -> None:
    """Why total variation is reported beside cosine everywhere in this project.

    A concentrated distribution and its square are close in direction and far apart as
    distributions, so the flattering metric passes and the conservative one does not.
    """
    quasispecies = np.array([0.70, 0.15, 0.10, 0.05])
    measured = quasispecies**2 / (quasispecies**2).sum()
    assert cosine_similarity(measured, quasispecies) > 0.95
    assert total_variation(measured, quasispecies) > 0.1
    assert total_variation(decode_from_measurement(measured), quasispecies) < 1e-12


def test_empty_bins_are_handled_rather_than_silently_asserting_impossibility() -> None:
    measured = np.array([0.5, 0.5, 0.0, 0.0])
    assert decode_from_measurement(measured)[2] == 0.0
    softened = decode_from_measurement(measured, floor=1e-6)
    assert softened[2] > 0.0


def test_negative_frequencies_are_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        decode_from_measurement(np.array([0.6, -0.1, 0.5]))
