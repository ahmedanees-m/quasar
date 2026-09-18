"""LCU block encoding of the mutation-selection operator.

A QSVT circuit acts on a block encoding: a unitary `U` on ancilla plus system registers whose
top-left block is the operator divided by a normalisation,

    ( <0|^m (x) I )  U  ( |0>^m (x) I )  =  A / alpha

G-2 checks this defining property to 1e-10.

Construction
------------

`A` is a sum of Pauli terms with real coefficients, `A = sum_j c_j P_j`. The linear combination
of unitaries uses `U = PREP_L^dagger . SELECT . PREP_R` with

    PREP_R |0> = sum_j sqrt(|c_j| / alpha) * sign(c_j) |j>
    PREP_L |0> = sum_j sqrt(|c_j| / alpha) |j>
    SELECT     = sum_j |j><j| (x) P_j

so the block is `sum_j c_j P_j / alpha` with `alpha = sum_j |c_j|`. Placing the signs in one
preparation lets negative coefficients through without a separate phase oracle.

`alpha`, the one-norm of the coefficients, sets the query complexity through `alpha / gap`. At
L = 12 the single peak has 4108 terms as a projector and 27 in sparse form, with the one-norm
following (`results/wp_r/g_r_10.json`).

Block extraction
----------------

Building the full `2^(m+n)` by `2^(m+n)` unitary would need 1 GB at L = 6 in the projector form.
Each column of the block is instead obtained by simulating `U` on `|0>^m |j>` and reading the
amplitudes with the ancillas in `|0>`: `2^n` statevector simulations with the same result to
machine precision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation, XGate, YGate, ZGate
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector

__all__ = [
    "BlockEncoding",
    "block_encoding_block",
    "encoding_qubit_count",
    "lcu_block_encoding",
    "one_norm",
    "verify_block_encoding",
]

# Coefficients below this are dropped: each would cost an ancilla index and a controlled Pauli
# while only increasing alpha.
COEFFICIENT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class BlockEncoding:
    """A block encoding and the numbers needed to read it.

    Attributes
    ----------
    circuit
        The unitary, on ``n_ancilla + n_system`` qubits. Ancillas are the low-index qubits,
        matching the little-endian convention in `quasarstack.io.conventions`.
    alpha
        Normalisation. The encoded operator is ``alpha`` times the top-left block.
    n_ancilla, n_system
        Register sizes.
    n_terms
        Pauli terms actually encoded, after dropping negligible coefficients.
    """

    circuit: QuantumCircuit
    alpha: float
    n_ancilla: int
    n_system: int
    n_terms: int

    @property
    def subnormalisation_cost(self) -> float:
        """``alpha``, which scales the query complexity.

        QSVT needs polynomial degree of order ``alpha / gap`` to resolve a gap in the unnormalised
        operator, because the block encoding divides by ``alpha``.
        """
        return self.alpha


def one_norm(operator: SparsePauliOp) -> float:
    """``alpha``, the sum of absolute Pauli coefficients, without building a circuit.

    Separate from the circuit construction because the resource estimate needs only this sum, while
    verifying a circuit costs `2^n` statevector simulations.
    """
    coefficients = np.asarray(operator.simplify().coeffs).real
    return float(np.sum(np.abs(coefficients[np.abs(coefficients) > COEFFICIENT_TOLERANCE])))


def encoding_qubit_count(operator: SparsePauliOp) -> int:
    """Total qubits an encoding of this operator would need, without building it.

    Used to decide in advance whether verifying a configuration is affordable.
    """
    coefficients = np.asarray(operator.simplify().coeffs).real
    n_terms = int(np.sum(np.abs(coefficients) > COEFFICIENT_TOLERANCE))
    return int(operator.num_qubits) + max(1, int(np.ceil(np.log2(max(n_terms, 1)))))


def lcu_block_encoding(operator: SparsePauliOp, symmetric: bool = False) -> BlockEncoding:
    """Build the LCU block encoding of a Hermitian Pauli sum.

    Parameters
    ----------
    operator
        Hermitian Pauli sum. Raises if any coefficient has a non-negligible imaginary part.
    symmetric
        Put the coefficient signs into SELECT rather than into one preparation, so that
        ``PREP_L == PREP_R`` and SELECT is self-inverse. The walk operator ``(2 Pi - I) U`` has
        Chebyshev polynomials of ``A / alpha`` in its block only for this symmetric form. The
        asymmetric form is the default because it is simpler to verify.
    """
    simplified = operator.simplify()
    coefficients = np.asarray(simplified.coeffs)
    if np.max(np.abs(coefficients.imag)) > COEFFICIENT_TOLERANCE:
        raise ValueError(
            "block encoding expects a Hermitian operator, but a Pauli coefficient has a "
            f"non-negligible imaginary part (max {np.max(np.abs(coefficients.imag)):.3e})"
        )

    real = coefficients.real
    keep = np.abs(real) > COEFFICIENT_TOLERANCE
    if not keep.any():
        raise ValueError("every coefficient is negligible; there is nothing to encode")

    paulis = [p for p, k in zip(simplified.paulis, keep, strict=True) if k]
    real = real[keep]

    n_system = simplified.num_qubits
    n_terms = len(paulis)
    n_ancilla = max(1, int(np.ceil(np.log2(n_terms))))
    alpha = float(np.sum(np.abs(real)))

    magnitudes = np.zeros(1 << n_ancilla)
    magnitudes[:n_terms] = np.sqrt(np.abs(real) / alpha)
    signed = magnitudes.copy()
    signed[:n_terms] *= np.sign(real)

    # The padding entries are zero, so both vectors are normalised only if the kept
    # coefficients account for all of alpha, which they do by construction.
    if symmetric:
        prepare_right = prepare_left = StatePreparation(magnitudes / np.linalg.norm(magnitudes))
        signs = np.sign(real)
    else:
        prepare_right = StatePreparation(signed / np.linalg.norm(signed))
        prepare_left = StatePreparation(magnitudes / np.linalg.norm(magnitudes))
        signs = np.ones_like(real)

    circuit = QuantumCircuit(n_ancilla + n_system, name="lcu_block_encoding")
    ancillas = list(range(n_ancilla))
    system = list(range(n_ancilla, n_ancilla + n_system))

    circuit.append(prepare_right, ancillas)
    for index, (pauli, sign) in enumerate(zip(paulis, signs, strict=True)):
        controlled = _controlled_pauli(pauli, index, n_ancilla, n_system, float(sign))
        if controlled is not None:
            circuit.compose(controlled, ancillas + system, inplace=True)
    circuit.append(prepare_left.inverse(), ancillas)

    return BlockEncoding(
        circuit=circuit,
        alpha=alpha,
        n_ancilla=n_ancilla,
        n_system=n_system,
        n_terms=n_terms,
    )


def _controlled_pauli(
    pauli: Any, index: int, n_ancilla: int, n_system: int, sign: float = 1.0
) -> QuantumCircuit | None:
    """``|index><index|`` on the ancillas, tensor ``sign * Pauli`` on the system.

    A positive identity term needs no gate. A negative identity term does, since a controlled global
    phase of pi is a relative phase; omitting it would flip the sign of the constant term, shifting
    every eigenvalue while leaving the eigenvectors unchanged.
    """
    label = pauli.to_label().replace("-", "").replace("i", "")
    if set(label) == {"I"} and sign > 0:
        return None

    ancillas = list(range(n_ancilla))
    wrapper = QuantumCircuit(n_ancilla + n_system)

    # One multi-controlled single-target gate per Pauli factor. The factors act on different
    # qubits under the same control condition, so this equals a multi-controlled multi-target
    # gate. Verification simulates 2^n statevectors through these gates: 0.6 s for 6 qubits,
    # 22 s for 9, 129 s for 11 and over twenty minutes for 13, so G-2 verifies a bounded set
    # of configurations.
    single = {"X": XGate(), "Y": YGate(), "Z": ZGate()}
    # Qiskit labels are printed most-significant first; qubit i is the i-th from the right.
    for position, character in enumerate(reversed(label)):
        if character in single:
            wrapper.append(
                single[character].control(n_ancilla, ctrl_state=index),
                [*ancillas, n_ancilla + position],
            )

    if sign < 0:
        # A conditional global phase of pi is a relative phase; it is applied as a one-qubit
        # identity carrying the phase, controlled on the ancilla register.
        phase = QuantumCircuit(1, name=f"minus{index}")
        phase.global_phase = np.pi
        wrapper.append(phase.to_gate().control(n_ancilla, ctrl_state=index), [*ancillas, n_ancilla])

    return wrapper


def block_encoding_block(encoding: BlockEncoding) -> NDArray[np.complex128]:
    """The top-left block of the encoding, one column at a time.

    Column ``j`` is the amplitude on ``|0>^m |i>`` after running the circuit on ``|0>^m |j>``. This
    costs ``2^n`` statevector simulations instead of one dense ``2^(m+n)`` squared unitary, 16 MB
    against 1 GB at L = 6.
    """
    dimension = 1 << encoding.n_system
    stride = 1 << encoding.n_ancilla
    block = np.zeros((dimension, dimension), dtype=np.complex128)

    for column in range(dimension):
        # Little-endian: ancillas are the low bits, so |0>^m |j> is index j * 2^m.
        initial = np.zeros(stride * dimension, dtype=np.complex128)
        initial[column * stride] = 1.0
        evolved = Statevector(initial).evolve(encoding.circuit).data
        block[:, column] = evolved[::stride]

    return block


def verify_block_encoding(encoding: BlockEncoding, target: SparsePauliOp) -> dict[str, object]:
    """Check the defining property and report the size of any violation.

    Requires agreement to 1e-10 between ``alpha`` times the top-left block and the operator. Where
    the full operator is affordable, the unitarity of the circuit is checked as well.
    """
    block = block_encoding_block(encoding)
    expected = np.asarray(target.to_matrix())
    difference = encoding.alpha * block - expected

    return {
        "n_system": encoding.n_system,
        "n_ancilla": encoding.n_ancilla,
        "n_terms": encoding.n_terms,
        "alpha": encoding.alpha,
        "max_abs_error": float(np.max(np.abs(difference))),
        "frobenius_error": float(np.linalg.norm(difference)),
        "block_spectral_norm": float(np.linalg.norm(block, ord=2)),
        # A block encoding is only useful if the block has norm at most one; a value above
        # one means alpha was computed wrongly and the encoding cannot be unitary.
        "block_norm_within_one": bool(np.linalg.norm(block, ord=2) <= 1.0 + 1e-9),
    }


def circuit_is_unitary(encoding: BlockEncoding, tolerance: float = 1e-10) -> bool:
    """Whether the circuit is unitary, at sizes where the full operator is affordable."""
    matrix = Operator(encoding.circuit).data
    identity = np.eye(matrix.shape[0])
    return bool(np.max(np.abs(matrix.conj().T @ matrix - identity)) < tolerance)
