"""LCU block-encoding of the mutation-selection operator. WP2 task T2.2.

Everything in Route B stands on this. A QSVT circuit does not act on a Hamiltonian; it acts
on a *block encoding* of one, a unitary `U` on ancilla plus system registers whose top-left
block is the operator divided by a normalisation:

    ( <0|^m (x) I )  U  ( |0>^m (x) I )  =  A / alpha

Get this wrong and every downstream number is wrong in a way that still looks plausible,
which is why `G-2` criterion 2 checks the defining property to 1e-10 rather than taking it
on trust.

The construction, and where the signs go
----------------------------------------

`A` arrives as a sum of Pauli terms with real coefficients, `A = sum_j c_j P_j`, because the
generator is Hermitian and every `P_j` is self-inverse. The standard linear-combination-of-
unitaries encoding uses `U = PREP_L^dagger . SELECT . PREP_R` with

    PREP_R |0> = sum_j sqrt(|c_j| / alpha) * sign(c_j) |j>
    PREP_L |0> = sum_j sqrt(|c_j| / alpha) |j>
    SELECT     = sum_j |j><j| (x) P_j

so that the block comes out as `sum_j c_j P_j / alpha` with `alpha = sum_j |c_j|`. Putting
the signs in one preparation and not the other is what lets negative coefficients through
without a separate phase oracle. The two preparations differ, which is allowed: nothing in
the definition asks them to be the same unitary.

`alpha` is the one-norm of the coefficients, and it is the price of the encoding: query
complexity scales with `alpha / gap`. This is the reason the spin convention matters so much.
At L = 12 the single peak written as a projector has 4108 terms and a correspondingly large
one-norm; written sparsely it has 27. That is not a constant factor, it is the difference
between a feasible and an infeasible circuit, and `results/wp_r/g_r_10.json` measures it.

Why the block is extracted column by column
-------------------------------------------

The obvious check builds the full `2^(m+n)` by `2^(m+n)` unitary and slices it. At L = 6 with
the projector form that is a 1 GB array for a matrix whose useful part is 64 by 64. Instead
each column of the block is obtained by simulating `U` on `|0>^m |j>` and reading the
amplitudes with the ancillas in `|0>`, which costs `2^n` statevector simulations of an
`(m + n)`-qubit circuit and gets the same answer to machine precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector

__all__ = [
    "BlockEncoding",
    "block_encoding_block",
    "lcu_block_encoding",
    "verify_block_encoding",
]

# Coefficients below this are dropped rather than encoded. They would each still cost an
# ancilla index and a controlled Pauli while contributing nothing but a larger alpha.
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
        """``alpha``, named for what it does to query complexity.

        QSVT needs polynomial degree of order ``alpha / gap`` to resolve a gap of ``gap`` in
        the *unnormalised* operator, because the block encoding has already divided by
        ``alpha``. Reported so that a resource estimate cannot quietly omit it.
        """
        return self.alpha


def lcu_block_encoding(operator: SparsePauliOp, symmetric: bool = False) -> BlockEncoding:
    """Build the LCU block encoding of a Hermitian Pauli sum.

    Parameters
    ----------
    operator
        Hermitian Pauli sum. Raises if any coefficient has a non-negligible imaginary part:
        a complex coefficient on a self-adjoint Pauli means the operator is not Hermitian,
        and the rest of Route B assumes it is.
    symmetric
        Put the coefficient signs into SELECT rather than into one of the two preparations,
        so that ``PREP_L == PREP_R`` and SELECT is self-inverse.

        This costs nothing and buys qubitisation. The walk operator ``(2 Pi - I) U`` has
        Chebyshev polynomials of ``A / alpha`` in its block *only* when the encoding has
        that symmetric form, and Chebyshev polynomials are how the eigenstate filter gets
        built without ever computing a phase factor. The asymmetric form is kept as the
        default because it is the simpler object to verify, and criterion 2 verifies the
        encoding rather than the walk.
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


def _controlled_pauli(pauli, index: int, n_ancilla: int, n_system: int, sign: float = 1.0):
    """``|index><index|`` on the ancillas, tensor ``sign * Pauli`` on the system.

    A positive-signed identity term needs no gate at all, which is worth skipping rather
    than emitting a controlled identity: for the sparse landscapes the constant term is
    always present and the saving is one multi-controlled block out of a handful. A
    *negative* identity is not skippable, because a controlled global phase of pi is a
    relative phase and changes the encoded operator. Getting that wrong would silently flip
    the sign of the constant term, which shifts every eigenvalue and leaves the eigenvectors
    intact, so it would pass an eigenvector check and fail nothing until the resource
    estimate.
    """
    label = pauli.to_label().replace("-", "").replace("i", "")
    if set(label) == {"I"} and sign > 0:
        return None

    gate = QuantumCircuit(n_system, name=f"P{index}")
    if sign < 0:
        # Qiskit lifts a circuit's global phase to a relative phase under .control().
        gate.global_phase = np.pi
    # Qiskit labels are printed most-significant first; qubit i is the i-th from the right.
    for position, character in enumerate(reversed(label)):
        if character == "X":
            gate.x(position)
        elif character == "Y":
            gate.y(position)
        elif character == "Z":
            gate.z(position)

    controlled = gate.to_gate().control(n_ancilla, ctrl_state=index)
    wrapper = QuantumCircuit(n_ancilla + n_system)
    wrapper.append(controlled, list(range(n_ancilla + n_system)))
    return wrapper


def block_encoding_block(encoding: BlockEncoding) -> NDArray[np.complex128]:
    """The top-left block of the encoding, one column at a time.

    Column ``j`` is the amplitude on ``|0>^m |i>`` after running the circuit on
    ``|0>^m |j>``. This costs ``2^n`` statevector simulations rather than one dense
    ``2^(m+n)`` squared unitary, which at L = 6 is the difference between 16 MB and 1 GB.
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
    """Check the defining property, and report how badly it is violated if it is.

    `G-2` criterion 2 asks for agreement to 1e-10 between ``alpha`` times the top-left block
    and the operator. The unitarity of the circuit is checked separately at sizes where the
    full operator is affordable, because a block that matches while the circuit is not
    unitary would mean the extraction is wrong rather than the encoding right.
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
    """Affordable only at small sizes, and worth doing there. See `verify_block_encoding`."""
    matrix = Operator(encoding.circuit).data
    identity = np.eye(matrix.shape[0])
    return bool(np.max(np.abs(matrix.conj().T @ matrix - identity)) < tolerance)
