"""The qubitised walk and its Chebyshev polynomials.

Given a symmetric LCU block encoding `U` of `A / alpha`, the walk operator

    W  =  (2 Pi - I) . U ,        Pi = |0>^m <0|^m  (x)  I

has top-left block of `W^d` equal to `T_d(A / alpha)`, the degree-`d` Chebyshev polynomial of the
first kind. Bounded polynomials are combinations of these, and QSVT phase factors produce such a
combination in one pass.

Checking `W^d` against `T_d` separately from the filter distinguishes an error in the walk from
an error in the polynomial.

Every polynomial reachable this way is bounded by 1 on the encoded spectrum, since the block of
a unitary cannot exceed norm 1. With the target eigenvalue inside that spectrum, a step
approximation costs degree of order `alpha / gap`, linear in that ratio. `alpha` is the one-norm
of the Pauli coefficients, so the sparse spin convention reduces the cost by a factor of 152 at
L = 12.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZGate
from qiskit.quantum_info import SparsePauliOp, Statevector

from quasarstack.qsvt.block_encoding import BlockEncoding, lcu_block_encoding

__all__ = [
    "chebyshev_block",
    "reflection_about_zero",
    "verify_chebyshev",
    "walk_operator",
]


def reflection_about_zero(n_ancilla: int, n_system: int) -> QuantumCircuit:
    """``2 |0><0|_anc (x) I  -  I``, with the global sign.

    Built as `X^m . MCZ . X^m`, which is `I - 2 Pi`, so the circuit carries a global phase of pi to
    give `2 Pi - I`. Without it `W^d` would differ from `T_d` by `(-1)^d`.
    """
    circuit = QuantumCircuit(n_ancilla + n_system, name="reflect")
    ancillas = list(range(n_ancilla))

    circuit.global_phase = np.pi
    circuit.x(ancillas)
    if n_ancilla == 1:
        circuit.z(0)
    else:
        circuit.append(ZGate().control(n_ancilla - 1), ancillas)
    circuit.x(ancillas)
    return circuit


def walk_operator(encoding: BlockEncoding) -> QuantumCircuit:
    """``(2 Pi - I) U`` for a symmetric block encoding.

    The encoding must be built with ``symmetric=True``; with the asymmetric form SELECT is not
    self-inverse and the block of the walk is not a Chebyshev polynomial. `verify_chebyshev` checks
    this.
    """
    circuit = QuantumCircuit(encoding.n_ancilla + encoding.n_system, name="qubitised_walk")
    circuit.compose(encoding.circuit, inplace=True)
    circuit.compose(reflection_about_zero(encoding.n_ancilla, encoding.n_system), inplace=True)
    return circuit


def chebyshev_block(encoding: BlockEncoding, degree: int) -> NDArray[np.complex128]:
    """Top-left block of ``W^degree``, extracted column by column.

    Degree 0 is the identity, which is `T_0`, and is returned without building a circuit.
    """
    dimension = 1 << encoding.n_system
    stride = 1 << encoding.n_ancilla
    if degree == 0:
        return np.eye(dimension, dtype=np.complex128)

    walk = walk_operator(encoding)
    powered = QuantumCircuit(encoding.n_ancilla + encoding.n_system)
    for _ in range(degree):
        powered.compose(walk, inplace=True)

    block = np.zeros((dimension, dimension), dtype=np.complex128)
    for column in range(dimension):
        initial = np.zeros(stride * dimension, dtype=np.complex128)
        initial[column * stride] = 1.0
        block[:, column] = Statevector(initial).evolve(powered).data[::stride]
    return block


def verify_chebyshev(operator: SparsePauliOp, degrees: list[int]) -> dict[str, object]:
    """Whether the walk produces ``T_d(A / alpha)``, measured against the matrix Chebyshev.

    The reference uses the recurrence ``T_0 = I``, ``T_1 = X``, ``T_{d+1} = 2 X T_d - T_{d-1}`` on
    the matrix, not an eigendecomposition.
    """
    encoding = lcu_block_encoding(operator, symmetric=True)
    scaled = np.asarray(operator.to_matrix()) / encoding.alpha

    dimension = scaled.shape[0]
    previous = np.eye(dimension, dtype=np.complex128)
    current = scaled.astype(np.complex128)
    reference = {0: previous, 1: current}
    for degree in range(2, max(degrees) + 1):
        previous, current = current, 2.0 * scaled @ current - previous
        reference[degree] = current

    per_degree = []
    worst = 0.0
    for degree in degrees:
        measured = chebyshev_block(encoding, degree)
        error = float(np.max(np.abs(measured - reference[degree])))
        worst = max(worst, error)
        per_degree.append({"degree": degree, "max_abs_error": error})

    return {
        "n_system": encoding.n_system,
        "n_ancilla": encoding.n_ancilla,
        "alpha": encoding.alpha,
        "worst_max_abs_error": worst,
        "per_degree": per_degree,
    }
