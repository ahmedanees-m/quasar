"""The qubitised walk, and the Chebyshev polynomials it produces. WP2, towards T2.3.

Given a symmetric LCU block encoding `U` of `A / alpha`, the walk operator

    W  =  (2 Pi - I) . U ,        Pi = |0>^m <0|^m  (x)  I

has the property that the top-left block of `W^d` is `T_d(A / alpha)`, the degree-`d`
Chebyshev polynomial of the first kind. This is the primitive underneath every QSVT
construction: an arbitrary bounded polynomial is a combination of these, and the phase-factor
machinery exists to produce that combination in one pass rather than by linear combination.

Verifying `W^d` against `T_d` is worth doing on its own, before any filter is built, because
it separates two failure modes that otherwise arrive together. If the walk is right and the
filtered state is wrong, the polynomial is wrong. If the walk is wrong, nothing downstream
means anything.

A caution that shapes the whole resource estimate
-------------------------------------------------

Every polynomial reachable this way is bounded by 1 on the encoded spectrum, because the
block of a unitary cannot have norm above 1. So the tempting Chebyshev-acceleration argument,
which puts the target eigenvalue *outside* the interval where the polynomial is bounded and
gets a square-root speedup, is not available here: the target eigenvalue is inside the
encoded spectrum by construction. What is available is a bounded polynomial approximating a
step, which costs degree of order `alpha / gap`, linear rather than square-root in that
ratio. The `alpha` in that expression is the one-norm of the Pauli coefficients, which is why
`block_encoding.BlockEncoding.subnormalisation_cost` exists and why the sparse spin
convention is worth 152 times at L = 12.
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
    """``2 |0><0|_anc (x) I  -  I``, with the global sign right.

    Built as `X^m . MCZ . X^m`, which flips the sign of everything *except* the all-zero
    ancilla state, and is therefore `I - 2 Pi`. The reflection wanted is `2 Pi - I`, the
    negative of that, so the circuit carries a global phase of pi. Under a single
    application the difference is unobservable; raised to the power `d` it is a factor of
    `(-1)^d`, which would put the Chebyshev comparison out by a sign on every odd degree and
    look like a broken walk.
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

    The encoding must have been built with ``symmetric=True``. With the asymmetric form the
    two preparations differ, SELECT is not self-inverse, and the walk's block is not a
    Chebyshev polynomial. Nothing here can detect that from the object alone, so
    `verify_chebyshev` is the check that matters.
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
    """Does the walk produce ``T_d(A / alpha)``? Measured against the matrix Chebyshev.

    The classical reference is built by the Chebyshev recurrence on the matrix,
    ``T_0 = I``, ``T_1 = X``, ``T_{d+1} = 2 X T_d - T_{d-1}``, rather than by
    diagonalising, so the comparison does not route through an eigendecomposition that the
    circuit never performs.
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
