"""Route A, second method: Motta-style quantum imaginary-time evolution.

varQITE is variational and its gradients vanish exponentially with system size. Motta-QITE has
no variational optimisation: each step's generator comes from a linear solve over expectation
values. Its cost is that the generator's support grows as correlations spread, so circuit depth
grows where varQITE's does not.

The step
--------

One imaginary-time step is ``psi -> exp(-dtau H) psi``, renormalised. It is reproduced by a
unitary ``exp(dtau G)`` whose generator ``G`` matches it to first order. Writing
``G = sum_I a_I K_I`` over a generator basis and matching the residual
``r = (phi - psi) / dtau``, with ``phi`` the normalised exact step, gives the normal equations

    M a = v,    M_IJ = (K_I psi) . (K_J psi),    v_I = (K_I psi) . r

Every entry is an expectation value in the current state.

The generator basis
-------------------

The state is real, because the Hamiltonian is real symmetric and the initial state is real. A
unitary mapping real vectors to real vectors is real orthogonal, so ``G`` must be real and
antisymmetric. A Pauli string is imaginary exactly when it contains an odd number of Y factors,
and ``-i`` times an imaginary Hermitian matrix is real antisymmetric. The basis is therefore the
Pauli strings with an odd number of Y factors.

In Motta's form of the right-hand side, ``b_I = Re(-i <psi| sigma_I |Delta>)``, the bracket is
real whenever ``sigma_I`` is real, so the real part vanishes for every Y-free or even-Y string:
such a basis gives a zero update. `motta_right_hand_side` and `zero_rhs_demonstration` compute
this quantity for both parities. The real-arithmetic solve used here is equivalent to Motta's
complex form up to an overall sign.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm

# Relative singular-value cutoff for the least-squares solve. The Gram matrix is rank deficient,
# since different generators can act identically on the current state, most strongly at the
# start, where the uniform superposition is maximally symmetric.
# A relative cutoff is scale invariant and discards the degenerate directions, where an absolute
# ridge would select an arbitrary vector from the near-null space. The Gram matrix reaches a
# condition number of 5.3e16, so directions below about 1e-8 relative carry no information.
DEFAULT_RCOND = 1e-8


def odd_y_strings(n_sites: int, max_weight: int) -> list[str]:
    """Pauli strings with an odd number of Y factors, up to the given support size.

    These Hermitian strings are purely imaginary, so multiplying by ``-i`` gives the real
    antisymmetric generators a real-to-real unitary needs. Strings use Qiskit's little-endian
    ordering, the rightmost character being qubit 0.

    The count grows as ``sum_k C(L, k) (3^k - 1) / 2``: 6 generators at L = 6 and weight 1, 66 at
    weight 2, 326 at weight 3.
    """
    if not 1 <= max_weight <= n_sites:
        raise ValueError(f"max_weight must be between 1 and {n_sites}, got {max_weight}")

    strings: list[str] = []
    for weight in range(1, max_weight + 1):
        for sites in combinations(range(n_sites), weight):
            for letters in product("XYZ", repeat=weight):
                if letters.count("Y") % 2 == 0:
                    continue
                label = ["I"] * n_sites
                for site, letter in zip(sites, letters, strict=True):
                    label[site] = letter
                # Qiskit reads the rightmost character as qubit 0.
                strings.append("".join(reversed(label)))
    return strings


@dataclass
class Generators:
    """The real antisymmetric generator basis, as sparse matrices."""

    labels: list[str]
    matrices: list[sp.csr_matrix]

    @property
    def size(self) -> int:
        return len(self.labels)


def build_generators(n_sites: int, max_weight: int) -> Generators:
    """Materialise ``K_I = -i * sigma_I`` for every odd-Y string up to ``max_weight``."""
    labels = odd_y_strings(n_sites, max_weight)
    matrices = []
    for label in labels:
        pauli = SparsePauliOp(label).to_matrix(sparse=True)
        real_part = (-1j * pauli).real
        matrices.append(sp.csr_matrix(real_part))
    return Generators(labels=labels, matrices=matrices)


def imaginary_time_propagator(hamiltonian: NDArray[np.float64], dtau: float) -> NDArray[np.float64]:
    """``exp(-dtau H)``, constant across a run and built once."""
    propagator: NDArray[np.float64] = expm(-dtau * hamiltonian)
    return propagator


def _exact_step(
    state: NDArray[np.float64],
    hamiltonian: NDArray[np.float64],
    dtau: float,
    propagator: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """The non-unitary target: ``exp(-dtau H) psi`` renormalised."""
    if propagator is None:
        propagator = imaginary_time_propagator(hamiltonian, dtau)
    evolved = propagator @ state
    normalised: NDArray[np.float64] = evolved / float(np.linalg.norm(evolved))
    return normalised


def solve_generator(
    state: NDArray[np.float64],
    hamiltonian: NDArray[np.float64],
    generators: Generators,
    dtau: float,
    rcond: float = DEFAULT_RCOND,
    propagator: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], float, float]:
    """Fit the step's generator coefficients by least squares.

    Solved by truncated SVD with a relative cutoff, since the Gram matrix is rank deficient (see
    ``DEFAULT_RCOND``). Returns the coefficients, the norm of the right-hand side, which vanishes for
    a basis of the wrong parity, and the condition number of the Gram matrix.
    """
    target = _exact_step(state, hamiltonian, dtau, propagator)
    residual = (target - state) / dtau

    applied = np.array([matrix @ state for matrix in generators.matrices])
    gram = applied @ applied.T
    rhs = applied @ residual

    coefficients, _, _, singular = np.linalg.lstsq(gram, rhs, rcond=rcond)
    condition = float(singular.max() / singular.min()) if singular.min() > 0 else float("inf")
    return coefficients, float(np.linalg.norm(rhs)), condition


def step(
    state: NDArray[np.float64],
    hamiltonian: NDArray[np.float64],
    generators: Generators,
    dtau: float,
    rcond: float = DEFAULT_RCOND,
    propagator: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], float, float]:
    """One Motta step: solve for the generator, then apply the unitary it defines."""
    coefficients, rhs_norm, condition = solve_generator(
        state, hamiltonian, generators, dtau, rcond, propagator
    )

    generator = sp.csr_matrix(hamiltonian.shape, dtype=np.float64)
    for coefficient, matrix in zip(coefficients, generators.matrices, strict=True):
        if coefficient != 0.0:
            generator = generator + coefficient * matrix

    unitary = expm(dtau * generator.toarray())
    evolved = unitary @ state
    return evolved / float(np.linalg.norm(evolved)), rhs_norm, condition


@dataclass
class Evolution:
    """The outcome of a Motta-QITE run."""

    probs: NDArray[np.float64]
    state: NDArray[np.float64]
    energies: list[float]
    tau_used: float
    steps: int
    converged: bool
    final_state_rate: float
    final_state_change: float
    min_rhs_norm: float
    max_gram_condition: float
    n_generators: int


def evolve(
    hamiltonian: NDArray[np.float64],
    n_sites: int,
    tau: float,
    dtau: float,
    max_weight: int = 2,
    rcond: float = DEFAULT_RCOND,
    tolerance: float = 1e-3,
    generators: Generators | None = None,
) -> Evolution:
    """Run Motta-QITE until the state stops moving, or until ``tau``.

    Convergence is judged on the rate of state change, ``||psi_new - psi_prev|| / dtau``, as in
    varQITE, so that the imaginary time reached does not depend on the step size. With a per-step
    criterion the accuracy at dtau 0.01 is 0.9999731 against 0.9999997 at dtau 0.1, because the
    finer run stops earlier in tau.
    """
    ratio = tau / dtau
    max_steps = int(round(ratio))
    if abs(ratio - max_steps) > 1e-9:
        raise ValueError(f"tau / dtau must be a whole number, got {ratio}")

    basis = generators if generators is not None else build_generators(n_sites, max_weight)
    dimension = 1 << n_sites
    state = np.full(dimension, 1.0 / np.sqrt(dimension), dtype=np.float64)
    propagator = imaginary_time_propagator(hamiltonian, dtau)

    energies: list[float] = []
    state_rate = float("inf")
    state_change = float("inf")
    smallest_rhs = float("inf")
    worst_condition = 0.0
    steps = 0

    for _ in range(max_steps):
        previous = state
        state, rhs_norm, condition = step(state, hamiltonian, basis, dtau, rcond, propagator)
        energies.append(float(state @ hamiltonian @ state))
        smallest_rhs = min(smallest_rhs, rhs_norm)
        worst_condition = max(worst_condition, condition)
        steps += 1
        state_rate = float(np.linalg.norm(state - previous)) / dtau
        state_change = 1.0 - abs(float(previous @ state))
        if state_rate < tolerance:
            break

    probs = np.abs(state)
    return Evolution(
        probs=probs / probs.sum(),
        state=state,
        energies=energies,
        tau_used=steps * dtau,
        steps=steps,
        converged=state_rate < tolerance,
        final_state_rate=state_rate,
        final_state_change=state_change,
        min_rhs_norm=smallest_rhs,
        max_gram_condition=worst_condition,
        n_generators=basis.size,
    )


def _even_y_strings(n_sites: int, max_weight: int) -> list[str]:
    """Pauli strings with an even, non-zero number of Y factors, plus the Y-free ones.

    The real symmetric Pauli strings, used to show that they give a zero right-hand side.
    """
    labels: list[str] = []
    for weight in range(1, max_weight + 1):
        for sites in combinations(range(n_sites), weight):
            for letters in product("XYZ", repeat=weight):
                if letters.count("Y") % 2 == 1:
                    continue
                label = ["I"] * n_sites
                for site, letter in zip(sites, letters, strict=True):
                    label[site] = letter
                labels.append("".join(reversed(label)))
    return labels


def motta_right_hand_side(
    state: NDArray[np.float64], labels: list[str], delta: NDArray[np.float64]
) -> NDArray[np.float64]:
    """The right-hand side in Motta's complex form: ``b_I = Re(-i <psi| sigma_I |Delta>)``.

    For a real state and a real residual, ``<psi| sigma_I |Delta>`` is real whenever ``sigma_I`` is
    real, so ``-i`` times it is purely imaginary and its real part is zero. Every Y-free string, and
    every string with an even number of Y factors, therefore contributes nothing. An odd number of Y
    factors makes ``sigma_I`` purely imaginary and the real part non-zero.
    """
    values = []
    for label in labels:
        matrix = SparsePauliOp(label).to_matrix(sparse=True)
        values.append(float(np.real(-1j * (state @ (matrix @ delta)))))
    return np.array(values)


def zero_rhs_demonstration(
    state: NDArray[np.float64],
    hamiltonian: NDArray[np.float64],
    n_sites: int,
    max_weight: int,
    dtau: float,
) -> dict[str, float]:
    """Compare the two parities on Motta's right-hand side.

    The even-Y set is expected to give exactly zero.
    """
    target = _exact_step(state, hamiltonian, dtau)
    delta = (target - state) / dtau

    odd_labels = odd_y_strings(n_sites, max_weight)
    even_labels = _even_y_strings(n_sites, max_weight)

    odd = motta_right_hand_side(state, odd_labels, delta)
    even = motta_right_hand_side(state, even_labels, delta)

    return {
        "odd_y_rhs_norm": float(np.linalg.norm(odd)),
        "even_y_rhs_norm": float(np.linalg.norm(even)),
        "even_y_rhs_max_abs": float(np.max(np.abs(even))) if even.size else 0.0,
        "n_odd_y_generators": len(odd_labels),
        "n_even_y_generators": len(even_labels),
    }
