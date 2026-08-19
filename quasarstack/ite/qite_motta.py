"""Route A, fallback: Motta-style quantum imaginary-time evolution.

Why the project carries a second imaginary-time method
------------------------------------------------------

varQITE is variational, so it inherits barren plateaus: its gradients vanish exponentially
in system size, and the planning documents put its ceiling near L = 10 to 12. Motta-QITE has
no variational optimisation at all. Each step's generator comes from a **linear solve** over
measured expectation values, so there is no landscape to get lost on and no plateau to stall
in. The price is that the generator's support grows as correlations spread, so the circuit
depth grows where varQITE's does not. The two methods fail in different directions, which is
why the project measures both.

The step
--------

One imaginary-time step is ``psi -> exp(-dtau H) psi``, renormalised. That is not unitary, so
it is reproduced by a unitary ``exp(dtau G)`` whose generator ``G`` is fitted to match it to
first order. Writing ``G = sum_I a_I K_I`` over a basis of generators and matching against
the residual ``r = (phi - psi) / dtau``, where ``phi`` is the normalised exact step, gives a
least-squares problem with normal equations

    M a = v,    M_IJ = (K_I psi) . (K_J psi),    v_I = (K_I psi) . r

Every entry is an expectation value in the current state, so nothing here needs a state
vector in principle.

The basis, and the bug that hides in it
---------------------------------------

**This is the part that matters, and it is where the planning documents record a real
failure:** "an element-wise gradient that vanishes for real states; the energy *ascended*
instead of descending".

Our state is real, because the Hamiltonian is real symmetric and the initial state is real.
A unitary that maps real vectors to real vectors must be a real orthogonal matrix, so its
generator ``G`` must be **real and antisymmetric**. A Pauli string is Hermitian; it is
*imaginary* exactly when it contains an odd number of Y factors, and ``-i`` times an
imaginary Hermitian matrix is real antisymmetric.

So the generator basis must be the Pauli strings with an **odd number of Y factors**, and
nothing else.

Where the failure hides is in Motta's own form of the right-hand side,
``b_I = Re(-i <psi| sigma_I |Delta>)``. For a real state and a real residual, that bracket is
**real** whenever ``sigma_I`` is real, so ``-i`` times it is purely imaginary and its real
part is exactly zero. Every Y-free string contributes nothing whatsoever, and the Y-free
strings are precisely the ones that feel natural to reach for, because the Hamiltonian is
built from X and Z. The solve then returns nothing, and the method stands still or, with a
sign slip, walks uphill. `motta_right_hand_side` and `zero_rhs_demonstration` compute that
quantity for both parities so the reason for the basis is evidenced rather than described.

The real-arithmetic solve used below is equivalent to Motta's complex form up to an overall
sign, and it hides the parity issue by construction, which is exactly why the complex form
is kept in the module as well.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from qiskit.quantum_info import SparsePauliOp
from scipy.linalg import expm

# Relative singular-value cutoff for the least-squares solve. The Gram matrix is routinely
# rank deficient, because different generators can act identically on the current state, and
# it is worst at the start, where the uniform superposition is maximally symmetric and whole
# families of generators become indistinguishable.
#
# This was a fixed absolute ridge, and that was wrong in a way worth recording. A ridge is an
# absolute quantity added to a matrix whose scale depends on the state, so which direction it
# picks out of the near-null space is arbitrary. Gate G-R.7 failed on it: the very first step
# of one instance raised the energy by 2.3e-3, and the failure was knife-edge, appearing at
# ridge 1e-8 but not at 1e-10 or 1e-6, and at dtau 0.05 but not at 0.1 or 0.02. Non-monotone
# sensitivity in every direction is the signature of an arbitrary choice in a degenerate
# subspace, not of a physical effect.
#
# A relative cutoff on the singular values is scale invariant and discards the degenerate
# directions rather than guessing in them, which is what the problem actually calls for.
#
# The value comes from the measured conditioning, not from what makes a gate pass. The Gram
# matrix reaches a condition number of 5.3e16, past the reciprocal of double precision, so it
# is numerically singular and directions below about 1e-8 relative carry no information. That
# cutoff also removes the overshoot at every step size tried rather than only at the
# registered one, while leaving accuracy unchanged at the seventh decimal.
DEFAULT_RCOND = 1e-8


def odd_y_strings(n_sites: int, max_weight: int) -> list[str]:
    """Pauli strings with an odd number of Y factors, up to the given support size.

    These are exactly the Hermitian Pauli strings that are purely imaginary, so multiplying
    by ``-i`` gives the real antisymmetric generators a real-to-real unitary needs. Strings
    are returned in Qiskit's little-endian ordering, rightmost character being qubit 0.

    The count grows as ``sum_k C(L, k) (3^k - 1) / 2``, which is where the method's cost
    lives: 6 generators at L = 6 and weight 1, 66 at weight 2, 326 at weight 3.
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
    """``exp(-dtau H)``, which is constant across a run and should be built once.

    Rebuilding it inside the step loop is the obvious way to write this and it costs a dense
    matrix exponential per step for no reason, on top of the one the method genuinely needs
    for its fitted generator.
    """
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

    Solved by truncated SVD with a relative cutoff rather than by adding a ridge, because
    the Gram matrix is rank deficient and a ridge chooses arbitrarily inside the degenerate
    subspace. See the note on ``DEFAULT_RCOND``.

    Returns the coefficients, the norm of the right-hand side, and the Gram matrix's
    condition number. The right-hand side is worth watching because it is identically zero
    for a basis of the wrong parity; the condition number is worth watching because it is
    what made this solve fragile in the first place.
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

    Convergence is judged on the **rate** of state change, ``||psi_new - psi_prev|| / dtau``,
    for the same reason as in varQITE. A per-step criterion trips sooner at a smaller step
    purely because each step moves less, which makes the imaginary time reached depend on the
    step size. This method is where that showed up: accuracy appeared to *fall* from 0.9999997
    to 0.9999731 as dtau went from 0.1 to 0.01, and the finer run was not worse, it had simply
    stopped earlier in tau. Since `tau_used` is the number docs/notes.md asks WP7 to compare across
    methods, a step-size-dependent one would be actively misleading.
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

    Exists only to evidence a negative. These are the real *symmetric* Pauli strings, and
    they are the natural set to reach for, because the Hamiltonian is built from X and Z.
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
    """The right-hand side in Motta's own complex form: ``b_I = Re(-i <psi| sigma_I |Delta>)``.

    Written out in this form because that is where the parity matters, and the module's
    real-arithmetic solve hides it.

    For a real state and a real residual, the bracket ``<psi| sigma_I |Delta>`` is **real**
    whenever ``sigma_I`` is real, so ``-i`` times it is purely imaginary and its real part is
    exactly zero. Every Y-free string, and every string with an even number of Y factors,
    therefore contributes nothing at all. That is the "element-wise gradient that vanishes
    for real states" the planning documents record for this method, and it is why the wrong
    basis makes it stand still or, with a sign slip, ascend.

    An odd number of Y factors makes ``sigma_I`` purely imaginary, the bracket purely
    imaginary, and the real part non-zero.
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
    """Compare the two parities on Motta's own right-hand side.

    Reproduces the mechanism of the recorded failure rather than describing it, so the basis
    choice is evidenced. The even-Y set is expected to come back at exactly zero.
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
