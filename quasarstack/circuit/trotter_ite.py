"""Trotterised imaginary-time evolution.

``exp(-H tau)`` is not unitary, so this propagator acts on a state vector rather than defining
a hardware circuit. It is the reference discretisation for the splitting error; the
hardware-compatible routes are varQITE and Motta-QITE in `quasarstack.ite`.

The propagator factorises into three parts:

- **Mutation.** ``exp(mu dtau X_i)`` on each site, the transverse field, which spreads the
  population across sequence space.
- **Selection.** ``exp(f(sigma) dtau)``, diagonal, carrying per-site fitness and epistatic
  couplings, which concentrates the population on the fittest genotype.
- **Schedule.** The interleaving of the two; their balance is mutation-selection balance, and
  their ratio sets the position relative to the error threshold.

The splitting
-------------

Symmetric, second order:

    exp(-H dtau) ~ exp(-H_S dtau/2) exp(-H_M dtau) exp(-H_S dtau/2)

with ``H = -W``, ``H_S = -diag(f)`` and ``H_M = -mu sum_i X_i + mu L I``. The diagonal factor is
an element-wise ``exp(f dtau/2)``, and the mutation factor is a product of single-site
``cosh(mu dtau) I + sinh(mu dtau) X_i``, since the site terms commute and each squares to the
identity. The error over a fixed total time is O(dtau^2).

The state is renormalised every step, since ``exp(f dtau)`` compounds and would overflow float64
at L = 8 and tau = 60. Renormalisation does not change the ray.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit

from quasarstack.analytic.exact_diag import infer_n_sites


def uniform_state(n_sites: int) -> NDArray[np.float64]:
    """Equal weight on every genotype.

    The Perron vector is strictly positive, so this state has non-zero overlap with it for any
    landscape.
    """
    dim = 1 << n_sites
    return np.full(dim, 1.0 / np.sqrt(dim), dtype=np.float64)


def apply_mutation(state: NDArray[np.float64], mu: float, dtau: float) -> NDArray[np.float64]:
    """Apply ``prod_i exp(mu dtau X_i)`` to the state.

    Single-site terms commute and ``exp(a X) = cosh(a) I + sinh(a) X`` exactly, so this is a
    sequence of L two-row updates.
    """
    n_sites = infer_n_sites(state)
    angle = mu * dtau
    cosh, sinh = float(np.cosh(angle)), float(np.sinh(angle))

    out = state
    for site in range(n_sites):
        # Site i occupies bit i, so this reshape puts it on the middle axis.
        view = out.reshape(1 << (n_sites - site - 1), 2, 1 << site)
        low, high = view[:, 0, :], view[:, 1, :]
        out = np.concatenate(
            [(cosh * low + sinh * high)[:, None, :], (sinh * low + cosh * high)[:, None, :]],
            axis=1,
        ).reshape(-1)
    return out


def apply_selection(
    state: NDArray[np.float64], fitness: NDArray[np.float64], dtau: float
) -> NDArray[np.float64]:
    """Apply the diagonal ``exp(f dtau)``.

    The exponent is shifted by its maximum before exponentiating, which changes the result by a
    positive scalar only and keeps the factor finite.
    """
    exponent = fitness * dtau
    scaled: NDArray[np.float64] = state * np.exp(exponent - exponent.max())
    return scaled


def _normalise(state: NDArray[np.float64]) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(state))
    if norm <= 0.0:
        raise ValueError("imaginary-time evolution collapsed the state to zero")
    normalised: NDArray[np.float64] = state / norm
    return normalised


def trotter_step(
    state: NDArray[np.float64], fitness: NDArray[np.float64], mu: float, dtau: float
) -> NDArray[np.float64]:
    """One symmetric second-order step, S(dtau/2) M(dtau) S(dtau/2)."""
    out = apply_selection(state, fitness, 0.5 * dtau)
    out = apply_mutation(out, mu, dtau)
    out = apply_selection(out, fitness, 0.5 * dtau)
    return _normalise(out)


def evolve(
    fitness: NDArray[np.float64],
    mu: float,
    tau: float,
    dtau: float,
    initial: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], int]:
    """Run imaginary-time evolution to ``tau`` in steps of ``dtau``.

    Returns
    -------
    probs
        Length ``2**L`` L1-normalised, non-negative distribution.
    n_steps
        Steps taken. ``tau / dtau`` must be a whole number, so runs at different step sizes reach
        the same total time.
    """
    ratio = tau / dtau
    n_steps = int(round(ratio))
    if abs(ratio - n_steps) > 1e-9:
        raise ValueError(f"tau / dtau must be a whole number, got {ratio}")

    n_sites = infer_n_sites(fitness)
    state = uniform_state(n_sites) if initial is None else _normalise(np.array(initial, float))

    for _ in range(n_steps):
        state = trotter_step(state, fitness, mu, dtau)

    probs = np.abs(state)
    return probs / probs.sum(), n_steps


def evolve_exact(
    fitness: NDArray[np.float64],
    mu: float,
    tau: float,
    initial: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Apply ``exp(-H tau)`` without splitting, by eigendecomposition.

    This is the reference for the step-size scaling fit. Comparing with it rather than with the
    quasispecies isolates the splitting error from the residual due to finite tau.
    """
    from quasarstack.analytic.exact_diag import mutation_selection_generator

    n_sites = infer_n_sites(fitness)
    state = uniform_state(n_sites) if initial is None else _normalise(np.array(initial, float))

    generator = mutation_selection_generator(fitness, mu).toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(generator)
    # exp(-H tau) = exp(+W tau); shift by the top eigenvalue so the exponential stays finite.
    weights = np.exp((eigenvalues - eigenvalues.max()) * tau)
    evolved = eigenvectors @ (weights * (eigenvectors.T @ state))

    probs = np.abs(evolved)
    normalised: NDArray[np.float64] = probs / probs.sum()
    return normalised


def trotter_circuit(
    n_sites: int,
    a: NDArray[np.float64],
    mu: float,
    dtau: float,
    b: NDArray[np.float64] | None = None,
) -> QuantumCircuit:
    """The circuit structure of one Trotter step, for resource reporting.

    This circuit does not implement imaginary-time evolution. It is the unitary real-time analogue
    with the same interaction pattern: Rz for per-site fitness, Rzz for epistatic couplings and Rx
    for mutation, in the symmetric S-M-S order, used for depth and two-qubit gate counts.
    """
    circuit = QuantumCircuit(n_sites, name="trotter_step")

    def selection_layer(scale: float) -> None:
        for site in range(n_sites):
            if abs(a[site]) > 0:
                circuit.rz(2.0 * a[site] * scale * dtau, site)
        if b is not None:
            for i in range(n_sites):
                for j in range(i + 1, n_sites):
                    if abs(b[i, j]) > 0:
                        circuit.rzz(2.0 * b[i, j] * scale * dtau, i, j)

    selection_layer(0.5)
    for site in range(n_sites):
        circuit.rx(2.0 * mu * dtau, site)
    selection_layer(0.5)
    return circuit
