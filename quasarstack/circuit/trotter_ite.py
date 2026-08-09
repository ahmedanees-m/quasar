"""Modules M, S and E: Trotterised imaginary-time evolution.

What this is, stated plainly
----------------------------

The correspondence demands imaginary-time evolution, ``exp(-H tau)``, and that operator is
**not unitary**. So the propagator here is not a hardware-runnable circuit, and calling it
one would be the kind of overclaim this project has already corrected twice. It is the
Trotterised imaginary-time propagator applied to a state vector: the reference
discretisation that establishes how the splitting error behaves, and the thing varQITE and
Motta-QITE are later measured against. Those two are the hardware-faithful routes, built
only from circuit expectation values, and they are gates G-R.6 and G-R.7.

What is genuine here is the *structure*. The propagator factorises into exactly the three
modules the correspondence suggests:

- **Module M, mutation.** ``exp(mu dtau X_i)`` on each site, the transverse field. On its
  own it diffuses the population uniformly across sequence space.
- **Module S, selection.** ``exp(f(sigma) dtau)``, diagonal, carrying both the per-site
  fitness and the epistatic couplings. On its own it collapses the population onto the
  fittest genotype.
- **Module E, the scheduler.** Interleaves them. Their competition *is* mutation-selection
  balance, and sweeping the ratio sweeps the mutation rate through the error threshold.

The splitting
-------------

Symmetric, second order:

    exp(-H dtau) ~ exp(-H_S dtau/2) exp(-H_M dtau) exp(-H_S dtau/2)

with ``H = -W``, so ``H_S = -diag(f)`` and ``H_M = -mu sum_i X_i + mu L I``. The diagonal
factor is therefore an element-wise ``exp(f dtau/2)``, and the mutation factor is a product
of single-site ``cosh(mu dtau) I + sinh(mu dtau) X_i``, since the site terms commute and
each squares to the identity. Error over a fixed total time is O(dtau^2), which gate G-R.3
measures rather than assumes.

The state is renormalised every step. That is required, not cosmetic: ``exp(f dtau)``
compounds, and at L = 8 with tau = 60 the unnormalised amplitude would overflow float64
long before the run finished. Renormalisation rescales and does not change the ray.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit

from quasarstack.analytic.exact_diag import infer_n_sites


def uniform_state(n_sites: int) -> NDArray[np.float64]:
    """The maximally uninformative starting population, equal weight on every genotype.

    Chosen because it has non-zero overlap with the Perron vector for any landscape, the
    Perron vector being strictly positive, so imaginary-time evolution is guaranteed to
    reach it rather than being trapped in an orthogonal subspace.
    """
    dim = 1 << n_sites
    return np.full(dim, 1.0 / np.sqrt(dim), dtype=np.float64)


def apply_mutation(state: NDArray[np.float64], mu: float, dtau: float) -> NDArray[np.float64]:
    """Module M: apply ``prod_i exp(mu dtau X_i)`` to the state.

    Single-site terms commute, and ``exp(a X) = cosh(a) I + sinh(a) X`` exactly, so this is
    a sequence of L two-row updates rather than any matrix exponential.
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
    """Module S: apply the diagonal ``exp(f dtau)``.

    The exponent is shifted by its maximum before exponentiating. That changes the result by
    a positive scalar only, which the renormalisation removes anyway, and it is what keeps
    the factor finite for a strongly selected landscape at a large step.
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
    """Module E: one symmetric second-order step, S(dtau/2) M(dtau) S(dtau/2)."""
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
        Steps actually taken. ``tau / dtau`` must be a whole number, so that a comparison
        across step sizes is at genuinely the same total time and not at whatever the
        rounding produced.
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

    This is the reference the Trotter result is measured against for the scaling fit. Using
    it rather than the analytic quasispecies is what isolates the splitting error: comparing
    to the quasispecies instead would fold in the residual from tau being finite, and that
    floor would flatten the fitted exponent.
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
    """The circuit *structure* of one Trotter step, for resource reporting only.

    **This circuit does not implement imaginary-time evolution.** It is the unitary
    real-time analogue with the same interaction pattern: Rz for per-site fitness, Rzz for
    epistatic couplings, Rx for mutation, in the same symmetric S-M-S order. It exists so
    that depth and two-qubit gate counts can be reported as structural resource measures,
    and for the correspondence schematic in the manuscript.

    The hardware-runnable imaginary-time routes are varQITE and Motta-QITE. Anything this
    function returns must be labelled as the structural analogue wherever it is reported.
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
