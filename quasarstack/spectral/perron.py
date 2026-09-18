"""Structure of the mutation-selection generator: Perron form and reversibility.

Two properties are distinct:

**Non-conservative.** The columns of the generator sum to the fitness, not to zero, so it is not
a Markov generator, and results stated for Markov kernels do not apply directly.

**Nonreversible.** A generator is reversible when a positive measure pi satisfies detailed
balance, ``pi_i W_ij = pi_j W_ji``, making it self-adjoint in the weighted inner product with a
real spectrum. Speedups for nonreversible Markov chains, such as that of Claudon, Piquemal and
Monmarche (2025), rely on its absence.

An operator can be non-conservative and reversible; the functions here determine which holds.

Testing reversibility
---------------------

On a connected graph detailed balance determines pi up to scale: fix ``pi`` at one vertex and
propagate ``pi_j = pi_i W_ij / W_ji`` along a spanning tree. For a reversible operator the
propagated measure satisfies detailed balance on every edge, including those outside the tree;
otherwise those edges disagree, and the size of the disagreement is the reversibility defect.
This is Kolmogorov's cycle criterion in constructive form.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray


def mutation_generator(
    n_sites: int,
    mu_forward: float,
    mu_backward: float | None = None,
    context_strength: float = 0.0,
) -> NDArray[np.float64]:
    """Mutation part of the generator, in three increasingly general forms.

    Parameters
    ----------
    n_sites
        Number of loci, L.
    mu_forward
        Rate of wild type turning into mutant at a site.
    mu_backward
        Rate of the reverse. Defaults to ``mu_forward``, the symmetric Crow-Kimura case.
    context_strength
        Multiplies the forward flip rate at site i by ``1 + context_strength`` when site ``i - 1``
        is mutated, leaving the back-mutation rate unchanged. Zero recovers independent per-site
        mutation.

        The factor applies to one direction, as in CpG hypermutation or APOBEC motif preference,
        which raise C to T in a motif but not T to C. A context factor applied to both directions
        cancels from Kolmogorov's cycle condition and leaves the chain reversible
        (`results/wp0/wp0_reversibility.json`).

    Returns
    -------
    ndarray
        Dense ``2**L`` by ``2**L`` array. Off-diagonal entry ``[i, j]`` is the rate from genotype j
        into genotype i, and the diagonal carries the negative column sums, so mutation alone
        conserves probability.
    """
    if mu_backward is None:
        mu_backward = mu_forward
    dim = 1 << n_sites
    operator = np.zeros((dim, dim), dtype=np.float64)

    for source in range(dim):
        for site in range(n_sites):
            target = source ^ (1 << site)
            mutated_at_site = bool(source >> site & 1)
            rate = mu_backward if mutated_at_site else mu_forward
            if context_strength != 0.0 and not mutated_at_site:
                neighbour = (site - 1) % n_sites
                if source >> neighbour & 1:
                    rate *= 1.0 + context_strength
            operator[target, source] += rate
            operator[source, source] -= rate

    return operator


def selection_generator(fitness: NDArray[np.float64]) -> NDArray[np.float64]:
    """Selection part of the generator: a diagonal matrix of Malthusian fitness."""
    return np.diag(np.asarray(fitness, dtype=np.float64))


def symmetrising_measure(
    generator: NDArray[np.float64], tolerance: float = 1e-12
) -> tuple[NDArray[np.float64] | None, float]:
    """Recover the measure that makes the generator self-adjoint, if one exists.

    Returns
    -------
    measure
        The positive measure satisfying detailed balance, normalised to sum to one, or ``None``
        when the generator is not reversible.
    defect
        Maximum relative violation of detailed balance over every edge, using the measure
        propagated along a spanning tree. Zero means reversible.

    Notes
    -----
    Only the off-diagonal structure matters; detailed balance says nothing about the diagonal,
    which is why a non-conservative operator can be reversible.
    """
    generator = np.asarray(generator, dtype=np.float64)
    dim = generator.shape[0]
    if generator.shape != (dim, dim):
        raise ValueError(f"expected a square matrix, got {generator.shape}")

    # Edges present in both directions. A one-directional edge makes detailed balance
    # impossible outright, so it is reported as an infinite defect rather than skipped.
    forward = generator.copy()
    np.fill_diagonal(forward, 0.0)
    present = np.abs(forward) > tolerance
    if not np.array_equal(present, present.T):
        return None, float("inf")

    log_measure = np.full(dim, np.nan)
    log_measure[0] = 0.0
    queue = deque([0])
    while queue:
        current = queue.popleft()
        for neighbour in np.flatnonzero(present[:, current]):
            if not np.isnan(log_measure[neighbour]):
                continue
            # detailed balance: m[current] * W[neighbour, current]
            #                 = m[neighbour] * W[current, neighbour]
            ratio = forward[neighbour, current] / forward[current, neighbour]
            if ratio <= 0.0:
                return None, float("inf")
            log_measure[neighbour] = log_measure[current] + np.log(ratio)
            queue.append(int(neighbour))

    if np.isnan(log_measure).any():
        raise ValueError("the transition graph is disconnected; detailed balance is not defined")

    log_measure -= log_measure.max()
    measure = np.exp(log_measure)

    rows, cols = np.nonzero(present)
    left = measure[cols] * forward[rows, cols]
    right = measure[rows] * forward[cols, rows]
    scale = np.maximum(np.abs(left), np.abs(right))
    defect = float(np.max(np.abs(left - right) / np.where(scale > 0, scale, 1.0)))

    if defect > 1e-9:
        return None, defect
    return measure / measure.sum(), defect


def reversibility_report(generator: NDArray[np.float64]) -> dict[str, object]:
    """Structure of one generator: non-conservation and non-reversibility, reported separately."""
    generator = np.asarray(generator, dtype=np.float64)
    measure, defect = symmetrising_measure(generator)

    off_diagonal = generator - np.diag(np.diag(generator))
    column_sums = generator.sum(axis=0)

    return {
        "dimension": int(generator.shape[0]),
        "is_symmetric": bool(np.allclose(generator, generator.T, atol=1e-12)),
        "symmetry_defect": float(np.max(np.abs(generator - generator.T))),
        "is_conservative": bool(np.allclose(column_sums, 0.0, atol=1e-12)),
        "max_abs_column_sum": float(np.max(np.abs(column_sums))),
        "off_diagonal_non_negative": bool(off_diagonal.min() >= -1e-15),
        "is_reversible": measure is not None,
        "reversibility_defect": defect,
        "stationary_measure_is_uniform": bool(
            measure is not None and np.allclose(measure, measure[0], atol=1e-10)
        ),
    }
