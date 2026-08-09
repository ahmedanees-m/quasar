"""Structure of the mutation-selection generator: Perron form, and reversibility.

WP1 task T1.1, which the execution plan flags with an explicit warning: "do not overclaim
nonreversibility; derive what is actually true for each landscape family". This module makes
that a measurement instead of an argument.

Why it matters
--------------

Execution plan v4 rests Route B on Claudon, Piquemal and Monmarche (2025), whose
beyond-quadratic speedup applies to *nonreversible* Markov chains. Two separate properties
get conflated easily and are worth stating apart:

**Non-conservative.** The rows of the mutation-selection generator sum to the fitness, not
to zero. It is not a Markov generator at all. This is true, and it is a real obstacle for
any result stated for Markov kernels.

**Nonreversible.** A generator is reversible when some positive measure pi satisfies
detailed balance, ``pi_i W_ij = pi_j W_ji``. Equivalently it is self-adjoint in the weighted
inner product, so its spectrum is real and the machinery for reversible chains applies. This
is the property Claudon et al. exploit the absence of.

The two are independent. An operator can be non-conservative and still perfectly reversible,
and the functions here determine which.

How reversibility is tested
---------------------------

Detailed balance on a connected graph determines pi up to scale: fix ``pi`` at one vertex and
propagate ``pi_j = pi_i W_ij / W_ji`` along a spanning tree. If the operator is reversible the
propagated measure satisfies detailed balance on *every* edge, including the ones outside the
tree. If it is not, the edges outside the tree disagree, and the size of that disagreement is
the reversibility defect. This is Kolmogorov's cycle criterion in constructive form: it both
decides the question and hands back the measure when the answer is yes.
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
        Rate of the reverse. Defaults to ``mu_forward``, which is the symmetric Crow-Kimura
        case the rest of this project implements.
    context_strength
        Multiplies the *forward* flip rate at site i by ``1 + context_strength`` when the
        neighbouring site ``i - 1`` is mutated, leaving the back-mutation rate untouched.
        Zero recovers independent per-site mutation.

        The one-sidedness is the whole point, and it is both the biologically faithful
        choice and the mathematically necessary one. CpG hypermutation and APOBEC motif
        preference raise C to T in a motif; they do not raise T to C by the same factor. And
        a context factor applied to *both* directions cancels out of Kolmogorov's cycle
        condition, because the rate then separates into a direction-dependent part times a
        context-dependent part, so the chain stays reversible. Measured, not assumed: see
        `results/wp0/prior_art_iv_4.json`.

    Returns
    -------
    ndarray
        Dense ``2**L`` by ``2**L`` array. Off-diagonal entry ``[i, j]`` is the rate of the
        transition from genotype j into genotype i, and the diagonal carries the negative
        column sums so that mutation alone conserves probability.
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
        The positive measure satisfying detailed balance, normalised to sum to one, or
        ``None`` when the generator is not reversible.
    defect
        Maximum relative violation of detailed balance over every edge, using the measure
        propagated along a spanning tree. Zero means reversible. This is the number to
        report; the boolean is a convenience derived from it.

    Notes
    -----
    Only the off-diagonal structure matters. The diagonal is whatever makes the generator
    conservative or not, and detailed balance says nothing about it, which is exactly why a
    non-conservative operator can still be reversible.
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
    """Everything WP1 needs to say about the structure of one generator.

    Reports non-conservation and non-reversibility separately, because they are separate
    properties and the execution plan's Route B argument depends on which one holds.
    """
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
