"""Comparing two distributions.

Two metrics are reported together.

**Cosine similarity** is dominated by where the mass is, so a method that gets the master
sequence right and the tail wrong scores well: on a concentrated quasispecies, cosine above 0.99
can coexist with a tail that is wrong by orders of magnitude.

**Total-variation distance** is the largest probability by which any event can disagree, so it
is sensitive to the tail.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _validate_pair(p: NDArray[np.float64], q: NDArray[np.float64]) -> None:
    if p.shape != q.shape:
        raise ValueError(f"shape mismatch: {p.shape} against {q.shape}")
    if p.size == 0:
        raise ValueError("cannot compare empty distributions")


def cosine_similarity(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """Cosine similarity between two non-negative vectors.

    Normalisation-free by construction, so it does not care whether the inputs are L1 or L2
    normalised, only about direction.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    _validate_pair(p, q)
    norm = float(np.linalg.norm(p) * np.linalg.norm(q))
    if norm <= 0.0:
        raise ValueError("cosine similarity is undefined when either vector is zero")
    return float(np.dot(p, q) / norm)


def total_variation(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """Total-variation distance between two probability distributions.

    Both inputs are expected to be L1-normalised; they are not renormalised here.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    _validate_pair(p, q)
    for name, vector in (("p", p), ("q", q)):
        total = float(vector.sum())
        if not np.isclose(total, 1.0, atol=1e-9):
            raise ValueError(
                f"{name} sums to {total}, not 1; total variation expects a distribution"
            )
    return float(0.5 * np.abs(p - q).sum())


def score(p: NDArray[np.float64], q: NDArray[np.float64]) -> dict[str, float]:
    """Both metrics at once."""
    return {
        "cosine": cosine_similarity(p, q),
        "tv": total_variation(p, q),
    }
