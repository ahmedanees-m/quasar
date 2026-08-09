"""Scoring: how two distributions are compared, and what each number hides.

Two metrics, reported together everywhere, because they fail differently.

**Cosine similarity** is the field's habit and is what the pre-registered thresholds are
written in. It is also flattering: it is dominated by wherever the mass is, so a method that
gets the master sequence right and the tail wrong scores well. On a concentrated
quasispecies, cosine above 0.99 can coexist with a tail that is wrong by orders of
magnitude.

**Total-variation distance** is the conservative one. It is the largest probability any
event can disagree by, so it notices the tail. Where the two metrics disagree about a
result, `GATES.md` section 11.4 makes total variation the one that decides.

Reporting only cosine would be the flattering choice, which is why both are returned by
the same call and both go into every record.
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

    Both inputs are expected to be L1-normalised already; this function does not renormalise
    them, because silently rescaling an input that was meant to sum to one would hide the
    bug that made it not sum to one.
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
    """Both metrics at once, which is the only way they should be reported."""
    return {
        "cosine": cosine_similarity(p, q),
        "tv": total_variation(p, q),
    }
