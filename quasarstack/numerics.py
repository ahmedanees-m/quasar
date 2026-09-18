"""Shared numerical settings.

`scipy.sparse.linalg.eigsh` starts its Krylov iteration from `v0`. Left unset, SciPy draws it
from NumPy's global random state, so repeated runs stop at slightly different points and results
differ in the last digits. Every `eigsh` call in the package passes the fixed start vector below.

The start vector is pseudo-random rather than constant: a vector such as all-ones overlaps
strongly with the Perron vector of this operator and would not generalise to operators whose
target eigenvector is orthogonal to it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["ARPACK_SEED", "deterministic_start"]

ARPACK_SEED = 20260810


def deterministic_start(dimension: int, seed: int = ARPACK_SEED) -> NDArray[np.float64]:
    """A fixed starting vector for ARPACK, so that repeated runs give identical digits.

    Pass as ``eigsh(..., v0=deterministic_start(n))``.
    """
    return np.random.default_rng(seed).standard_normal(dimension)
