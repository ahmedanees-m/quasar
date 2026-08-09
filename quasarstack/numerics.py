"""Numerical house rules that more than one module has to obey identically.

Currently one rule, and it exists because breaking it is invisible.

Why `eigsh` needs a starting vector
-----------------------------------

`scipy.sparse.linalg.eigsh` accepts `v0`, the vector ARPACK starts its Krylov iteration from.
Left unset, SciPy draws one from **NumPy's global random state**, which is seeded from OS
entropy when NumPy is imported. Every process therefore starts from a different vector, and
ARPACK stops as soon as its residual is under tolerance, so it stops at a slightly different
point each time.

The result is a computation that is correct to about `1e-14` and **not reproducible**. It
usually agrees to the last bit, which is worse than never agreeing, because the disagreement
only appears occasionally and looks like a real change when it does. Gate G-R.4 reproduced
bit-identically across several reruns and then, on the run that closed WP-R, moved its
measured gap decay from `0.7167436421588269` to `0.7167436421588261`. Nothing had changed but
the starting vector.

The project's first engineering principle is that every claim maps to a re-runnable artefact.
An artefact that re-runs to a different number in its fifteenth digit does not satisfy that,
and ADR-0009's rule for telling a provenance-only rerun from a real finding cannot work if
the numbers move on their own. See ADR-0016.

A fixed pseudo-random start is used rather than a constant vector such as all-ones. All-ones
happens to have large overlap with the Perron vector of this particular operator, which would
work here and hide the problem the day someone points these functions at an operator whose
target eigenvector is orthogonal to it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["ARPACK_SEED", "deterministic_start"]

ARPACK_SEED = 20260810


def deterministic_start(dimension: int, seed: int = ARPACK_SEED) -> NDArray[np.float64]:
    """A fixed starting vector for ARPACK, so that two runs give the same digits.

    Pass as ``eigsh(..., v0=deterministic_start(n))``. Every `eigsh` call in `quasarstack`
    does; `tests/unit/test_numerics.py` fails if one stops.
    """
    return np.random.default_rng(seed).standard_normal(dimension)
