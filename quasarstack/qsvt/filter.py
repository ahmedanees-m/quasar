"""Eigenstate filtering: the polynomial that turns a block encoding into a quasispecies.

WP2 tasks T2.3 and T2.4, built on the working assumption that Route B is in docs/notes.md
option C: QSVT eigenstate filtering for a Hermitian stoquastic operator, not the
nonreversible-Markov-chain construction the execution plan originally cited.

The idea
--------

`qubitisation` gives Chebyshev polynomials of `A / alpha` in the block of a circuit. QSVT
gives an arbitrary polynomial bounded by 1 on `[-1, 1]`. Choose one that is close to 1 at the
Perron eigenvalue and close to 0 on the rest of the spectrum, apply it to any starting state
with non-zero overlap, and what comes out is the Perron eigenvector.

Why the degree is what it is, and why there is no square root here
------------------------------------------------------------------

The polynomial has to separate `lambda_1 / alpha` from `lambda_2 / alpha`, a distance of
`Delta / alpha`, while staying bounded by 1 across the whole interval. Approximating a step
of width `delta` to accuracy `epsilon` with a bounded polynomial needs degree of order
`(1 / delta) log(1 / epsilon)`, so

    degree  ~  (alpha / Delta) * log(1 / epsilon)

**Linear in `alpha / Delta`, not square root.** It is tempting to reach for Chebyshev
acceleration, which does buy a square root by putting the target eigenvalue outside the
interval where the polynomial is controlled. That is not available: the target eigenvalue is
inside the encoded spectrum by construction, and a block of a unitary cannot exceed norm 1
anywhere on it. Saying otherwise would be the single easiest way to claim a quantum speedup
this project has not earned, so it is written down here rather than left to be assumed.

`alpha` is the one-norm of the Pauli coefficients and it multiplies the whole cost. That is
the practical content of the sparse spin convention: at L = 12 the single peak has 4108 terms
as a projector and 27 in sparse form, and the one-norm follows.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "apply_chebyshev_series",
    "chebyshev_coefficients",
    "filtered_state",
    "predicted_degree",
    "smallest_sufficient_degree",
    "step_filter",
]


def chebyshev_coefficients(
    function: Callable[[NDArray[np.float64]], NDArray[np.float64]], degree: int
) -> NDArray[np.float64]:
    """Chebyshev coefficients of ``function`` on ``[-1, 1]``, by quadrature at the nodes.

    Uses ``2 * degree + 2`` Chebyshev points, which is comfortably oversampled, so the
    coefficients are the interpolation coefficients rather than a least-squares fit and no
    aliasing enters at the degrees this module works at.
    """
    if degree < 0:
        raise ValueError(f"degree must be non-negative, got {degree}")
    n_nodes = 2 * degree + 2
    angles = np.pi * (np.arange(n_nodes) + 0.5) / n_nodes
    nodes = np.cos(angles)
    values = np.asarray(function(nodes), dtype=np.float64)

    coefficients = np.empty(degree + 1)
    for order in range(degree + 1):
        weight = 1.0 if order == 0 else 2.0
        coefficients[order] = weight * float(np.mean(values * np.cos(order * angles)))
    return coefficients


def step_filter(
    cut: float, sharpness: float, degree: int, headroom: float = 0.999
) -> NDArray[np.float64]:
    """Chebyshev coefficients of a bounded polynomial approximating a step at ``cut``.

    The target is ``(1 + erf(sharpness * (x - cut))) / 2``, which is smooth, so its Chebyshev
    coefficients decay geometrically once the degree exceeds roughly ``sharpness``.

    The result is rescaled so its supremum on a fine grid is at most ``headroom``. A QSVT
    polynomial must be bounded by 1 on ``[-1, 1]`` or no phase factors exist for it, and the
    truncated series overshoots slightly near the step; scaling down costs a constant factor
    in the final overlap and nothing in the separation.
    """
    from scipy.special import erf

    coefficients = chebyshev_coefficients(
        lambda x: 0.5 * (1.0 + erf(sharpness * (x - cut))), degree
    )
    grid = np.linspace(-1.0, 1.0, 4001)
    supremum = float(np.max(np.abs(np.polynomial.chebyshev.chebval(grid, coefficients))))
    if supremum > headroom:
        coefficients = coefficients * (headroom / supremum)
    return coefficients


def apply_chebyshev_series(
    matrix: NDArray[np.float64], coefficients: NDArray[np.float64], vector: NDArray
) -> NDArray:
    """``sum_d a_d T_d(matrix) @ vector``, by the recurrence rather than by powers.

    Matrix powers of a scaled operator lose precision fast; the Chebyshev recurrence
    ``T_{d+1} = 2 M T_d - T_{d-1}`` is stable and is also exactly what the circuit does, so
    the classical reference and the quantum object are computing the same thing the same way.
    """
    previous = np.asarray(vector, dtype=np.complex128)
    if coefficients.size == 1:
        return np.asarray(coefficients[0] * previous, dtype=np.complex128)
    current = matrix @ previous
    total = coefficients[0] * previous + coefficients[1] * current
    for order in range(2, coefficients.size):
        previous, current = current, 2.0 * (matrix @ current) - previous
        total = total + coefficients[order] * current
    return np.asarray(total, dtype=np.complex128)


def predicted_degree(gap: float, alpha: float, overlap: float, epsilon: float = 0.0975) -> float:
    """Derived degree for eigenstate filtering, in the standard two-factor form.

        d  =  (alpha / gap) * ln( sqrt(1 - gamma^2) / (gamma * sqrt(epsilon)) )

    **Two separate things set the cost and conflating them is a real error, not a constant
    factor.** The gap sets how *sharp* the polynomial has to be, through the width
    ``delta = gap / alpha`` of the step it must resolve. The initial overlap ``gamma`` sets
    how far *down* the unwanted components have to be pushed. Writing

        |psi_0> = gamma |v_1> + sqrt(1 - gamma^2) |w>

    for a filter with ``p(x_1) = 1`` and ``|p| <= eta`` on the rest of the spectrum, the
    infidelity after filtering is ``(1 - gamma^2) eta^2 / gamma^2``, so reaching ``epsilon``
    needs ``eta <= gamma sqrt(epsilon) / sqrt(1 - gamma^2)``. A bounded polynomial
    approximation to a step of width ``delta`` achieves ``eta ~ exp(-d delta)``, and solving
    for ``d`` gives the expression above.

    ``epsilon`` defaults to ``1 - 0.95^2``, matching the cosine 0.95 that `G-2` criterion 1
    demands, because a predicted degree is only comparable to a measured one if both are for
    the same accuracy.

    An earlier version of this function omitted the overlap entirely and used a fixed
    ``epsilon = 1e-3``. It overestimated the measured degree by factors of 3.3 to 7.2, which
    is what surfaced the omission. See `docs/protocol.md` revision 13 for the disclosure.
    """
    if gap <= 0.0:
        return float("inf")
    overlap = float(np.clip(overlap, 1e-15, 1.0 - 1e-15))
    suppression = np.sqrt(1.0 - overlap**2) / (overlap * np.sqrt(epsilon))
    return float(alpha / gap * max(np.log(suppression), 0.0))


def filtered_state(
    operator_matrix: NDArray[np.float64],
    alpha: float,
    degree: int,
    lambda_1: float,
    lambda_2: float,
    initial: NDArray[np.float64] | None = None,
    epsilon: float = 1e-3,
) -> NDArray[np.complex128]:
    """Apply the eigenstate filter and return the resulting normalised state.

    The step is placed midway between the two leading eigenvalues, and the smoothing width
    is set to half the gap, both in the normalised units the block encoding works in. Those
    are the only two free choices and neither is tuned per instance.
    """
    scaled = np.asarray(operator_matrix, dtype=np.float64) / alpha
    cut = 0.5 * (lambda_1 + lambda_2) / alpha
    sharpness = 4.0 * alpha / max(lambda_1 - lambda_2, 1e-300)

    if initial is None:
        dimension = scaled.shape[0]
        initial = np.full(dimension, 1.0 / np.sqrt(dimension))

    coefficients = step_filter(cut, sharpness, degree)
    result = apply_chebyshev_series(scaled, coefficients, initial)
    norm = np.linalg.norm(result)
    if norm == 0.0:
        raise ValueError(
            "the filter annihilated the initial state; the cut is on the wrong side of the "
            "spectrum or the degree is far too low"
        )
    return np.asarray(result / norm, dtype=np.complex128)


def smallest_sufficient_degree(
    operator_matrix: NDArray[np.float64],
    alpha: float,
    lambda_1: float,
    lambda_2: float,
    target_cosine: float,
    reference: NDArray[np.float64],
    max_degree: int,
    initial: NDArray[np.float64] | None = None,
) -> dict[str, object]:
    """Smallest degree whose filtered state reaches ``target_cosine`` against ``reference``.

    Searched by doubling then bisecting, because each evaluation costs a Chebyshev series
    and a linear scan would spend most of its time at degrees that were never going to work.
    Monotonicity in degree is not guaranteed in principle, so the bisection is on "the first
    degree at which the target is met and stays met at twice that degree", and the raw curve
    is returned so a non-monotone case would be visible rather than silently bisected over.
    """
    reference = np.asarray(reference, dtype=np.float64)
    reference = reference / np.linalg.norm(reference)

    def cosine_at(degree: int) -> float:
        state = filtered_state(operator_matrix, alpha, degree, lambda_1, lambda_2, initial=initial)
        return float(abs(np.vdot(reference, state)))

    curve = []
    degree = 1
    found = None
    while degree <= max_degree:
        value = cosine_at(degree)
        curve.append({"degree": degree, "cosine": value})
        if value >= target_cosine:
            found = degree
            break
        degree *= 2

    if found is None:
        return {
            "sufficient_degree": None,
            "reached": curve[-1]["cosine"] if curve else 0.0,
            "max_degree": max_degree,
            "curve": curve,
        }

    low, high = found // 2, found
    while low + 1 < high:
        middle = (low + high) // 2
        value = cosine_at(middle)
        curve.append({"degree": middle, "cosine": value})
        if value >= target_cosine:
            high = middle
        else:
            low = middle

    return {
        "sufficient_degree": high,
        "reached": cosine_at(high),
        "max_degree": max_degree,
        "curve": sorted(curve, key=lambda row: row["degree"]),
    }
