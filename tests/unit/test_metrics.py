"""Scoring metrics, including the case that motivates reporting both.

Cosine is the flattering one and total variation is the conservative one, which is why
`GATES.md` section 11.4 lets total variation decide where they disagree. That divergence is
worth a test of its own, so the reason both are carried is visible in the suite and not only
in a docstring.
"""

from __future__ import annotations

import numpy as np
import pytest

from quasarstack.scoring.metrics import cosine_similarity, score, total_variation

pytestmark = pytest.mark.fast


def test_identical_distributions() -> None:
    p = np.array([0.5, 0.3, 0.2])
    assert cosine_similarity(p, p) == pytest.approx(1.0)
    assert total_variation(p, p) == pytest.approx(0.0)


def test_orthogonal_distributions() -> None:
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert cosine_similarity(p, q) == pytest.approx(0.0)
    assert total_variation(p, q) == pytest.approx(1.0)


def test_cosine_ignores_scale_and_total_variation_does_not() -> None:
    p = np.array([0.6, 0.4])
    assert cosine_similarity(p, 5.0 * p) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="not 1"):
        total_variation(p, 5.0 * p)


def test_cosine_flatters_a_wrong_tail_that_total_variation_catches() -> None:
    """The reason both are reported.

    Two distributions that agree on where the mass is but differ by an order of magnitude
    across the tail: cosine barely moves, total variation registers the disagreement.
    """
    p = np.zeros(64)
    p[0] = 0.9
    p[1:] = 0.1 / 63

    q = np.zeros(64)
    q[0] = 0.9
    q[1:32] = 0.09 / 31
    q[32:] = 0.01 / 32

    cos = cosine_similarity(p, q)
    tv = total_variation(p, q)
    assert cos > 0.999, "cosine should look reassuring here"
    assert tv > 20 * (1.0 - cos), "total variation should be the louder of the two"


def test_score_returns_both() -> None:
    p = np.array([0.7, 0.3])
    q = np.array([0.6, 0.4])
    result = score(p, q)
    assert set(result) == {"cosine", "tv"}
    assert result["tv"] == pytest.approx(0.1)


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        cosine_similarity(np.ones(3), np.ones(4))


def test_zero_vector_is_rejected() -> None:
    with pytest.raises(ValueError, match="undefined"):
        cosine_similarity(np.zeros(3), np.ones(3))
