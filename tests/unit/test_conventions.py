"""Lock the index, ordering, and normalisation conventions.

These tests exist because a wrong convention here is silent: it produces a distribution
that is a permutation of the correct one, non-negative and summing to one, which passes
every eyeball check. The project's history includes two convention bugs of exactly this
shape.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from quasarstack.io.conventions import (
    assert_dense_allowed,
    genotype_to_index,
    hamming_class_collapse,
    hamming_weight,
    index_to_genotype,
    normalise_l1,
    qiskit_bitstring_to_genotype,
)

pytestmark = pytest.mark.fast


@pytest.mark.parametrize(
    ("genotype", "index"),
    [
        ("0", 0),
        ("1", 1),
        ("000", 0),
        ("100", 1),
        ("010", 2),
        ("001", 4),
        ("111", 7),
        ("1010", 5),
    ],
)
def test_genotype_index_hand_computed(genotype: str, index: int) -> None:
    """Site 0 is the leftmost character and contributes 2**0."""
    assert genotype_to_index(genotype) == index
    assert index_to_genotype(index, len(genotype)) == genotype


@given(st.integers(min_value=1, max_value=10), st.integers(min_value=0))
def test_index_genotype_round_trip(n_sites: int, raw: int) -> None:
    index = raw % (1 << n_sites)
    assert genotype_to_index(index_to_genotype(index, n_sites)) == index


def test_qiskit_bitstring_is_reversed() -> None:
    """A Qiskit counts key read directly as a genotype would be reversed."""
    assert qiskit_bitstring_to_genotype("001") == "100"
    assert qiskit_bitstring_to_genotype("1100") == "0011"
    assert qiskit_bitstring_to_genotype("0 1 0") == "010"


def test_qiskit_bitstring_round_trips_through_index() -> None:
    """Qubit 0 carries site 0, so a single excitation lands on site 0 either way."""
    genotype = qiskit_bitstring_to_genotype("001")
    assert genotype_to_index(genotype) == 1


@pytest.mark.parametrize(("index", "weight"), [(0, 0), (1, 1), (3, 2), (7, 3), (255, 8)])
def test_hamming_weight(index: int, weight: int) -> None:
    assert hamming_weight(index) == weight


def test_hamming_class_collapse_preserves_mass() -> None:
    rng = np.random.default_rng(0)
    n_sites = 5
    dist = rng.random(1 << n_sites)
    dist /= dist.sum()
    collapsed = hamming_class_collapse(dist, n_sites)
    assert collapsed.shape == (n_sites + 1,)
    assert collapsed.sum() == pytest.approx(1.0, abs=1e-12)


def test_hamming_class_collapse_on_a_point_mass() -> None:
    n_sites = 4
    dist = np.zeros(1 << n_sites)
    dist[genotype_to_index("1010")] = 1.0
    collapsed = hamming_class_collapse(dist, n_sites)
    assert collapsed[2] == pytest.approx(1.0)
    assert collapsed.sum() == pytest.approx(1.0)


def test_hamming_class_collapse_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="expected shape"):
        hamming_class_collapse(np.ones(7), 3)


def test_normalise_l1_sums_to_one_and_is_non_negative() -> None:
    vector = np.array([-0.5, 0.25, -0.25, 0.0])
    out = normalise_l1(vector)
    assert out.sum() == pytest.approx(1.0)
    assert (out >= 0).all()


def test_normalise_l1_rejects_zero_vector() -> None:
    with pytest.raises(ValueError, match="sum to zero"):
        normalise_l1(np.zeros(4))


def test_dense_guard_allows_small_and_blocks_large() -> None:
    assert_dense_allowed(12)
    with pytest.raises(RuntimeError, match="dense construction forbidden"):
        assert_dense_allowed(14)


@pytest.mark.parametrize("bad", ["", "012", "abc"])
def test_genotype_rejects_non_binary(bad: str) -> None:
    with pytest.raises(ValueError):
        genotype_to_index(bad)


def test_index_to_genotype_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        index_to_genotype(8, 3)
