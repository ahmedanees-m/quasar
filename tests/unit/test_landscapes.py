"""Landscape construction, checked against hand-computed values.

The spin convention is the thing being locked here. A projector-convention slip changes
every fitness by a constant and a factor, which shifts the quasispecies without making it
look wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from quasarstack.classical.landscapes import (
    additive_fitness,
    class_fitness,
    single_peak_classes,
    spin_matrix,
    uniform_additive_classes,
)
from quasarstack.io.conventions import genotype_to_index

pytestmark = pytest.mark.fast


def test_spin_matrix_is_plus_one_for_wild_type() -> None:
    z = spin_matrix(2)
    assert z.shape == (2, 4)
    # genotype "00" is index 0, both sites wild type
    assert z[0, 0] == 1 and z[1, 0] == 1
    # genotype "10" is index 1: site 0 mutated, site 1 wild type
    assert z[0, 1] == -1 and z[1, 1] == 1
    # genotype "01" is index 2: site 0 wild type, site 1 mutated
    assert z[0, 2] == 1 and z[1, 2] == -1


def test_additive_fitness_hand_computed_l2() -> None:
    """f = a0 z0 + a1 z1 with a = (1.0, 0.5), evaluated on all four genotypes."""
    a = np.array([1.0, 0.5])
    f = additive_fitness(a)
    assert f[genotype_to_index("00")] == pytest.approx(1.5)
    assert f[genotype_to_index("10")] == pytest.approx(-0.5)
    assert f[genotype_to_index("01")] == pytest.approx(0.5)
    assert f[genotype_to_index("11")] == pytest.approx(-1.5)


def test_pairwise_epistasis_hand_computed_l2() -> None:
    """Only the strict upper triangle of b is read, so a symmetric b is not double counted."""
    a = np.zeros(2)
    b = np.array([[0.0, 0.7], [0.7, 0.0]])
    f = additive_fitness(a, b)
    assert f[genotype_to_index("00")] == pytest.approx(0.7)
    assert f[genotype_to_index("10")] == pytest.approx(-0.7)
    assert f[genotype_to_index("11")] == pytest.approx(0.7)


def test_class_fitness_assigns_by_hamming_weight() -> None:
    f_by_class = np.array([5.0, 3.0, 1.0])
    f = class_fitness(f_by_class)
    assert f[genotype_to_index("00")] == pytest.approx(5.0)
    assert f[genotype_to_index("10")] == pytest.approx(3.0)
    assert f[genotype_to_index("01")] == pytest.approx(3.0)
    assert f[genotype_to_index("11")] == pytest.approx(1.0)


def test_single_peak_is_a_spike_on_the_master_sequence() -> None:
    f = class_fitness(single_peak_classes(4, 2.5))
    assert f[0] == pytest.approx(2.5)
    assert np.all(f[1:] == 0.0)


@pytest.mark.parametrize("n_sites", [2, 3, 5, 8])
@pytest.mark.parametrize("a_value", [0.25, 1.0, 2.0])
def test_uniform_additive_matches_the_class_form(n_sites: int, a_value: float) -> None:
    """The two constructions of the same landscape must agree entry by entry.

    This equivalence is what lets gate G-R.1 cross-check the closed-form product solution
    against the Hamming-class reduction, so it is worth a test of its own.
    """
    direct = additive_fitness(np.full(n_sites, a_value))
    via_classes = class_fitness(uniform_additive_classes(n_sites, a_value))
    assert np.allclose(direct, via_classes, atol=1e-14)


def test_additive_rejects_wrong_shaped_coupling() -> None:
    with pytest.raises(ValueError, match="b must have shape"):
        additive_fitness(np.zeros(3), np.zeros((2, 2)))


def test_guard_rejects_absurd_size() -> None:
    with pytest.raises(ValueError, match="guard is set at"):
        spin_matrix(40)
