"""Reproducibility of the sparse eigensolver, which is not free and was not there.

`scipy.sparse.linalg.eigsh` starts from NumPy's global random state unless told otherwise,
so every process starts somewhere different and stops at a slightly different point. These
tests fail if any `eigsh` call in `quasarstack` stops passing a fixed start. See ADR-0016.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from quasarstack.analytic.exact_diag import perron_vector
from quasarstack.classical.landscapes import nk_fitness
from quasarstack.numerics import deterministic_start
from quasarstack.spectral.gap import sparse_gap

pytestmark = pytest.mark.fast

ROOT = Path(__file__).resolve().parents[2]


def test_start_vector_is_the_same_every_call() -> None:
    first = deterministic_start(64)
    np.random.seed(12345)  # noqa: NPY002 - deliberately disturbing the global state
    np.random.random(1000)  # noqa: NPY002
    assert np.array_equal(first, deterministic_start(64))


@pytest.mark.parametrize("n_sites", [10, 12])
def test_sparse_gap_is_bit_identical_across_global_rng_states(n_sites: int) -> None:
    """The failure this catches: G-R.4's measured gap decay moved from
    ...588269 to ...588261 between two runs of identical code, because ARPACK had started
    somewhere else. A fifteenth-digit wobble is not a rounding detail here; it is the
    difference between ADR-0009 classifying a rerun as provenance-only and as a finding."""
    fitness = nk_fitness(n_sites, 2, seed=0)

    np.random.seed(1)  # noqa: NPY002
    first = sparse_gap(fitness, 0.1)
    np.random.seed(999)  # noqa: NPY002
    np.random.random(5000)  # noqa: NPY002
    second = sparse_gap(fitness, 0.1)

    assert first == second, f"sparse_gap moved by {abs(first - second):.3e} between runs"


def test_perron_vector_is_bit_identical_across_global_rng_states() -> None:
    fitness = nk_fitness(13, 2, seed=0)  # above the dense limit, so the sparse path runs

    np.random.seed(7)  # noqa: NPY002
    first = perron_vector(fitness, 0.1)
    np.random.seed(4242)  # noqa: NPY002
    np.random.random(5000)  # noqa: NPY002
    second = perron_vector(fitness, 0.1)

    # perron_vector returns a tuple mixing an array with scalars, so compare part by part;
    # np.array_equal on the tuple itself is False whatever the contents are.
    assert len(first) == len(second)
    for left, right in zip(first, second, strict=True):
        if isinstance(left, np.ndarray):
            assert np.array_equal(left, right)
        else:
            assert left == right


def test_no_eigsh_call_in_the_package_omits_its_start_vector() -> None:
    """A grep, because the two tests above only cover the call sites they happen to reach.

    A new `eigsh` without `v0` would reintroduce the defect somewhere neither test looks,
    and it would pass every accuracy check while doing so.
    """
    offenders = []
    for path in (ROOT / "quasarstack").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"eigsh\s*\(", text):
            # Take the call's argument text up to the matching close paren.
            depth, index = 0, match.end() - 1
            while index < len(text):
                if text[index] == "(":
                    depth += 1
                elif text[index] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                index += 1
            if "v0=" not in text[match.end() : index]:
                line = text[: match.start()].count("\n") + 1
                offenders.append(f"{path.relative_to(ROOT).as_posix()}:{line}")
    assert not offenders, f"eigsh called without a fixed v0 at: {offenders}"
