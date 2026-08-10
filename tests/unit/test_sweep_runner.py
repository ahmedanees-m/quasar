"""The sweep runner's bookkeeping, which is what makes a long run trustworthy.

None of these run a sweep. They check the properties that decide whether a multi-hour run
can be believed afterwards: that the cell enumeration matches the declared grid, that keys
are stable across restarts, and that nothing silently disappears.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("sweep_runner", ROOT / "scripts" / "sweep_runner.py")
sweep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sweep)

pytestmark = pytest.mark.fast


def test_cell_count_matches_the_declared_grid() -> None:
    """If this drifts, the manifest's 'planned' count stops meaning anything and a shrunk
    grid would report itself as fully covered."""
    grid = sweep.REGISTERED_GRID
    seeded = [
        f for f in grid["families"] if f["family"] not in {"single_peak", "additive_pairwise"}
    ]
    unseeded = len(grid["families"]) - len(seeded)
    expected = (
        len(grid["sizes"]) * len(grid["mu_ratios"]) * (len(seeded) * len(grid["seeds"]) + unseeded)
    )
    assert len(list(sweep.cells(grid))) == expected


def test_cell_keys_are_unique_and_stable() -> None:
    """Resumption skips by key, so a collision would silently drop a cell and a key that
    changed between runs would repeat every cell forever."""
    keys = [sweep.cell_key(c) for c in sweep.cells(sweep.REGISTERED_GRID)]
    assert len(keys) == len(set(keys))
    again = [sweep.cell_key(c) for c in sweep.cells(sweep.REGISTERED_GRID)]
    assert keys == again


def test_every_declared_family_can_be_built() -> None:
    """A family named in the grid but missing from the builder would fail hours into a run."""
    for spec in sweep.REGISTERED_GRID["families"]:
        fitness = sweep.build_fitness(spec, 6, seed=0)
        assert fitness.shape == (64,)
        assert np.all(np.isfinite(fitness))


def test_every_family_has_a_positive_threshold() -> None:
    """mu_c scales the whole mutation axis, so a zero or negative value would put every
    cell of that family at mu = 0 without anything complaining."""
    for spec in sweep.REGISTERED_GRID["families"]:
        fitness = sweep.build_fitness(spec, 8, seed=0)
        assert sweep.threshold_for(spec, fitness, 8) > 0.0, spec


def test_scoring_reports_both_metrics() -> None:
    """Section 11.4 requires cosine and total variation both, and makes total variation the
    deciding one where they disagree. Storing only the flattering metric is the failure."""
    reference = np.array([0.7, 0.2, 0.06, 0.04])
    scored = sweep.score(np.array([0.6, 0.3, 0.06, 0.04]), reference)
    assert set(scored) == {"cosine", "total_variation"}
    assert 0.0 <= scored["cosine"] <= 1.0
    assert scored["total_variation"] == pytest.approx(0.1)


def test_budget_table_covers_every_declared_size() -> None:
    for n_sites in sweep.REGISTERED_GRID["sizes"]:
        assert n_sites in sweep.BUDGET_SECONDS, n_sites


def test_baseline_b_refuses_outside_its_class_rather_than_guessing() -> None:
    """The refusal is the point: a baseline that quietly solved an out-of-class cell would
    report coverage it does not have, and the boundary map would inherit that."""
    rugged = sweep.build_fitness({"family": "nk", "K": 2}, 6, seed=0)
    outcome = sweep.method_baseline_b(rugged, 0.2, budget=10.0)
    assert outcome["applicable"] is False
    assert "polynomial-time class" in outcome["reason"]

    easy = sweep.build_fitness({"family": "single_peak"}, 6, seed=0)
    assert sweep.method_baseline_b(easy, 0.2, budget=10.0)["applicable"] is True
