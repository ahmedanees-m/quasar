"""Cell enumeration, keys, builders and scoring in the sweep runner."""

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


def test_cell_count_matches_the_grid() -> None:
    grid = sweep.FULL_GRID
    seeded = [
        f for f in grid["families"] if f["family"] not in {"single_peak", "additive_pairwise"}
    ]
    unseeded = len(grid["families"]) - len(seeded)
    expected = (
        len(grid["sizes"]) * len(grid["mu_ratios"]) * (len(seeded) * len(grid["seeds"]) + unseeded)
    )
    assert len(list(sweep.cells(grid))) == expected


def test_cell_keys_are_unique_and_stable() -> None:
    """Resumption skips cells by key."""
    keys = [sweep.cell_key(c) for c in sweep.cells(sweep.FULL_GRID)]
    assert len(keys) == len(set(keys))
    again = [sweep.cell_key(c) for c in sweep.cells(sweep.FULL_GRID)]
    assert keys == again


def test_every_family_in_the_grid_can_be_built() -> None:
    for spec in sweep.FULL_GRID["families"]:
        fitness = sweep.build_fitness(spec, 6, seed=0)
        assert fitness.shape == (64,)
        assert np.all(np.isfinite(fitness))


def test_every_family_has_a_positive_threshold() -> None:
    for spec in sweep.FULL_GRID["families"]:
        fitness = sweep.build_fitness(spec, 8, seed=0)
        assert sweep.threshold_for(spec, fitness, 8) > 0.0, spec


def test_scoring_reports_both_metrics() -> None:
    reference = np.array([0.7, 0.2, 0.06, 0.04])
    scored = sweep.score(np.array([0.6, 0.3, 0.06, 0.04]), reference)
    assert set(scored) == {"cosine", "total_variation"}
    assert 0.0 <= scored["cosine"] <= 1.0
    assert scored["total_variation"] == pytest.approx(0.1)


def test_budget_table_covers_every_size() -> None:
    for n_sites in sweep.FULL_GRID["sizes"]:
        assert n_sites in sweep.BUDGET_SECONDS, n_sites


def test_baseline_b_is_inapplicable_outside_its_class() -> None:
    rugged = sweep.build_fitness({"family": "nk", "K": 2}, 6, seed=0)
    outcome = sweep.method_baseline_b(rugged, 0.2, budget=10.0)
    assert outcome["applicable"] is False
    assert "polynomial-time class" in outcome["reason"]

    easy = sweep.build_fitness({"family": "single_peak"}, 6, seed=0)
    assert sweep.method_baseline_b(easy, 0.2, budget=10.0)["applicable"] is True
