"""G-6's per-cell deadline, and the distinction it exists to protect.

revision 23 gives each cell at `L >= 14` a wall-clock allotment, because the chi ladder is
climbed from 1 on every cell and the rugged families at the largest size cannot be afforded
otherwise. The risk in any such limit is that it manufactures a scientific result: a cell the
clock stopped looks identical, in a record, to a cell where the tensor network genuinely could
not represent the state. Those are opposite findings and only one is about physics.

So the tests here are less about the timer than about the bookkeeping. A stopped cell must say
it was stopped, must carry the largest chi it did try and the best cosine it saw, and must not
be counted against criterion 1.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quasarstack.analytic.exact_diag import perron_vector
from quasarstack.classical.landscapes import nk_fitness

pytestmark = pytest.mark.fast

ROOT = Path(__file__).resolve().parents[2]


def load_gate() -> Any:
    path = ROOT / "experiments" / "wp6_mps" / "g_6_tensor_network.py"
    spec = importlib.util.spec_from_file_location("g_6_budget_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = load_gate()


@pytest.fixture
def rugged_cell() -> tuple[np.ndarray, float, np.ndarray]:
    """An NK instance rugged enough that chi = 1 will not do."""
    fitness = nk_fitness(8, 2, seed=0)
    mu = 0.2
    reference = np.abs(perron_vector(fitness, mu)[0])
    return fitness, mu, reference / reference.sum()


class TestWhereTheLimitApplies:
    def test_no_limit_below_fourteen(self) -> None:
        assert gate.budget_for(8) is None
        assert gate.budget_for(12) is None

    def test_the_limit_is_section_11_3s_number(self) -> None:
        assert gate.budget_for(14) == gate.CELL_BUDGET_SECONDS == 900.0


class TestAStoppedCellSaysSo:
    def test_without_a_budget_the_search_is_unchanged(
        self, rugged_cell: tuple[np.ndarray, float, np.ndarray]
    ) -> None:
        fitness, mu, reference = rugged_cell
        found = gate.smallest_sufficient_chi(fitness, mu, reference, ceiling=16)
        assert found["chi_needed"] is not None, "this cell should be solvable within chi = 16"
        assert found["budget_limited"] is False

    def test_an_exhausted_budget_stops_the_climb_and_reports_what_it_reached(
        self, rugged_cell: tuple[np.ndarray, float, np.ndarray]
    ) -> None:
        fitness, mu, reference = rugged_cell
        found = gate.smallest_sufficient_chi(fitness, mu, reference, ceiling=16, budget=0.0)
        assert found["budget_limited"] is True
        assert found["chi_needed"] is None, "nothing was established, so no chi is claimed"
        assert found["largest_chi_attempted"] == gate.CHI_SWEEP[0]
        assert found["budget_seconds"] == 0.0
        assert 0.0 <= found["cosine"] <= 1.0, "the best cosine seen is still reported"

    def test_the_first_rung_always_runs(
        self, rugged_cell: tuple[np.ndarray, float, np.ndarray]
    ) -> None:
        """A zero budget must still produce a measurement, not an empty cell."""
        fitness, mu, reference = rugged_cell
        found = gate.smallest_sufficient_chi(fitness, mu, reference, ceiling=16, budget=0.0)
        assert found["largest_chi_attempted"] >= 1
        assert found["cosine"] > 0.0

    def test_a_generous_budget_behaves_exactly_as_no_budget(
        self, rugged_cell: tuple[np.ndarray, float, np.ndarray]
    ) -> None:
        fitness, mu, reference = rugged_cell
        unlimited = gate.smallest_sufficient_chi(fitness, mu, reference, ceiling=16)
        generous = gate.smallest_sufficient_chi(fitness, mu, reference, ceiling=16, budget=1e6)
        assert unlimited["chi_needed"] == generous["chi_needed"]
        assert unlimited["cosine"] == pytest.approx(generous["cosine"])


class TestTheBookkeeping:
    def test_a_stopped_cell_is_not_scored_as_a_convergence_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The whole point: the clock must not be able to fail criterion 1."""
        monkeypatch.setattr(gate, "CHECKPOINT", tmp_path / "scratch" / "cells.jsonl")
        monkeypatch.setattr(gate, "SIZES", [6])
        monkeypatch.setattr(gate, "MU_RATIOS", [1.0])
        monkeypatch.setattr(gate, "DTAU_SUBSET_SIZES", [6])
        monkeypatch.setattr(gate, "DTAU_SWEEP", [0.1])
        monkeypatch.setattr(gate, "CHI_SWEEP", [1, 2])
        monkeypatch.setattr(gate, "MAX_STEPS", 40)
        # Force every cell to be stopped by the clock.
        monkeypatch.setattr(gate, "budget_for", lambda n_sites: 0.0)

        _, measured, cases = gate.run()
        first = measured["criterion_1_converges_to_exact"]
        stopped = first["configurations_stopped_by_the_budget"]
        assert stopped, "the setup forces every cell to be stopped, so some must be recorded"
        assert (
            first["configurations_never_reaching_threshold"] == []
        ), "cells the clock stopped must not appear as convergence failures"
        assert first["passed"] is True, "a budget must not be able to fail criterion 1"
        assert all("largest_chi_attempted" in row for row in stopped)
        assert len(cases) > 0
