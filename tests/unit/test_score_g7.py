"""The G-7 scorer, which turns the sweep into the project's headline decision.

It had no test. That is the wrong thing for this function to be missing, because it is the
one piece of code whose output is a claim rather than a number: it decides whether the
boundary map is a positive result or a registered null, and a scorer that is too lenient
manufactures an advantage while one that is too strict manufactures a null. Both failures
look like a plausible answer.

The cases below fix the four conditions of section 11.5 in place, one test per way of
failing them, plus the budget rule from ADR-0019 and the recomputation of `over_budget` that
ADR-0013's amendment introduced after a method reported its own budget state incorrectly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.fast

ROOT = Path(__file__).resolve().parents[2]


def load_scorer():
    path = ROOT / "scripts" / "score_g7.py"
    spec = importlib.util.spec_from_file_location("score_g7_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scorer = load_scorer()


def cell(
    seed: int,
    *,
    quantum: float | None = 0.95,
    tensor: float | None = 0.70,
    exact_class: bool = False,
    tensor_seconds: float = 100.0,
    allotted: float = 300.0,
    error: bool = False,
    n_sites: int = 12,
) -> dict[str, Any]:
    """One sweep cell. Defaults are a cell that satisfies every condition."""
    methods: dict[str, Any] = {
        scorer.TENSOR_NETWORK: {
            "applicable": True,
            "seconds_used": tensor_seconds,
            "seconds_allotted": allotted,
        },
        scorer.EXACT_CLASS: {"applicable": exact_class},
        "route_b_qsvt_filter": {
            "applicable": True,
            "seconds_used": 1.0,
            "seconds_allotted": allotted,
        },
    }
    if tensor is not None:
        methods[scorer.TENSOR_NETWORK]["cosine"] = tensor
    if quantum is not None:
        methods["route_b_qsvt_filter"]["cosine"] = quantum
    if error:
        methods[scorer.TENSOR_NETWORK]["error"] = "RuntimeError('boom')"
    return {
        "family": "nk",
        "K": 4,
        "roughness": None,
        "block_size": None,
        "L": n_sites,
        "seed": seed,
        "mu_over_mu_c": 1.0,
        "methods": methods,
    }


class TestTheFourConditions:
    def test_a_clean_advantage_is_reported_as_positive(self) -> None:
        verdict = scorer.score([cell(s) for s in range(5)])
        assert verdict["verdict"] == "positive"
        assert len(verdict["positive_region"]) == 1
        assert verdict["positive_region"][0]["route"] == "route_b_qsvt_filter"

    def test_a_tensor_network_that_does_not_fail_blocks_the_positive(self) -> None:
        """The situation the real sweep is in: the reference is essentially exact.

        Asserting only that the verdict is null would not test this. A mutation run showed
        the same cells return null through the interval condition as well, so a scorer that
        had dropped condition 2 entirely would still pass a bare null assertion. The reason
        is checked, not just the outcome.
        """
        verdict = scorer.score([cell(s, tensor=0.9995) for s in range(5)])
        assert verdict["verdict"] == "null"
        assert verdict["null_bound"]["largest_L_with_a_valid_reference"] == 12
        assert (
            "the tensor network stays at or above 0.8"
            in verdict["conditions_failed_by_group_count"]
        )

    def test_a_quantum_route_below_threshold_blocks_the_positive(self) -> None:
        verdict = scorer.score([cell(s, quantum=0.85) for s in range(5)])
        assert verdict["verdict"] == "null"
        assert "no quantum route reaches 0.9" in verdict["conditions_failed_by_group_count"]

    def test_baseline_b_applying_blocks_the_positive(self) -> None:
        """A cell a polynomial-time method solves exactly cannot demonstrate advantage."""
        verdict = scorer.score([cell(s, exact_class=True) for s in range(5)])
        assert verdict["verdict"] == "null"
        assert verdict["positive_region"] == []
        assert verdict["conditions_failed_by_group_count"] == {"baseline B applies": 1}

    def test_too_few_seeds_blocks_the_positive(self) -> None:
        verdict = scorer.score([cell(s) for s in range(4)])
        assert verdict["verdict"] == "null"
        assert "fewer than 5 seeds" in verdict["conditions_failed_by_group_count"]

    def test_overlapping_intervals_block_the_positive(self) -> None:
        """Condition 4: the bootstrap intervals must separate, not merely the means."""
        cells = [cell(s, quantum=0.95, tensor=0.79) for s in range(5)]
        cells[0]["methods"][scorer.TENSOR_NETWORK]["cosine"] = 0.99
        cells[1]["methods"]["route_b_qsvt_filter"]["cosine"] = 0.60
        verdict = scorer.score(cells)
        assert verdict["verdict"] == "null"


class TestTheBudgetRule:
    def test_an_over_budget_cell_is_excluded_and_counted(self) -> None:
        cells = [cell(s) for s in range(5)] + [cell(9, tensor_seconds=985.0)]
        verdict = scorer.score(cells)
        by_size = {row["L"]: row for row in verdict["excluded_cells_by_size"]}
        assert by_size[12]["cells_excluded"] == 1
        assert by_size[12]["cells"] == 6
        assert by_size[12]["by_reason"] == {"over budget": 1}

    def test_over_budget_is_recomputed_rather_than_trusted(self) -> None:
        """A method that overran and says otherwise must still be excluded. ADR-0013."""
        over = cell(9, tensor_seconds=985.0)
        over["methods"][scorer.TENSOR_NETWORK]["over_budget"] = False
        over["methods"][scorer.TENSOR_NETWORK]["budget_exhausted"] = False
        assert scorer.over_budget(over["methods"][scorer.TENSOR_NETWORK]) is True

    def test_a_cell_with_no_timing_recorded_is_still_judged(self) -> None:
        """Cells written before the field existed are scored, not silently dropped."""
        assert scorer.over_budget({"applicable": True, "cosine": 0.9}) is False

    def test_an_errored_cell_is_excluded_with_its_own_reason(self) -> None:
        verdict = scorer.score([cell(s) for s in range(5)] + [cell(9, error=True)])
        by_size = {row["L"]: row for row in verdict["excluded_cells_by_size"]}
        assert by_size[12]["by_reason"] == {"a method errored": 1}

    def test_every_size_appears_in_the_summary_even_with_nothing_excluded(self) -> None:
        """A size missing from the table reads as a size nobody looked at."""
        cells = [cell(s, n_sites=8) for s in range(5)] + [cell(s, n_sites=12) for s in range(5)]
        sizes = {row["L"] for row in scorer.score(cells)["excluded_cells_by_size"]}
        assert sizes == {8, 12}


class TestTheNullCarriesItsBound:
    def test_a_null_states_the_largest_size_it_reached(self) -> None:
        """Section 11.5 requires a delimitation, not a shrug."""
        cells = [cell(s, tensor=0.9995, n_sites=8) for s in range(5)]
        cells += [cell(s, tensor=0.9995, n_sites=12) for s in range(5)]
        bound = scorer.score(cells)["null_bound"]
        assert bound["largest_L_with_a_valid_reference"] == 12
        assert "beyond the largest L" in bound["statement"]

    def test_a_positive_carries_no_null_bound(self) -> None:
        assert scorer.score([cell(s) for s in range(5)])["null_bound"] is None
