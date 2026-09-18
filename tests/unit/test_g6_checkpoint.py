"""G-6's checkpoint and its grid fingerprint.

Cells are appended to a checkpoint as they complete, and a rerun resumes from it. A checkpoint
written under different constants, such as another `CHI_SWEEP` or cross-check setting, is
ignored, so a record never mixes cells from two configurations.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.fast

ROOT = Path(__file__).resolve().parents[2]


def load_gate() -> Any:
    path = ROOT / "experiments" / "wp6_mps" / "g_6_tensor_network.py"
    spec = importlib.util.spec_from_file_location("g_6_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = load_gate()


@pytest.fixture
def checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    target = tmp_path / "scratch" / "g_6_cells.jsonl"
    monkeypatch.setattr(gate, "CHECKPOINT", target)
    return target


class TestRoundTrip:
    def test_nothing_written_means_nothing_to_resume(self, checkpoint: Path) -> None:
        assert gate.load_checkpoint() == {}

    def test_a_cell_survives_the_round_trip(self, checkpoint: Path) -> None:
        gate.append_checkpoint({"_key": "cell|12|nk|4|None|None|0|1.0", "chi_needed": 16})
        loaded = gate.load_checkpoint()
        assert list(loaded) == ["cell|12|nk|4|None|None|0|1.0"]
        assert loaded["cell|12|nk|4|None|None|0|1.0"]["chi_needed"] == 16

    def test_the_header_is_written_once_and_is_not_a_cell(self, checkpoint: Path) -> None:
        gate.append_checkpoint({"_key": "a", "chi_needed": 1})
        gate.append_checkpoint({"_key": "b", "chi_needed": 2})
        lines = checkpoint.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3, "header plus two cells"
        assert json.loads(lines[0]) == {"fingerprint": gate.grid_fingerprint()}
        assert set(gate.load_checkpoint()) == {"a", "b"}


class TestTheGuard:
    def test_a_checkpoint_from_another_grid_is_refused(
        self, checkpoint: Path, capsys: pytest.CaptureFixture
    ) -> None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(
            json.dumps({"fingerprint": "0000deadbeef0000"})
            + "\n"
            + json.dumps({"_key": "cell|12|nk|4|None|None|0|1.0", "chi_needed": 2})
            + "\n",
            encoding="utf-8",
        )
        assert gate.load_checkpoint() == {}, "cells from another grid must not be reused"
        assert "ignoring it" in capsys.readouterr().err

    def test_changing_a_constant_changes_the_fingerprint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every constant in the digest changes it."""
        before = gate.grid_fingerprint()
        for name, value in (
            ("CHI_SWEEP", [1, 2, 4]),
            ("COSINE_THRESHOLD", 0.5),
            ("SIZES", [8]),
            ("MU_RATIOS", [1.0]),
            ("SEEDS", [0]),
            ("CROSS_CHECK_SIZES", [8]),
            ("CROSS_CHECK_RATIOS", [1.0]),
            ("CROSS_CHECK_DTAU", 0.01),
            ("CROSS_CHECK_MAX_STEPS", 10),
            ("RUNG_TOL", 1e-6),
            ("RUNG_MAX_SWEEPS", 5),
        ):
            with monkeypatch.context() as patch:
                patch.setattr(gate, name, value)
                assert gate.grid_fingerprint() != before, f"{name} does not move the fingerprint"

    def test_an_empty_file_is_treated_as_no_checkpoint(self, checkpoint: Path) -> None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text("", encoding="utf-8")
        assert gate.load_checkpoint() == {}


class TestTheKey:
    def test_cells_differing_only_in_seed_do_not_collide(self) -> None:
        first = gate.cell_key("cell", 12, "nk", 4, None, None, 0, 1.0)
        second = gate.cell_key("cell", 12, "nk", 4, None, None, 1, 1.0)
        assert first != second

    def test_a_cross_check_row_cannot_collide_with_a_grid_cell(self) -> None:
        assert gate.cell_key("cross_check", 8, "nk", 2, None, None, 1.0) != gate.cell_key(
            "cell", 8, "nk", 2, None, None, 1.0
        )


class TestResumeSkipsTheWork:
    """A resumed run skips the cells already in the checkpoint."""

    @pytest.fixture
    def tiny_grid(self, monkeypatch: pytest.MonkeyPatch, checkpoint: Path) -> None:
        # L = 6 rather than 4: families() yields NK at K = 4, which needs L >= 5.
        monkeypatch.setattr(gate, "SIZES", [6])
        monkeypatch.setattr(gate, "MU_RATIOS", [1.0])
        monkeypatch.setattr(gate, "CROSS_CHECK_SIZES", [6])
        monkeypatch.setattr(gate, "CROSS_CHECK_RATIOS", [1.0])
        monkeypatch.setattr(gate, "CROSS_CHECK_MAX_STEPS", 40)
        monkeypatch.setattr(gate, "CHI_SWEEP", [1, 2, 4])

    def count_evolutions(self, monkeypatch: pytest.MonkeyPatch) -> list[int]:
        calls = [0]
        real = gate.dominant_state

        def counting(*args: Any, **kwargs: Any) -> Any:
            calls[0] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(gate, "dominant_state", counting)
        return calls

    def test_the_second_run_computes_nothing_and_agrees_with_the_first(
        self, tiny_grid: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first_calls = self.count_evolutions(monkeypatch)
        _, _, first_cases = gate.run()
        assert first_calls[0] > 0, "the first run should have done real work"

        second_calls = self.count_evolutions(monkeypatch)
        _, _, second_cases = gate.run()
        assert second_calls[0] == 0, (
            f"the resumed run called DMRG {second_calls[0]} times; every cell was already "
            f"in the checkpoint, so it should have called it none"
        )
        assert first_cases == second_cases, "a resumed run must reproduce the first exactly"

    def test_a_partial_checkpoint_only_recomputes_what_is_missing(
        self, tiny_grid: None, monkeypatch: pytest.MonkeyPatch, checkpoint: Path
    ) -> None:
        self.count_evolutions(monkeypatch)
        gate.run()
        lines = checkpoint.read_text(encoding="utf-8").strip().splitlines()
        # Drop the last two rows and confirm the rerun pays for those and nothing else. The
        # last rows are cross-checks, which run DMRG once each.
        checkpoint.write_text("\n".join(lines[:-2]) + "\n", encoding="utf-8")

        calls = self.count_evolutions(monkeypatch)
        gate.run()
        assert (
            0 < calls[0] <= 2 * len(gate.CHI_SWEEP)
        ), f"expected work for two missing rows at most, got {calls[0]} DMRG runs"
