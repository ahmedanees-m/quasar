"""G-6's checkpoint, and the guard that stops it blending two different measurements.

The gate used to write nothing until it finished. That is what turned a crash fourteen hours
in into fourteen hours lost, and it is what makes any decision to change the grid mid-run cost
the entire run rather than the remainder of it. Cells are now appended as they land.

The interesting part is not the resume, it is the refusal. A checkpoint written under one
`CHI_SWEEP` or `MAX_STEPS` describes a different measurement, and silently reusing those cells
would produce a single artefact whose rows came from two methods. That is worse than losing
the run, because losing a run is visible and a blended artefact is not.
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

    def test_changing_a_registered_constant_changes_the_fingerprint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every constant in the digest must actually move it, or the guard is decorative."""
        before = gate.grid_fingerprint()
        for name, value in (
            ("CHI_SWEEP", [1, 2, 4]),
            ("MAX_STEPS", 10),
            ("COSINE_THRESHOLD", 0.5),
            ("SIZES", [8]),
            ("MU_RATIOS", [1.0]),
            ("SEEDS", [0]),
            ("DTAU", 0.01),
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

    def test_a_trotter_row_cannot_collide_with_a_grid_cell(self) -> None:
        assert gate.cell_key("trotter", 12, "nk_k2", 0.05) != gate.cell_key(
            "cell", 12, "nk_k2", 0.05
        )


class TestResumeSkipsTheWork:
    """The point of a checkpoint is not that it loads, it is that it stops recomputing."""

    @pytest.fixture
    def tiny_grid(self, monkeypatch: pytest.MonkeyPatch, checkpoint: Path) -> None:
        # L = 6 rather than 4: families() yields NK at K = 4, which needs L >= 5. The
        # registered grid starts at 8 so this never bites in practice.
        monkeypatch.setattr(gate, "SIZES", [6])
        monkeypatch.setattr(gate, "MU_RATIOS", [1.0])
        monkeypatch.setattr(gate, "DTAU_SUBSET_SIZES", [6])
        monkeypatch.setattr(gate, "DTAU_SWEEP", [0.1])
        monkeypatch.setattr(gate, "CHI_SWEEP", [1, 2, 4])
        monkeypatch.setattr(gate, "MAX_STEPS", 40)

    def count_evolutions(self, monkeypatch: pytest.MonkeyPatch) -> list[int]:
        calls = [0]
        real = gate.evolve

        def counting(*args: Any, **kwargs: Any) -> Any:
            calls[0] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(gate, "evolve", counting)
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
            f"the resumed run called evolve {second_calls[0]} times; every cell was already "
            f"in the checkpoint, so it should have called it none"
        )
        assert first_cases == second_cases, "a resumed run must reproduce the first exactly"

    def test_a_partial_checkpoint_only_recomputes_what_is_missing(
        self, tiny_grid: None, monkeypatch: pytest.MonkeyPatch, checkpoint: Path
    ) -> None:
        self.count_evolutions(monkeypatch)
        gate.run()
        lines = checkpoint.read_text(encoding="utf-8").strip().splitlines()
        # Drop the last two cells and confirm the rerun pays for those and nothing else.
        checkpoint.write_text("\n".join(lines[:-2]) + "\n", encoding="utf-8")

        calls = self.count_evolutions(monkeypatch)
        gate.run()
        assert (
            0 < calls[0] <= 2 * len(gate.CHI_SWEEP)
        ), f"expected work for two missing cells at most, got {calls[0]} evolutions"
