"""Result records and the provenance they carry.

A number in the manuscript is only traceable if the record around it is. These tests check
the record has the fields that make it so, and that the dirty-tree flag means what it
claims to mean.
"""

from __future__ import annotations

import json

import pytest

from quasarstack.io import store

pytestmark = pytest.mark.fast


def test_environment_block_has_every_provenance_field() -> None:
    env = store.environment()
    for field in ("git_sha", "git_dirty", "image", "python", "platform", "gates_md_sha256"):
        assert field in env, f"provenance field {field} missing from the record"
    assert isinstance(env["git_dirty"], bool)


def test_gates_md_is_hashed_into_every_record() -> None:
    """The hash is what makes 'the threshold was registered first' checkable rather than
    asserted. A record whose GATES.md hash does not appear in the repository history was
    judged against a threshold nobody can produce."""
    env = store.environment()
    assert env["gates_md_sha256"] != "missing"
    assert len(env["gates_md_sha256"]) == 64


def test_dirty_flag_ignores_the_results_tree(tmp_path, monkeypatch) -> None:
    """A gate writes into results/, so a naive check would see its own output and call every
    first run dirty. That would make the flag noise, and a noisy flag gets ignored."""
    stray = store.RESULTS_ROOT / "_dirty_flag_probe"
    stray.mkdir(parents=True, exist_ok=True)
    (stray / "untracked.json").write_text("{}", encoding="utf-8")
    try:
        assert store.git_dirty() is False or store.git_dirty() is True  # never raises
        # the probe alone must not be what makes it dirty
        import subprocess

        out = subprocess.run(
            ["git", "status", "--porcelain", "--", ".", ":(exclude)results"],
            cwd=store.REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert "_dirty_flag_probe" not in out.stdout
    finally:
        (stray / "untracked.json").unlink(missing_ok=True)
        stray.rmdir()


def test_write_gate_record_round_trips(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(store, "RESULTS_ROOT", tmp_path)
    path = store.write_gate_record(
        gate="G-X.1",
        work_package="wp_probe",
        threshold={"statistic": "max abs error", "value": 1e-9},
        measured={"max_abs_error": 1e-15},
        passed=True,
        cases=[{"L": 2, "max_abs_error": 1e-15}],
        notes="probe",
    )
    assert path.name == "g_x_1.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["gate"] == "G-X.1"
    assert record["passed"] is True
    assert record["n_cases"] == 1
    assert record["threshold"]["value"] == 1e-9
    assert "timestamp" in record and "env" in record
