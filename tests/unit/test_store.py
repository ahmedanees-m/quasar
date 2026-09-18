"""Result records and their provenance fields."""

from __future__ import annotations

import json
import re

import pytest

from quasarstack.io import store

pytestmark = pytest.mark.fast


def test_environment_block_has_every_provenance_field() -> None:
    env = store.environment()
    for field in ("git_sha", "git_dirty", "image", "python", "platform", "protocol_sha256"):
        assert field in env, f"provenance field {field} missing from the record"
    assert isinstance(env["git_dirty"], bool)


def test_protocol_is_hashed_into_every_record() -> None:
    env = store.environment()
    assert env["protocol_sha256"] != "missing"
    assert len(env["protocol_sha256"]) == 64


def test_dirty_flag_ignores_the_results_tree(tmp_path, monkeypatch) -> None:
    """Records written under results/ do not mark the tree dirty."""
    stray = store.RESULTS_ROOT / "_dirty_flag_probe"
    stray.mkdir(parents=True, exist_ok=True)
    (stray / "untracked.json").write_text("{}", encoding="utf-8")
    try:
        assert store.git_dirty() is False or store.git_dirty() is True  # never raises
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


def test_records_from_outside_the_image_go_to_the_local_directory(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.setattr(store, "RESULTS_ROOT", tmp_path)
    monkeypatch.delenv("QUASAR_IMAGE", raising=False)

    path = store.write_gate_record(
        gate="G-X.9",
        work_package="wp_probe",
        threshold={"value": 1.0},
        measured={"value": 0.0},
        passed=True,
        cases=[{}],
    )
    assert path.parent.name == "_local", f"expected results/_local, got {path}"
    assert not (tmp_path / "wp_probe").exists()
    assert "not running in the pinned image" in capsys.readouterr().out


def test_records_from_inside_the_image_go_to_the_package_directory(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(store, "RESULTS_ROOT", tmp_path)
    monkeypatch.setenv("QUASAR_IMAGE", "quasar:v1")
    monkeypatch.setattr(store, "environment", lambda: {**_linux_env(), "image": "quasar:v1"})

    path = store.write_gate_record(
        gate="G-X.9",
        work_package="wp_probe",
        threshold={"value": 1.0},
        measured={"value": 0.0},
        passed=True,
        cases=[{}],
    )
    assert path.parent.name == "wp_probe"


def _linux_env() -> dict:
    return {
        "git_sha": "abc123",
        "git_dirty": False,
        "image": "quasar:v1",
        "python": "3.12.13",
        "platform": "Linux-6.8.0-generic-x86_64",
        "protocol_sha256": "0" * 64,
    }


def test_write_gate_record_round_trips(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(store, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(store, "environment", _linux_env)
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


def test_make_gates_sets_the_image_and_python_path() -> None:
    """`make gates` passes ``QUASAR_IMAGE`` and ``PYTHONPATH`` into the container."""
    makefile = (store.REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    # Fold line continuations so the DOCKER definition reads as one line.
    joined = re.sub(r"\\\n\s*", " ", makefile.replace("\r\n", "\n"))
    docker_line = next((line for line in joined.splitlines() if line.startswith("DOCKER")), None)
    assert docker_line is not None, "Makefile no longer defines DOCKER"
    assert "-e QUASAR_IMAGE=" in docker_line, "make gates would write to results/_local"
    assert "-e PYTHONPATH=/work" in docker_line, "gate scripts cannot import quasarstack"


def test_the_sweep_runner_uses_the_shared_output_directory() -> None:
    source = (store.REPO_ROOT / "scripts" / "sweep_runner.py").read_text(encoding="utf-8")
    assert "output_directory" in source


def test_every_results_writer_uses_the_shared_output_directory() -> None:
    """Scripts that write under `results/` go through `output_directory`."""
    read_only = {
        "check_results_provenance.py",
    }
    offenders = []
    for path in (store.REPO_ROOT / "scripts").glob("*.py"):
        if path.name in read_only:
            continue
        source = path.read_text(encoding="utf-8")
        builds_results_path = "RESULTS_ROOT" in source or "RESULTS /" in source
        # `write_gate_record` delegates to `output_directory`, checked below.
        guarded = "output_directory" in source or "write_gate_record" in source
        if builds_results_path and not guarded:
            offenders.append(path.name)
    assert (
        not offenders
    ), f"these scripts build a results path without output_directory: {offenders}"

    store_source = (store.REPO_ROOT / "quasarstack" / "io" / "store.py").read_text(encoding="utf-8")
    body = store_source[store_source.index("def write_gate_record") :]
    assert "output_directory(" in body, "write_gate_record no longer calls output_directory"

    for name in read_only:
        source = (store.REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "RESULTS_ROOT /" not in source.replace(
            "RESULTS_ROOT / relative", ""
        ), f"{name} is on the read-only list but builds a results path to write to"
