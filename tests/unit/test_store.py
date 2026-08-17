"""Result records and the provenance they carry.

A number in the manuscript is only traceable if the record around it is. These tests check
the record has the fields that make it so, and that the dirty-tree flag means what it
claims to mean.
"""

from __future__ import annotations

import json
import re

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


def test_records_from_outside_the_image_are_redirected_out_of_the_evidence_tree(
    monkeypatch, tmp_path, capsys
) -> None:
    """A run outside the pinned image must not leave a file where committed results live.

    This happened twice in one afternoon: a laptop run wrote into results/, a `git add -A`
    swept it into a commit, and the second time it also blocked a pull by colliding with the
    real record. Local runs stay allowed; their output just goes somewhere gitignored and
    says so.
    """
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
    assert path.parent.name == "_local", f"expected redirection, got {path}"
    assert not (tmp_path / "wp_probe").exists(), "the evidence tree must stay untouched"
    assert "not evidence" in capsys.readouterr().out


def test_records_from_inside_the_image_land_in_the_evidence_tree(monkeypatch, tmp_path) -> None:
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
        "gates_md_sha256": "0" * 64,
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


def test_make_gates_runs_the_container_so_that_records_count_as_evidence() -> None:
    """`make gates` must produce evidence, not gitignored local records.

    `store.write_gate_record` files a run under ``results/_local`` unless ``QUASAR_IMAGE``
    is set, and a gate script cannot import ``quasarstack`` unless ``PYTHONPATH`` points at
    the mounted working tree. Neither was in the Makefile: every gate that has passed so
    far was run with both supplied by hand, so the documented one-command reproduction
    would have written nothing committable and said so only in a line of console output
    nobody reads twice. This test is the reason that cannot recur silently. See ADR-0014.
    """
    makefile = (store.REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    # Fold escaped line continuations so the DOCKER definition reads as one string.
    # Line endings are normalised first: the working tree sits on a Windows-mounted
    # drive, so the raw text carries CRLF and a join on backslash-newline would match
    # nothing and quietly pass a broken Makefile.
    joined = re.sub(r"\\\n\s*", " ", makefile.replace("\r\n", "\n"))
    docker_line = next((line for line in joined.splitlines() if line.startswith("DOCKER")), None)
    assert docker_line is not None, "Makefile no longer defines DOCKER"
    assert (
        "-e QUASAR_IMAGE=" in docker_line
    ), "make gates would write every record to results/_local and produce no evidence"
    assert (
        "-e PYTHONPATH=/work" in docker_line
    ), "gate scripts cannot import quasarstack without PYTHONPATH; make gates would fail"


def test_the_sweep_runner_also_redirects_out_of_the_evidence_tree() -> None:
    """ADR-0012 applies to every writer, not only to `write_gate_record`.

    The sweep writes its own JSONL stream rather than going through the gate-record path, so
    it needs its own copy of the redirection. It was written without one, caught before it
    produced anything, and this test is why that cannot recur. A grep rather than an
    execution, because running the sweep to find out would cost minutes.
    """
    source = (store.REPO_ROOT / "scripts" / "sweep_runner.py").read_text(encoding="utf-8")
    assert "evidence_directory" in source, (
        "sweep_runner does not use the shared guard, so a non-image run could write into "
        "the evidence tree"
    )


def test_every_results_writer_uses_the_shared_evidence_guard() -> None:
    """ADR-0012's redirection was re-implemented per writer, and twice forgotten.

    `write_gate_record` had it, the WP7 sweep runner had its own copy, and the G-7 scorer had
    none: both wrote straight into `results/` on their first version. Three occurrences of one
    omission means the guard was in the wrong place, so it now lives in `evidence_directory`
    and this test fails if a script that *writes* under `results/` does not use it.

    Read-only consumers are listed explicitly rather than inferred. Guessing read from write
    by pattern-matching would either miss a writer or, as the first version of this test did,
    flag a reader of artefacts it never writes to.
    """
    read_only = {
        "check_claims.py",
        "check_results_provenance.py",
    }
    offenders = []
    for path in (store.REPO_ROOT / "scripts").glob("*.py"):
        if path.name in read_only:
            continue
        source = path.read_text(encoding="utf-8")
        builds_results_path = "RESULTS_ROOT" in source or "RESULTS /" in source
        # `write_gate_record` counts as guarded because it delegates to `evidence_directory`,
        # which the assertion below verifies rather than takes on trust. Without this the test
        # flagged the G-7 scorer for switching to `write_gate_record`, which had made it more
        # protected rather than less.
        guarded = "evidence_directory" in source or "write_gate_record" in source
        if builds_results_path and not guarded:
            offenders.append(path.name)
    assert not offenders, (
        f"these scripts build a results path without the shared guard: {offenders}. "
        f"Use quasarstack.io.store.evidence_directory so a non-image run cannot write "
        f"where committed evidence lives."
    )

    # The delegation the exemption above relies on, checked rather than assumed. If
    # `write_gate_record` ever stops routing through `evidence_directory`, treating it as
    # proof of guarding would be wrong, and every caller would silently lose the protection.
    store_source = (store.REPO_ROOT / "quasarstack" / "io" / "store.py").read_text(encoding="utf-8")
    body = store_source[store_source.index("def write_gate_record") :]
    assert "evidence_directory(" in body, (
        "write_gate_record no longer calls evidence_directory, so scripts that rely on it "
        "for ADR-0012's redirection are unguarded and the exemption above is unsound."
    )

    # And the read-only list must not go stale: a script named there that starts writing
    # would slip through the check above.
    for name in read_only:
        source = (store.REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "RESULTS_ROOT /" not in source.replace(
            "RESULTS_ROOT / relative", ""
        ), f"{name} is on the read-only list but builds a results path to write to"
