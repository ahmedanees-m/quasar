"""Result records and their provenance.

Each record carries the git commit, the image tag, the interpreter version and the seeds, and the
SHA-256 of `docs/protocol.md`. Modules in the package compute; this module writes.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"


def git_sha() -> str:
    """Current commit, or ``"unknown"`` outside a checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def git_dirty() -> bool:
    """True when the code that produced this result has uncommitted changes.

    ``results/`` is excluded, since a check writes its own record there and would otherwise always
    report a dirty tree.
    """
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain", "--", ".", ":(exclude)results"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(out.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return False


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, or ``"missing"`` if it is not there."""
    if not path.is_file():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment() -> dict[str, Any]:
    """The provenance block embedded in every result record."""
    return {
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "image": os.environ.get("QUASAR_IMAGE", "unknown"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "protocol_sha256": file_sha256(REPO_ROOT / "docs" / "protocol.md"),
    }


def in_pinned_image() -> bool:
    """Whether this process runs inside the execution image."""
    env = environment()
    return env["image"] != "unknown" and str(env["platform"]).startswith("Linux")


def output_directory(work_package: str, announce: bool = True) -> Path:
    """Where this run may write: ``results/<work_package>`` inside the image, ``results/_local`` outside.

    Every writer asks for its directory here, so records written outside the image never land in the
    committed tree.
    """
    inside = in_pinned_image()
    if inside:
        directory = RESULTS_ROOT / work_package if work_package else RESULTS_ROOT
    else:
        directory = RESULTS_ROOT / "_local"
    directory.mkdir(parents=True, exist_ok=True)
    if not inside and announce:
        print(f"NOTE: not running in the pinned image, so output goes to {directory}.")
    return directory


def write_gate_record(
    gate: str,
    work_package: str,
    threshold: dict[str, Any],
    measured: dict[str, Any],
    passed: bool,
    cases: list[dict[str, Any]],
    notes: str = "",
) -> Path:
    """Write a check's record to ``results/<work_package>/<gate>.json`` and return the path.

    Parameters
    ----------
    gate
        Check identifier, for example ``"G-R.1"``.
    work_package
        Directory under ``results/``, for example ``"wp_r"``.
    threshold
        The acceptance criterion, as listed in `docs/protocol.md`.
    measured
        What the run produced.
    passed
        Whether ``measured`` satisfies ``threshold``.
    cases
        One entry per configuration tested.
    notes
        Anything the other fields do not carry.
    """
    env = environment()

    # Outside the image, records go to results/_local.
    directory = output_directory(work_package, announce=False)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{gate.lower().replace('-', '_').replace('.', '_')}.json"
    if not in_pinned_image():
        try:
            shown = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:  # results root redirected elsewhere, as tests do
            shown = str(path)
        print(
            f"NOTE: not running in the pinned image, so this record is written to {shown}. "
            f"Run `make gates` to write to results/."
        )

    record = {
        "gate": gate,
        "work_package": work_package,
        "passed": passed,
        "threshold": threshold,
        "measured": measured,
        "n_cases": len(cases),
        "cases": cases,
        "notes": notes,
        "env": env,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    path.write_text(json.dumps(record, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path
