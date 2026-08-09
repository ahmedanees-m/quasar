"""Writing result records, with the provenance that makes a number traceable.

Every number that reaches the manuscript has to trace back to exact code in an exact
environment. So no result is written as a bare value: each record carries the git commit,
the image tag, the interpreter version, and the seeds, and `GATES.md` is hashed into the
record so it is provable that the threshold was registered before the run rather than after
it.

Side effects live here and nowhere else in the package. Modules compute, `io/store.py`
writes.
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

    Recorded rather than blocked. A gate run from a dirty tree is not reproducible, and the
    record should say so plainly instead of the fact being invisible later.

    ``results/`` is excluded, and the exclusion is load bearing rather than cosmetic. A gate
    writes its record into that directory, so the record would otherwise observe itself
    being untracked and report every first run as dirty, which would make the flag noise and
    train the reader to ignore it. Result artefacts are outputs; they say nothing about
    whether the inputs were committed.
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
        "gates_md_sha256": file_sha256(REPO_ROOT / "GATES.md"),
    }


def write_gate_record(
    gate: str,
    work_package: str,
    threshold: dict[str, Any],
    measured: dict[str, Any],
    passed: bool,
    cases: list[dict[str, Any]],
    notes: str = "",
) -> Path:
    """Write a gate result to ``results/<work_package>/<gate>.json`` and return the path.

    Parameters
    ----------
    gate
        Gate identifier as written in `GATES.md`, for example ``"G-R.1"``.
    work_package
        Directory under ``results/``, for example ``"wp_r"``.
    threshold
        The registered acceptance criterion, copied verbatim from `GATES.md`.
    measured
        What the run actually produced.
    passed
        Whether ``measured`` satisfies ``threshold``.
    cases
        One entry per configuration tested, so a failure can be localised without a rerun.
    notes
        Anything a reader of the record needs that the fields do not carry.
    """
    env = environment()

    # A run outside the pinned image writes somewhere gitignored, not into the tree where
    # committed evidence lives. Twice in one afternoon a laptop run left a record in
    # results/, and the second time it also blocked a pull by colliding with the real one.
    # Local runs are useful and stay allowed; what stops is their output sitting where a
    # `git add -A` can mistake it for evidence. See DECISIONS.md ADR-0012.
    in_pinned_image = env["image"] != "unknown" and str(env["platform"]).startswith("Linux")
    directory = RESULTS_ROOT / work_package if in_pinned_image else RESULTS_ROOT / "_local"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{gate.lower().replace('-', '_').replace('.', '_')}.json"
    if not in_pinned_image:
        try:
            shown = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:  # results root redirected elsewhere, as tests do
            shown = str(path)
        print(
            f"NOTE: not running in the pinned image, so this record is written to "
            f"{shown} and is not evidence. "
            f"Rerun with `make gates` on the compute VM to produce a committable record."
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
