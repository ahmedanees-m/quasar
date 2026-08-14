"""Compare every regenerated artefact against its committed version.

A reproduction is only meaningful if something reads the result. This is that something: for
each file `git` reports as modified under `results/`, it fetches the committed version with
`git show HEAD:<path>` and compares content, ignoring timing and provenance fields that a rerun
is expected to rewrite.

Two things this exists to get right, both learned by getting them wrong.

**`.jsonl` is not `.json`.** The previous version called `json.loads` on the whole file, which
works for a record and raises `Extra data: line 2` on a line-delimited sweep. The comparison
then aborted **partway through**, after twelve of twenty-four artefacts, having printed a
reassuring run of `IDENTICAL` lines. A crash that follows good news reads like an ending.

**A partial comparison must not look like a complete one.** Every artefact is now reported as
identical, changed, new, or unreadable, and the exit code is non-zero if any file could not be
compared at all. Silence is not agreement.

    python scripts/compare_reproduction.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Rewritten by every rerun by construction, so they are not evidence of a real change.
#
# Matched as substrings rather than exact names, and the first version of this file got that
# wrong. It stripped `seconds` and missed `seconds_per_1000_generations_by_population` and
# `seconds_to_reach_target`, so G-4 was reported CHANGED on six leaves that were all wall-clock
# measurements: 0.478 against 0.4655 seconds per thousand generations, and so on. A reproduction
# check that cries wolf over a machine being fractionally faster trains its reader to skim it,
# which is the same failure mode as one that stays silent.
#
# Timing is not thereby unguarded. Where a duration is itself a gate criterion, as G-4's
# throughput is, the gate decides pass or fail against its own threshold and `passed` is
# compared below. That is the right division: this script asks whether the science is the same,
# and the gate asks whether the timing is acceptable.
VOLATILE_SUBSTRINGS = ("second", "elapsed", "timestamp", "duration", "wall_clock", "runtime")


def is_volatile(key: str) -> bool:
    lowered = str(key).lower()
    return any(token in lowered for token in VOLATILE_SUBSTRINGS)


def strip_volatile(value):
    """Drop timing keys at any depth, so a slower machine does not read as a different result."""
    if isinstance(value, dict):
        return {k: strip_volatile(v) for k, v in value.items() if not is_volatile(k)}
    if isinstance(value, list):
        return [strip_volatile(v) for v in value]
    return value


def digest(payload: str, is_lines: bool) -> str:
    """A content fingerprint that ignores volatile fields and key order."""
    if is_lines:
        rows = [strip_volatile(json.loads(line)) for line in payload.splitlines() if line.strip()]
        canonical = json.dumps(rows, sort_keys=True, default=str)
    else:
        record = json.loads(payload)
        # `measured` is the scientific content. `env` is provenance and changes every run.
        body = record.get("measured", record) if isinstance(record, dict) else record
        canonical = json.dumps(strip_volatile(body), sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def passed_of(payload: str, is_lines: bool):
    if is_lines:
        return None
    try:
        record = json.loads(payload)
        return record.get("passed") if isinstance(record, dict) else None
    except Exception:  # noqa: BLE001
        return None


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout


def main() -> int:
    changed = [p for p in git("diff", "--name-only", "--", "results/").split() if p.strip()]
    print(f"{len(changed)} artefacts regenerated\n")

    identical = moved = new = unreadable = 0
    for path in sorted(changed):
        committed = git("show", f"HEAD:{path}")
        if not committed.strip():
            print(f"  NEW        {path}")
            new += 1
            continue
        live_path = ROOT / path
        if not live_path.is_file():
            print(f"  DELETED    {path}")
            moved += 1
            continue
        is_lines = path.endswith(".jsonl")
        try:
            live = live_path.read_text(encoding="utf-8")
            before, after = digest(committed, is_lines), digest(live, is_lines)
            was, now = passed_of(committed, is_lines), passed_of(live, is_lines)
        except Exception as error:  # noqa: BLE001
            # Counted and reported. An artefact that cannot be compared is not an artefact that
            # agrees, and the exit code below says so.
            print(f"  UNREADABLE {path}: {type(error).__name__}: {str(error)[:80]}")
            unreadable += 1
            continue
        if before == after and was == now:
            print(f"  IDENTICAL  {path}  passed={now}")
            identical += 1
        else:
            print(f"  CHANGED    {path}: passed {was} -> {now}, digest {before} -> {after}")
            moved += 1

    print(f"\n{identical} identical, {moved} changed, {new} new, {unreadable} unreadable")
    if moved:
        print("A CHANGED artefact is a finding to read, not something to rerun until it agrees.")
    if unreadable:
        print("An artefact that could not be compared is not an artefact that agreed.")
    return 1 if (moved or unreadable) else 0


if __name__ == "__main__":
    sys.exit(main())
