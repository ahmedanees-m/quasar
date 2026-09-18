"""Compare regenerated result files against the committed versions.

For each file under `results/` that `git` reports as modified, the committed version is read with
`git show HEAD:<path>` and the two are compared with timing and provenance fields removed. Each
file is reported as identical, changed, new, deleted or unreadable. The exit code is non-zero
if any file changed or could not be read.

    python scripts/compare_reproduction.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Keys containing any of these are wall-clock measurements and are ignored. Gates whose criteria
# involve timing still report the outcome in `passed`, which is compared.
VOLATILE_SUBSTRINGS = ("second", "elapsed", "timestamp", "duration", "wall_clock", "runtime")


def is_volatile(key: str) -> bool:
    lowered = str(key).lower()
    return any(token in lowered for token in VOLATILE_SUBSTRINGS)


def strip_volatile(value):
    """Drop timing keys at any depth."""
    if isinstance(value, dict):
        return {k: strip_volatile(v) for k, v in value.items() if not is_volatile(k)}
    if isinstance(value, list):
        return [strip_volatile(v) for v in value]
    return value


def digest(payload: str, is_lines: bool) -> str:
    """Content fingerprint independent of timing fields and key order."""
    if is_lines:
        rows = [strip_volatile(json.loads(line)) for line in payload.splitlines() if line.strip()]
        canonical = json.dumps(rows, sort_keys=True, default=str)
    else:
        record = json.loads(payload)
        # `measured` holds the results; `env` is provenance.
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
    print(f"{len(changed)} files regenerated\n")

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
    return 1 if (moved or unreadable) else 0


if __name__ == "__main__":
    sys.exit(main())
