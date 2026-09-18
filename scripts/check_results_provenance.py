"""Check the provenance block of every committed result record.

A record passes when it was produced in the `quasar:v1` image, on Linux, from a clean working
tree, at a commit present in this repository.

    python scripts/check_results_provenance.py

Exit code 0 means every committed record passes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

REQUIRED_PLATFORM_PREFIX = "Linux"
FORBIDDEN_IMAGE_VALUES = {"unknown", "", None}

# Run summaries, not gate records.
SKIP_NAMES = {"gate_run_manifest.json"}

# Output of runs outside the image. Gitignored.
SKIP_DIRS = {"_local"}


def problems_with(path: Path) -> list[str]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"unreadable: {exc}"]

    env = record.get("env")
    if not isinstance(env, dict):
        return ["no env block"]

    found = []
    if env.get("image") in FORBIDDEN_IMAGE_VALUES:
        found.append(
            f"image is {env.get('image')!r}: produced outside the pinned image, or the run "
            f"did not set QUASAR_IMAGE"
        )
    platform = str(env.get("platform", ""))
    if not platform.startswith(REQUIRED_PLATFORM_PREFIX):
        found.append(f"platform is {platform!r}, not {REQUIRED_PLATFORM_PREFIX}")
    if env.get("git_dirty"):
        found.append("produced from a dirty working tree")
    sha = env.get("git_sha")
    if sha in {"unknown", "", None}:
        found.append("no git commit recorded")
    elif not _commit_exists(str(sha)):
        found.append(f"records commit {str(sha)[:12]}, which is not in this repository")
    return found


def _commits_resolvable() -> bool:
    """Whether this checkout can tell if a commit exists.

    A shallow clone or an export without `.git` cannot, so the commit check is skipped there and
    the other fields are still checked.
    """
    inside = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if inside != "true":
        return False
    shallow = subprocess.run(
        ["git", "rev-parse", "--is-shallow-repository"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return shallow != "true"


COMMITS_RESOLVABLE = _commits_resolvable()


def _commit_exists(sha: str) -> bool:
    if not COMMITS_RESOLVABLE:
        return True
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
            cwd=ROOT,
            capture_output=True,
        ).returncode
        == 0
    )


def main() -> int:
    if not RESULTS.is_dir():
        print("no results/ directory")
        return 0

    records = sorted(
        p
        for p in RESULTS.rglob("*.json")
        if p.name not in SKIP_NAMES and not SKIP_DIRS.intersection(p.parts)
    )
    if not records:
        print("no result records")
        return 0

    failures = 0
    for path in records:
        rel = path.relative_to(ROOT).as_posix()
        found = problems_with(path)
        if found:
            failures += 1
            print(f"REJECTED {rel}")
            for problem in found:
                print(f"    {problem}")
        else:
            print(f"ok       {rel}")

    print(f"\n{len(records) - failures}/{len(records)} records have valid provenance")
    if failures:
        print("\nRerun the affected gates with `make gates` in the pinned image.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
