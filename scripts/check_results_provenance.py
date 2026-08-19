"""Refuse to accept a committed result record that was not produced in the pinned image.

docs/notes.md says every result record comes from `quasar:v1` on the declared hardware. That was
policy with nothing enforcing it, and it broke the first time it was tested: a gate run outside the
image wrote a record that a `git add -A` swept into a commit. The record's own provenance
block said so plainly, with `image: unknown` and `git_dirty: true`, and it was not being read.

So it is read here, and in CI. A result whose provenance does not check out is not evidence,
however good the number inside it looks.

    python scripts/check_results_provenance.py

Exit code 0 means every committed record was produced in the image, from a clean tree, on
the declared platform.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# The declared execution environment, from docs/notes.md and docs/protocol.md section 11.3.
REQUIRED_PLATFORM_PREFIX = "Linux"
FORBIDDEN_IMAGE_VALUES = {"unknown", "", None}

# Not every JSON under results/ is a gate record. These are driver output, not evidence.
SKIP_NAMES = {"gate_run_manifest.json"}

# Records from runs outside the pinned image land here, and the directory is gitignored.
# They are not evidence and are not pretending to be, so failing on them would make this
# check fire during ordinary development and train people to ignore it. What matters is what
# is committed.
SKIP_DIRS = {"_local"}


def problems_with(path: Path) -> list[str]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"unreadable: {exc}"]

    env = record.get("env")
    if not isinstance(env, dict):
        return ["no env block, so the record carries no provenance at all"]

    found = []
    if env.get("image") in FORBIDDEN_IMAGE_VALUES:
        found.append(
            f"image is {env.get('image')!r}: produced outside the pinned image, or the run "
            f"did not set QUASAR_IMAGE"
        )
    platform = str(env.get("platform", ""))
    if not platform.startswith(REQUIRED_PLATFORM_PREFIX):
        found.append(f"platform is {platform!r}, not the declared {REQUIRED_PLATFORM_PREFIX}")
    if env.get("git_dirty"):
        found.append("produced from a dirty working tree, so the code is not identified")
    sha = env.get("git_sha")
    if sha in {"unknown", "", None}:
        found.append("no git commit recorded")
    elif not _commit_exists(str(sha)):
        # A recorded commit that no longer exists identifies nothing. This is not hypothetical:
        # rewriting commit messages across the history changed every hash, and all twenty-one
        # records were left pointing at commits that had ceased to exist. Every other field
        # still looked correct, so the checker passed while the chain it exists to protect was
        # broken end to end.
        found.append(
            f"records commit {str(sha)[:12]}, which is not in this repository, so the record "
            f"identifies no code. If the history was rewritten, repoint the record at the "
            f"commit carrying the same tree rather than leaving it dangling."
        )
    return found


def _commits_resolvable() -> bool:
    """Can this tree answer whether a commit exists?

    A shallow clone holds one commit, and an export of the released tree holds no `.git` at
    all. In both cases every record would look like it names a missing commit, which says
    nothing about the record. The other fields are still checked.
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
        print("no results/ directory yet")
        return 0

    records = sorted(
        p
        for p in RESULTS.rglob("*.json")
        if p.name not in SKIP_NAMES and not SKIP_DIRS.intersection(p.parts)
    )
    if not records:
        print("no result records committed yet")
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
        print(
            "\nA record produced outside the pinned image is not evidence. Rerun the gate "
            "with `make gates` in the pinned image and commit that record instead."
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
