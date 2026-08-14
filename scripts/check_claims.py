"""Verify that every entry in CLAIMS.md resolves.

A claim without a resolvable artefact does not go in the paper. This script parses the
ledger and checks, for each row, that the named script exists and that the named artefact
exists when the row is marked `pass`.

    python scripts/check_claims.py                 strict: `pass` rows must have artefacts
    python scripts/check_claims.py --allow-planned CI mode: `planned` rows need only a path

Exit code 0 means every checked entry resolves.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "CLAIMS.md"

# Identifiers may carry a letter suffix, as C4b and C5b do, for a claim that belongs to the
# same gate as its parent. An earlier form of this pattern required digits only, so those
# rows matched nothing and were skipped in silence: they sat in the ledger looking checked
# and were not. A checker that quietly ignores what it cannot parse is worse than no checker,
# so unparsed rows are now also counted and reported.
ROW = re.compile(r"^\|\s*(C\d+[a-z]*)\s*\|(.+)\|\s*$")
CLAIM_LIKE = re.compile(r"^\|\s*(C\S*)\s*\|")
CODE = re.compile(r"`([^`]+)`")
VALID_STATUS = {"planned", "pass", "fail", "dropped"}


@dataclass
class Claim:
    ident: str
    statement: str
    gate: str
    artefacts: list[str]
    scripts: list[str]
    status: str


def parse_ledger(text: str) -> tuple[list[Claim], list[str]]:
    """Return the parsed claims, and any claim-looking rows that could not be parsed.

    The second half matters as much as the first. A row that looks like a claim but does not
    parse must be surfaced, not skipped, or the ledger can grow entries nothing is checking.
    """
    claims: list[Claim] = []
    unparsed: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        match = ROW.match(stripped)
        if not match:
            if CLAIM_LIKE.match(stripped) and "---" not in stripped:
                unparsed.append(stripped[:80])
            continue
        ident = match.group(1)
        cells = [c.strip() for c in match.group(2).split("|")]
        if len(cells) < 5:
            # Identifier fine, columns short. Another way to be skipped in silence.
            unparsed.append(stripped[:80])
            continue
        statement, gate, artefact_cell, script_cell, status = cells[:5]
        claims.append(
            Claim(
                ident=ident,
                statement=statement,
                gate=gate,
                artefacts=CODE.findall(artefact_cell),
                scripts=CODE.findall(script_cell),
                status=status.lower(),
            )
        )
    return claims, unparsed


def _is_tracked(path: Path) -> bool:
    """Is this file committed, rather than merely present on disk?"""
    return (
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", str(path)],
            cwd=ROOT,
            capture_output=True,
        ).returncode
        == 0
    )


def check(claims: list[Claim], allow_planned: bool) -> list[str]:
    problems: list[str] = []
    for claim in claims:
        if claim.status not in VALID_STATUS:
            problems.append(f"{claim.ident}: unknown status {claim.status!r}")
            continue
        if claim.status == "dropped":
            continue
        if not claim.artefacts and not claim.scripts:
            problems.append(f"{claim.ident}: no artefact and no script named")
            continue

        # A `pass` row must resolve fully. A `planned` row is allowed to name paths that do
        # not exist yet, but the paths must still be well formed and inside the repository.
        must_exist = claim.status in {"pass", "fail"} or not allow_planned
        for path_str in claim.artefacts + claim.scripts:
            if path_str.endswith("/"):
                continue
            target = ROOT / path_str
            try:
                target.resolve().relative_to(ROOT.resolve())
            except ValueError:
                problems.append(f"{claim.ident}: path escapes the repository: {path_str}")
                continue
            if must_exist and target.exists() and not _is_tracked(target):
                # Existing on the author's disk is not the same as existing in the repository.
                # Two claims named artefacts under the gitignored `results/_local/` tree, which
                # resolved locally and failed in CI on a fresh checkout: the ledger appeared to
                # hold while pointing at files no reader could obtain.
                problems.append(
                    f"{claim.ident}: {claim.status} but {path_str} is not committed, so it "
                    f"resolves only on the author's machine"
                )
            elif must_exist and not target.exists():
                problems.append(f"{claim.ident}: {claim.status} but missing {path_str}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-planned",
        action="store_true",
        help="do not require artefacts for rows still marked planned",
    )
    args = parser.parse_args()

    if not LEDGER.is_file():
        print(f"CLAIMS.md not found at {LEDGER}")
        return 1

    claims, unparsed = parse_ledger(LEDGER.read_text(encoding="utf-8"))
    if not claims:
        print("CLAIMS.md parsed but contained no claim rows")
        return 1

    problems = check(claims, allow_planned=args.allow_planned)
    problems.extend(f"row looks like a claim but did not parse: {row}" for row in unparsed)

    by_status: dict[str, int] = {}
    for claim in claims:
        by_status[claim.status] = by_status.get(claim.status, 0) + 1
    summary = ", ".join(f"{n} {s}" for s, n in sorted(by_status.items()))
    print(f"{len(claims)} claims: {summary}")

    if problems:
        print()
        for p in problems:
            print(f"UNRESOLVED: {p}")
        return 1

    print("every checked claim resolves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
