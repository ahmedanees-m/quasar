"""Every committed record says where it came from, and this reads it.

A record produced outside the pinned image once reached a commit because nothing was checking.
The record itself said so (`image: unknown`, `platform: Windows-10`, `git_dirty: true`) and
the fields went unread. Running it here means it runs on every push with the rest of the suite.
"""

from __future__ import annotations

import pytest

from scripts.check_results_provenance import RESULTS, SKIP_DIRS, SKIP_NAMES, problems_with

pytestmark = pytest.mark.fast


def committed_records() -> list:
    return [
        path
        for path in sorted(RESULTS.rglob("*.json"))
        if path.name not in SKIP_NAMES
        and not any(part in SKIP_DIRS for part in path.relative_to(RESULTS).parts)
    ]


def test_there_are_records_to_check() -> None:
    """A checker that finds nothing passes trivially, which is the failure mode to avoid."""
    assert len(committed_records()) >= 20


@pytest.mark.parametrize("path", committed_records(), ids=lambda p: p.stem)
def test_record_was_produced_in_the_pinned_image(path) -> None:
    problems = problems_with(path)
    assert not problems, f"{path.name}: " + "; ".join(problems)
