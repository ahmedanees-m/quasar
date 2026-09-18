"""The provenance block of every committed record."""

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
    """The committed records are found."""
    assert len(committed_records()) >= 20


@pytest.mark.parametrize("path", committed_records(), ids=lambda p: p.stem)
def test_record_was_produced_in_the_pinned_image(path) -> None:
    problems = problems_with(path)
    assert not problems, f"{path.name}: " + "; ".join(problems)
