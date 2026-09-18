"""Every test file carries a declared marker.

``make test`` runs ``pytest -m fast`` and ``make test-all`` runs
``pytest -m "fast or slow or gate"``, so a file without a marker is collected by neither.
"""

from __future__ import annotations

import pathlib
import re

import pytest

pytestmark = pytest.mark.fast

TESTS = pathlib.Path(__file__).resolve().parents[1]
DECLARED = {"fast", "slow", "gate", "hardware"}
MARKER = re.compile(r"pytest\.mark\.(\w+)")


def suite_files() -> list[pathlib.Path]:
    return sorted(TESTS.rglob("test_*.py"))


def test_the_search_found_the_suite() -> None:
    """The glob finds the suite."""
    assert len(suite_files()) >= 20, f"only {len(suite_files())} test files found"


def test_every_test_file_carries_a_declared_marker() -> None:
    unmarked = []
    for path in suite_files():
        text = path.read_text(encoding="utf-8")
        if not re.search(r"^\s*def test_", text, re.M):
            continue
        if not (set(MARKER.findall(text)) & DECLARED):
            unmarked.append(path.relative_to(TESTS).as_posix())
    assert not unmarked, (
        f"{unmarked} declare no marker from {sorted(DECLARED)}. Add "
        f"`pytestmark = pytest.mark.fast` at module level, or the marker that fits."
    )


def test_no_file_invents_a_marker_that_pyproject_does_not_declare() -> None:
    """Every marker used is declared in pyproject.toml."""
    root = TESTS.parent / "pyproject.toml"
    declared = set(re.findall(r'"(\w+):', root.read_text(encoding="utf-8")))
    invented = {}
    for path in suite_files():
        used = set(MARKER.findall(path.read_text(encoding="utf-8")))
        # pytest's built-in markers.
        unknown = used - declared - {"parametrize", "skip", "skipif", "xfail", "usefixtures"}
        if unknown:
            invented[path.relative_to(TESTS).as_posix()] = sorted(unknown)
    assert not invented, f"markers not declared in pyproject.toml: {invented}"
