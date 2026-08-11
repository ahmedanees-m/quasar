"""Every test file must carry a marker, because the documented targets select by marker.

`pyproject.toml` declares four markers and the Makefile runs exactly two selections:
``make test`` is ``pytest -m fast`` and ``make test-all`` is
``pytest -m "fast or slow or gate"``. Both are marker expressions, so a file that declares no
marker is collected by neither. It is not skipped and not reported. It simply never runs, and
the suite says nothing about it.

That happened. ``tests/regression/test_gate_reporting.py`` was written to catch gates whose
reporting had silently stopped working, and it went in unmarked, so under both documented
entry points it silently did nothing itself. It passed only because it was invoked by path
during development, which is the one way nobody runs it afterwards.

This is the fourth instance in this project of a mechanism that exists, looks correct, and is
never reached: `make gates` set no image variable and produced no evidence, `run_all_gates.py`
skipped four work packages behind an allowlist and reported success, the sweep runner had no
quantum route while answering a question about quantum routes, and now this. The pattern is
always the same and always invisible from the outside, because the absence of output looks
exactly like the absence of a problem.
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
    """If the glob breaks, everything below passes vacuously."""
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
        f"{unmarked} declare no marker from {sorted(DECLARED)}. Neither `make test` nor "
        f"`make test-all` selects them, so they never run under any documented entry point. "
        f"Add `pytestmark = pytest.mark.fast` at module level, or the marker that fits."
    )


def test_no_file_invents_a_marker_that_pyproject_does_not_declare() -> None:
    """`--strict-markers` catches this at runtime, but only for files that get collected."""
    root = TESTS.parent / "pyproject.toml"
    declared = set(re.findall(r'"(\w+):', root.read_text(encoding="utf-8")))
    invented = {}
    for path in suite_files():
        used = set(MARKER.findall(path.read_text(encoding="utf-8")))
        # parametrize, skip, xfail and friends are pytest's own, not acceptance markers.
        unknown = used - declared - {"parametrize", "skip", "skipif", "xfail", "usefixtures"}
        if unknown:
            invented[path.relative_to(TESTS).as_posix()] = sorted(unknown)
    assert not invented, f"markers not declared in pyproject.toml: {invented}"
