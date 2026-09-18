"""Every gate's ``main`` can print its own committed record.

The committed record is replayed through the real ``main``: ``run`` is stubbed to return what
the record holds, so the reporting path runs against real data in milliseconds. Two properties
are checked:

* ``main`` completes without raising, so every key it prints exists in the record.
* its exit code agrees with the record's ``passed``.

A gate without a committed record is skipped.
"""

from __future__ import annotations

import importlib.util
import io
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import pytest

from quasarstack.io.store import RESULTS_ROOT

# The first case takes about 3 s: G-R.8 imports qiskit-aer once per session.
pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT / "experiments"

GATE_CALL = re.compile(r'gate="(?P<gate>[^"]+)"[^)]*?work_package="(?P<wp>[^"]+)"', re.S)

# A script that computes its gate id (G-8 appends a device suffix) declares GATE and
# WORK_PACKAGE as module-level literals instead.
GATE_DECLARATION = re.compile(
    r'^GATE\s*=\s*"(?P<gate>[^"]+)"\s*$.*?^WORK_PACKAGE\s*=\s*"(?P<wp>[^"]+)"\s*$',
    re.S | re.M,
)


def artefact_name(gate: str) -> str:
    """The file name rule of `write_gate_record`."""
    return f"{gate.lower().replace('-', '_').replace('.', '_')}.json"


def gate_scripts() -> list[tuple[Path, str, str]]:
    """Every script that writes a gate record, with the gate and work package it declares."""
    found = []
    for path in sorted(EXPERIMENTS.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "write_gate_record(" not in source:
            continue
        match = GATE_CALL.search(source) or GATE_DECLARATION.search(source)
        assert match is not None, (
            f"{path.name} writes a record but declares no gate id. Pass gate= and "
            f"work_package= as literals, or declare GATE and WORK_PACKAGE at module level."
        )
        found.append((path, match["gate"], match["wp"]))
    return found


SCRIPTS = gate_scripts()


def restore_integer_keys(value: Any) -> Any:
    """Undo JSON's stringification of integer dict keys.

    ``measured["ruggedness_by_k_at_L8"]`` is keyed by K and G-R.5 prints it with ``{k:>2d}``.
    """
    if isinstance(value, dict):
        return {
            (int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k): restore_integer_keys(
                v
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [restore_integer_keys(v) for v in value]
    return value


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"gate_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_at_least_one_gate_script_was_discovered() -> None:
    """Discovery walks the experiments tree."""
    assert len(SCRIPTS) >= 10, f"only {len(SCRIPTS)} gate scripts found, the glob is wrong"


# Gates that compute inside main and so cannot be replayed. Empty.
COMPUTES_INSIDE_MAIN: set[str] = set()


def test_every_gate_separates_measuring_from_reporting() -> None:
    unsplit = {gate for path, gate, _ in SCRIPTS if "def run(" not in path.read_text("utf-8")}
    assert unsplit <= COMPUTES_INSIDE_MAIN, (
        f"{sorted(unsplit - COMPUTES_INSIDE_MAIN)} compute inside main, so their reporting "
        f"cannot be replayed. Split run from main."
    )


@pytest.mark.parametrize(
    ("script", "gate", "work_package"),
    SCRIPTS,
    ids=[gate for _, gate, _ in SCRIPTS],
)
def test_main_can_print_its_own_artefact(
    script: Path, gate: str, work_package: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    record_path = RESULTS_ROOT / work_package / artefact_name(gate)
    if not record_path.is_file():
        pytest.skip(f"{gate} has no committed record")

    record = restore_integer_keys(json.loads(record_path.read_text(encoding="utf-8")))
    module = load_module(script)
    if not hasattr(module, "run"):
        pytest.skip(f"{gate} computes inside main, so its reporting cannot be replayed")

    monkeypatch.setattr(
        module, "run", lambda *a, **k: (record["passed"], record["measured"], record["cases"])
    )
    # The reporting path needs no file on disk.
    monkeypatch.setattr(module, "write_gate_record", lambda **k: tmp_path / artefact_name(gate))

    captured = io.StringIO()
    try:
        with redirect_stdout(captured):
            exit_code = module.main()
    except KeyError as missing:
        ran_at = str(record.get("env", {}).get("git_sha", "unknown"))[:8]
        raise AssertionError(
            f"{gate}'s main asks for {missing}, which its committed record (produced at "
            f"{ran_at}) does not contain. Either main and run disagree on a key, or the record "
            f"predates the key and {gate} needs to be rerun in the pinned image."
        ) from missing

    assert exit_code == (
        0 if record["passed"] else 1
    ), f"{gate} exits {exit_code} while its record has passed={record['passed']}"
    assert captured.getvalue().strip(), f"{gate} printed nothing"
