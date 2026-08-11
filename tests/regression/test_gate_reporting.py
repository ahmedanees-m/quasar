"""Every gate's ``main`` must survive printing its own committed artefact.

G-5 provoked this. Its computation was right, the record it wrote was right, and it then
died with a ``KeyError`` in the block that prints the summary, because ``run`` had been
renamed to say ``in_class_instances_the_baseline_refused`` and ``main`` still asked for
``in_class_configurations_the_predicate_rejected``. The exit code was 1, the chain that
invoked it recorded a failure, and the gate had in fact passed at 1.375e-10.

The smoke test that was supposed to catch this called ``run`` directly, so it never touched
the line that broke. This test closes that hole for every gate at once by replaying the
committed artefact through the real ``main``: ``run`` is stubbed to hand back exactly what
the artefact recorded, and the reporting path runs against real data in milliseconds
instead of the minutes or hours a recomputation would cost.

Two properties are checked, and the second is the one with teeth:

* ``main`` completes without raising, so no key it prints has drifted from ``run``.
* its exit code agrees with the artefact's ``passed``, so a gate cannot report failure
  while its own evidence says otherwise, which is precisely the way G-5 went wrong.

A gate with no artefact yet is skipped, and the skip is derived from the filesystem rather
than from a list someone has to remember to prune. The count is asserted so the whole suite
cannot quietly degrade to zero the way `run_all_gates.py` once did.
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

# Marked, and the marker matters more than it looks. `make test` selects `-m fast` and
# `make test-all` selects `-m "fast or slow or gate"`, so a file carrying no marker at all is
# run by neither. This file was unmarked when it was written: a test against gates that
# silently do nothing, itself silently doing nothing under both documented entry points. It
# only ever ran because it was invoked by path. tests/unit/test_suite_markers.py now refuses
# to let an unmarked file be added again.
#
# One case takes about 3 s rather than the sub-second `fast` promises. It is not the replay:
# G-R.8 imports the hardware backends, which pull in qiskit-aer, and that import costs 3.0 s
# measured on its own, once per session, paid by whichever test reaches it first. The replay
# parses a twelve-case artefact in 0.01 s. Recorded so the number is not mistaken later for a
# slow test worth removing.
pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT / "experiments"

GATE_CALL = re.compile(r'gate="(?P<gate>[^"]+)"[^)]*?work_package="(?P<wp>[^"]+)"', re.S)


def artefact_name(gate: str) -> str:
    """The same rule `write_gate_record` uses, so the two cannot drift apart."""
    return f"{gate.lower().replace('-', '_').replace('.', '_')}.json"


def gate_scripts() -> list[tuple[Path, str, str]]:
    """Every script that writes a gate record, with the gate and work package it declares."""
    found = []
    for path in sorted(EXPERIMENTS.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "write_gate_record(" not in source:
            continue
        match = GATE_CALL.search(source)
        assert match is not None, f"{path.name} writes a record but declares no gate id"
        found.append((path, match["gate"], match["wp"]))
    return found


SCRIPTS = gate_scripts()


def restore_integer_keys(value: Any) -> Any:
    """Undo JSON's stringification of dict keys, so the replay is faithful.

    ``measured["ruggedness_by_k_at_L8"]`` is keyed by K, an integer, and G-R.5 prints it with
    ``{k:>2d}``. JSON has no integer keys, so a naive replay hands ``main`` the string ``"2"``
    and the format code fails. That failure is the harness lying about what the gate received,
    not a defect in the gate, and the distinction matters: repairing the gate to satisfy an
    unfaithful test would be the G-5 mistake with the roles swapped.
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
    """The discovery above walks the tree, so an empty result means the walk broke."""
    assert len(SCRIPTS) >= 10, f"only {len(SCRIPTS)} gate scripts found, the glob is wrong"


# G-1 and G-6.3 used to compute inside main and were exempted here by name. Both have since
# been split, so the exemption list is empty and stays empty: every gate is replayable.
COMPUTES_INSIDE_MAIN: set[str] = set()


def test_every_gate_separates_measuring_from_reporting() -> None:
    unsplit = {gate for path, gate, _ in SCRIPTS if "def run(" not in path.read_text("utf-8")}
    assert unsplit <= COMPUTES_INSIDE_MAIN, (
        f"{sorted(unsplit - COMPUTES_INSIDE_MAIN)} compute inside main, so their summary "
        f"printing cannot be replayed and is untested. Split run from main as every other "
        f"gate does. Do not add the gate to the exemption set instead: it is empty on purpose."
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
        pytest.skip(f"{gate} has no committed artefact yet")

    record = restore_integer_keys(json.loads(record_path.read_text(encoding="utf-8")))
    module = load_module(script)
    if not hasattr(module, "run"):
        pytest.skip(f"{gate} computes inside main, so its reporting cannot be replayed")

    monkeypatch.setattr(
        module, "run", lambda *a, **k: (record["passed"], record["measured"], record["cases"])
    )
    # The real writer would put a laptop-provenance record into results/_local. Nothing about
    # the reporting path needs a file on disk, so it gets a path that is never written.
    monkeypatch.setattr(module, "write_gate_record", lambda **k: tmp_path / artefact_name(gate))

    captured = io.StringIO()
    try:
        with redirect_stdout(captured):
            exit_code = module.main()
    except KeyError as missing:
        ran_at = str(record.get("env", {}).get("git_sha", "unknown"))[:8]
        raise AssertionError(
            f"{gate}'s main asks for {missing} and its committed artefact does not have it. "
            f"The artefact was produced at {ran_at}. Two things look like this and they have "
            f"opposite fixes. If the key was renamed in run and not in main, fix main: that is "
            f"the bug this test was written for, and it cost G-5 a gate run. If the key is new "
            f"in run, the code is ahead of the evidence and the fix is to rerun {gate} in the "
            f"pinned image, not to soften either side. Committed evidence its own gate can no "
            f"longer read is not evidence anyone should trust."
        ) from missing

    assert exit_code == (
        0 if record["passed"] else 1
    ), f"{gate} exits {exit_code} while its artefact records passed={record['passed']}"
    assert captured.getvalue().strip(), f"{gate} printed nothing"
