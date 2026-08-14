"""`--recover` pairs a finished job's results with a freshly rebuilt plan, positionally.

That is safe only if rebuilding is deterministic. If varQITE converged to different parameters
on a second run, or the transpiler chose a different layout, recovery would attach each result
to the wrong circuit and produce a record that looks entirely reasonable and describes
something else. The docstring in `qpu_sweep.attach_from_job` asserts determinism; this checks
it, because an assumption that only exists in prose is not one a paid-for result should rest on.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments" / "wp8_live_qpu" / "qpu_sweep.py"


def load():
    spec = importlib.util.spec_from_file_location("qpu_sweep_determinism", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.fast
def test_state_preparation_is_deterministic() -> None:
    """Two independent varQITE convergences must bind identical parameters."""
    module = load()
    first, reference_a = module.prepare_state(2, 0.5 * module.mu_critical(2))
    second, reference_b = module.prepare_state(2, 0.5 * module.mu_critical(2))

    assert first.data == second.data, (
        "varQITE bound different parameters on a second run of the same point. Recovery "
        "attaches a finished job's results to a rebuilt plan by position, so a plan that "
        "rebuilds differently would pair every result with the wrong circuit and record it "
        "without complaint."
    )
    assert (reference_a == reference_b).all()


@pytest.mark.fast
def test_transpilation_is_deterministic_and_calibration_shares_the_layout() -> None:
    """The seed must pin the layout, and calibration must land on the data circuit's qubits."""
    module = load()
    from qiskit_ibm_runtime import fake_provider

    backend = getattr(fake_provider, module.FAKE_BACKEND)()
    module.SIZES = [2]
    module.MU_RATIOS = {2: [1.0]}

    first = module.transpile_with_shared_layout(module.build_plan(), backend)
    second = module.transpile_with_shared_layout(module.build_plan(), backend)

    assert [e["layout"] for e in first] == [e["layout"] for e in second], (
        "the transpiler chose different physical qubits on a second run despite "
        "SEED_TRANSPILER, so the free `--mode isa` inspection would describe circuits other "
        "than the ones submitted"
    )
    assert [e["depth"] for e in first] == [e["depth"] for e in second]

    # The assignment matrix corrects the qubits it was measured on. Calibration landing
    # elsewhere would mitigate the wrong qubits, and do it quietly.
    layouts = {e["kind"]: e["layout"] for e in first}
    assert layouts["sweep"] == layouts["calibration"], (
        "calibration circuits were transpiled onto different physical qubits than the data "
        "circuits, so the readout correction would describe qubits it was not measured on"
    )
