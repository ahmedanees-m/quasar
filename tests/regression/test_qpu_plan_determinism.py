"""Plan rebuilding for `--recover` is deterministic.

`--recover` attaches a completed job's results to a rebuilt plan by position, which requires
varQITE to converge to the same parameters and the transpiler to choose the same layout.
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

    assert (
        first.data == second.data
    ), "varQITE bound different parameters on a second run of the same point"
    assert (reference_a == reference_b).all()


@pytest.mark.fast
def test_transpilation_is_deterministic_and_calibration_shares_the_layout() -> None:
    """The seed must pin the layout, and calibration must land on the data circuit's qubits."""
    module = load()
    # `qiskit-ibm-runtime` is the optional `qpu` extra, installed in CI.
    fake_provider = pytest.importorskip(
        "qiskit_ibm_runtime.fake_provider",
        reason="requires the optional qpu extra",
    )

    backend = getattr(fake_provider, module.FAKE_BACKEND)()
    module.SIZES = [2]
    module.MU_RATIOS = {2: [1.0]}

    first = module.transpile_with_shared_layout(module.build_plan(), backend)
    second = module.transpile_with_shared_layout(module.build_plan(), backend)

    assert [e["layout"] for e in first] == [e["layout"] for e in second], (
        "the transpiler chose different physical qubits on a second run despite " "SEED_TRANSPILER"
    )
    assert [e["depth"] for e in first] == [e["depth"] for e in second]

    # Calibration circuits share the data circuits' physical qubits.
    layouts = {e["kind"]: e["layout"] for e in first}
    assert layouts["sweep"] == layouts["calibration"], (
        "calibration circuits were transpiled onto different physical qubits than the data "
        "circuits"
    )
