"""G-8: the error-threshold sweep on IBM Quantum hardware.

G-8 is a feasibility check with no accuracy threshold. The record carries job IDs, backend
name, calibration date, transpiled depth, two-qubit gate count, shots, and both raw and
mitigated distributions.

The dry run and the hardware submission share one code path. `--mode dry` uses
`FakeMarrakesh`, a calibration snapshot of the target device.

What is submitted
-----------------

The mutation-rate sweep across the error threshold at `L = 2, 3, 4` on the single-peak
landscape. varQITE converges in simulation, as in G-R.8, and the optimised parameters are bound
into the ansatz, so the device runs one shallow circuit per point plus readout calibration.

Readout calibration uses the full 2^n assignment matrix, 4 + 8 + 16 = 28 circuits, with the
constrained NNLS estimator of G-R.8. Calibration circuits are pinned to the physical qubits of
the data circuits.

Modes
-----

    python experiments/wp8_live_qpu/qpu_sweep.py --mode dry     # FakeMarrakesh, no network
    python experiments/wp8_live_qpu/qpu_sweep.py --mode isa     # real target, transpile only
    python experiments/wp8_live_qpu/qpu_sweep.py --mode pilot   # one circuit per size, timed
    python experiments/wp8_live_qpu/qpu_sweep.py --mode main    # the full set, one batch
    python experiments/wp8_live_qpu/qpu_sweep.py --recover ID   # analyse a completed job

`isa` transpiles against the real target and submits nothing, so depth and two-qubit counts
can be read before submission. `pilot` submits one circuit per size to estimate the cost of the
main run. `pilot` and `main` submit to hardware and require `--i-mean-it`. `--recover` rebuilds
the plan deterministically, fetches a completed job and runs the same analysis. Job IDs are
written to disk as soon as they exist. The IBM Quantum token is read from ``QISKIT_IBM_TOKEN``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import sys
import time
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager

from quasarstack.analytic.crow_kimura import class_quasispecies
from quasarstack.backends.execution import mitigate_readout
from quasarstack.classical.landscapes import class_fitness, single_peak_classes
from quasarstack.hamiltonian.builder import diagonal_hamiltonian
from quasarstack.io.conventions import (
    decode_from_measurement,
    genotype_to_index,
    qiskit_bitstring_to_genotype,
)
from quasarstack.io.progress import Progress
from quasarstack.io.store import write_gate_record
from quasarstack.ite.varqite import Ansatz, evolve
from quasarstack.scoring.metrics import score

# Target device: IBM Heron r2, 156 qubits. The record carries its calibration timestamp, error
# rates and layout. `FakeMarrakesh` is a calibration snapshot of the same device dated
# 2025-02-26; the dry run checks the code path, and fidelity is compared against the G-R.8 noise
# model instead.
TARGET_BACKEND = "ibm_marrakesh"
FAKE_BACKEND = "FakeMarrakesh"
# Fixed so that the transpilation inspected with `--mode isa` is the one submitted.
SEED_TRANSPILER = 20260813

SIZES = [2, 3, 4]
MU_RATIOS = {
    2: [0.4, 0.7, 1.0, 1.3, 1.6],
    3: [0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.45, 1.6],
    4: [0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.45, 1.6],
}
PEAK_HEIGHT = 1.0
SHOTS = 4096
OPTIMISATION_LEVEL = 3
# varQITE settings of G-R.8.
TAU_CAP = 40.0
DTAU = 0.05
TOLERANCE = 1e-9

JOB_LOG = pathlib.Path.home() / "quasar_qpu_jobs.jsonl"

# Declared statically for the gate-reporting test. `gate_name` appends a suffix for other modes
# and devices; the main run on the target device writes G-8.
GATE = "G-8"
WORK_PACKAGE = "wp8"


def mu_critical(n_sites: int) -> float:
    """The single-peak threshold, `height / L`."""
    return PEAK_HEIGHT / n_sites


def prepare_state(n_sites: int, mu: float) -> tuple[QuantumCircuit, np.ndarray]:
    """Converge varQITE in simulation, bind the parameters, return circuit and reference."""
    classes = single_peak_classes(n_sites, PEAK_HEIGHT)
    matrix = np.asarray(diagonal_hamiltonian(class_fitness(classes), mu).to_matrix()).real
    reference, _, _ = class_quasispecies(classes, mu)

    ansatz = Ansatz(n_sites, reps=n_sites + 2)
    evolution = evolve(ansatz, matrix, tau=TAU_CAP, dtau=DTAU, tolerance=TOLERANCE)
    circuit = ansatz.circuit(evolution.params)
    measured = circuit.copy()
    measured.measure_all()
    return measured, np.asarray(reference, dtype=np.float64)


def calibration_circuits(n_sites: int) -> list[QuantumCircuit]:
    """One circuit per computational basis state, matching `execution.assignment_matrix`."""
    circuits = []
    for prepared in range(1 << n_sites):
        circuit = QuantumCircuit(n_sites)
        for qubit in range(n_sites):
            if prepared >> qubit & 1:
                circuit.x(qubit)
        circuit.measure_all()
        circuits.append(circuit)
    return circuits


def counts_to_distribution(counts: dict[str, int], n_sites: int) -> np.ndarray:
    """Counts to a genotype distribution. Qiskit bitstrings are big-endian; site i is bit i."""
    probabilities = np.zeros(1 << n_sites, dtype=np.float64)
    for bitstring, count in counts.items():
        genotype = qiskit_bitstring_to_genotype(bitstring.replace(" ", ""))
        probabilities[genotype_to_index(genotype)] += count
    total = probabilities.sum()
    if total <= 0:
        raise ValueError("no counts returned")
    return probabilities / total


def build_plan() -> list[dict[str, Any]]:
    """Every circuit to be submitted. Converging varQITE for the 23 sweep points takes several
    minutes; progress goes to stderr per point."""
    plan: list[dict[str, Any]] = []
    total = sum(len(MU_RATIOS[n]) for n in SIZES)
    progress = Progress(total, "varqite")
    for n_sites in SIZES:
        mu_c = mu_critical(n_sites)
        for ratio in MU_RATIOS[n_sites]:
            mu = ratio * mu_c
            circuit, reference = prepare_state(n_sites, mu)
            progress.step(f"L={n_sites} mu/mu_c={ratio:.2f}")
            plan.append(
                {
                    "kind": "sweep",
                    "L": n_sites,
                    "mu": mu,
                    "mu_over_mu_c": ratio,
                    "circuit": circuit,
                    "reference": reference,
                }
            )
        for index, circuit in enumerate(calibration_circuits(n_sites)):
            plan.append(
                {
                    "kind": "calibration",
                    "L": n_sites,
                    "prepared_index": index,
                    "circuit": circuit,
                    "reference": None,
                }
            )
    progress.finish()
    return plan


def resolve_backend(mode: str, backend_name: str | None = None):
    """The fake device for a dry run, the real one otherwise.

    `backend_name` selects a device other than `TARGET_BACKEND`; such runs are written to a
    suffixed record.
    """
    if mode == "dry":
        from qiskit_ibm_runtime import fake_provider

        return getattr(fake_provider, FAKE_BACKEND)(), None

    from qiskit_ibm_runtime import QiskitRuntimeService

    token = os.environ.get("QISKIT_IBM_TOKEN", "")
    if not token:
        raise SystemExit("set QISKIT_IBM_TOKEN")
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    return service.backend(backend_name or TARGET_BACKEND), service


def transpile_with_shared_layout(plan: list[dict[str, Any]], backend) -> list[dict[str, Any]]:
    """Transpile every circuit, pinning calibration circuits to the layout of the data circuits."""
    manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=OPTIMISATION_LEVEL,
        seed_transpiler=SEED_TRANSPILER,
    )

    layouts: dict[int, list[int]] = {}
    for entry in plan:
        if entry["kind"] != "sweep":
            continue
        if entry["L"] in layouts:
            continue
        isa = manager.run(entry["circuit"])
        layouts[entry["L"]] = _physical_qubits(isa, entry["L"])

    for entry in plan:
        n_sites = entry["L"]
        pinned = generate_preset_pass_manager(
            backend=backend,
            optimization_level=OPTIMISATION_LEVEL,
            initial_layout=layouts[n_sites],
            seed_transpiler=SEED_TRANSPILER,
        )
        isa = pinned.run(entry["circuit"])
        entry["isa"] = isa
        entry["layout"] = layouts[n_sites]
        entry["depth"] = isa.depth()
        operations = isa.count_ops()
        entry["two_qubit_gates"] = sum(
            count for name, count in operations.items() if name in {"cz", "cx", "ecr", "rzz"}
        )
    return plan


def _physical_qubits(isa: QuantumCircuit, n_sites: int) -> list[int]:
    """Which physical qubits the transpiler chose for the n virtual ones."""
    layout = isa.layout
    if layout is None:
        return list(range(n_sites))
    mapping = layout.final_index_layout(filter_ancillas=True)
    return list(mapping[:n_sites])


def make_sampler(backend, mode: str, shots: int):
    """A SamplerV2 configured identically for fake and real backends, with XY4 dynamical
    decoupling."""
    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = shots
    try:
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XY4"
    except Exception as exc:  # noqa: BLE001
        print(f"  note: dynamical decoupling unavailable here ({type(exc).__name__})")
    return sampler


def record_job(job_id: str, mode: str, backend_name: str, n_circuits: int) -> None:
    """Append the job ID to the job log as soon as the job is submitted."""
    JOB_LOG.parent.mkdir(parents=True, exist_ok=True)
    with JOB_LOG.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "job_id": job_id,
                    "mode": mode,
                    "backend": backend_name,
                    "circuits": n_circuits,
                    "submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
            + "\n"
        )
    print(f"  job {job_id} recorded in {JOB_LOG}")


def execute(plan: list[dict[str, Any]], backend, mode: str, shots: int) -> dict[str, Any]:
    """Submit every circuit as one job and attach the counts back onto the plan."""
    sampler = make_sampler(backend, mode, shots)
    circuits = [entry["isa"] for entry in plan]

    print(f"  submitting {len(circuits)} circuits to {backend.name}")
    job = sampler.run(circuits, shots=shots)
    job_id = getattr(job, "job_id", lambda: "local")()
    record_job(str(job_id), mode, backend.name, len(circuits))

    result = result_with_retry(job)
    for entry, pub in zip(plan, result, strict=True):
        entry["counts"] = pub.data.meas.get_counts()

    usage: dict[str, Any] = {"job_id": str(job_id)}
    # Usage fields differ across runtime versions and are absent on a fake backend.
    for source, getter in (
        ("metrics", lambda: job.metrics()),
        ("usage_estimation", lambda: job.usage_estimation),
    ):
        with contextlib.suppress(Exception):
            usage[source] = getter()
    return usage


def result_with_retry(job, attempts: int = 40, pause: float = 60.0):
    """Wait for a job's result, retrying on network errors. The submitted job is unaffected."""
    for attempt in range(1, attempts + 1):
        try:
            return job.result()
        except Exception as error:  # noqa: BLE001
            if attempt == attempts:
                raise SystemExit(
                    f"gave up collecting job {job.job_id()} after {attempts} attempts: {error!r}\n"
                    f"The job itself is untouched. Recover it with:\n"
                    f"    python experiments/wp8_live_qpu/qpu_sweep.py --recover {job.job_id()}"
                ) from error
            print(
                f"  attempt {attempt} to collect the result failed "
                f"({type(error).__name__}), retrying in {pause:.0f}s. "
                f"The job is unaffected."
            )
            time.sleep(pause)
    raise SystemExit("unreachable")


def recorded_job(job_id: str) -> dict[str, Any] | None:
    """The job log entry for this job, which gives the mode and device to rebuild the plan."""
    if not JOB_LOG.is_file():
        return None
    for line in JOB_LOG.read_text(encoding="utf-8").splitlines():
        with contextlib.suppress(Exception):
            entry = json.loads(line)
            if entry.get("job_id") == job_id:
                return entry
    return None


def attach_from_job(plan: list[dict[str, Any]], job) -> dict[str, Any]:
    """Attach a completed job's counts onto a rebuilt plan.

    Matching is positional. The rebuild is deterministic, since varQITE converges to the same
    parameters and `SEED_TRANSPILER` fixes the layout, and the circuit count is checked.
    """
    result = result_with_retry(job)
    if len(result) != len(plan):
        raise SystemExit(
            f"job returned {len(result)} results for a rebuilt plan of {len(plan)} circuits. "
            f"Check the recorded mode and shot count."
        )
    for entry, pub in zip(plan, result, strict=True):
        entry["counts"] = pub.data.meas.get_counts()

    usage: dict[str, Any] = {"job_id": str(job.job_id())}
    for source, getter in (("metrics", lambda: job.metrics()), ("usage", lambda: job.usage())):
        with contextlib.suppress(Exception):
            usage[source] = getter()
    return usage


def shot_noise_floor(reference: np.ndarray, shots: int, draws: int = 200) -> dict[str, float]:
    """Mean score of a noiseless device at this shot count, by sampling the exact reference.

    The circuit holds the reference in its amplitudes, so samples are drawn from `reference**2`.
    The square-root decode magnifies sampling error on rare outcomes: a basis state with
    probability `1e-4` carries amplitude `1e-2`.
    """
    probabilities = np.clip(np.asarray(reference, dtype=np.float64) ** 2, 0.0, None)
    total = probabilities.sum()
    if total <= 0:
        return {"tv": float("nan"), "cosine": float("nan")}
    probabilities = probabilities / total
    generator = np.random.default_rng(abs(hash((shots, probabilities.size))) % (2**32))
    tv, cosine = [], []
    for _ in range(draws):
        sampled = generator.multinomial(shots, probabilities) / shots
        scored = score(decode_from_measurement(sampled), reference)
        tv.append(scored["tv"])
        cosine.append(scored["cosine"])
    return {"tv": float(np.mean(tv)), "cosine": float(np.mean(cosine))}


def analyse(plan: list[dict[str, Any]], shots: int) -> list[dict[str, Any]]:
    """Assignment matrix per size, then mitigate, decode and score every sweep point."""
    assignment: dict[int, np.ndarray] = {}
    for n_sites in sorted({e["L"] for e in plan}):
        calibration = sorted(
            (e for e in plan if e["kind"] == "calibration" and e["L"] == n_sites),
            key=lambda e: e["prepared_index"],
        )
        if not calibration:
            continue
        dimension = 1 << n_sites
        matrix = np.zeros((dimension, dimension), dtype=np.float64)
        for entry in calibration:
            matrix[:, entry["prepared_index"]] = counts_to_distribution(entry["counts"], n_sites)
        assignment[n_sites] = matrix

    cases: list[dict[str, Any]] = []
    for entry in plan:
        if entry["kind"] != "sweep":
            continue
        n_sites = entry["L"]
        raw = counts_to_distribution(entry["counts"], n_sites)
        matrix = assignment.get(n_sites)
        mitigated = mitigate_readout(raw, matrix) if matrix is not None else raw

        reference = entry["reference"]
        decoded_raw = decode_from_measurement(raw)
        decoded_mitigated = decode_from_measurement(mitigated)
        scored_raw = score(decoded_raw, reference)
        scored_mitigated = score(decoded_mitigated, reference)
        floor = shot_noise_floor(reference, shots)
        cases.append(
            {
                "L": n_sites,
                "shot_noise_floor_tv": floor["tv"],
                "shot_noise_floor_cosine": floor["cosine"],
                "total_variation_above_floor": scored_mitigated["tv"] - floor["tv"],
                "mu": entry["mu"],
                "mu_over_mu_c": entry["mu_over_mu_c"],
                "layout": entry["layout"],
                "transpiled_depth": entry["depth"],
                "two_qubit_gates": entry["two_qubit_gates"],
                "shots": shots,
                "raw_distribution": raw.tolist(),
                "mitigated_distribution": mitigated.tolist(),
                "decoded_raw_cosine": scored_raw["cosine"],
                "decoded_mitigated_cosine": scored_mitigated["cosine"],
                "decoded_raw_total_variation": scored_raw["tv"],
                "decoded_mitigated_total_variation": scored_mitigated["tv"],
                "assignment_diagonal_min": (
                    float(np.min(np.diag(matrix))) if matrix is not None else None
                ),
                # The pilot submits data circuits only, without calibration.
                "readout_mitigation_applied": matrix is not None,
            }
        )
    return cases


def jsonable(value: Any) -> Any:
    """Make a value safe for `json.dumps`, recursively. `job.metrics()` contains datetimes."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, bool, int, float, type(None))):
        return value
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def backend_provenance(backend) -> dict[str, Any]:
    """Backend name, qubit count, basis gates and calibration timestamp."""
    provenance: dict[str, Any] = {"backend": backend.name}
    for field, getter in (
        ("num_qubits", lambda: backend.num_qubits),
        ("processor_type", lambda: getattr(backend, "processor_type", None)),
        ("basis_gates", lambda: sorted(backend.basis_gates)),
    ):
        try:
            provenance[field] = getter()
        except Exception:  # noqa: BLE001
            provenance[field] = None
    for name in ("last_update_date", "updated"):
        try:
            properties = backend.properties()
            stamp = getattr(properties, name, None)
            if stamp is not None:
                provenance["calibration_timestamp"] = str(stamp)
                break
        except Exception:  # noqa: BLE001
            continue
    provenance.setdefault("calibration_timestamp", "unavailable")
    return provenance


def build_and_transpile(mode: str, backend_name: str | None = None):
    """Build and transpile the circuit set, shared by the run and by `--mode isa`."""
    print(f"building the circuit set ({mode})")
    plan = build_plan()
    sweep = [e for e in plan if e["kind"] == "sweep"]
    calibration = [e for e in plan if e["kind"] == "calibration"]
    print(f"  {len(sweep)} sweep circuits, {len(calibration)} calibration, {len(plan)} total")

    backend, _service = resolve_backend("pilot" if mode == "isa" else mode, backend_name)
    print(f"  backend {backend.name}")

    device = backend_provenance(backend)
    print(f"  calibration {device['calibration_timestamp']}")

    plan = transpile_with_shared_layout(plan, backend)
    print(
        f"  transpiled: worst depth {max(e['depth'] for e in plan)}, "
        f"worst two-qubit count {max(e['two_qubit_gates'] for e in plan)}"
    )
    for n_sites in SIZES:
        entries = [e for e in plan if e["L"] == n_sites and e["kind"] == "sweep"]
        print(
            f"    L={n_sites}: layout {entries[0]['layout']}, "
            f"depth {max(e['depth'] for e in entries)}, "
            f"2q {max(e['two_qubit_gates'] for e in entries)}"
        )
    return backend, device, plan


def inspect_only() -> int:
    """Transpile against the real target and submit nothing."""
    backend, device, plan = build_and_transpile("isa")
    report = {
        "device": device,
        "seed_transpiler": SEED_TRANSPILER,
        "optimisation_level": OPTIMISATION_LEVEL,
        "circuits": [
            {
                "kind": e["kind"],
                "L": e["L"],
                "mu_over_mu_c": e.get("mu_over_mu_c"),
                "prepared_index": e.get("prepared_index"),
                "layout": e["layout"],
                "depth": e["depth"],
                "two_qubit_gates": e["two_qubit_gates"],
            }
            for e in plan
        ],
    }
    destination = JOB_LOG.parent / f"quasar_isa_{backend.name}.json"
    destination.write_text(json.dumps(jsonable(report), indent=2), encoding="utf-8")
    print(f"  nothing submitted. report {destination}")
    return 0


def run(
    mode: str,
    shots: int,
    recover_id: str | None = None,
    backend_name: str | None = None,
) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
    """Build, submit or recover, and analyse. `main` prints from the returned values."""
    started = time.monotonic()
    recovering = recover_id is not None
    # A recovered job is rebuilt with the mode and device from the job log.
    if recovering:
        entry = recorded_job(recover_id) or {}
        mode = entry.get("mode") or mode
        backend_name = entry.get("backend") or backend_name
        print(f"  recovering job {recover_id}, mode {mode!r} on {backend_name!r}")
    backend, device, plan = build_and_transpile(mode, backend_name)

    if (
        not recovering
        and mode in {"pilot", "main"}
        and (device["calibration_timestamp"] == "unavailable")
    ):
        raise SystemExit("refusing to submit: the backend does not report a calibration date")

    if mode == "pilot":
        # One circuit per size, so the cost estimate spans depth 13 at L = 2 to 58 at L = 4.
        chosen = []
        for n_sites in SIZES:
            at_size = [e for e in plan if e["kind"] == "sweep" and e["L"] == n_sites]
            if at_size:
                chosen.append(at_size[len(at_size) // 2])
        plan = chosen
        print(f"  pilot: submitting {len(plan)} circuits, one per size, L={[e['L'] for e in plan]}")

    print(f"  {len(plan)} circuits x {shots} shots = {shots * len(plan)} shots")

    if recovering:
        _, service = resolve_backend("main", backend_name)
        usage = attach_from_job(plan, service.job(recover_id))
    else:
        usage = execute(plan, backend, mode, shots)
    cases = analyse(plan, shots)
    if not cases:
        raise SystemExit("no sweep cases were analysed")

    mitigated = [c["decoded_mitigated_cosine"] for c in cases]
    raw = [c["decoded_raw_cosine"] for c in cases]
    measured = {
        "mode": mode,
        "device": device,
        "simulated": mode == "dry",
        "circuits_submitted": len(plan),
        "shots_per_circuit": shots,
        "total_shots": shots * len(plan),
        "transpiled_worst_depth": max(e["depth"] for e in plan),
        "transpiled_worst_two_qubit_gates": max(e["two_qubit_gates"] for e in plan),
        "optimisation_level": OPTIMISATION_LEVEL,
        "dynamical_decoupling": "XY4",
        "readout_mitigation": "full assignment matrix, constrained NNLS, as G-R.8",
        "usage": jsonable(usage),
        "decoded_mitigated_cosine_min": float(np.min(mitigated)),
        "decoded_mitigated_cosine_median": float(np.median(mitigated)),
        "decoded_raw_cosine_min": float(np.min(raw)),
        "decoded_raw_cosine_median": float(np.median(raw)),
        "client_environment": {
            "qiskit": _version("qiskit"),
            "qiskit_ibm_runtime": _version("qiskit-ibm-runtime"),
        },
        "note": (
            "Feasibility result with no accuracy threshold. Route B is not run on hardware: "
            "1024 walk-operator queries on 5 to 9 ancillas is a deep coherent circuit."
        ),
        "seconds": round(time.monotonic() - started, 2),
    }
    # Passes when every provenance field is present.
    required = (
        "device",
        "circuits_submitted",
        "shots_per_circuit",
        "transpiled_worst_depth",
        "transpiled_worst_two_qubit_gates",
        "usage",
    )
    passed = (
        all(measured.get(field) is not None for field in required)
        and measured["device"]["calibration_timestamp"] != "unavailable"
        and all(c.get("raw_distribution") and c.get("mitigated_distribution") for c in cases)
        and bool(measured["usage"].get("job_id"))
    )
    return passed, measured, cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Defaults to the dry run, so `main()` can be called without arguments.
    parser.add_argument("--mode", choices=["dry", "isa", "pilot", "main"], default="dry")
    parser.add_argument(
        "--i-mean-it",
        action="store_true",
        help="required for pilot and main, which submit to hardware",
    )
    parser.add_argument("--shots", type=int, default=SHOTS)
    parser.add_argument(
        "--backend",
        default=None,
        help=f"device to submit to, default {TARGET_BACKEND}",
    )
    parser.add_argument(
        "--recover",
        metavar="JOB_ID",
        help="fetch a completed job by id and analyse it, submitting nothing",
    )
    arguments, _unknown = parser.parse_known_args()

    # Recovery submits nothing and needs no confirmation.
    if arguments.recover:
        passed, measured, cases = run(
            arguments.mode, arguments.shots, arguments.recover, arguments.backend
        )
        return report(passed, measured, cases)

    if arguments.mode in {"pilot", "main"} and not arguments.i_mean_it:
        print(
            f"--mode {arguments.mode} submits to {TARGET_BACKEND}.\n"
            f"Re-run with --i-mean-it if that is what you intend."
        )
        return 2

    if arguments.mode == "isa":
        return inspect_only()

    passed, measured, cases = run(arguments.mode, arguments.shots, backend_name=arguments.backend)
    return report(passed, measured, cases)


def gate_name(measured: dict[str, Any]) -> str:
    """G-8 for the main run on the target device, suffixed otherwise."""
    base = "G-8" if measured["mode"] == "main" else f"G-8-{measured['mode']}"
    device = measured["device"]["backend"]
    if not measured["simulated"] and device != TARGET_BACKEND:
        return f"{base}-{device.replace('_', '-')}"
    return base


def report(passed: bool, measured: dict[str, Any], cases: list[dict[str, Any]]) -> int:
    """Write the record and print from it. Nothing here recomputes anything."""
    path = write_gate_record(
        gate=gate_name(measured),
        work_package="wp8",
        threshold={
            "statistic": "feasibility: job ids, backend, calibration date, transpiled depth, "
            "two-qubit count, shots, and both raw and mitigated distributions recorded",
            "accuracy_threshold": None,
            "defined_in": "docs/protocol.md, G-8",
        },
        measured=measured,
        passed=passed,
        cases=cases,
        notes=measured["note"],
    )

    raw = [c["decoded_raw_cosine"] for c in cases]
    mitigated = [c["decoded_mitigated_cosine"] for c in cases]
    floor = [c["shot_noise_floor_tv"] for c in cases]
    above = [c["total_variation_above_floor"] for c in cases]
    print(f"\n  decoded cosine, raw       min {min(raw):.4f}  median {np.median(raw):.4f}")
    print(
        f"  decoded cosine, mitigated min {min(mitigated):.4f}  median {np.median(mitigated):.4f}"
    )
    print(f"  total variation above the shot-noise floor, worst {max(above):+.4f}")
    print(f"    (floor itself runs {min(floor):.4f} to {max(floor):.4f} at this shot count)")
    for size in sorted({c["L"] for c in cases}):
        at_size = [c for c in cases if c["L"] == size]
        print(
            f"    L={size}: mitigated cosine min {min(c['decoded_mitigated_cosine'] for c in at_size):.5f}"
            f"  depth {at_size[0]['transpiled_depth']}  2q {at_size[0]['two_qubit_gates']}"
        )
    print(f"  {'PASSED' if passed else 'FAILED'}  record  {path}")
    print(f"elapsed {measured['seconds']:.1f} s")
    return 0 if passed else 1


def _version(package: str) -> str:
    import importlib.metadata as metadata

    try:
        return metadata.version(package)
    except Exception:  # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
