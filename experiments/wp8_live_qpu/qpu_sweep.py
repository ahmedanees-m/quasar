"""G-8: the error-threshold sweep on a real QPU, and the dry run that must pass first.

`GATES.md` section 12 registers G-8 as a **feasibility** gate. It sets no accuracy threshold,
requires job IDs, backend name, calibration date, transpiled depth, two-qubit gate count,
shots, and both raw and mitigated distributions, and states that framing the result as
evidence of advantage fails the gate. Nothing here claims advantage.

One rule shapes this file: **the dry run and the real submission are the same code path.**
`--mode dry` uses `FakeMarrakesh`, a calibration snapshot of the target device itself, and
every other line is identical. A dry run that exercised different code would prove nothing
about the run that spends irreplaceable budget.

What is submitted
-----------------

Not a single distribution, which would only show the circuit runs. The **mutation-rate sweep
across the error threshold** at `L = 2, 3, 4` on the single-peak landscape, which is a
scientific figure rather than a smoke test.

The circuit is not iterative on the device. Route A's expense is its classical optimisation
loop, and that loop runs in simulation here as it does in G-R.8: varQITE converges, the
optimised parameters are bound into the ansatz, and the QPU sees one shallow circuit per
point plus readout calibration.

Readout calibration is the **full 2^n assignment matrix**, 4 + 8 + 16 = 28 circuits, not the
2 per qubit a tensored scheme would need. That is 10 circuits more than the cheaper scheme and
it is the right trade: G-R.8 validated its mitigated cosines with this estimator and a
constrained NNLS solve, and the most useful outcome of this run is whether the device matches
those simulated predictions. Swapping the estimator for the one hardware run would compare two
different methods and answer a different question.

**Calibration circuits are pinned to the data circuit's physical qubits.** A layout chosen
independently would produce an assignment matrix for the wrong qubits, and the mitigation
would then be confidently wrong rather than obviously broken.

Modes
-----

    python experiments/wp8_live_qpu/qpu_sweep.py --mode dry     # FakeMarrakesh, no network
    python experiments/wp8_live_qpu/qpu_sweep.py --mode isa     # real target, transpile only
    python experiments/wp8_live_qpu/qpu_sweep.py --mode pilot   # one circuit per size, timed
    python experiments/wp8_live_qpu/qpu_sweep.py --mode main    # the full set, one batch
    python experiments/wp8_live_qpu/qpu_sweep.py --recover ID   # analyse a job already paid for

`isa` reaches the network but submits nothing: it transpiles against the real target so the
depth and two-qubit counts can be read before any budget is committed, which is step 3 of the
run plan. It is free.

`pilot` submits one circuit per size rather than two of the smallest. Its whole purpose is to
predict the cost of the main run, and the `L = 4` circuit is 58 deep against 13 at `L = 2`, so
timing only the cheap end and multiplying would repeat a mistake this project has already made
once, in section 4.27.

`pilot` and `main` submit to hardware and consume budget. Both refuse to run without
`--i-mean-it`, because a mode flag is too small a thing to stand between a typo and an
irreplaceable resource.

`--recover` spends nothing and needs no confirmation: it rebuilds the plan deterministically,
fetches a job that has already been paid for, and runs the identical analysis. Job ids are
written to disk the instant they exist so that a crash between submission and retrieval cannot
lose spent budget, and this is the path that makes that promise real rather than decorative.
"""

from __future__ import annotations

import argparse
import contextlib
import json
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

# The target device. Heron r2, 156 qubits, chosen because it executes: it begins running
# within seconds of submission rather than sitting in a queue that does not clear. It is not
# the lowest-error device on the platform, and the artefact records its calibration timestamp,
# error rates and layout so a reader can weigh that.
#
# `FakeMarrakesh` is a calibration snapshot of this same device, so `--mode dry` exercises the
# target rather than a stand-in. The snapshot is dated 2025-02-26 and is about eighteen months
# stale, and the hardware bore that out: the real device ran below the fake by 1.4e-4, 6.5e-4
# and 2.7e-3 at L = 2, 3, 4, widening with depth. So the dry run proves the code path and is
# never used to predict fidelity. Fidelity is compared against G-R.8's noise model, which the
# hardware matched to within 2.2e-3 at all three points where a prediction existed.
TARGET_BACKEND = "ibm_marrakesh"
FAKE_BACKEND = "FakeMarrakesh"
# Fixed so a transpile can be checked in one process and reproduced in the next. Without it,
# SabreLayout is free to choose differently at submission than at inspection, which would make
# the pre-flight check describe circuits other than the ones that ran.
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
# varQITE settings copied from G-R.8 so the state prepared here is the state that gate
# validated, rather than a differently converged one that happens to look similar.
TAU_CAP = 40.0
DTAU = 0.05
TOLERANCE = 1e-9

JOB_LOG = pathlib.Path.home() / "quasar_qpu_jobs.jsonl"

# Declared statically so the gate-reporting test can discover this script's gate without
# importing it. `gate_name` appends a device suffix when a run goes to a backend other than the
# registered target, so that a cross-device comparison cannot overwrite the registered record,
# but the canonical artefact this gate is judged on is G-8 in wp8.
GATE = "G-8"
WORK_PACKAGE = "wp8"


def mu_critical(n_sites: int) -> float:
    """The single-peak threshold, `height / L`, as section 11.1 defines it."""
    return PEAK_HEIGHT / n_sites


def prepare_state(n_sites: int, mu: float) -> tuple[QuantumCircuit, np.ndarray]:
    """Converge varQITE in simulation, bind the parameters, return circuit and reference.

    The device never sees the optimisation. This mirrors `g_r_8_noise.py` exactly.
    """
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
    """Counts to a genotype distribution, in the project's indexing.

    Qiskit hands back big-endian bitstrings and this project indexes site i by bit i, so the
    conversion goes through `qiskit_bitstring_to_genotype`. Doing it by hand here would be a
    second place for an endianness bug to live.
    """
    probabilities = np.zeros(1 << n_sites, dtype=np.float64)
    for bitstring, count in counts.items():
        genotype = qiskit_bitstring_to_genotype(bitstring.replace(" ", ""))
        probabilities[genotype_to_index(genotype)] += count
    total = probabilities.sum()
    if total <= 0:
        raise ValueError("no counts returned")
    return probabilities / total


def build_plan() -> list[dict[str, Any]]:
    """Every circuit that will be submitted, built once so it can be counted before it runs.

    This phase converges varQITE for all 23 sweep points in simulation and takes several
    minutes, during which nothing has been submitted and no budget is at risk. It used to print
    nothing at all, which made a healthy build indistinguishable from a hung process: the run
    was reported as failed by someone watching the workloads page, correctly, because there was
    no other evidence available. Progress now goes to stderr per point.
    """
    plan: list[dict[str, Any]] = []
    total = sum(len(MU_RATIOS[n]) for n in SIZES)
    progress = Progress(total, "varqite")
    for n_sites in SIZES:
        mu_c = mu_critical(n_sites)
        for ratio in MU_RATIOS[n_sites]:
            mu = ratio * mu_c
            circuit, reference = prepare_state(n_sites, mu)
            progress.step(f"L={n_sites} mu/mu_c={ratio:.2f} (nothing submitted yet)")
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
    """The fake device for a dry run, the real one otherwise. Same class of object either way.

    `backend_name` exists so a second device can be run alongside the registered target without
    editing `TARGET_BACKEND`. Step 5 of the run plan holds budget back precisely for a
    cross-device comparison, so a flag is the honest way to do it. Anything submitted elsewhere
    records its own device provenance and lands in its own suffixed artefact, so a comparison
    run can never overwrite or be confused with the target's record.
    """
    if mode == "dry":
        from qiskit_ibm_runtime import fake_provider

        return getattr(fake_provider, FAKE_BACKEND)(), None

    from qiskit_ibm_runtime import QiskitRuntimeService

    key_file = pathlib.Path("G:/My Drive/Qubis_HiQ/QUASAR/IBM/apikey.json")
    if not key_file.is_file():
        raise SystemExit(f"no credential at {key_file}")
    token = json.loads(key_file.read_text(encoding="utf-8"))["apikey"]
    service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
    return service.backend(backend_name or TARGET_BACKEND), service


def transpile_with_shared_layout(plan: list[dict[str, Any]], backend) -> list[dict[str, Any]]:
    """Transpile every circuit, pinning calibration to the layout its data circuits use.

    The assignment matrix corrects the qubits it was measured on. If the calibration circuits
    were transpiled independently they would land on a different set, and the correction would
    be applied to qubits it does not describe: wrong, and wrong quietly.
    """
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
    """A SamplerV2 configured identically for fake and real backends.

    Dynamical decoupling is on with XY4. It is error *suppression* rather than mitigation, it
    costs nothing in extra circuits, and QuBiS-HiQ measured it working on this device family.
    Turning it on for the real run but not the dry run would mean the dry run validated a
    different pipeline, which is the one thing this file refuses to do.
    """
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
    """Write the job id to disk the moment it exists.

    Before the result is fetched, before anything is analysed. A crash or a reboot between
    submission and retrieval would otherwise lose budget that has already been spent, and this
    machine rebooted three times in two hours on 13 August.
    """
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
    # Usage is reported differently across runtime versions and is absent on a fake backend,
    # so each source is attempted and a miss is not an error. Measuring real usage is the
    # entire point of the pilot mode, so whatever the service does report is kept verbatim.
    for source, getter in (
        ("metrics", lambda: job.metrics()),
        ("usage_estimation", lambda: job.usage_estimation),
    ):
        with contextlib.suppress(Exception):
            usage[source] = getter()
    return usage


def result_with_retry(job, attempts: int = 40, pause: float = 60.0):
    """Wait for a job's result, treating a network failure as weather rather than an outcome.

    This is not defensive programming for its own sake. The pilot's client died exactly here:
    `job.result()` polls the API, DNS failed to resolve `quantum.cloud.ibm.com` once, and the
    exception propagated out of a process that was otherwise doing nothing but waiting. The
    job was untouched on the service and finished later regardless, but the process that was
    supposed to collect it was gone.

    A queue wait on the open plan runs to hours, and the odds of a laptop keeping an unbroken
    connection for that long are not good. Retrying costs nothing, because the job is already
    submitted and already paid for: the only thing at risk is the client's ability to collect
    what it bought.
    """
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
    """The log line written when this job was submitted: its mode and its device.

    Both matter for recovery. The mode decides how many circuits the rebuilt plan holds, and
    the device decides the layouts they are transpiled onto. Reading either from a flag typed
    by a person after a crash is how a recovery quietly describes the wrong circuits.
    """
    if not JOB_LOG.is_file():
        return None
    for line in JOB_LOG.read_text(encoding="utf-8").splitlines():
        with contextlib.suppress(Exception):
            entry = json.loads(line)
            if entry.get("job_id") == job_id:
                return entry
    return None


def attach_from_job(plan: list[dict[str, Any]], job) -> dict[str, Any]:
    """Attach an already-finished job's counts onto a freshly rebuilt plan.

    The reason job ids are written to disk the moment they exist is so that a crash between
    submission and retrieval does not lose budget already spent. That reason only holds if
    something can act on a recorded id, and until now nothing could: the id was a receipt for
    results no code path would ever fetch. This is the path that redeems it.

    Matching is positional, which is safe only because the rebuild is deterministic: varQITE
    converges to the same parameters and `SEED_TRANSPILER` pins the layout, so circuit `i` of a
    rebuilt plan is circuit `i` of the submitted one. The count is checked rather than assumed,
    because a positional match against the wrong plan would produce results that look fine and
    describe different circuits.
    """
    result = result_with_retry(job)
    if len(result) != len(plan):
        raise SystemExit(
            f"job returned {len(result)} results for a rebuilt plan of {len(plan)} circuits. "
            f"The plan does not match the job, so attaching counts positionally would silently "
            f"pair each result with the wrong circuit. Check the recorded mode and shot count."
        )
    for entry, pub in zip(plan, result, strict=True):
        entry["counts"] = pub.data.meas.get_counts()

    usage: dict[str, Any] = {"job_id": str(job.job_id())}
    for source, getter in (("metrics", lambda: job.metrics()), ("usage", lambda: job.usage())):
        with contextlib.suppress(Exception):
            usage[source] = getter()
    return usage


def shot_noise_floor(reference: np.ndarray, shots: int, draws: int = 200) -> dict[str, float]:
    """What a perfect device would score at this shot count, by sampling the exact reference.

    Without this the record invites a wrong reading. The dry run scored total variation 0.125 at
    `L = 4`, which looks like heavy device error and is mostly sampling: a noiseless device
    scores 0.082 at the same shot count. The cause is the square-root decode. A basis state with
    true probability `1e-4` carries amplitude `1e-2`, so the sampling error on a rare outcome is
    magnified on its way into the decoded distribution, and total variation feels it.

    The reference is a probability vector while the circuit holds it in **amplitudes**, so the
    draw is from `reference**2` renormalised. Sampling the reference directly would be the
    amplitude-for-probability substitution this project treats as its most dangerous silent
    error, and it produces a floor worse than the measurement it is supposed to bound.
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
                # The pilot submits data circuits only, so no assignment matrix exists and the
                # mitigated columns hold unmitigated values. Saying so in the record is cheaper
                # than a reader discovering later that a "mitigated cosine" was nothing of the
                # kind.
                "readout_mitigation_applied": matrix is not None,
            }
        )
    return cases


def jsonable(value: Any) -> Any:
    """Make a value safe for `json.dumps`, recursively.

    The runtime returns `job.metrics()` with `datetime` objects nested inside it. Serialising
    the gate record then raises, and it raises *after* the QPU has run: the budget is spent, the
    results are in memory, and the process dies before anything reaches disk. The dry run found
    this, which is what a dry run is for.
    """
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
    """Backend name and calibration timestamp, which section 12 requires by name."""
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
    """Everything up to the point where a circuit could be submitted, and no further.

    Shared by the run and by `--mode isa`, so the free inspection describes the same circuits
    the paid submission sends. The transpiler is seeded for the same reason.
    """
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
    """Transpile against the real target and submit nothing. Step 3 of the run plan, free."""
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
    print(f"  nothing submitted, nothing spent. report {destination}")
    return 0


def run(
    mode: str,
    shots: int,
    recover_id: str | None = None,
    backend_name: str | None = None,
) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
    """Measure. Everything printed by `main` comes from what this returns.

    The split is not decoration. G-5 renamed keys here and not in its reporting, and printed a
    summary that disagreed with the artefact it had just written. For a run that spends an
    irreplaceable budget, a printed summary that does not come from the record is worse than no
    summary: it is the thing a reader will believe.
    """
    started = time.monotonic()
    recovering = recover_id is not None
    # A recovered job was submitted in some earlier mode, and the plan must be rebuilt as that
    # mode built it or the positional match in `attach_from_job` pairs results with the wrong
    # circuits. The mode comes from the job log rather than from a flag, because a person
    # retyping it after a crash is exactly who gets it wrong.
    if recovering:
        entry = recorded_job(recover_id) or {}
        mode = entry.get("mode") or mode
        # The backend must come from the log too. A plan transpiled against the wrong device
        # would carry the wrong layouts, and recovery would then report a correct set of
        # counts against physical qubits that never ran them.
        backend_name = entry.get("backend") or backend_name
        print(f"  recovering job {recover_id}, mode {mode!r} on {backend_name!r}")
    backend, device, plan = build_and_transpile(mode, backend_name)

    if (
        not recovering
        and mode in {"pilot", "main"}
        and (device["calibration_timestamp"] == "unavailable")
    ):
        raise SystemExit(
            "refusing to submit: section 12 requires the calibration date by name and this "
            "backend will not report one. Failing now costs nothing; failing after the run "
            "costs budget that cannot be bought back."
        )

    if mode == "pilot":
        # One circuit per size, not two of the smallest. The pilot exists to extrapolate the
        # main run's cost, and the L = 4 circuit is 58 deep against 13 at L = 2. Timing only the
        # cheapest circuits and scaling by 51 would repeat the mistake of section 4.27, where a
        # ratio measured on one mix of sizes was extended to a different mix and came out wrong
        # by orders of magnitude. Three circuits instead of two is a trivial extra cost and it
        # brackets the range rather than sampling one end of it.
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
        "what_this_is_not": (
            "A feasibility and validation result, as GATES.md section 12 registers G-8. No "
            "accuracy threshold is set and no advantage is claimed. Route B is absent from "
            "this run because it cannot be run here: 1024 walk-operator queries on 5 to 9 "
            "ancillas is a deep coherent circuit and squarely fault-tolerant territory. That "
            "absence is the two-currency comparison of section 4.35 made concrete, not a gap "
            "in the experiment."
        ),
        "seconds": round(time.monotonic() - started, 2),
    }
    # G-8 is a feasibility gate. It passes when the run is recorded with the provenance section
    # 12 names, not when the numbers are good, and every one of those fields is checked here
    # rather than assumed. A gate that reports its own success without looking is not a gate.
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
    # Defaulted rather than required so the gate-reporting replay test can call `main()` with no
    # argument vector, and defaulted to the mode that spends nothing.
    parser.add_argument("--mode", choices=["dry", "isa", "pilot", "main"], default="dry")
    parser.add_argument(
        "--i-mean-it",
        action="store_true",
        help="required for pilot and main, which spend irreplaceable QPU budget",
    )
    parser.add_argument("--shots", type=int, default=SHOTS)
    parser.add_argument(
        "--backend",
        default=None,
        help=f"device to submit to, default {TARGET_BACKEND} as revision 25 registers. "
        "A second device is what run-plan step 5 reserves budget for.",
    )
    parser.add_argument(
        "--recover",
        metavar="JOB_ID",
        help="fetch an already-submitted job by id and analyse it, submitting nothing. "
        "This is what the job log is for.",
    )
    arguments, _unknown = parser.parse_known_args()

    # Recovery reads a job that has already been paid for, so it spends nothing and needs
    # no confirmation. Requiring --i-mean-it here would put a speed bump in front of the
    # one action a person takes while trying to rescue results after a crash.
    if arguments.recover:
        passed, measured, cases = run(
            arguments.mode, arguments.shots, arguments.recover, arguments.backend
        )
        return report(passed, measured, cases)

    if arguments.mode in {"pilot", "main"} and not arguments.i_mean_it:
        print(
            f"--mode {arguments.mode} submits to {TARGET_BACKEND} and spends QPU budget.\n"
            f"Re-run with --i-mean-it if that is what you intend."
        )
        return 2

    if arguments.mode == "isa":
        return inspect_only()

    passed, measured, cases = run(arguments.mode, arguments.shots, backend_name=arguments.backend)
    return report(passed, measured, cases)


def gate_name(measured: dict[str, Any]) -> str:
    """G-8 for the registered target, suffixed elsewhere so records cannot collide."""
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
            "registered_in": "GATES.md section 12, revision 25",
        },
        measured=measured,
        passed=passed,
        cases=cases,
        notes=measured["what_this_is_not"],
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
