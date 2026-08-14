"""G-R.8: feasibility under simulated device noise, with readout mitigation.

**Every number here comes from a simulated noise model. Nothing ran on hardware.** The live
run is WP8 and reports job identifiers. `QUASAR_engineering_standards.md` section 11.2
requires the label wherever the figures appear, and the planning documents already list
simulated-not-live among the things this project does not claim.

The comparison is made in the **decoded** domain, against the analytic quasispecies, because
that is the object the biology asks for. That choice matters more than it sounds. The circuit
holds the quasispecies in its *amplitudes*, so a computational-basis measurement returns the
distribution *squared*, and the two differ by a total-variation distance of 0.22 while their
cosine similarity is 0.987. Scoring the undecoded measurement would look almost fine and
would be measuring the wrong object. The square-root decode in
`quasarstack.io.conventions.decode_from_measurement` inverts it.

Thresholds and configurations are in GATES.md section 3 and revision 10, committed before
this ran.

    python experiments/wp_r_rebuild/g_r_8_noise.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

from quasarstack.analytic.crow_kimura import additive_quasispecies, class_quasispecies
from quasarstack.backends.execution import run_pipeline
from quasarstack.backends.hardware import HERON_LIKE, TRAPPED_ION_LIKE
from quasarstack.classical.landscapes import class_fitness, single_peak_classes
from quasarstack.hamiltonian.builder import additive_hamiltonian, diagonal_hamiltonian
from quasarstack.io.conventions import decode_from_measurement
from quasarstack.io.store import write_gate_record
from quasarstack.ite.varqite import Ansatz, evolve
from quasarstack.scoring.metrics import score

# Registered in GATES.md section 3.
MITIGATED_COSINE_THRESHOLD = 0.98

# Registered in GATES.md revision 10.
MU = 0.20
SIZES = [2, 3, 4]
SHOTS = 40000
CALIBRATION_SHOTS = 40000
SEED = 0
DEVICES = [HERON_LIKE, TRAPPED_ION_LIKE]
TAU_CAP = 60.0
DTAU = 0.05
TOLERANCE = 1e-3


def _prepare(name: str, n_sites: int) -> tuple[Ansatz, np.ndarray, np.ndarray]:
    """Return the ansatz, its converged parameters, and the analytic reference."""
    if name == "additive_random":
        a = np.random.default_rng(0).uniform(0.25, 2.00, size=n_sites)
        matrix = np.asarray(additive_hamiltonian(a, MU).to_matrix()).real
        reference = additive_quasispecies(a, MU)
    elif name == "single_peak":
        f_by_class = single_peak_classes(n_sites, 1.0)
        matrix = np.asarray(diagonal_hamiltonian(class_fitness(f_by_class), MU).to_matrix()).real
        reference, _, _ = class_quasispecies(f_by_class, MU)
    else:
        raise ValueError(f"unregistered landscape {name!r}")

    ansatz = Ansatz(n_sites, reps=n_sites + 2)
    evolution = evolve(ansatz, matrix, tau=TAU_CAP, dtau=DTAU, tolerance=TOLERANCE)
    return ansatz, evolution.params, reference


def run() -> tuple[bool, dict, list[dict]]:
    started = time.monotonic()
    cases: list[dict] = []

    for n_sites in SIZES:
        for name in ("additive_random", "single_peak"):
            ansatz, params, reference = _prepare(name, n_sites)
            circuit = ansatz.circuit(params)
            for device in DEVICES:
                report = run_pipeline(
                    circuit,
                    device,
                    shots=SHOTS,
                    seed=SEED,
                    calibration_shots=CALIBRATION_SHOTS,
                )

                decoded = {
                    key: decode_from_measurement(report[key])
                    for key in ("noiseless", "raw", "mitigated")
                }
                scored = {key: score(value, reference) for key, value in decoded.items()}
                # Also score in the measured domain, which answers a different and easier
                # question: did mitigation recover what the ideal circuit would have given?
                against_ideal = score(report["mitigated"], report["noiseless"])

                cases.append(
                    {
                        "landscape": name,
                        "L": n_sites,
                        "mu": MU,
                        "device": device.name,
                        "simulated": True,
                        "provenance": device.provenance,
                        "shots": SHOTS,
                        "resources": report["resources"],
                        "decoded_noiseless": scored["noiseless"],
                        "decoded_raw": scored["raw"],
                        "decoded_mitigated": scored["mitigated"],
                        "mitigated_against_ideal_sampling": against_ideal,
                        "mitigation_improved_cosine": bool(
                            scored["mitigated"]["cosine"] >= scored["raw"]["cosine"]
                        ),
                        "assignment_diagonal_min": report["assignment_matrix_diagonal_min"],
                    }
                )

    elapsed = time.monotonic() - started
    mitigated = np.array([c["decoded_mitigated"]["cosine"] for c in cases])
    raw = np.array([c["decoded_raw"]["cosine"] for c in cases])
    worst = int(np.argmin(mitigated))

    by_device = {}
    for device in DEVICES:
        subset = [c for c in cases if c["device"] == device.name]
        by_device[device.name] = {
            "min_mitigated_cosine": float(min(c["decoded_mitigated"]["cosine"] for c in subset)),
            "max_mitigated_tv": float(max(c["decoded_mitigated"]["tv"] for c in subset)),
            "max_two_qubit_gates": int(max(c["resources"]["two_qubit_gates"] for c in subset)),
            "max_depth": int(max(c["resources"]["depth"] for c in subset)),
        }

    measured = {
        "simulated_noise_only": True,
        "n_cases": len(cases),
        "min_mitigated_cosine": float(mitigated.min()),
        "min_raw_cosine": float(raw.min()),
        "max_mitigated_tv": float(max(c["decoded_mitigated"]["tv"] for c in cases)),
        "max_raw_tv": float(max(c["decoded_raw"]["tv"] for c in cases)),
        "n_cases_mitigation_helped": int(sum(c["mitigation_improved_cosine"] for c in cases)),
        "min_mitigated_against_ideal_sampling": float(
            min(c["mitigated_against_ideal_sampling"]["cosine"] for c in cases)
        ),
        "by_device": by_device,
        "worst_case": cases[worst],
        "seconds": round(elapsed, 2),
    }
    return bool(mitigated.min() >= MITIGATED_COSINE_THRESHOLD), measured, cases


def main() -> int:
    passed, measured, cases = run()

    path = write_gate_record(
        gate="G-R.8",
        work_package="wp_r",
        threshold={
            "statistic": "cosine between the mitigated, decoded distribution and the "
            "analytic quasispecies",
            "value": MITIGATED_COSINE_THRESHOLD,
            "registered_in": "GATES.md section 3, configurations in revision 10",
        },
        measured=measured,
        passed=passed,
        cases=cases,
        notes=(
            "SIMULATED NOISE ONLY. Nothing here ran on hardware; the live run is WP8. "
            "Scoring is done in the decoded domain against the analytic quasispecies, "
            "because the circuit holds the distribution in its amplitudes and a "
            "computational-basis measurement returns it squared. Total variation is "
            "reported beside cosine throughout, and it is much the less flattering of the "
            "two here, because the square-root decode amplifies the noise floor in the tail."
        ),
    )

    print(f"G-R.8 (SIMULATED NOISE): {len(cases)} cases in {measured['seconds']} s\n")
    header = (
        f"{'landscape':16s} {'L':>2s} {'device':28s} {'depth':>6s} {'2q':>4s} "
        f"{'raw cos':>9s} {'mit cos':>9s} {'mit tv':>9s}"
    )
    print(header)
    print("-" * len(header))
    for case in cases:
        print(
            f"{case['landscape']:16s} {case['L']:>2d} {case['device']:28s} "
            f"{case['resources']['depth']:>6d} {case['resources']['two_qubit_gates']:>4d} "
            f"{case['decoded_raw']['cosine']:>9.6f} {case['decoded_mitigated']['cosine']:>9.6f} "
            f"{case['decoded_mitigated']['tv']:>9.4f}"
        )

    print(
        f"\n  min mitigated cosine       {measured['min_mitigated_cosine']:.6f}  "
        f"(threshold {MITIGATED_COSINE_THRESHOLD})"
    )
    print(f"  min raw cosine             {measured['min_raw_cosine']:.6f}")
    print(
        f"  max mitigated TV           {measured['max_mitigated_tv']:.4f}  "
        f"(the less flattering metric, and the one that matters)"
    )
    print(
        f"  mitigation helped in       {measured['n_cases_mitigation_helped']} of "
        f"{measured['n_cases']} cases"
    )
    print(
        f"  mitigated vs ideal sampling {measured['min_mitigated_against_ideal_sampling']:.6f} "
        f"(the easier question: did mitigation recover the ideal circuit output)"
    )
    print(f"  record                     {path.relative_to(path.parents[2])}")
    print(f"  {'PASS' if passed else 'FAIL'}  [SIMULATED NOISE, NOT HARDWARE]")
    if not passed:
        print(f"  worst case: {measured['worst_case']}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
