"""G-R.6: varQITE reproduces the quasispecies at circuit depth constant in imaginary time.

Two registered criteria, and the second is the one that matters for hardware. Accuracy alone
would be satisfied by any competent solver. What makes varQITE a near-term method is that the
circuit never changes: only its parameters move, so evolving ten times longer costs no extra
depth. The run therefore evolves each configuration twice, to tau = 2.5 and tau = 20, and
compares the circuits.

Depths are compared after transpilation as well as before. A run that happened to leave an
angle near zero could have that rotation optimised away, which would change the transpiled
depth even though the ansatz is identical, so comparing only the written circuit would be
the weaker check.

Thresholds, ansatz rule and configurations are in docs/protocol.md section 3 and revision 6,
committed before this ran.

    python experiments/wp_r_rebuild/g_r_6_varqite.py
"""

from __future__ import annotations

import sys
import time

import numpy as np
from qiskit import transpile

from quasarstack.analytic.crow_kimura import additive_quasispecies, class_quasispecies
from quasarstack.analytic.exact_diag import perron_vector
from quasarstack.classical.landscapes import (
    class_fitness,
    nk_fitness,
    single_peak_classes,
)
from quasarstack.hamiltonian.builder import additive_hamiltonian, diagonal_hamiltonian
from quasarstack.io.store import write_gate_record
from quasarstack.ite.varqite import Ansatz, evolve, verify_hardware_route
from quasarstack.scoring.metrics import score

# Registered in docs/protocol.md section 3.
COSINE_THRESHOLD = 0.999

# Registered in docs/protocol.md revision 6.
MU = 0.20
SIZES = [3, 4, 5, 6]
TAU_CAP = 60.0
TAU_SHORT = 2.5
TAU_LONG = 20.0
DTAU = 0.05
# A rate, so its scale is set by matching the old per-step infidelity criterion: infidelity
# below 1e-9 corresponds to a step of about 4.5e-5, hence a rate of about 9e-4 at this step
# size. 1e-3 is the like-for-like value. Carrying the old number across the change of
# quantity made it six orders too strict, and every configuration ran to the tau cap.
TOLERANCE = 1e-3
RIDGE = 1e-6
BASIS = ["rz", "sx", "x", "cx"]
HARDWARE_CHECK_MAX_SITES = 4
# Three seeds, not one. A single instance understates the requirement badly: at L = 6 and
# reps = 4, seed 0 reaches 0.99996 while the worst of three seeds reaches only 0.9913, and it
# was the worst that justified the reps = L + 2 rule. Reporting the lucky seed would have put
# a misleading number in the record next to a rule it did not support.
REPS_SCAN = {"L": 6, "K": 2, "reps": [4, 6, 8], "seeds": [0, 1, 2]}
# Step sizes for the energy-descent refinement. The continuous flow cannot raise the energy,
# so any rise is the explicit Euler integrator overshooting, and the test of that is whether
# it shrinks with the step. Run at L = 4 to keep the finest step affordable.
DESCENT_REFINEMENT = {"L": 4, "K": 2, "dtaus": [0.05, 0.02, 0.01, 0.005]}


def reps_for(n_sites: int) -> int:
    """The registered ansatz rule: reps = L + 2, the shallowest choice that cleared 0.999."""
    return n_sites + 2


def _configuration(name: str, n_sites: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (hamiltonian_matrix, reference_distribution), or None if not applicable."""
    if name == "additive_random":
        a = np.random.default_rng(0).uniform(0.25, 2.00, size=n_sites)
        matrix = np.asarray(additive_hamiltonian(a, MU).to_matrix()).real
        return matrix, additive_quasispecies(a, MU)

    if name == "single_peak":
        f_by_class = single_peak_classes(n_sites, 1.0)
        matrix = np.asarray(diagonal_hamiltonian(class_fitness(f_by_class), MU).to_matrix()).real
        reference, _, _ = class_quasispecies(f_by_class, MU)
        return matrix, reference

    if name.startswith("nk"):
        k = int(name.split("K")[1])
        if k > n_sites - 1:
            return None
        if k == 4 and n_sites < 5:
            return None
        fitness = nk_fitness(n_sites, k, seed=0)
        matrix = np.asarray(diagonal_hamiltonian(fitness, MU).to_matrix()).real
        reference, _, _ = perron_vector(fitness, MU)
        return matrix, reference

    raise ValueError(f"unregistered configuration {name!r}")


def _descent_report(energies: list[float]) -> dict[str, float | int | bool]:
    """How far the energy ever went the wrong way, and by how much relative to the descent.

    The continuous McLachlan flow cannot raise the energy: ``dE/dtau`` equals
    ``-(1/2) grad(E)^T (A + delta I)^-1 grad(E)``, and ``A`` is a Gram matrix so the whole
    quadratic form is non-negative. Any rise is therefore the explicit Euler integrator
    overshooting at finite step, and the honest thing to record is its size rather than a
    boolean that will always read False at a usable step size.
    """
    series = np.asarray(energies, dtype=np.float64)
    rises = np.diff(series)
    up = rises[rises > 0.0]
    span = float(series[0] - series[-1])
    largest = float(up.max()) if up.size else 0.0
    return {
        "n_rises": int(up.size),
        "n_steps": int(series.size),
        "largest_rise": largest,
        "largest_rise_relative_to_span": largest / span if span > 0 else 0.0,
        "first_rise_at_step": int(np.argmax(rises > 0.0)) if up.size else -1,
        "strictly_monotone": bool(up.size == 0),
    }


def _shape(ansatz: Ansatz, params: np.ndarray) -> dict[str, int]:
    circuit = ansatz.circuit(params)
    transpiled = transpile(circuit, basis_gates=BASIS, optimization_level=1, seed_transpiler=0)
    return {
        "depth": int(circuit.depth()),
        "two_qubit_gates": int(circuit.count_ops().get("cx", 0)),
        "transpiled_depth": int(transpiled.depth()),
        "transpiled_two_qubit_gates": int(transpiled.count_ops().get("cx", 0)),
    }


def run() -> tuple[bool, dict, list[dict]]:
    started = time.monotonic()
    cases: list[dict] = []

    for n_sites in SIZES:
        for name in ("additive_random", "single_peak", "nkK2", "nkK4"):
            built = _configuration(name, n_sites)
            if built is None:
                continue
            matrix, reference = built
            ansatz = Ansatz(n_sites, reps=reps_for(n_sites))

            evolution = evolve(
                ansatz, matrix, tau=TAU_CAP, dtau=DTAU, ridge=RIDGE, tolerance=TOLERANCE
            )
            scores = score(evolution.probs, reference)
            descent = _descent_report(evolution.energies)

            short = evolve(ansatz, matrix, tau=TAU_SHORT, dtau=DTAU, ridge=RIDGE, tolerance=0.0)
            long = evolve(ansatz, matrix, tau=TAU_LONG, dtau=DTAU, ridge=RIDGE, tolerance=0.0)
            shape_short, shape_long = _shape(ansatz, short.params), _shape(ansatz, long.params)

            hardware = None
            if n_sites <= HARDWARE_CHECK_MAX_SITES:
                hardware = verify_hardware_route(ansatz, evolution.params, matrix)

            cases.append(
                {
                    "landscape": name,
                    "L": n_sites,
                    "mu": MU,
                    "reps": ansatz.reps,
                    "n_parameters": ansatz.n_parameters,
                    "cosine": scores["cosine"],
                    "tv": scores["tv"],
                    "tau_used": evolution.tau_used,
                    "converged": evolution.converged,
                    "final_state_change": evolution.final_state_change,
                    "final_parameter_change": evolution.final_parameter_change,
                    "descent": descent,
                    "shape_at_short_tau": shape_short,
                    "shape_at_long_tau": shape_long,
                    "depth_unchanged": shape_short == shape_long,
                    "hardware_route": hardware,
                }
            )

    # Diagnostic: how much ansatz does the hardest size actually need?
    # Worst over three seeds, because that is what the rule was chosen against.
    reps_scan = []
    for reps in REPS_SCAN["reps"]:
        ansatz = Ansatz(REPS_SCAN["L"], reps=reps)
        cosines, taus = [], []
        for seed in REPS_SCAN["seeds"]:
            fitness = nk_fitness(REPS_SCAN["L"], REPS_SCAN["K"], seed=seed)
            matrix = np.asarray(diagonal_hamiltonian(fitness, MU).to_matrix()).real
            reference, _, _ = perron_vector(fitness, MU)
            evolution = evolve(
                ansatz, matrix, tau=TAU_CAP, dtau=DTAU, ridge=RIDGE, tolerance=TOLERANCE
            )
            cosines.append(score(evolution.probs, reference)["cosine"])
            taus.append(evolution.tau_used)
        reps_scan.append(
            {
                "reps": reps,
                "n_parameters": ansatz.n_parameters,
                "worst_cosine": float(min(cosines)),
                "best_cosine": float(max(cosines)),
                "clears_threshold_on_every_seed": bool(min(cosines) >= COSINE_THRESHOLD),
                "max_tau_used": float(max(taus)),
            }
        )

    # Diagnostic: is the energy rise discretisation or a defect? The continuous flow cannot
    # raise the energy, so a rise that shrinks with the step is the Euler integrator, and one
    # that does not would be a bug of the kind the planning documents record for Motta-QITE.
    fitness = nk_fitness(DESCENT_REFINEMENT["L"], DESCENT_REFINEMENT["K"], seed=0)
    matrix = np.asarray(diagonal_hamiltonian(fitness, MU).to_matrix()).real
    refinement = []
    for dtau in DESCENT_REFINEMENT["dtaus"]:
        evolution = evolve(
            Ansatz(DESCENT_REFINEMENT["L"], reps=reps_for(DESCENT_REFINEMENT["L"])),
            matrix,
            tau=TAU_CAP,
            dtau=dtau,
            ridge=RIDGE,
            tolerance=TOLERANCE,
        )
        refinement.append({"dtau": dtau, **_descent_report(evolution.energies)})
    rises = [row["largest_rise"] for row in refinement]
    rise_shrinks_with_step = all(
        later <= earlier for earlier, later in zip(rises, rises[1:], strict=False)
    )

    elapsed = time.monotonic() - started
    cosines = np.array([c["cosine"] for c in cases])
    worst = int(np.argmin(cosines))
    depths_unchanged = all(c["depth_unchanged"] for c in cases)
    hardware_cases = [c for c in cases if c["hardware_route"] is not None]

    measured = {
        "n_configurations": len(cases),
        "min_cosine": float(cosines.min()),
        "n_below_threshold": int((cosines < COSINE_THRESHOLD).sum()),
        "max_tv": float(max(c["tv"] for c in cases)),
        "all_depths_unchanged": depths_unchanged,
        "n_strictly_monotone": sum(1 for c in cases if c["descent"]["strictly_monotone"]),
        "largest_energy_rise_relative_to_span": float(
            max(c["descent"]["largest_rise_relative_to_span"] for c in cases)
        ),
        "descent_refinement": refinement,
        "energy_rise_shrinks_with_step": rise_shrinks_with_step,
        "all_converged": all(c["converged"] for c in cases),
        "max_tau_used": float(max(c["tau_used"] for c in cases)),
        "min_tau_used": float(min(c["tau_used"] for c in cases)),
        "hardware_route_max_force_error": float(
            max(c["hardware_route"]["force_max_abs_error"] for c in hardware_cases)
        ),
        "hardware_route_max_tensor_error": float(
            max(c["hardware_route"]["tensor_max_abs_error"] for c in hardware_cases)
        ),
        "expressibility_scan": reps_scan,
        "worst_case": {k: v for k, v in cases[worst].items() if k != "hardware_route"},
        "seconds": round(elapsed, 2),
    }
    passed = bool(cosines.min() >= COSINE_THRESHOLD and depths_unchanged)
    return passed, measured, cases


def main() -> int:
    passed, measured, cases = run()

    path = write_gate_record(
        gate="G-R.6",
        work_package="wp_r",
        threshold={
            "accuracy": {"statistic": "cosine against the reference", "value": COSINE_THRESHOLD},
            "depth": {
                "statistic": "circuit depth and two-qubit count identical at tau = 2.5 and "
                "tau = 20, before and after transpilation",
                "value": "identical",
            },
            "registered_in": "docs/protocol.md section 3, ansatz rule and configurations in revision 6",
        },
        measured=measured,
        passed=passed,
        cases=cases,
        notes=(
            "The ansatz depth rule reps = L + 2 was chosen by a pre-run scan, disclosed in "
            "revision 6 with its numbers, because an ansatz too shallow to hold the answer "
            "fails for reasons unrelated to the method. Convergence is judged on the state "
            "rather than the parameters: the ansatz has gauge directions, so parameters "
            "drift long after the state has settled. tau_used is reported per configuration "
            "as the budget-needed-for-accuracy half of docs/notes.md."
        ),
    )

    print(f"G-R.6: {len(cases)} configurations in {measured['seconds']} s\n")
    header = (
        f"{'landscape':16s} {'L':>2s} {'reps':>4s} {'P':>3s} {'cosine':>11s} "
        f"{'tau_used':>8s} {'conv':>5s} {'depth':>6s} {'2q':>3s} {'same':>5s}"
    )
    print(header)
    print("-" * len(header))
    for case in cases:
        shape = case["shape_at_long_tau"]
        print(
            f"{case['landscape']:16s} {case['L']:>2d} {case['reps']:>4d} "
            f"{case['n_parameters']:>3d} {case['cosine']:>11.7f} {case['tau_used']:>8.2f} "
            f"{str(case['converged']):>5s} {shape['transpiled_depth']:>6d} "
            f"{shape['transpiled_two_qubit_gates']:>3d} {str(case['depth_unchanged']):>5s}"
        )

    print("\nansatz depth needed at L = 6, K = 2, worst over three seeds:")
    for row in measured["expressibility_scan"]:
        print(
            f"  reps={row['reps']:>2d} P={row['n_parameters']:>3d} "
            f"worst={row['worst_cosine']:.7f} best={row['best_cosine']:.7f} "
            f"clears_every_seed={row['clears_threshold_on_every_seed']}"
        )

    print("\nenergy descent against step size (the continuous flow cannot ascend):")
    for row in measured["descent_refinement"]:
        print(
            f"  dtau={row['dtau']:<6} steps={row['n_steps']:>5d} rises={row['n_rises']:>3d} "
            f"largest={row['largest_rise']:.3e} "
            f"relative={row['largest_rise_relative_to_span']:.2e}"
        )
    print(f"  rise shrinks with step: {measured['energy_rise_shrinks_with_step']}")

    print(
        f"\n  min cosine              {measured['min_cosine']:.7f}  (threshold {COSINE_THRESHOLD})"
    )
    print(f"  depth constant in tau   {measured['all_depths_unchanged']}")
    print(
        f"  strictly monotone       {measured['n_strictly_monotone']} of "
        f"{measured['n_configurations']}, largest rise "
        f"{measured['largest_energy_rise_relative_to_span']:.2e} of the descent"
    )
    print(f"  all converged           {measured['all_converged']}")
    print(
        f"  tau needed              {measured['min_tau_used']:.2f} to "
        f"{measured['max_tau_used']:.2f}"
    )
    print(
        f"  hardware route errors   C {measured['hardware_route_max_force_error']:.2e}, "
        f"A {measured['hardware_route_max_tensor_error']:.2e}"
    )
    print(f"  record                  {path.relative_to(path.parents[2])}")
    print(f"  {'PASS' if passed else 'FAIL'}")
    if not passed:
        print(f"  worst case: {measured['worst_case']}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
