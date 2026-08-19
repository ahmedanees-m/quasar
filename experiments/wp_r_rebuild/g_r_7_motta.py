"""G-R.7: Motta-QITE reaches the quasispecies, and the energy descends.

Two registered criteria. Accuracy at cosine >= 0.95, which is deliberately looser than
varQITE's because the generator is truncated to a finite support and the truncation costs
accuracy. And **no energy increase beyond 1e-10 on any step**, which is the criterion that
matters, because an ascending energy is precisely the failure the planning documents record
for this method.

The parity demonstration is recorded per configuration rather than described. The recorded
failure came from a generator basis of the wrong parity, where Motta's right-hand side
``Re(-i <psi| sigma_I |Delta>)`` vanishes identically for a real state. The run computes that
quantity for both parities and records both, so the reason the basis is what it is sits in
the artefact.

Thresholds and configurations are in docs/protocol.md section 3 and revision 7, committed before
this ran.

    python experiments/wp_r_rebuild/g_r_7_motta.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

from quasarstack.analytic.crow_kimura import additive_quasispecies, class_quasispecies
from quasarstack.analytic.exact_diag import perron_vector
from quasarstack.classical.landscapes import (
    class_fitness,
    nk_fitness,
    single_peak_classes,
)
from quasarstack.hamiltonian.builder import additive_hamiltonian, diagonal_hamiltonian
from quasarstack.io.store import write_gate_record
from quasarstack.ite.qite_motta import (
    build_generators,
    evolve,
    odd_y_strings,
    zero_rhs_demonstration,
)
from quasarstack.scoring.metrics import score

# Registered in docs/protocol.md section 3.
COSINE_THRESHOLD = 0.95
ENERGY_RISE_TOLERANCE = 1e-10

# Registered in docs/protocol.md revision 7. Filled in from the pre-run scan.
MU = 0.20
SIZES = [3, 4, 5, 6]
MAX_WEIGHT = 2
TAU_CAP = 40.0
DTAU = 0.05
# A rate, so its scale is set by matching the old per-step infidelity criterion: infidelity
# below 1e-9 corresponds to a step of about 4.5e-5, hence a rate of about 9e-4 at this step
# size. Registered as 1e-3, the like-for-like value, rather than carrying the old number
# across a change of quantity, which made it six orders too strict.
TOLERANCE = 1e-3
RCOND = 1e-8
SUPPORT_SCAN = {"L": 5, "K": 2, "weights": [1, 2, 3]}


def _configuration(name: str, n_sites: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (hamiltonian_matrix, reference_distribution), or None if not applicable."""
    if name == "additive_random":
        a = np.random.default_rng(0).uniform(0.25, 2.00, size=n_sites)
        return np.asarray(additive_hamiltonian(a, MU).to_matrix()).real, additive_quasispecies(
            a, MU
        )

    if name == "single_peak":
        f_by_class = single_peak_classes(n_sites, 1.0)
        matrix = np.asarray(diagonal_hamiltonian(class_fitness(f_by_class), MU).to_matrix()).real
        reference, _, _ = class_quasispecies(f_by_class, MU)
        return matrix, reference

    if name.startswith("nk"):
        k = int(name.split("K")[1])
        if k > n_sites - 1:
            return None
        fitness = nk_fitness(n_sites, k, seed=0)
        matrix = np.asarray(diagonal_hamiltonian(fitness, MU).to_matrix()).real
        reference, _, _ = perron_vector(fitness, MU)
        return matrix, reference

    raise ValueError(f"unregistered configuration {name!r}")


def _descent_report(energies: list[float]) -> dict[str, float | int | bool]:
    """Energy rises measured against the registered tolerance, not against zero."""
    series = np.asarray(energies, dtype=np.float64)
    rises = np.diff(series)
    beyond = rises[rises > ENERGY_RISE_TOLERANCE]
    span = float(series[0] - series[-1])
    largest = float(beyond.max()) if beyond.size else 0.0
    return {
        "n_rises_beyond_tolerance": int(beyond.size),
        "largest_rise": largest,
        "largest_rise_relative_to_span": largest / span if span > 0 else 0.0,
        "descends": bool(beyond.size == 0),
        "n_steps": int(series.size),
    }


def run() -> tuple[bool, dict, list[dict]]:
    started = time.monotonic()
    cases: list[dict] = []

    for n_sites in SIZES:
        basis = build_generators(n_sites, MAX_WEIGHT)
        for name in ("additive_random", "single_peak", "nkK2", "nkK4"):
            built = _configuration(name, n_sites)
            if built is None:
                continue
            matrix, reference = built

            evolution = evolve(
                matrix,
                n_sites=n_sites,
                tau=TAU_CAP,
                dtau=DTAU,
                tolerance=TOLERANCE,
                rcond=RCOND,
                generators=basis,
            )
            scores = score(evolution.probs, reference)
            descent = _descent_report(evolution.energies)

            uniform = np.full(1 << n_sites, 1.0 / np.sqrt(1 << n_sites))
            parity = zero_rhs_demonstration(uniform, matrix, n_sites, MAX_WEIGHT, DTAU)

            cases.append(
                {
                    "landscape": name,
                    "L": n_sites,
                    "mu": MU,
                    "max_weight": MAX_WEIGHT,
                    "n_generators": evolution.n_generators,
                    "cosine": scores["cosine"],
                    "tv": scores["tv"],
                    "tau_used": evolution.tau_used,
                    "converged": evolution.converged,
                    "max_gram_condition": evolution.max_gram_condition,
                    "descent": descent,
                    "parity": parity,
                }
            )

    # Diagnostic: accuracy against how much generator support the method is allowed. This is
    # Motta's cost curve, the counterpart to varQITE's ansatz-depth curve in G-R.6, and it is
    # where the two methods' opposite failure modes show up.
    support_scan = []
    fitness = nk_fitness(SUPPORT_SCAN["L"], SUPPORT_SCAN["K"], seed=0)
    matrix = np.asarray(diagonal_hamiltonian(fitness, MU).to_matrix()).real
    reference, _, _ = perron_vector(fitness, MU)
    for weight in SUPPORT_SCAN["weights"]:
        evolution = evolve(
            matrix,
            n_sites=SUPPORT_SCAN["L"],
            tau=TAU_CAP,
            dtau=DTAU,
            max_weight=weight,
            tolerance=TOLERANCE,
        )
        support_scan.append(
            {
                "max_weight": weight,
                "n_generators": len(odd_y_strings(SUPPORT_SCAN["L"], weight)),
                "cosine": score(evolution.probs, reference)["cosine"],
                "tau_used": evolution.tau_used,
                **_descent_report(evolution.energies),
            }
        )

    elapsed = time.monotonic() - started
    cosines = np.array([c["cosine"] for c in cases])
    worst = int(np.argmin(cosines))
    all_descend = all(c["descent"]["descends"] for c in cases)

    measured = {
        "n_configurations": len(cases),
        "min_cosine": float(cosines.min()),
        "n_below_threshold": int((cosines < COSINE_THRESHOLD).sum()),
        "max_tv": float(max(c["tv"] for c in cases)),
        "all_energies_descend": all_descend,
        "total_rises_beyond_tolerance": int(
            sum(c["descent"]["n_rises_beyond_tolerance"] for c in cases)
        ),
        "largest_energy_rise": float(max(c["descent"]["largest_rise"] for c in cases)),
        "all_converged": all(c["converged"] for c in cases),
        "max_tau_used": float(max(c["tau_used"] for c in cases)),
        "max_gram_condition": float(max(c["max_gram_condition"] for c in cases)),
        "parity_even_y_max_abs": float(max(c["parity"]["even_y_rhs_max_abs"] for c in cases)),
        "parity_odd_y_min_norm": float(min(c["parity"]["odd_y_rhs_norm"] for c in cases)),
        "support_scan": support_scan,
        "worst_case": {k: v for k, v in cases[worst].items() if k != "parity"},
        "seconds": round(elapsed, 2),
    }
    passed = bool(cosines.min() >= COSINE_THRESHOLD and all_descend)
    return passed, measured, cases


def main() -> int:
    passed, measured, cases = run()

    path = write_gate_record(
        gate="G-R.7",
        work_package="wp_r",
        threshold={
            "accuracy": {"statistic": "cosine against the reference", "value": COSINE_THRESHOLD},
            "descent": {
                "statistic": "no energy increase on any step",
                "value": ENERGY_RISE_TOLERANCE,
            },
            "registered_in": "docs/protocol.md section 3, configurations in revision 7",
        },
        measured=measured,
        passed=passed,
        cases=cases,
        notes=(
            "The generator basis is the odd-Y Pauli strings, because a real state needs a "
            "real orthogonal unitary and hence a real antisymmetric generator. The parity "
            "block records Motta's own right-hand side for both parities per configuration: "
            "the even-Y set contributes exactly zero, which is the mechanism of the failure "
            "the planning documents record for this method. The support scan is Motta's cost "
            "curve, the counterpart to varQITE's ansatz-depth curve in G-R.6."
        ),
    )

    print(f"G-R.7: {len(cases)} configurations in {measured['seconds']} s\n")
    header = (
        f"{'landscape':16s} {'L':>2s} {'gens':>5s} {'cosine':>11s} {'tv':>10s} "
        f"{'tau':>6s} {'conv':>5s} {'rises':>6s}"
    )
    print(header)
    print("-" * len(header))
    for case in cases:
        print(
            f"{case['landscape']:16s} {case['L']:>2d} {case['n_generators']:>5d} "
            f"{case['cosine']:>11.7f} {case['tv']:>10.2e} {case['tau_used']:>6.2f} "
            f"{str(case['converged']):>5s} "
            f"{case['descent']['n_rises_beyond_tolerance']:>6d}"
        )

    print("\naccuracy against generator support, nk L = 5, K = 2:")
    for row in measured["support_scan"]:
        print(
            f"  weight={row['max_weight']} generators={row['n_generators']:>4d} "
            f"cosine={row['cosine']:.7f} rises={row['n_rises_beyond_tolerance']}"
        )

    print(f"\n  min cosine            {measured['min_cosine']:.7f}  (threshold {COSINE_THRESHOLD})")
    print(
        f"  energy descends       {measured['all_energies_descend']}  "
        f"({measured['total_rises_beyond_tolerance']} rises beyond "
        f"{ENERGY_RISE_TOLERANCE:.0e})"
    )
    print(f"  all converged         {measured['all_converged']}")
    print(
        f"  parity: even-Y max    {measured['parity_even_y_max_abs']:.2e}  "
        f"odd-Y min norm {measured['parity_odd_y_min_norm']:.3e}"
    )
    print(f"  worst Gram condition  {measured['max_gram_condition']:.2e}")
    print(f"  record                {path.relative_to(path.parents[2])}")
    print(f"  {'PASS' if passed else 'FAIL'}")
    if not passed:
        print(f"  worst case: {measured['worst_case']}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
