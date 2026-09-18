"""G-4: the Wright-Fisher finite-population baseline.

Criteria:

1. Reproduces the analytic single-peak quasispecies as population size grows: total variation
   < 0.02 at N = 1e6, L = 8.
2. Reaches total variation <= 0.02 at L = 8 within 300 s, the per-cell time allocation of the
   WP7 sweep.

The implementation runs in genotype-count space at `O(L 2^L)` per generation, independent of N.
Absolute throughput and the convergence in N are recorded, together with a time-step study.

    python experiments/wp4_wright_fisher/g_4_wright_fisher.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

from quasarstack.analytic.crow_kimura import single_peak_quasispecies
from quasarstack.classical.wright_fisher import sample_stationary, time_step_bias
from quasarstack.io.store import write_gate_record

TV_THRESHOLD = 0.02
N_SITES = 8
MU = 0.10
PEAK_HEIGHT = 1.0
POPULATIONS = [10**3, 10**4, 10**5, 10**6]
GENERATIONS = 4000
BURN_IN_FRACTION = 0.2
SEEDS = list(range(10))
DT = 0.01
TIME_STEPS = [0.04, 0.02, 0.01, 0.005]
# Per-cell per-method time allocation of the WP7 sweep at L <= 12.
WP7_BUDGET_SECONDS = 300.0


def run() -> tuple[bool, dict, list[dict]]:
    started = time.monotonic()
    cases: list[dict] = []

    reference, _, _ = single_peak_quasispecies(N_SITES, PEAK_HEIGHT, MU)
    fitness = np.zeros(1 << N_SITES)
    fitness[0] = PEAK_HEIGHT

    largest_tv = None
    for population in POPULATIONS:
        began = time.monotonic()
        result = sample_stationary(
            fitness,
            MU,
            population,
            GENERATIONS,
            SEEDS,
            dt=DT,
            burn_in_fraction=BURN_IN_FRACTION,
        )
        elapsed = time.monotonic() - began
        distribution = np.asarray(result["distribution"])
        total_variation = 0.5 * float(np.abs(distribution - reference).sum())
        if population == POPULATIONS[-1]:
            largest_tv = total_variation

        cases.append(
            {
                "criterion": 1,
                "population": population,
                "generations": GENERATIONS,
                "dt": DT,
                "total_variation": total_variation,
                "max_pairwise_tv_between_seeds": result["max_pairwise_tv_between_seeds"],
                "max_burn_in_drift": result["max_burn_in_drift"],
                "seconds": round(elapsed, 2),
                "generations_per_second": round(GENERATIONS * len(SEEDS) / elapsed, 1),
            }
        )

    # Convergence in N: one fitted slope, and the local slope between adjacent populations.
    populations = np.array([c["population"] for c in cases], dtype=float)
    variations = np.array([c["total_variation"] for c in cases])
    slope = float(np.polyfit(np.log(populations), np.log(variations), 1)[0])
    local_slopes = [
        {
            "from_population": int(populations[i]),
            "to_population": int(populations[i + 1]),
            "slope": float(
                np.log(variations[i + 1] / variations[i])
                / np.log(populations[i + 1] / populations[i])
            ),
        }
        for i in range(len(populations) - 1)
    ]

    bias_rows = time_step_bias(
        fitness, MU, reference, TIME_STEPS, 10**5, 2000, SEEDS[:3], scale_population=True
    )
    unscaled_rows = time_step_bias(
        fitness, MU, reference, TIME_STEPS, 10**5, 2000, SEEDS[:3], scale_population=False
    )
    for row in bias_rows:
        cases.append({"study": "time_step_population_scaled", **row})
    for row in unscaled_rows:
        cases.append({"study": "time_step_population_fixed", **row})

    criterion_1 = bool(largest_tv is not None and largest_tv < TV_THRESHOLD)

    # Criterion 2: the cheapest population that reaches the target accuracy, and whether it fits
    # the time allocation. Read from the criterion 1 runs.
    ladder = sorted(
        (c for c in cases if c.get("criterion") == 1), key=lambda c: float(c["population"])
    )
    sufficient = [c for c in ladder if float(c["total_variation"]) <= TV_THRESHOLD]
    cheapest = min(sufficient, key=lambda c: float(c["seconds"])) if sufficient else None
    criterion_2 = bool(cheapest is not None and float(cheapest["seconds"]) <= WP7_BUDGET_SECONDS)

    measured = {
        "criterion_1_accuracy": {
            "passed": criterion_1,
            "total_variation_at_largest_budget": largest_tv,
            "threshold": TV_THRESHOLD,
            "largest_population": POPULATIONS[-1],
            "convergence_exponent_in_N": slope,
            "expected_exponent": -0.5,
            "local_slopes_between_adjacent_populations": local_slopes,
            "exponent_note": (
                "The local slopes run from about -1.0 at the smallest populations, where the "
                "distribution is not yet resolved, to about -0.34 at the largest, where the "
                "remaining error is a finite-generation floor rather than sampling noise."
            ),
        },
        "criterion_2_time_to_accuracy": {
            "passed": criterion_2,
            "target_total_variation": TV_THRESHOLD,
            "budget_seconds": WP7_BUDGET_SECONDS,
            "cheapest_sufficient_population": cheapest["population"] if cheapest else None,
            "seconds_to_reach_target": cheapest["seconds"] if cheapest else None,
            "total_variation_reached": cheapest["total_variation"] if cheapest else None,
        },
        "throughput": {
            "seconds_per_1000_generations_by_population": {
                str(c["population"]): round(1000 * c["seconds"] / (GENERATIONS * len(SEEDS)), 4)
                for c in cases
                if c.get("criterion") == 1
            },
            "cost_model": "O(L * 2**L) per generation, independent of N",
        },
        "time_step_study": {
            "population_scaled_with_inverse_dt": bias_rows,
            "population_held_fixed": unscaled_rows,
            "note": "Genetic drift is 1/N per generation, so 1/(N dt) per unit simulated "
            "time. Shrinking dt at fixed N halves the discretisation bias and doubles the "
            "drift, so the fixed-population rows plateau while their equilibration drift "
            "climbs.",
        },
        "seconds": round(time.monotonic() - started, 2),
    }
    return bool(criterion_1 and criterion_2), measured, cases


def main() -> int:
    passed, measured, cases = run()

    path = write_gate_record(
        gate="G-4",
        work_package="wp4",
        threshold={
            "criterion_1": f"total variation < {TV_THRESHOLD} against the analytic "
            f"quasispecies at N = {POPULATIONS[-1]}, L = {N_SITES}",
            "criterion_2": f"reaches total variation <= {TV_THRESHOLD} at L = {N_SITES} "
            f"within {WP7_BUDGET_SECONDS:.0f} s",
            "defined_in": "docs/protocol.md, G-4",
        },
        measured=measured,
        passed=passed,
        cases=cases,
        notes=(
            "The implementation runs in genotype-count space at O(L 2^L) per generation, "
            "independent of N. Throughput is recorded in measured.throughput."
        ),
    )

    accuracy = measured["criterion_1_accuracy"]
    print(f"G-4: {len(cases)} cases in {measured['seconds']} s\n")
    print("  Criterion 1, accuracy against the analytic quasispecies")
    print(f"    {'N':>9} {'TV':>10} {'seed spread':>12} {'drift':>9} {'gen/s':>9}")
    for case in cases:
        if case.get("criterion") == 1:
            print(
                f"    {case['population']:>9} {case['total_variation']:>10.5f} "
                f"{case['max_pairwise_tv_between_seeds']:>12.5f} "
                f"{case['max_burn_in_drift']:>9.5f} "
                f"{case['generations_per_second']:>9.1f}"
            )
    print(
        f"    convergence exponent in N   {accuracy['convergence_exponent_in_N']:.3f} "
        f"(central limit {accuracy['expected_exponent']})"
    )
    for step in accuracy["local_slopes_between_adjacent_populations"]:
        print(
            f"      {step['from_population']:>9} to {step['to_population']:<9} "
            f"slope {step['slope']:>7.3f}"
        )
    print(
        f"    TV at N = {accuracy['largest_population']}          "
        f"{accuracy['total_variation_at_largest_budget']:.5f}  "
        f"(threshold {TV_THRESHOLD})"
    )
    print(f"    {'PASS' if accuracy['passed'] else 'FAIL'}\n")

    print("  Time-step study, same simulated time per row")
    for label, key in (
        ("population scaled as 1/dt", "population_scaled_with_inverse_dt"),
        ("population held fixed", "population_held_fixed"),
    ):
        print(f"    {label}")
        for row in measured["time_step_study"][key]:
            print(
                f"      dt={row['dt']:<6} N={row['population']:>8} "
                f"TV={row['total_variation']:.5f} drift={row['max_burn_in_drift']:.5f}"
            )

    accuracy_in_time = measured["criterion_2_time_to_accuracy"]
    print("\n  Criterion 2, time to accuracy")
    print(
        f"    cheapest N reaching TV <= {accuracy_in_time['target_total_variation']}   "
        f"{accuracy_in_time['cheapest_sufficient_population']}"
    )
    print(
        f"    wall clock                     "
        f"{accuracy_in_time['seconds_to_reach_target']} s  "
        f"(budget {accuracy_in_time['budget_seconds']:.0f} s)"
    )
    print(f"    {'PASS' if accuracy_in_time['passed'] else 'FAIL'}")

    print(f"\n  record  {path}")
    print(f"  G-4: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
