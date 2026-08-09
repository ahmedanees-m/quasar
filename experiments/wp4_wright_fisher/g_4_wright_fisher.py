"""G-4: the Wright-Fisher finite-population baseline. WP4.

Criteria, registered in `GATES.md` section 8 with configurations in Amendment 15:

1. Reproduces the analytic single-peak quasispecies as population size and sample budget
   grow: total variation < 0.02 at the largest declared budget, L = 8.
2. Throughput within 5x of the reference community implementation.

**Criterion 2 is reported as blocked, not as passed.** The implementation runs in
genotype-count space at `O(L 2^L)` per generation, independent of N, while a community
forward simulator is individual-based at `O(N L)`. At the top of the declared sweep the two
differ by about three orders of magnitude by construction, so the test would pass by a factor
of a thousand while establishing nothing. No reference implementation is present in the
pinned image either. ADR-0018 has the reasoning and the recommended replacement; this gate
records absolute throughput and the measured scaling so the comparison can be completed later
without rerunning anything.

    python experiments/wp4_wright_fisher/g_4_wright_fisher.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

from quasarstack.analytic.crow_kimura import single_peak_quasispecies
from quasarstack.classical.wright_fisher import sample_stationary, time_step_bias
from quasarstack.io.store import write_gate_record

# Registered in GATES.md section 8 and Amendment 15.
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

CRITERION_2_STATUS = (
    "BLOCKED, not passed. This implementation is O(L 2^L) per generation and independent of "
    "N; a community forward simulator is individual-based at O(N L). At N = 1e6 the two "
    "differ by about three orders of magnitude by construction, so a throughput-within-5x "
    "test would pass by a factor of a thousand and establish nothing about whether the "
    "baseline is well built. No reference implementation is present in the pinned image, and "
    "ADR-0006 forbids installing one outside Docker. See ADR-0018 for the recommended "
    "replacement: time-to-accuracy at matched total variation."
)


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

    # Is the convergence the 1/sqrt(N) it should be, or is it plateauing on something else?
    populations = np.array([c["population"] for c in cases], dtype=float)
    variations = np.array([c["total_variation"] for c in cases])
    slope = float(np.polyfit(np.log(populations), np.log(variations), 1)[0])

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

    measured = {
        "criterion_1_accuracy": {
            "passed": criterion_1,
            "total_variation_at_largest_budget": largest_tv,
            "threshold": TV_THRESHOLD,
            "largest_population": POPULATIONS[-1],
            "convergence_exponent_in_N": slope,
            "expected_exponent": -0.5,
        },
        "criterion_2_throughput": {
            "passed": None,
            "status": CRITERION_2_STATUS,
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
            "drift, which is why the fixed-population rows plateau while their equilibration "
            "drift climbs.",
        },
        "seconds": round(time.monotonic() - started, 2),
    }
    # The gate as a whole is not claimed as passed while criterion 2 is blocked.
    return False, measured, cases


def main() -> int:
    _, measured, cases = run()

    path = write_gate_record(
        gate="G-4",
        work_package="wp4",
        threshold={
            "criterion_1": f"total variation < {TV_THRESHOLD} against the analytic "
            f"quasispecies at N = {POPULATIONS[-1]}, L = {N_SITES}",
            "criterion_2": "throughput within 5x of the reference community implementation",
            "registered_in": "GATES.md section 8, configurations in Amendment 15",
        },
        measured=measured,
        passed=False,
        cases=cases,
        notes=(
            "Criterion 1 is met. Criterion 2 is blocked rather than passed, so the gate as a "
            "whole is not claimed. " + CRITERION_2_STATUS
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
        f"(expected {accuracy['expected_exponent']})"
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

    print("\n  Criterion 2, throughput: BLOCKED")
    print("    " + CRITERION_2_STATUS.replace(". ", ".\n    "))
    print(f"\n  record  {path}")
    print("  G-4: NOT CLAIMED (criterion 1 met, criterion 2 blocked)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
