"""G-R.1: the analytic oracle against brute-force exact diagonalisation.

This is the first gate, and it is the one the rest of the project leans on. Everything
downstream is validated against the oracle, so the oracle itself has to be validated against
something that shares none of its assumptions. Exact diagonalisation is that something: it
assembles the full 2^L generator and takes its Perron eigenvector, knowing nothing about
product states or Hamming classes.

Threshold and case set are registered in docs/protocol.md section 3 and revision 1, both committed
before this script was run.

    python experiments/wp_r_rebuild/g_r_1_oracle_vs_ed.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

from quasarstack.analytic.crow_kimura import (
    additive_mean_fitness,
    additive_quasispecies,
    class_quasispecies,
)
from quasarstack.analytic.exact_diag import perron_vector
from quasarstack.classical.landscapes import (
    additive_fitness,
    class_fitness,
    single_peak_classes,
    uniform_additive_classes,
)
from quasarstack.io.store import write_gate_record

# Registered in docs/protocol.md section 3.
THRESHOLD = 1e-9

# Registered in docs/protocol.md revision 1.
SIZES = [2, 3, 4, 5, 6, 7, 8, 9, 10]
MUS = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]
ADDITIVE_SEEDS = list(range(10))
UNIFORM_A = [0.25, 0.50, 1.00, 2.00]
HEIGHTS = [1.0, 2.0, 5.0]


def _max_abs(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.max(np.abs(x - y)))


def run() -> tuple[bool, dict, list[dict]]:
    cases: list[dict] = []
    started = time.monotonic()

    for n_sites in SIZES:
        for mu in MUS:
            # Family 1: additive with random coefficients. Closed-form product route.
            for seed in ADDITIVE_SEEDS:
                rng = np.random.default_rng(seed)
                a = rng.uniform(0.25, 2.00, size=n_sites)
                oracle = additive_quasispecies(a, mu)
                reference, ref_lambda, gap = perron_vector(additive_fitness(a), mu)
                cases.append(
                    {
                        "family": "additive_random",
                        "L": n_sites,
                        "mu": mu,
                        "seed": seed,
                        "comparison": "closed_form_vs_ed",
                        "max_abs_error": _max_abs(oracle, reference),
                        "mean_fitness_error": abs(additive_mean_fitness(a, mu) - ref_lambda),
                        "spectral_gap": gap,
                    }
                )

            # Family 2: uniform additive. Reachable by both analytic routes, so this is
            # where the gate becomes a three-way agreement rather than a pairwise one.
            for a_value in UNIFORM_A:
                a = np.full(n_sites, a_value)
                closed_form = additive_quasispecies(a, mu)
                reference, ref_lambda, gap = perron_vector(additive_fitness(a), mu)
                by_class, _, class_lambda = class_quasispecies(
                    uniform_additive_classes(n_sites, a_value), mu
                )
                cases.append(
                    {
                        "family": "additive_uniform",
                        "L": n_sites,
                        "mu": mu,
                        "a": a_value,
                        "comparison": "closed_form_vs_ed",
                        "max_abs_error": _max_abs(closed_form, reference),
                        "mean_fitness_error": abs(additive_mean_fitness(a, mu) - ref_lambda),
                        "spectral_gap": gap,
                    }
                )
                cases.append(
                    {
                        "family": "additive_uniform",
                        "L": n_sites,
                        "mu": mu,
                        "a": a_value,
                        "comparison": "closed_form_vs_class_reduction",
                        "max_abs_error": _max_abs(closed_form, by_class),
                        "mean_fitness_error": abs(additive_mean_fitness(a, mu) - class_lambda),
                        "spectral_gap": gap,
                    }
                )

            # Families 3 to 5: permutation-symmetric landscapes, class-reduction route.
            d = np.arange(n_sites + 1, dtype=np.float64)
            for height in HEIGHTS:
                shapes = {
                    "single_peak": single_peak_classes(n_sites, height),
                    "class_quadratic": height * (1.0 - d / n_sites) ** 2,
                    "class_exponential": height * np.exp(-2.0 * d / n_sites),
                }
                for family, f_by_class in shapes.items():
                    oracle, _, oracle_lambda = class_quasispecies(f_by_class, mu)
                    reference, ref_lambda, gap = perron_vector(class_fitness(f_by_class), mu)
                    cases.append(
                        {
                            "family": family,
                            "L": n_sites,
                            "mu": mu,
                            "height": height,
                            "comparison": "class_reduction_vs_ed",
                            "max_abs_error": _max_abs(oracle, reference),
                            "mean_fitness_error": abs(oracle_lambda - ref_lambda),
                            "spectral_gap": gap,
                        }
                    )

    elapsed = time.monotonic() - started
    errors = np.array([c["max_abs_error"] for c in cases])
    fitness_errors = np.array([c["mean_fitness_error"] for c in cases])
    gaps = np.array([c["spectral_gap"] for c in cases])
    worst = int(np.argmax(errors))

    measured = {
        "max_abs_error": float(errors.max()),
        "median_abs_error": float(np.median(errors)),
        "max_mean_fitness_error": float(fitness_errors.max()),
        "min_spectral_gap": float(gaps.min()),
        "n_cases_over_threshold": int((errors >= THRESHOLD).sum()),
        "worst_case": cases[worst],
        "seconds": round(elapsed, 2),
    }
    return bool(errors.max() < THRESHOLD), measured, cases


def main() -> int:
    passed, measured, cases = run()

    path = write_gate_record(
        gate="G-R.1",
        work_package="wp_r",
        threshold={
            "statistic": "max absolute difference between oracle and exact-diagonalisation "
            "genotype distributions, over every case",
            "value": THRESHOLD,
            "registered_in": "docs/protocol.md section 3, case set in revision 1",
        },
        measured=measured,
        passed=passed,
        cases=cases,
        notes=(
            "The oracle never forms the 2^L generator and exact diagonalisation never uses "
            "the structure the oracle exploits, so agreement is evidence about both. The "
            "additive_uniform family is checked by both analytic routes as well as against "
            "exact diagonalisation. Spectral gaps are recorded for every case and no case "
            "was excluded on the basis of its gap."
        ),
    )

    print(f"G-R.1: {len(cases)} cases in {measured['seconds']} s")
    print(f"  max abs error        {measured['max_abs_error']:.3e}  (threshold {THRESHOLD:.0e})")
    print(f"  median abs error     {measured['median_abs_error']:.3e}")
    print(f"  max mean-fitness err {measured['max_mean_fitness_error']:.3e}")
    print(f"  min spectral gap     {measured['min_spectral_gap']:.3e}")
    print(f"  record               {path.relative_to(path.parents[2])}")
    print(f"  {'PASS' if passed else 'FAIL'}")

    if not passed:
        print(f"  worst case: {measured['worst_case']}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
