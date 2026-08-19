"""G-5: Baseline B, the polynomial-time landscape class. WP5.

Criteria, registered in `docs/protocol.md` section 9 with the class predicate in revision 16:

1. Matches the analytic oracle to at most 1e-6 on every landscape in the declared class.
2. The applicability boundary is an explicit predicate in code, and the covered set of WP7
   grid cells is emitted as a machine-readable map before the sweep runs.

Refusal is part of the gate. A baseline that quietly solved out-of-class instances by brute
force would report itself as covering cells it does not cover, and the boundary map would
inherit that in the direction that flatters the quantum method.

    python experiments/wp5_exact_class/g_5_exact_class.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

from quasarstack.analytic.exact_diag import perron_vector
from quasarstack.classical.exact_class import applicability, coverage_map, solve
from quasarstack.classical.landscapes import (
    additive_fitness,
    block_fitness,
    class_fitness,
    house_of_cards_fitness,
    nk_fitness,
    pairwise_uniform_classes,
    rough_mount_fuji_fitness,
    single_peak_classes,
    spin_glass_fitness,
)
from quasarstack.io.store import write_gate_record

# Registered in docs/protocol.md section 9 and revision 16.
ORACLE_TOLERANCE = 1e-6
SIZES = [4, 6, 8, 10]
MUS = [0.05, 0.10, 0.20]
SEEDS = list(range(10))
PEAK_HEIGHTS = [1.0, 2.5]
EPISTASIS_B = [0.05, 0.1]
NK_K = [1, 2, 4]


def in_class(n_sites: int):
    for seed in SEEDS:
        rng = np.random.default_rng(50_000 + 100 * n_sites + seed)
        yield (
            {"family": "additive", "seed": seed},
            additive_fitness(rng.uniform(0.3, 1.5, size=n_sites)),
        )
    for height in PEAK_HEIGHTS:
        yield (
            {"family": "single_peak", "height": height},
            class_fitness(single_peak_classes(n_sites, height)),
        )
    for b in EPISTASIS_B:
        yield (
            {"family": "additive_pairwise", "b": b},
            class_fitness(pairwise_uniform_classes(n_sites, 1.0, b)),
        )


def out_of_class(n_sites: int):
    for seed in SEEDS:
        for k in NK_K:
            if k <= n_sites - 1:
                yield {"family": "nk", "K": k, "seed": seed}, nk_fitness(n_sites, k, seed=seed)
        yield {"family": "spin_glass", "seed": seed}, spin_glass_fitness(n_sites, seed=seed)
        yield {"family": "house_of_cards", "seed": seed}, house_of_cards_fitness(n_sites, seed=seed)
        yield (
            {"family": "rough_mount_fuji", "seed": seed},
            rough_mount_fuji_fitness(n_sites, seed=seed, roughness=0.5),
        )
        if n_sites >= 2:
            yield (
                {"family": "block", "block_size": 2, "seed": seed},
                block_fitness(n_sites, 2, seed=seed),
            )


def run() -> tuple[bool, dict, list[dict]]:
    """Classify by the predicate, then check the right thing for each verdict.

    An earlier version keyed the check on the family label, expecting every spin glass, NK
    and house-of-cards instance to be out of class. **That expectation is false at small L**
    and the gate failed on it: the spin glass at L = 4 has six couplings of `+/- 1`, so one
    draw in thirty-two gives them all the same sign, and `sum_{i<j} z_i z_j` then depends
    only on `sum_i z_i`, which is a function of Hamming weight. Seeds 4 and 7 produced
    exactly that, fitness `[2.449, 0, -0.816, 0, 2.449]` by Hamming class, with a symmetry
    residual of exactly zero.

    The baseline solved them, correctly, and the gate called that a misclassification. The
    predicate was right and the test was wrong: section 9's criterion says "every landscape
    in its declared applicability class", and the class is what the predicate decides rather
    than what the family name suggests. So each instance is classified first, and then
    accuracy is required where it applies and refusal where it does not.

    The refusal check is not weakened by this, only keyed correctly. A landscape the
    predicate rejects must still raise.
    """
    started = time.monotonic()
    cases: list[dict] = []
    worst = 0.0
    solved_out_of_class = 0
    refused_in_class = 0
    by_family: dict[str, dict[str, int]] = {}

    for n_sites in SIZES:
        for label, fitness in list(in_class(n_sites)) + list(out_of_class(n_sites)):
            verdict = applicability(fitness)
            family = label["family"]
            counts = by_family.setdefault(family, {"in_class": 0, "out_of_class": 0})
            counts["in_class" if verdict["applies"] else "out_of_class"] += 1

            if verdict["applies"]:
                for mu in MUS:
                    try:
                        computed = np.asarray(solve(fitness, mu)["distribution"])
                    except ValueError:
                        refused_in_class += 1
                        cases.append(
                            {
                                **label,
                                "L": n_sites,
                                "mu": mu,
                                "applies": True,
                                "refused_despite_applying": True,
                            }
                        )
                        continue
                    reference = np.abs(perron_vector(fitness, mu)[0])
                    reference = reference / reference.sum()
                    error = float(np.max(np.abs(computed - reference)))
                    worst = max(worst, error)
                    cases.append(
                        {
                            **label,
                            "L": n_sites,
                            "mu": mu,
                            "applies": True,
                            "class": verdict["class"],
                            "max_abs_error": error,
                        }
                    )
            else:
                refused = False
                try:
                    solve(fitness, MUS[0])
                except ValueError:
                    refused = True
                if not refused:
                    solved_out_of_class += 1
                cases.append(
                    {
                        **label,
                        "L": n_sites,
                        "applies": False,
                        "refused": refused,
                        "additive_residual": verdict["additive_residual"],
                        "symmetric_residual": verdict["symmetric_residual"],
                    }
                )

    survey_size = 8
    cells = [
        {
            "family": label.get("family"),
            "L": survey_size,
            **{k: v for k, v in label.items() if k != "family"},
            "fitness": fitness,
        }
        for label, fitness in list(in_class(survey_size)) + list(out_of_class(survey_size))
    ]
    coverage = coverage_map(cells)

    criterion_1 = bool(
        worst <= ORACLE_TOLERANCE and solved_out_of_class == 0 and refused_in_class == 0
    )
    criterion_2 = bool(coverage["n_cells"] > 0)

    measured = {
        "criterion_1_matches_the_oracle": {
            "passed": criterion_1,
            "worst_max_abs_error": worst,
            "tolerance": ORACLE_TOLERANCE,
            "in_class_instances_the_baseline_refused": refused_in_class,
            "out_of_class_instances_the_baseline_solved": solved_out_of_class,
        },
        "criterion_2_coverage_map": {
            "passed": criterion_2,
            "n_cells": coverage["n_cells"],
            "n_covered": coverage["n_covered"],
            "fraction_covered": coverage["fraction_covered"],
            "map": coverage["cells"],
        },
        # Which families land in the class, decided instance by instance. A family is not a
        # class: a small spin glass is sometimes permutation symmetric by chance.
        "instances_in_class_by_family": by_family,
        "attribution": (
            "Execution plan v4 names this baseline Dixit-Srivastava-Vishnoi after PRIOR_ART "
            "entry II.1, which is still flagged to-verify. This gate does not claim the class "
            "it implements is theirs. If their class is strictly larger, WP7 has cells it "
            "currently believes are classically hard, and the boundary map would be wrong in "
            "the direction that favours the quantum method."
        ),
        "seconds": round(time.monotonic() - started, 2),
    }
    return bool(criterion_1 and criterion_2), measured, cases


def main() -> int:
    passed, measured, cases = run()

    path = write_gate_record(
        gate="G-5",
        work_package="wp5",
        threshold={
            "criterion_1": f"max abs error <= {ORACLE_TOLERANCE} against the analytic oracle "
            f"on every in-class landscape, and refusal on every out-of-class one",
            "criterion_2": "applicability as an explicit predicate, covered set emitted "
            "before the sweep",
            "registered_in": "docs/protocol.md section 9, class predicate in revision 16",
        },
        measured=measured,
        passed=passed,
        cases=cases,
        notes=measured["attribution"],
    )

    one = measured["criterion_1_matches_the_oracle"]
    two = measured["criterion_2_coverage_map"]
    print(f"G-5: {len(cases)} cases in {measured['seconds']} s\n")
    print("  Criterion 1, agreement with the oracle inside the class")
    print(
        f"    worst max abs error        {one['worst_max_abs_error']:.3e}  "
        f"(tolerance {ORACLE_TOLERANCE})"
    )
    print(f"    in-class refused           {one['in_class_instances_the_baseline_refused']}")
    print(f"    out-of-class solved anyway {one['out_of_class_instances_the_baseline_solved']}")
    print(f"    {'PASS' if one['passed'] else 'FAIL'}\n")
    print("  Criterion 2, coverage map")
    print(f"    cells                      {two['n_cells']}")
    print(f"    covered                    {two['n_covered']} ({two['fraction_covered']:.1%})")
    print(f"    {'PASS' if two['passed'] else 'FAIL'}")
    print("\n  Instances in the class, by family")
    for family, counts in sorted(measured["instances_in_class_by_family"].items()):
        total = counts["in_class"] + counts["out_of_class"]
        print(f"    {family:22s} {counts['in_class']:>4} of {total:>4}")
    print(f"\n  record  {path}")
    print(f"  G-5: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
