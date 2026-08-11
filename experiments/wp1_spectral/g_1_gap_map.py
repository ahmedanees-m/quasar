"""G-1: the spectral-gap map, and the three criteria that judge it. WP1 tasks T1.2 and T1.3.

The gap governs convergence for every eigenvector-extraction method, quantum or classical, so
this map is the object that says where a quantum method could possibly help. It is also the
input to the WP7 budget protocol (ADR-0013) and to the WP2 resource estimate.

The three criteria are registered in `GATES.md` section 5; Amendment 12 fixes what section 5
left ambiguous and discloses the exploratory scans that preceded it.

1. Every closed form that exists is reproduced to relative error < 1e-6.
2. The gap minimum lies within 5% of the analytic mu_c at L = 6, 8, 10, under **both**
   readings of "analytic mu_c": the asymptotic `height / L`, and the project's own
   susceptibility-peak locator computed from the exact class reduction.
3. Every operator-structure claim is derived in `docs/theory.md` and resolves to a test or
   an artefact. Asserted-but-underived claims fail.

Criterion 2 is expected to fail, and Amendment 12 registers that expectation rather than
adjusting the threshold. The two locators differ by 29%, 14% and 7% at L = 6, 8, 10 and
converge to grid resolution by L = 24, because they are different finite-size locators of a
crossover that only becomes sharp as L grows.

    python experiments/wp1_spectral/g_1_gap_map.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np

from quasarstack.analytic.crow_kimura import class_distribution
from quasarstack.classical.landscapes import additive_fitness, nk_fitness
from quasarstack.io.store import write_gate_record
from quasarstack.spectral.gap import (
    additive_gap,
    class_gap,
    class_gap_extended,
    eigenvector_condition_number,
    locate_gap_minimum,
    pure_mutation_gap,
    spectral_gap,
    symmetric_sector_holds_lambda2,
)
from quasarstack.spectral.order_parameter import locate_threshold, magnetisation_from_classes

ROOT = Path(__file__).resolve().parents[2]

# Registered in GATES.md section 5 and Amendment 12.
CLOSED_FORM_TOLERANCE = 1e-6
THRESHOLD_TOLERANCE = 0.05
CRITERION_2_SIZES = [6, 8, 10]
GRID_POINTS = 41
GRID_SPAN = (0.2, 2.0)
FINE_GRID_POINTS = 1500
PEAK_HEIGHTS = [1.0, 2.5]
SEEDS = list(range(10))
DENSE_SIZES = [4, 6, 8, 10, 12]
CLASS_SIZES = [4, 6, 8, 10, 12, 16, 24, 32, 48, 64]
NK_K = [0, 1, 2, 4]
EXTENDED_PRECISION_BELOW = 1e-9
EXTENDED_DPS = 60
CLOSING_SEARCH_ITERATIONS = 160


def single_peak_classes(n_sites: int, height: float) -> np.ndarray:
    classes = np.zeros(n_sites + 1)
    classes[0] = height
    return classes


def registered_grid(mu_c: float, points: int = GRID_POINTS) -> np.ndarray:
    return np.linspace(GRID_SPAN[0] * mu_c, GRID_SPAN[1] * mu_c, points)


def _trusted_class_gap(classes: np.ndarray, mu: float) -> tuple[float, bool]:
    """Class gap, recomputed at 60 digits when float64 cannot be trusted.

    Returns ``(gap, used_extended)``. Amendment 12: any gap below 1e-9 is recomputed, and
    the float64 value is discarded rather than averaged with it.
    """
    gap = class_gap(classes, mu)
    if gap < EXTENDED_PRECISION_BELOW:
        return float(class_gap_extended(classes, repr(float(mu)), dps=EXTENDED_DPS)), True
    return gap, False


# --------------------------------------------------------------------------------------
# Criterion 1: the closed forms


def criterion_one() -> tuple[bool, dict, list[dict]]:
    cases: list[dict] = []
    worst = 0.0

    # (a) Additive: Delta = 2 min_i sqrt(a_i^2 + mu^2), exactly, independent of L.
    for n_sites in DENSE_SIZES:
        for seed in SEEDS:
            rng = np.random.default_rng(1000 * n_sites + seed)
            a = rng.uniform(0.2, 2.0, size=n_sites)
            for mu in (0.05, 0.2, 0.5, 1.0):
                predicted = additive_gap(a, mu)
                measured = spectral_gap(additive_fitness(a), mu)
                relative = abs(measured - predicted) / abs(predicted)
                worst = max(worst, relative)
                cases.append(
                    {
                        "criterion": 1,
                        "closed_form": "additive",
                        "L": n_sites,
                        "seed": seed,
                        "mu": mu,
                        "predicted": predicted,
                        "measured": measured,
                        "relative_error": relative,
                    }
                )

    # (b) Zero fitness: the generator is the mutation operator and its gap is exactly 2 mu.
    for n_sites in CLASS_SIZES:
        for mu in (0.05, 0.2, 0.5):
            predicted = pure_mutation_gap(mu)
            measured = class_gap(np.zeros(n_sites + 1), mu)
            relative = abs(measured - predicted) / predicted
            worst = max(worst, relative)
            cases.append(
                {
                    "criterion": 1,
                    "closed_form": "pure_mutation",
                    "L": n_sites,
                    "mu": mu,
                    "predicted": predicted,
                    "measured": measured,
                    "relative_error": relative,
                }
            )

    # (c) Single peak far above threshold saturates at 2 mu. A limit, not an identity, so
    #     it is checked only where the limit has been reached: mu L >> height.
    saturation = []
    for n_sites in (32, 48, 64):
        for mu in (0.3, 0.5):
            measured = class_gap(single_peak_classes(n_sites, 1.0), mu)
            relative = abs(measured - pure_mutation_gap(mu)) / pure_mutation_gap(mu)
            worst = max(worst, relative)
            saturation.append({"L": n_sites, "mu": mu, "relative_error": relative})
            cases.append(
                {
                    "criterion": 1,
                    "closed_form": "single_peak_saturation",
                    "L": n_sites,
                    "mu": mu,
                    "mu_times_L_over_height": mu * n_sites,
                    "predicted": pure_mutation_gap(mu),
                    "measured": measured,
                    "relative_error": relative,
                }
            )

    return (
        bool(worst < CLOSED_FORM_TOLERANCE),
        {
            "worst_relative_error": worst,
            "tolerance": CLOSED_FORM_TOLERANCE,
            "n_comparisons": len(cases),
            "saturation_checks": saturation,
        },
        cases,
    )


# --------------------------------------------------------------------------------------
# Criterion 2: does the gap minimum locate the threshold?


def criterion_two() -> tuple[bool, dict, list[dict]]:
    cases: list[dict] = []
    rows = []
    passes = True

    # The registered sizes decide the gate; the rest are reported so the convergence is
    # visible rather than asserted.
    for n_sites in sorted(set(CRITERION_2_SIZES) | {12, 16, 24, 32, 48, 64}):
        for height in PEAK_HEIGHTS:
            classes = single_peak_classes(n_sites, height)
            asymptotic = height / n_sites

            grid = registered_grid(asymptotic)
            gaps = [_trusted_class_gap(classes, mu)[0] for mu in grid]
            mu_gap_registered = float(grid[int(np.argmin(gaps))])

            fine = registered_grid(asymptotic, FINE_GRID_POINTS)
            fine_gaps = [_trusted_class_gap(classes, mu)[0] for mu in fine]
            mu_gap_fine = float(fine[int(np.argmin(fine_gaps))])

            magnetisations = np.array(
                [magnetisation_from_classes(class_distribution(classes, mu)[0]) for mu in fine]
            )
            located = locate_threshold(fine, magnetisations)
            mu_chi = located["mu_c"]

            deviation_a = abs(mu_gap_registered - asymptotic) / asymptotic
            deviation_b = abs(mu_gap_registered - mu_chi) / mu_chi
            within = bool(deviation_a <= THRESHOLD_TOLERANCE and deviation_b <= THRESHOLD_TOLERANCE)
            if n_sites in CRITERION_2_SIZES and not within:
                passes = False

            row = {
                "criterion": 2,
                "L": n_sites,
                "height": height,
                "decides_gate": n_sites in CRITERION_2_SIZES,
                "mu_min_gap_registered_grid": mu_gap_registered,
                "mu_min_gap_fine_grid": mu_gap_fine,
                "mu_c_reading_a_asymptotic": asymptotic,
                "mu_c_reading_b_susceptibility": mu_chi,
                "deviation_from_reading_a": deviation_a,
                "deviation_from_reading_b": deviation_b,
                "within_five_percent_of_both": within,
                "susceptibility_peak_is_interior": bool(located["peak_is_interior"]),
                "registered_grid_spacing_as_fraction_of_mu_c": float(
                    (grid[1] - grid[0]) / asymptotic
                ),
                "mu_star_times_L_over_height": mu_gap_fine * n_sites / height,
            }
            rows.append(row)
            cases.append(row)

    deciding = [r for r in rows if r["decides_gate"]]
    return (
        passes,
        {
            "tolerance": THRESHOLD_TOLERANCE,
            "sizes_that_decide": CRITERION_2_SIZES,
            "worst_deviation_reading_a": max(r["deviation_from_reading_a"] for r in deciding),
            "worst_deviation_reading_b": max(r["deviation_from_reading_b"] for r in deciding),
            "n_deciding_cases_within": sum(r["within_five_percent_of_both"] for r in deciding),
            "n_deciding_cases": len(deciding),
            "largest_L_outside_tolerance": max(
                (r["L"] for r in rows if not r["within_five_percent_of_both"]), default=0
            ),
        },
        cases,
    )


# --------------------------------------------------------------------------------------
# Criterion 3: is every structural claim derived and checked?


def criterion_three() -> tuple[bool, dict, list[dict]]:
    """Parse the claim index in `docs/theory.md` and resolve every reference.

    A derivation nobody checked is an assertion with extra steps, which is what this
    criterion exists to prevent. A row whose reference names a file that does not exist is
    a failure, not a warning.
    """
    theory = ROOT / "docs" / "theory.md"
    cases: list[dict] = []
    if not theory.is_file():
        return False, {"error": "docs/theory.md is missing"}, cases

    text = theory.read_text(encoding="utf-8")
    index = text.split("## 10. Claim-to-check index", 1)
    if len(index) != 2:
        return False, {"error": "docs/theory.md has no claim-to-check index"}, cases

    unresolved = []
    for line in index[1].splitlines():
        if not line.startswith("| S"):
            continue
        columns = [c.strip() for c in line.strip("|").split("|")]
        if len(columns) < 3:
            continue
        claim_id, claim, references = columns[0], columns[1], columns[2]
        targets = re.findall(r"`([^`]+)`", references)
        resolved = []
        for target in targets:
            candidates = [
                ROOT / target,
                ROOT / "tests" / "unit" / target,
                ROOT / "tests" / "integration" / target,
            ]
            hit = next((c for c in candidates if c.is_file()), None)
            resolved.append({"target": target, "found": hit is not None})
            if hit is None:
                unresolved.append(f"{claim_id}: {target}")
        # A claim carrying no reference at all is exactly the failure mode this criterion
        # is for: it reads as derived and is only asserted.
        if not targets:
            unresolved.append(f"{claim_id}: no reference given")
        cases.append(
            {
                "criterion": 3,
                "claim": claim_id,
                "statement": claim,
                "references": resolved,
                "resolved": bool(targets) and all(r["found"] for r in resolved),
            }
        )

    return (
        bool(cases and not unresolved),
        {
            "n_claims": len(cases),
            "n_resolved": sum(c["resolved"] for c in cases),
            "unresolved": unresolved,
        },
        cases,
    )


# --------------------------------------------------------------------------------------
# The gap map itself, which is the scientific output regardless of the gate verdict


def gap_map() -> tuple[dict, list[dict]]:
    cases: list[dict] = []

    for n_sites in CLASS_SIZES:
        for height in PEAK_HEIGHTS:
            classes = single_peak_classes(n_sites, height)
            mu_c = height / n_sites
            for mu in registered_grid(mu_c):
                gap, extended = _trusted_class_gap(classes, mu)
                cases.append(
                    {
                        "family": "single_peak",
                        "L": n_sites,
                        "height": height,
                        "mu": float(mu),
                        "mu_over_mu_c": float(mu / mu_c),
                        "gap": gap,
                        "eigenvector_condition": eigenvector_condition_number(gap),
                        "extended_precision": extended,
                        "route": "class_reduction",
                    }
                )

    for n_sites in DENSE_SIZES:
        for seed in SEEDS:
            rng = np.random.default_rng(1000 * n_sites + seed)
            a = rng.uniform(0.2, 2.0, size=n_sites)
            fitness = additive_fitness(a)
            mu_c = float((fitness.max() - fitness.mean()) / n_sites)
            for mu in registered_grid(mu_c):
                gap = spectral_gap(fitness, mu)
                cases.append(
                    {
                        "family": "additive",
                        "L": n_sites,
                        "seed": seed,
                        "mu": float(mu),
                        "mu_over_mu_c": float(mu / mu_c),
                        "gap": gap,
                        "closed_form": additive_gap(a, mu),
                        "eigenvector_condition": eigenvector_condition_number(gap),
                        "route": "sparse" if n_sites > 8 else "dense",
                    }
                )

    for n_sites in DENSE_SIZES:
        for k in NK_K:
            if k >= n_sites:
                continue
            for seed in SEEDS:
                fitness = nk_fitness(n_sites, k, seed=seed)
                mu_c = float((fitness.max() - fitness.mean()) / n_sites)
                for mu in registered_grid(mu_c):
                    gap = spectral_gap(fitness, mu)
                    cases.append(
                        {
                            "family": "nk",
                            "L": n_sites,
                            "K": k,
                            "seed": seed,
                            "mu": float(mu),
                            "mu_over_mu_c": float(mu / mu_c),
                            "gap": gap,
                            "eigenvector_condition": eigenvector_condition_number(gap),
                            "route": "sparse" if n_sites > 8 else "dense",
                        }
                    )

    # Does the cheap route see the true second eigenvalue? Measured, not assumed.
    sector = [
        symmetric_sector_holds_lambda2(single_peak_classes(n_sites, 1.0), mu)
        for n_sites in (4, 6, 8, 10)
        for mu in (0.05, 0.1, 0.2, 0.4)
    ]

    # How the minimum gap over the sweep closes with L: the conditioning statement WP1 task
    # T1.3 is about, and the input WP7's budget protocol needs.
    #
    # This deliberately does NOT read the minimum off the grid. The minimum is an avoided
    # crossing with a slope of order 30 in mu at L = 32, so a 1500-point grid whose spacing
    # is 1e-3 of mu_c lands 19 times above the true minimum there, and worse as L grows. The
    # grid locates mu* perfectly well, which is all criterion 2 asks of it; measuring the
    # depth needs the arbitrary-precision search.
    closing = []
    for n_sites in CLASS_SIZES:
        found = locate_gap_minimum(
            lambda size: single_peak_classes(size, 1.0),
            n_sites,
            repr(0.3 / n_sites),
            repr(3.0 / n_sites),
            dps=EXTENDED_DPS,
            iterations=CLOSING_SEARCH_ITERATIONS,
        )
        smallest = found["min_gap_float"]
        grid = registered_grid(1.0 / n_sites, FINE_GRID_POINTS)
        grid_minimum = float(
            min(_trusted_class_gap(single_peak_classes(n_sites, 1.0), mu)[0] for mu in grid)
        )
        closing.append(
            {
                "L": n_sites,
                "min_gap": smallest,
                "mu_star": found["mu_star"],
                "mu_star_times_L": found["mu_star_times_L"],
                "worst_eigenvector_condition": eigenvector_condition_number(smallest),
                # Reported so the size of the trap is on the record rather than in a comment.
                "min_gap_off_the_fine_grid": grid_minimum,
                "grid_overestimates_by": (
                    grid_minimum / smallest if smallest > 0 else float("inf")
                ),
            }
        )

    sizes = np.array([c["L"] for c in closing], dtype=float)
    logs = np.log([c["min_gap"] for c in closing])
    slope, intercept = np.polyfit(sizes, logs, 1)
    r_squared = float(np.corrcoef(sizes, logs)[0, 1] ** 2)

    return (
        {
            "n_cells": len(cases),
            "sector_check": sector,
            "sector_check_all_symmetric": all(s["lambda2_is_symmetric"] for s in sector),
            "gap_closing_at_threshold": closing,
            "closing_base": float(np.exp(slope)),
            "closing_prefactor": float(np.exp(intercept)),
            "closing_r_squared": r_squared,
        },
        cases,
    )


def run() -> tuple[bool, dict, list[dict]]:
    """Measure the three criteria and the gap map, separately from reporting them.

    Split out after G-5 passed its science and then died formatting the result. The replay
    in tests/regression/test_gate_reporting.py can only reach gates shaped this way, and this
    was one of the two it could not cover. The split moves no computation and changes no
    recorded key, so the artefact already committed stays readable and needs no rerun.
    """
    started = time.monotonic()

    one_ok, one, one_cases = criterion_one()
    two_ok, two, two_cases = criterion_two()
    three_ok, three, three_cases = criterion_three()
    map_summary, map_cases = gap_map()

    passed = bool(one_ok and two_ok and three_ok)
    measured = {
        "criterion_1_closed_forms": {"passed": one_ok, **one},
        "criterion_2_threshold_location": {"passed": two_ok, **two},
        "criterion_3_derivations": {"passed": three_ok, **three},
        "gap_map": map_summary,
        "seconds": round(time.monotonic() - started, 2),
    }
    return passed, measured, one_cases + two_cases + three_cases + map_cases


def main() -> int:
    passed, measured, cases = run()
    one = measured["criterion_1_closed_forms"]
    two = measured["criterion_2_threshold_location"]
    three = measured["criterion_3_derivations"]
    map_summary = measured["gap_map"]
    one_ok, two_ok, three_ok = one["passed"], two["passed"], three["passed"]

    path = write_gate_record(
        gate="G-1",
        work_package="wp1",
        threshold={
            "criterion_1": f"every closed form reproduced to relative error < "
            f"{CLOSED_FORM_TOLERANCE}",
            "criterion_2": f"gap minimum within {THRESHOLD_TOLERANCE:.0%} of the analytic "
            f"mu_c under both readings, at L = {CRITERION_2_SIZES}",
            "criterion_3": "every operator-structure claim derived in docs/theory.md and "
            "resolving to a test or artefact",
            "registered_in": "GATES.md section 5, readings and grid in Amendment 12",
        },
        measured=measured,
        passed=passed,
        cases=cases,
        notes=(
            "Criterion 2 was registered in Amendment 12 in the expectation that it fails, "
            "with the exploratory numbers disclosed there, rather than adjusted to fit. The "
            "gap minimum and the susceptibility peak are different finite-size locators of a "
            "crossover that only becomes sharp as L grows; they converge to the grid "
            "resolution by L = 24 and cannot be made to agree at L = 6 by any correct "
            "implementation. The physics the criterion was written to test, that the gap "
            "closes where the population delocalises, is supported by the closing fit."
        ),
    )

    print(f"G-1: {len(cases)} cases in {measured['seconds']} s\n")

    print("  Criterion 1, closed forms")
    print(
        f"    worst relative error   {one['worst_relative_error']:.3e}  "
        f"(tolerance {CLOSED_FORM_TOLERANCE})"
    )
    print(f"    comparisons            {one['n_comparisons']}")
    print(f"    {'PASS' if one_ok else 'FAIL'}\n")

    print("  Criterion 2, threshold location")
    print(f"    {'L':>4} {'height':>7} {'vs asymptotic':>14} {'vs chi peak':>12} {'within':>7}")
    # Picked out of cases by a key only criterion 2 records, so this works on a record read
    # back from disk as well as on one just computed.
    for row in [c for c in cases if "deviation_from_reading_a" in c]:
        mark = "  <-" if row["decides_gate"] else ""
        print(
            f"    {row['L']:>4} {row['height']:>7.1f} "
            f"{row['deviation_from_reading_a']:>13.2%} "
            f"{row['deviation_from_reading_b']:>12.2%} "
            f"{str(row['within_five_percent_of_both']):>7}{mark}"
        )
    print(f"    {'PASS' if two_ok else 'FAIL'}\n")

    print("  Criterion 3, derivations")
    print(f"    claims resolved        {three['n_resolved']} of {three['n_claims']}")
    if three["unresolved"]:
        for item in three["unresolved"]:
            print(f"      unresolved: {item}")
    print(f"    {'PASS' if three_ok else 'FAIL'}\n")

    print("  Gap map")
    print(f"    cells                  {map_summary['n_cells']}")
    print(
        f"    lambda2 in symmetric sector everywhere: {map_summary['sector_check_all_symmetric']}"
    )
    print(
        f"    gap at threshold ~ {map_summary['closing_prefactor']:.4f} x "
        f"{map_summary['closing_base']:.5f}^L, R2 = {map_summary['closing_r_squared']:.6f}"
    )
    print(f"    {'L':>4} {'min gap':>12} {'1/gap':>14} {'grid err':>9}")
    for row in map_summary["gap_closing_at_threshold"]:
        print(
            f"    {row['L']:>4} {row['min_gap']:>12.4e} "
            f"{row['worst_eigenvector_condition']:>14.4e} "
            f"{row['grid_overestimates_by']:>9.1f}x"
        )

    print(f"\n  record                   {path}")
    print(f"  G-1: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
