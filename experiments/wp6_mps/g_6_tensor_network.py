"""G-6: the matrix-product baseline. WP6 criteria 1, 2 and 4.

Criteria, registered in `GATES.md` section 10 with the sweep in Amendment 19:

1. Converges to sparse exact diagonalisation where both run: cosine >= 0.999 at sufficient
   chi, for L in {8, 10, 12, 14} across all families.
2. The bond dimension needed to hold cosine >= 0.999 is mapped across (family, K, mu, L).
   Section 10 calls this map a primary deliverable.
3. MPO bond dimension per family, with two site orderings. Recorded separately in
   `results/wp6/g_6_3.json` by `mpo_analysis.py`, and its headline is folded in here.
4. Truncation error tracked and recorded at every step, not only at the end.

    python experiments/wp6_mps/g_6_tensor_network.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

from quasarstack.analytic.exact_diag import perron_vector
from quasarstack.classical.landscapes import (
    additive_fitness,
    block_fitness,
    class_fitness,
    house_of_cards_fitness,
    nk_fitness,
    rough_mount_fuji_fitness,
    single_peak_classes,
    spin_glass_fitness,
)
from quasarstack.classical.mps_ite import evolve, step_operator_bond_dimension
from quasarstack.io.store import write_gate_record

# Registered in GATES.md section 10 and Amendment 19.
COSINE_THRESHOLD = 0.999
SIZES = [8, 10, 12, 14]
CHI_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128]
DTAU = 0.05
DTAU_SWEEP = [0.1, 0.05, 0.02]
DTAU_SUBSET_SIZES = [8, 12]
MU_RATIOS = [0.4, 0.7, 1.0, 1.3, 1.6]
SEEDS = [0, 1]
# One seed at the largest size: see the second addendum to Amendment 19. The cost of a step
# scales with the operator's bond dimension, which saturates for the dense families.
SEEDS_AT_LARGEST_SIZE = [0]
# perron_vector goes dense at or below dense_limit, and a dense 4096 by 4096 solve in the
# single-threaded image costs 37 s against 0.2 s for the sparse path at L = 14. Forcing
# the sparse route at L = 12 makes the reference cheaper than the evolution it checks,
# which is the right way round. The two agree to machine precision, which is what G-R.1
# and tests/unit/test_numerics.py establish.
REFERENCE_DENSE_LIMIT = 10
MAX_STEPS = 3000


def families(n_sites: int):
    seeds = SEEDS_AT_LARGEST_SIZE if n_sites >= max(SIZES) else SEEDS
    rng = np.random.default_rng(9000 + n_sites)
    yield {"family": "additive"}, additive_fitness(rng.uniform(0.3, 1.5, size=n_sites))
    yield {"family": "single_peak"}, class_fitness(single_peak_classes(n_sites, 1.0))
    for k in (1, 2, 4):
        for seed in seeds:
            yield {"family": "nk", "K": k, "seed": seed}, nk_fitness(n_sites, k, seed=seed)
    for seed in seeds:
        yield {"family": "spin_glass", "seed": seed}, spin_glass_fitness(n_sites, seed=seed)
        yield {"family": "house_of_cards", "seed": seed}, house_of_cards_fitness(n_sites, seed=seed)
        yield {"family": "rough_mount_fuji", "roughness": 0.5, "seed": seed}, (
            rough_mount_fuji_fitness(n_sites, seed=seed, roughness=0.5)
        )
        yield {"family": "block", "block_size": 2, "seed": seed}, block_fitness(
            n_sites, 2, seed=seed
        )


def threshold_for(label: dict, fitness: np.ndarray, n_sites: int) -> float:
    """mu_c per instance, as section 11.1 and Amendment 12's addendum define it."""
    if label["family"] == "single_peak":
        return 1.0 / n_sites
    return float((fitness.max() - fitness.mean()) / n_sites)


def smallest_sufficient_chi(
    fitness: np.ndarray, mu: float, reference: np.ndarray, ceiling: int
) -> dict:
    """First chi in the registered sweep reaching the cosine threshold, and the diagnostics."""
    best_cosine = 0.0
    worst_truncation = 0.0
    for chi in CHI_SWEEP:
        if chi > ceiling:
            break
        result = evolve(fitness, mu, chi, dtau=DTAU, max_steps=MAX_STEPS)
        distribution = np.asarray(result["distribution"])
        cosine = float(
            distribution @ reference / (np.linalg.norm(distribution) * np.linalg.norm(reference))
        )
        best_cosine = max(best_cosine, cosine)
        worst_truncation = max(worst_truncation, float(result["max_discarded_weight_in_one_step"]))
        if cosine >= COSINE_THRESHOLD:
            return {
                "chi_needed": chi,
                "cosine": cosine,
                "converged": bool(result["converged"]),
                "steps": int(result["steps"]),
                "total_discarded_weight": float(result["total_discarded_weight"]),
                "max_discarded_weight_in_one_step": float(
                    result["max_discarded_weight_in_one_step"]
                ),
                "truncation_recorded_every_step": len(result["truncation_history"])
                == int(result["steps"]),
            }
    return {
        "chi_needed": None,
        "cosine": best_cosine,
        "converged": False,
        "steps": 0,
        "total_discarded_weight": 0.0,
        "max_discarded_weight_in_one_step": worst_truncation,
        "truncation_recorded_every_step": True,
    }


def run() -> tuple[bool, dict, list[dict]]:
    started = time.monotonic()
    cases: list[dict] = []
    unreached: list[dict] = []
    truncation_gaps = 0

    for n_sites in SIZES:
        ceiling = 1 << (n_sites // 2)
        for label, fitness in families(n_sites):
            mu_c = threshold_for(label, fitness, n_sites)
            operator = step_operator_bond_dimension(fitness, DTAU)
            for ratio in MU_RATIOS:
                mu = ratio * mu_c
                reference = np.abs(perron_vector(fitness, mu, dense_limit=REFERENCE_DENSE_LIMIT)[0])
                reference = reference / reference.sum()
                began = time.monotonic()
                found = smallest_sufficient_chi(fitness, mu, reference, ceiling)
                found["seconds"] = round(time.monotonic() - began, 3)
                if found["chi_needed"] is None:
                    unreached.append({**label, "L": n_sites, "mu_over_mu_c": ratio})
                if not found["truncation_recorded_every_step"]:
                    truncation_gaps += 1
                cases.append(
                    {
                        **label,
                        "L": n_sites,
                        "mu": mu,
                        "mu_over_mu_c": ratio,
                        "state_ceiling": ceiling,
                        **operator,
                        **found,
                    }
                )

    # The Trotter floor, on a fixed subset, so it is separable from truncation error.
    trotter = []
    for n_sites in DTAU_SUBSET_SIZES:
        for name, fitness in (
            ("single_peak", class_fitness(single_peak_classes(n_sites, 1.0))),
            ("nk_k2", nk_fitness(n_sites, 2, seed=0)),
        ):
            mu = 0.2
            reference = np.abs(perron_vector(fitness, mu, dense_limit=REFERENCE_DENSE_LIMIT)[0])
            reference = reference / reference.sum()
            for dtau in DTAU_SWEEP:
                result = evolve(fitness, mu, 1 << (n_sites // 2), dtau=dtau, max_steps=MAX_STEPS)
                distribution = np.asarray(result["distribution"])
                trotter.append(
                    {
                        "family": name,
                        "L": n_sites,
                        "dtau": dtau,
                        "total_variation": 0.5 * float(np.abs(distribution - reference).sum()),
                        "total_discarded_weight": float(result["total_discarded_weight"]),
                        "steps": int(result["steps"]),
                    }
                )

    reached = [c for c in cases if c["chi_needed"] is not None]
    criterion_1 = bool(not unreached)
    criterion_2 = bool(len(reached) == len(cases) and cases)
    criterion_4 = truncation_gaps == 0

    by_size = {}
    for n_sites in SIZES:
        needed = [c["chi_needed"] for c in reached if c["L"] == n_sites]
        by_size[str(n_sites)] = {
            "max_chi_needed": max(needed) if needed else None,
            "median_chi_needed": float(np.median(needed)) if needed else None,
        }

    # Does the requirement peak at the threshold, as a localisation transition would suggest?
    by_ratio = {
        str(r): max((c["chi_needed"] for c in reached if c["mu_over_mu_c"] == r), default=None)
        for r in MU_RATIOS
    }

    measured = {
        "criterion_1_converges_to_exact": {
            "passed": criterion_1,
            "threshold": COSINE_THRESHOLD,
            "configurations": len(cases),
            "configurations_never_reaching_threshold": unreached,
        },
        "criterion_2_bond_dimension_map": {
            "passed": criterion_2,
            "max_chi_needed_by_size": by_size,
            "max_chi_needed_by_mu_over_mu_c": by_ratio,
            "chi_sweep": CHI_SWEEP,
        },
        "criterion_4_truncation_tracked": {
            "passed": criterion_4,
            "configurations_missing_step_history": truncation_gaps,
            "worst_single_step_discarded_weight": max(
                (c["max_discarded_weight_in_one_step"] for c in cases), default=0.0
            ),
        },
        "trotter_floor": trotter,
        "criterion_3_reference": "recorded separately in results/wp6/g_6_3.json",
        "seconds": round(time.monotonic() - started, 2),
    }
    return bool(criterion_1 and criterion_2 and criterion_4), measured, cases


def main() -> int:
    passed, measured, cases = run()

    path = write_gate_record(
        gate="G-6",
        work_package="wp6",
        threshold={
            "criterion_1": f"cosine >= {COSINE_THRESHOLD} against sparse exact "
            f"diagonalisation at sufficient chi, L = {SIZES}",
            "criterion_2": "bond dimension needed, mapped across family, K, mu and L",
            "criterion_4": "truncation error recorded at every step",
            "registered_in": "GATES.md section 10, sweep in Amendment 19",
        },
        measured=measured,
        passed=passed,
        cases=cases,
        notes=(
            "The operator's bond dimension is a poor predictor of the state's: Rough Mount "
            "Fuji and house-of-cards saturate the operator ceiling while their states need "
            "single digits. Criterion 3 is recorded separately in results/wp6/g_6_3.json."
        ),
    )

    one = measured["criterion_1_converges_to_exact"]
    two = measured["criterion_2_bond_dimension_map"]
    four = measured["criterion_4_truncation_tracked"]

    print(f"G-6: {len(cases)} configurations in {measured['seconds']} s\n")
    print(f"  Criterion 1, convergence to exact: {'PASS' if one['passed'] else 'FAIL'}")
    print(f"    configurations                 {one['configurations']}")
    print(
        f"    never reached the threshold    {len(one['configurations_never_reaching_threshold'])}"
    )
    print(f"\n  Criterion 2, bond-dimension map: {'PASS' if two['passed'] else 'FAIL'}")
    print(f"    {'L':>4} {'max chi':>9} {'median chi':>11}")
    for size, row in two["max_chi_needed_by_size"].items():
        print(f"    {size:>4} {str(row['max_chi_needed']):>9} {str(row['median_chi_needed']):>11}")
    print(f"    max chi by mu/mu_c: {two['max_chi_needed_by_mu_over_mu_c']}")
    print(f"\n  Criterion 4, truncation tracked: {'PASS' if four['passed'] else 'FAIL'}")
    print(
        f"    worst single-step discarded weight "
        f"{four['worst_single_step_discarded_weight']:.3e}"
    )
    print("\n  Trotter floor at full chi (total variation, so truncation is not the cause)")
    for row in measured["trotter_floor"]:
        print(
            f"    {row['family']:12s} L={row['L']:>3} dtau={row['dtau']:<6} "
            f"TV={row['total_variation']:.3e}"
        )
    print(f"\n  record  {path}")
    print(f"  G-6: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
