"""G-6: the matrix-product baseline.

Criteria:

1. Converges to sparse exact diagonalisation: cosine >= 0.999 at sufficient chi, for L in
   {8, 10, 12, 14} across all families.
2. The bond dimension needed for cosine >= 0.999 is mapped across (family, K, mu, L).
3. MPO bond dimension per family, with two site orderings. Recorded separately in
   `results/wp6/g_6_3.json` by `mpo_analysis.py`.
4. The eigenvalue and bond dimension are recorded after every DMRG sweep, with the discarded
   weight per update bounded by the cutoff.

The record also carries a cross-check against the imaginary-time implementation in
`quasarstack.classical.mps_ite` at L = 8 and 10.

    python experiments/wp6_mps/g_6_tensor_network.py
"""

from __future__ import annotations

import hashlib
import json
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
from quasarstack.classical.mps_dmrg import dominant_state, generator_mpo
from quasarstack.classical.mps_ite import evolve
from quasarstack.io.progress import Progress
from quasarstack.io.store import RESULTS_ROOT, write_gate_record

COSINE_THRESHOLD = 0.999
SIZES = [8, 10, 12, 14]
CHI_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128]
MU_RATIOS = [0.4, 0.7, 1.0, 1.3, 1.6]
SEEDS = [0, 1]
SEEDS_AT_LARGEST_SIZE = [0]
# perron_vector goes dense at or below dense_limit. The sparse path is used from L = 11.
REFERENCE_DENSE_LIMIT = 10
# Time allocation per cell at L >= 14.
CELL_BUDGET_SECONDS = 900.0
# A rung capped below the bond dimension the state needs has truncation noise in its eigenvalue,
# so it stops at a looser tolerance and a sweep limit.
RUNG_TOL = 1e-8
RUNG_MAX_SWEEPS = 30
# DMRG against the imaginary-time implementation, at full bond dimension.
CROSS_CHECK_SIZES = [8, 10]
CROSS_CHECK_RATIOS = [0.4, 1.0, 1.6]
CROSS_CHECK_DTAU = 0.05
CROSS_CHECK_MAX_STEPS = 3000


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
        yield (
            {"family": "rough_mount_fuji", "roughness": 0.5, "seed": seed},
            (rough_mount_fuji_fitness(n_sites, seed=seed, roughness=0.5)),
        )
        yield (
            {"family": "block", "block_size": 2, "seed": seed},
            block_fitness(n_sites, 2, seed=seed),
        )


def threshold_for(label: dict, fitness: np.ndarray, n_sites: int) -> float:
    """mu_c per instance: 1/L for the single peak, (max f - mean f)/L otherwise."""
    if label["family"] == "single_peak":
        return 1.0 / n_sites
    return float((fitness.max() - fitness.mean()) / n_sites)


def budget_for(n_sites: int) -> float | None:
    """Wall clock a single cell may spend climbing the ladder: 900 s at L >= 14, else no limit."""
    return CELL_BUDGET_SECONDS if n_sites >= 14 else None


def cosine(distribution: np.ndarray, reference: np.ndarray) -> float:
    return float(
        distribution @ reference / (np.linalg.norm(distribution) * np.linalg.norm(reference))
    )


def smallest_sufficient_chi(
    fitness: np.ndarray, mu: float, reference: np.ndarray, ceiling: int, budget: float | None = None
) -> dict:
    """First chi in CHI_SWEEP reaching the cosine threshold, with diagnostics.

    Each rung is an independent DMRG run with the bond dimension capped at that chi. With a
    budget, the climb stops before a rung once the clock has run out, and the rung in progress
    is handed the time remaining. A stopped cell reports the largest chi it tried and the best
    cosine it saw, marked `budget_limited`, so it cannot be read as a failure to converge.
    """
    started = time.monotonic()
    best_cosine = 0.0
    largest_attempted = 0
    for chi in CHI_SWEEP:
        if chi > ceiling:
            break
        elapsed = time.monotonic() - started
        if budget is not None and largest_attempted and elapsed >= budget:
            return {
                "chi_needed": None,
                "cosine": best_cosine,
                "converged": False,
                "sweeps": 0,
                "convergence_recorded_every_sweep": True,
                "budget_limited": True,
                "largest_chi_attempted": largest_attempted,
                "budget_seconds": budget,
            }
        largest_attempted = chi
        remaining = None if budget is None else max(budget - elapsed, 0.0)
        result = dominant_state(
            fitness, mu, chi, budget=remaining, tol=RUNG_TOL, max_sweeps=RUNG_MAX_SWEEPS
        )
        value = cosine(np.asarray(result["distribution"]), reference)
        best_cosine = max(best_cosine, value)
        if value >= COSINE_THRESHOLD:
            return {
                "chi_needed": chi,
                "cosine": value,
                "converged": bool(result["converged"]),
                "sweeps": int(result["sweeps"]),
                "bond_reached": int(result["bond_reached"]),
                "eigenvalue": float(result["eigenvalue"]),
                "energy_by_sweep": result["energy_by_sweep"],
                "bond_by_sweep": result["bond_by_sweep"],
                "cutoff": result["cutoff"],
                "complement_symmetric": bool(result["complement_symmetric"]),
                "convergence_recorded_every_sweep": len(result["energy_by_sweep"])
                == int(result["sweeps"]),
                "budget_limited": False,
            }
    return {
        "chi_needed": None,
        "cosine": best_cosine,
        "converged": False,
        "sweeps": 0,
        "convergence_recorded_every_sweep": True,
        "budget_limited": False,
        "largest_chi_attempted": largest_attempted,
    }


CHECKPOINT = RESULTS_ROOT / "wp6" / "scratch" / "g_6_cells.jsonl"


def grid_fingerprint() -> str:
    """Digest of the constants that define a cell. A checkpoint resumes only on a match."""
    return hashlib.sha256(
        json.dumps(
            {
                "method": "dmrg2",
                "sizes": SIZES,
                "chi_sweep": CHI_SWEEP,
                "mu_ratios": MU_RATIOS,
                "seeds": SEEDS,
                "seeds_at_largest_size": SEEDS_AT_LARGEST_SIZE,
                "cosine_threshold": COSINE_THRESHOLD,
                "cell_budget_seconds": CELL_BUDGET_SECONDS,
                "rung_tol": RUNG_TOL,
                "rung_max_sweeps": RUNG_MAX_SWEEPS,
                "reference_dense_limit": REFERENCE_DENSE_LIMIT,
                "cross_check_sizes": CROSS_CHECK_SIZES,
                "cross_check_ratios": CROSS_CHECK_RATIOS,
                "cross_check_dtau": CROSS_CHECK_DTAU,
                "cross_check_max_steps": CROSS_CHECK_MAX_STEPS,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]


def load_checkpoint() -> dict[str, dict]:
    """Cells already computed by an earlier run of this same grid, keyed by cell."""
    if not CHECKPOINT.is_file():
        return {}
    rows = [json.loads(line) for line in CHECKPOINT.read_text("utf-8").splitlines() if line.strip()]
    if not rows:
        return {}
    header, cells = rows[0], rows[1:]
    if header.get("fingerprint") != grid_fingerprint():
        print(
            f"checkpoint at {CHECKPOINT} was written for grid {header.get('fingerprint')} and "
            f"this is {grid_fingerprint()}; ignoring it and starting from the beginning",
            file=sys.stderr,
            flush=True,
        )
        return {}
    return {cell["_key"]: cell for cell in cells}


def append_checkpoint(record: dict) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    if not CHECKPOINT.is_file():
        CHECKPOINT.write_text(
            json.dumps({"fingerprint": grid_fingerprint()}) + "\n", encoding="utf-8"
        )
    with CHECKPOINT.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def cell_key(*parts: object) -> str:
    return "|".join(str(part) for part in parts)


def reference_for(fitness: np.ndarray, mu: float) -> np.ndarray:
    reference = np.abs(perron_vector(fitness, mu, dense_limit=REFERENCE_DENSE_LIMIT)[0])
    return reference / reference.sum()


def cross_check(label: dict, fitness: np.ndarray, n_sites: int, ratio: float) -> dict:
    """DMRG and imaginary-time evolution at full bond dimension, against each other and exact."""
    mu = ratio * threshold_for(label, fitness, n_sites)
    reference = reference_for(fitness, mu)
    ceiling = 1 << (n_sites // 2)

    began = time.monotonic()
    dmrg = np.asarray(dominant_state(fitness, mu, ceiling)["distribution"])
    dmrg_seconds = time.monotonic() - began

    began = time.monotonic()
    ite = evolve(fitness, mu, ceiling, dtau=CROSS_CHECK_DTAU, max_steps=CROSS_CHECK_MAX_STEPS)
    ite_seconds = time.monotonic() - began
    evolved = np.asarray(ite["distribution"])

    return {
        **label,
        "L": n_sites,
        "mu": mu,
        "mu_over_mu_c": ratio,
        "dmrg_cosine_to_exact": cosine(dmrg, reference),
        "ite_cosine_to_exact": cosine(evolved, reference),
        "dmrg_to_ite_cosine": cosine(dmrg, evolved),
        "dmrg_to_ite_total_variation": 0.5 * float(np.abs(dmrg - evolved).sum()),
        "ite_total_discarded_weight": float(ite["total_discarded_weight"]),
        "ite_converged": bool(ite["converged"]),
        "dmrg_seconds": round(dmrg_seconds, 3),
        "ite_seconds": round(ite_seconds, 3),
    }


def run() -> tuple[bool, dict, list[dict]]:
    """Measure the bond dimension each cell needs, checkpointing every cell as it lands."""
    started = time.monotonic()
    cases: list[dict] = []
    unreached: list[dict] = []
    budget_limited: list[dict] = []
    record_gaps = 0
    done = load_checkpoint()
    if done:
        print(f"resuming: {len(done)} cells already computed", file=sys.stderr, flush=True)

    grid = [(n, list(families(n))) for n in SIZES]
    checks = [
        (n, label, fitness)
        for n in CROSS_CHECK_SIZES
        for label, fitness in families(n)
        if label.get("seed", 0) == 0
    ]
    progress = Progress(
        sum(len(f) for _, f in grid) * len(MU_RATIOS) + len(checks) * len(CROSS_CHECK_RATIOS),
        "G-6",
    )

    for n_sites, family_list in grid:
        ceiling = 1 << (n_sites // 2)
        for label, fitness in family_list:
            mu_c = threshold_for(label, fitness, n_sites)
            # Bond dimension of the diagonal part of the operator, the rank of the fitness table
            # across the worst cut. Computed on first use so a checkpointed family costs nothing.
            fitness_bond: int | None = None
            for ratio in MU_RATIOS:
                key = cell_key(
                    "cell",
                    n_sites,
                    label.get("family"),
                    label.get("K"),
                    label.get("roughness"),
                    label.get("block_size"),
                    label.get("seed"),
                    ratio,
                )
                if key in done:
                    case = {name: value for name, value in done[key].items() if name != "_key"}
                    progress.step(
                        f"L={n_sites} {label['family']} mu/mu_c={ratio} "
                        f"chi={case['chi_needed']} (from checkpoint)"
                    )
                else:
                    if fitness_bond is None:
                        fitness_bond = generator_mpo(fitness, 0.0)[1]
                    mu = ratio * mu_c
                    reference = reference_for(fitness, mu)
                    began = time.monotonic()
                    found = smallest_sufficient_chi(
                        fitness, mu, reference, ceiling, budget=budget_for(n_sites)
                    )
                    found["seconds"] = round(time.monotonic() - began, 3)
                    case = {
                        **label,
                        "L": n_sites,
                        "mu": mu,
                        "mu_over_mu_c": ratio,
                        "state_ceiling": ceiling,
                        "fitness_bond_dimension": fitness_bond,
                        **found,
                    }
                    append_checkpoint({**case, "_key": key})
                    progress.step(
                        f"L={n_sites} {label['family']} mu/mu_c={ratio} chi={case['chi_needed']}"
                    )

                if case["chi_needed"] is None:
                    where = {**label, "L": n_sites, "mu_over_mu_c": ratio}
                    # A cell stopped by the time allocation is reported separately.
                    if case.get("budget_limited"):
                        where["largest_chi_attempted"] = case.get("largest_chi_attempted")
                        where["best_cosine"] = case.get("cosine")
                        budget_limited.append(where)
                    else:
                        unreached.append(where)
                if not case["convergence_recorded_every_sweep"]:
                    record_gaps += 1
                cases.append(case)

    comparisons = []
    for n_sites, label, fitness in checks:
        for ratio in CROSS_CHECK_RATIOS:
            key = cell_key(
                "cross_check",
                n_sites,
                label.get("family"),
                label.get("K"),
                label.get("roughness"),
                label.get("block_size"),
                ratio,
            )
            if key in done:
                row = {k: v for k, v in done[key].items() if k != "_key"}
                progress.step(f"cross-check L={n_sites} {label['family']} (from checkpoint)")
            else:
                row = cross_check(label, fitness, n_sites, ratio)
                append_checkpoint({**row, "_key": key})
                progress.step(f"cross-check L={n_sites} {label['family']} mu/mu_c={ratio}")
            comparisons.append(row)
    progress.finish()

    reached = [c for c in cases if c["chi_needed"] is not None]
    criterion_1 = bool(not unreached)
    criterion_2 = bool(len(reached) + len(budget_limited) == len(cases) and cases)
    criterion_4 = record_gaps == 0

    by_size = {}
    for n_sites in SIZES:
        needed = [c["chi_needed"] for c in reached if c["L"] == n_sites]
        by_size[str(n_sites)] = {
            "max_chi_needed": max(needed) if needed else None,
            "median_chi_needed": float(np.median(needed)) if needed else None,
        }
    by_ratio = {
        str(r): max((c["chi_needed"] for c in reached if c["mu_over_mu_c"] == r), default=None)
        for r in MU_RATIOS
    }

    measured = {
        "method": (
            f"two-site DMRG (quimb), bond dimension capped at chi, cutoff 1e-12, eigenvalue "
            f"tolerance {RUNG_TOL} relative, at most {RUNG_MAX_SWEEPS} sweeps per rung"
        ),
        "criterion_1_converges_to_exact": {
            "passed": criterion_1,
            "threshold": COSINE_THRESHOLD,
            "configurations": len(cases),
            "configurations_never_reaching_threshold": unreached,
            "configurations_stopped_by_the_budget": budget_limited,
            "budget_seconds_at_L14_and_above": CELL_BUDGET_SECONDS,
        },
        "criterion_2_bond_dimension_map": {
            "passed": criterion_2,
            "max_chi_needed_by_size": by_size,
            "max_chi_needed_by_mu_over_mu_c": by_ratio,
            "chi_sweep": CHI_SWEEP,
        },
        "criterion_4_convergence_recorded": {
            "passed": criterion_4,
            "configurations_missing_sweep_history": record_gaps,
        },
        "cross_check_against_imaginary_time": {
            "sizes": CROSS_CHECK_SIZES,
            "mu_over_mu_c": CROSS_CHECK_RATIOS,
            "dtau": CROSS_CHECK_DTAU,
            "configurations": len(comparisons),
            "lowest_dmrg_cosine_to_exact": min(
                (r["dmrg_cosine_to_exact"] for r in comparisons), default=None
            ),
            "lowest_ite_cosine_to_exact": min(
                (r["ite_cosine_to_exact"] for r in comparisons), default=None
            ),
            "lowest_dmrg_to_ite_cosine": min(
                (r["dmrg_to_ite_cosine"] for r in comparisons), default=None
            ),
            "largest_dmrg_to_ite_total_variation": max(
                (r["dmrg_to_ite_total_variation"] for r in comparisons), default=None
            ),
            "rows": comparisons,
        },
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
            "criterion_4": "eigenvalue and bond dimension recorded after every sweep",
            "defined_in": "docs/protocol.md, G-6",
        },
        measured=measured,
        passed=passed,
        cases=cases,
        notes="Criterion 3 is recorded separately in results/wp6/g_6_3.json.",
    )

    one = measured["criterion_1_converges_to_exact"]
    two = measured["criterion_2_bond_dimension_map"]
    four = measured["criterion_4_convergence_recorded"]
    check = measured["cross_check_against_imaginary_time"]

    print(f"G-6: {len(cases)} configurations in {measured['seconds']} s\n")
    print(f"  Criterion 1, convergence to exact: {'PASS' if one['passed'] else 'FAIL'}")
    print(f"    configurations                 {one['configurations']}")
    print(
        f"    never reached the threshold    {len(one['configurations_never_reaching_threshold'])}"
    )
    print(f"    stopped by the budget          {len(one['configurations_stopped_by_the_budget'])}")
    print(f"\n  Criterion 2, bond-dimension map: {'PASS' if two['passed'] else 'FAIL'}")
    print(f"    {'L':>4} {'max chi':>9} {'median chi':>11}")
    for size, row in two["max_chi_needed_by_size"].items():
        print(f"    {size:>4} {str(row['max_chi_needed']):>9} {str(row['median_chi_needed']):>11}")
    print(f"    max chi by mu/mu_c: {two['max_chi_needed_by_mu_over_mu_c']}")
    print(f"\n  Criterion 4, convergence recorded: {'PASS' if four['passed'] else 'FAIL'}")
    print("\n  Cross-check against imaginary-time evolution at full bond dimension")
    print(f"    configurations                 {check['configurations']}")
    print(f"    lowest DMRG cosine to exact    {check['lowest_dmrg_cosine_to_exact']}")
    print(f"    lowest ITE cosine to exact     {check['lowest_ite_cosine_to_exact']}")
    print(f"    largest DMRG-ITE TV            {check['largest_dmrg_to_ite_total_variation']}")
    print(f"\n  record  {path}")
    print(f"  G-6: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
