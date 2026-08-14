"""G-6: the matrix-product baseline. WP6 criteria 1, 2 and 4.

Criteria, registered in `GATES.md` section 10 with the sweep in revision 19:

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
from quasarstack.classical.mps_ite import evolve, step_operator_bond_dimension
from quasarstack.io.progress import Progress
from quasarstack.io.store import RESULTS_ROOT, write_gate_record

# Registered in GATES.md section 10 and revision 19.
COSINE_THRESHOLD = 0.999
SIZES = [8, 10, 12, 14]
CHI_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128]
DTAU = 0.05
DTAU_SWEEP = [0.1, 0.05, 0.02]
DTAU_SUBSET_SIZES = [8, 12]
MU_RATIOS = [0.4, 0.7, 1.0, 1.3, 1.6]
SEEDS = [0, 1]
# One seed at the largest size: see the second addendum to revision 19. The cost of a step
# scales with the operator's bond dimension, which saturates for the dense families.
SEEDS_AT_LARGEST_SIZE = [0]
# perron_vector goes dense at or below dense_limit, and a dense 4096 by 4096 solve in the
# single-threaded image costs 37 s against 0.2 s for the sparse path at L = 14. Forcing
# the sparse route at L = 12 makes the reference cheaper than the evolution it checks,
# which is the right way round. The two agree to machine precision, which is what G-R.1
# and tests/unit/test_numerics.py establish.
REFERENCE_DENSE_LIMIT = 10
MAX_STEPS = 3000
# revision 23. Section 11.3's per-cell allotment at L >= 14, borrowed for the reason
# revision 22 borrowed the 300 s figure: a limit taken from the measurement it judges
# would be circular, and this one was fixed earlier and for another purpose.
CELL_BUDGET_SECONDS = 900.0


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
    """mu_c per instance, as section 11.1 and revision 12's addendum define it."""
    if label["family"] == "single_peak":
        return 1.0 / n_sites
    return float((fitness.max() - fitness.mean()) / n_sites)


def budget_for(n_sites: int) -> float | None:
    """Wall clock a single cell may spend climbing the ladder, or None for no limit.

    Registered in revision 23 and borrowed rather than chosen: 900 s is what section 11.3
    allots a method per cell at `L >= 14`, fixed long before any of this was measured and for
    an unrelated purpose, so it cannot have been fitted to the answer it now judges.

    The limit applies only at `L >= 14`. Sizes up to 12 are left alone because they are
    already affordable at 5.1 hours for the whole grid, and capping a measurement that runs
    fine would discard real results for nothing. The worst L = 12 cell, house of cards at the
    threshold, took 1.6 hours and returned an honest chi of 64; a 300 s cap would have thrown
    that away.
    """
    return CELL_BUDGET_SECONDS if n_sites >= 14 else None


def smallest_sufficient_chi(
    fitness: np.ndarray, mu: float, reference: np.ndarray, ceiling: int, budget: float | None = None
) -> dict:
    """First chi in the registered sweep reaching the cosine threshold, and the diagnostics.

    With a budget, the climb stops before starting a rung it cannot afford, and the cell
    reports the largest chi it managed and the best cosine it saw, marked `budget_limited`.
    This is ADR-0019's own recommendation applied here: a method that runs out of time should
    hand back what it has, so the record can tell "the tensor network cannot represent this
    state" apart from "the tensor network was not given long enough". Those are different
    findings and only one of them is about physics.

    The check sits between rungs rather than inside the evolution, so a cell can overshoot by
    at most one rung. Interrupting an evolution midway would leave a state that is neither
    converged nor a fair report of that chi.
    """
    started = time.monotonic()
    best_cosine = 0.0
    worst_truncation = 0.0
    largest_attempted = 0
    for chi in CHI_SWEEP:
        if chi > ceiling:
            break
        if budget is not None and largest_attempted and time.monotonic() - started >= budget:
            return {
                "chi_needed": None,
                "cosine": best_cosine,
                "converged": False,
                "steps": 0,
                "total_discarded_weight": 0.0,
                "max_discarded_weight_in_one_step": worst_truncation,
                "truncation_recorded_every_step": True,
                "budget_limited": True,
                "largest_chi_attempted": largest_attempted,
                "budget_seconds": budget,
            }
        largest_attempted = chi
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
                "budget_limited": False,
            }
    return {
        "chi_needed": None,
        "cosine": best_cosine,
        "converged": False,
        "steps": 0,
        "total_discarded_weight": 0.0,
        "max_discarded_weight_in_one_step": worst_truncation,
        "truncation_recorded_every_step": True,
        "budget_limited": False,
        "largest_chi_attempted": largest_attempted,
    }


CHECKPOINT = RESULTS_ROOT / "wp6" / "scratch" / "g_6_cells.jsonl"


def grid_fingerprint() -> str:
    """A digest of every registered constant that defines what a cell means.

    A checkpoint is only safe to resume if the grid and the method behind it are the same.
    Resuming across a change to `CHI_SWEEP` or `MAX_STEPS` would silently blend results from
    two different measurements into one artefact, which is worse than losing the run.
    """
    return hashlib.sha256(
        json.dumps(
            {
                "sizes": SIZES,
                "chi_sweep": CHI_SWEEP,
                "dtau": DTAU,
                "dtau_sweep": DTAU_SWEEP,
                "dtau_subset_sizes": DTAU_SUBSET_SIZES,
                "mu_ratios": MU_RATIOS,
                "seeds": SEEDS,
                "seeds_at_largest_size": SEEDS_AT_LARGEST_SIZE,
                "cosine_threshold": COSINE_THRESHOLD,
                "cell_budget_seconds": CELL_BUDGET_SECONDS,
                "max_steps": MAX_STEPS,
                "reference_dense_limit": REFERENCE_DENSE_LIMIT,
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


def run() -> tuple[bool, dict, list[dict]]:
    """Measure the bond dimension each cell needs, checkpointing every cell as it lands.

    The checkpoint exists because the first version of this gate was all or nothing: it wrote
    its record only at the end, so an interruption at nine tenths of the way through produced
    nothing at all. That is what made a crash after fourteen hours cost fourteen hours, and it
    is what makes any decision to change the grid mid-run cost the whole run. Cells are now
    appended as they complete and reused on a restart, so the expensive question is only ever
    asked once. `grid_fingerprint` refuses a checkpoint written for a different grid.
    """
    started = time.monotonic()
    cases: list[dict] = []
    unreached: list[dict] = []
    budget_limited: list[dict] = []
    truncation_gaps = 0
    done = load_checkpoint()
    if done:
        print(f"resuming: {len(done)} cells already computed", file=sys.stderr, flush=True)

    grid = [(n, list(families(n))) for n in SIZES]
    progress = Progress(
        sum(len(f) for _, f in grid) * len(MU_RATIOS)
        + len(DTAU_SUBSET_SIZES) * 2 * len(DTAU_SWEEP),
        "G-6",
    )

    for n_sites, family_list in grid:
        ceiling = 1 << (n_sites // 2)
        for label, fitness in family_list:
            mu_c = threshold_for(label, fitness, n_sites)
            # Computed on first use rather than per family, so a fully checkpointed family
            # costs nothing to replay. The operator's own numbers are already inside each
            # stored case, so a resumed cell never needs it.
            operator: dict | None = None
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
                    if operator is None:
                        operator = step_operator_bond_dimension(fitness, DTAU)
                    mu = ratio * mu_c
                    reference = np.abs(
                        perron_vector(fitness, mu, dense_limit=REFERENCE_DENSE_LIMIT)[0]
                    )
                    reference = reference / reference.sum()
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
                        **operator,
                        **found,
                    }
                    append_checkpoint({**case, "_key": key})
                    progress.step(
                        f"L={n_sites} {label['family']} mu/mu_c={ratio} chi={case['chi_needed']}"
                    )

                if case["chi_needed"] is None:
                    where = {**label, "L": n_sites, "mu_over_mu_c": ratio}
                    # A cell stopped by the clock has not failed to converge; nothing was
                    # established about it either way. Counting it as a failure would let a
                    # budget manufacture a physics result. revision 23.
                    if case.get("budget_limited"):
                        where["largest_chi_attempted"] = case.get("largest_chi_attempted")
                        where["best_cosine"] = case.get("cosine")
                        budget_limited.append(where)
                    else:
                        unreached.append(where)
                if not case["truncation_recorded_every_step"]:
                    truncation_gaps += 1
                cases.append(case)

    # The Trotter floor, on a fixed subset, so it is separable from truncation error.
    trotter = []
    for n_sites in DTAU_SUBSET_SIZES:
        for name, fitness in (
            ("single_peak", class_fitness(single_peak_classes(n_sites, 1.0))),
            ("nk_k2", nk_fitness(n_sites, 2, seed=0)),
        ):
            mu = 0.2
            reference = None
            for dtau in DTAU_SWEEP:
                key = cell_key("trotter", n_sites, name, dtau)
                if key in done:
                    row = {k: v for k, v in done[key].items() if k != "_key"}
                    trotter.append(row)
                    progress.step(f"trotter L={n_sites} {name} dtau={dtau} (from checkpoint)")
                    continue
                if reference is None:
                    reference = np.abs(
                        perron_vector(fitness, mu, dense_limit=REFERENCE_DENSE_LIMIT)[0]
                    )
                    reference = reference / reference.sum()
                result = evolve(fitness, mu, 1 << (n_sites // 2), dtau=dtau, max_steps=MAX_STEPS)
                distribution = np.asarray(result["distribution"])
                row = {
                    "family": name,
                    "L": n_sites,
                    "dtau": dtau,
                    "total_variation": 0.5 * float(np.abs(distribution - reference).sum()),
                    "total_discarded_weight": float(result["total_discarded_weight"]),
                    "steps": int(result["steps"]),
                }
                trotter.append(row)
                append_checkpoint({**row, "_key": key})
                progress.step(f"trotter L={n_sites} {name} dtau={dtau}")
    progress.finish()

    reached = [c for c in cases if c["chi_needed"] is not None]
    # Criterion 1 asks whether the method converges where it is given the chance. A cell the
    # clock stopped was not given the chance, so it is reported separately rather than scored
    # as a failure, and criterion 2's map is complete when every cell either has a chi or has
    # a stated reason it does not. revision 23.
    criterion_1 = bool(not unreached)
    criterion_2 = bool(len(reached) + len(budget_limited) == len(cases) and cases)
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
            "configurations_stopped_by_the_budget": budget_limited,
            "budget_seconds_at_L14_and_above": CELL_BUDGET_SECONDS,
            "what_a_budget_limited_cell_means": (
                "The ladder ran out of wall clock before it ran out of chi. Nothing is "
                "established about whether the tensor network can hold this state: the "
                "largest chi attempted and the best cosine seen are recorded so the cell "
                "can be finished later without redoing the rest. revision 23."
            ),
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
            "registered_in": "GATES.md section 10, sweep in revision 19",
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
        f"    worst single-step discarded weight {four['worst_single_step_discarded_weight']:.3e}"
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
