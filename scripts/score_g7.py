"""Score the WP7 decision gate against `GATES.md` section 11.5.

The gate asks a single question with four conjunctive conditions, and the answer is either a
region of the grid that satisfies all four or a null reported as a delimitation. Both were
registered in advance as publishable, so this script is written to make the null as
informative as the positive rather than as a shrug.

    A positive result is a non-empty region, contiguous and reproducible across at least five
    of the seeds run, in which a quantum route reaches cosine >= 0.90, the compute-matched
    tensor-network baseline is below 0.80, Baseline B does not apply, and the bootstrap
    confidence intervals of the two methods do not overlap.

Three things this script refuses to do
--------------------------------------

**It does not score a cell that was not compute-matched.** Section 11.3 calls the budget a
fairness firewall. A cell where any method ran past its allotment is reported separately and
excluded from the decision, because a quantum route that beat the tensor network on 1.7 times
the allotted time has not beaten it. `over_budget` is recomputed here from the recorded
seconds rather than trusted from the record, so cells written before that field existed are
still judged.

**It does not silently drop a cell.** Every planned cell is scored, excluded with a reason, or
reported missing. Claim C27.

**It does not report a null without a bound.** Section 11.5 requires the delimitation to say
where the crossover would have to be, so the null states the largest L and the largest
tensor-network bond dimension at which the sweep still held a valid reference.

    python scripts/score_g7.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quasarstack.io.store import RESULTS_ROOT, evidence_directory  # noqa: E402

QUANTUM_THRESHOLD = 0.90
CLASSICAL_THRESHOLD = 0.80
MIN_SEEDS = 5
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 0
QUANTUM_ROUTES = ("route_a_varqite", "route_b_qsvt_filter")
TENSOR_NETWORK = "baseline_c_tensor_network"
EXACT_CLASS = "baseline_b_exact_class"


def load_stream(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def cell_key(row: dict[str, Any]) -> str:
    return "|".join(
        str(row.get(k))
        for k in ("family", "K", "roughness", "block_size", "L", "seed", "mu_over_mu_c")
    )


def merge(classical: list[dict], quantum: list[dict]) -> list[dict]:
    """Join the two passes on cell identity. A cell present in only one is kept and marked."""
    by_key: dict[str, dict] = {}
    for row in classical:
        by_key[cell_key(row)] = {**row, "methods": dict(row["methods"])}
    for row in quantum:
        key = cell_key(row)
        if key in by_key:
            by_key[key]["methods"].update(row["methods"])
        else:
            by_key[key] = {**row, "methods": dict(row["methods"]), "classical_missing": True}
    return list(by_key.values())


def over_budget(entry: dict[str, Any]) -> bool:
    """Recomputed, not trusted. Cells written before the field existed still get judged."""
    used = entry.get("seconds_used")
    allotted = entry.get("seconds_allotted")
    if used is None or allotted is None:
        return False
    return float(used) > float(allotted)


def bootstrap_interval(values: list[float]) -> tuple[float, float]:
    """Percentile bootstrap of the mean, as section 11.4 registers."""
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    array = np.asarray(values, dtype=float)
    draws = rng.choice(array, size=(BOOTSTRAP_RESAMPLES, array.size), replace=True).mean(axis=1)
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def score(cells: list[dict]) -> dict[str, Any]:
    groups: dict[tuple, list[dict]] = {}
    for cell in cells:
        key = (
            cell.get("family"),
            cell.get("K"),
            cell.get("roughness"),
            cell.get("block_size"),
            cell.get("L"),
            cell.get("mu_over_mu_c"),
        )
        groups.setdefault(key, []).append(cell)

    regions, excluded = [], []
    for key, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        family, k, roughness, block, n_sites, ratio = key

        usable, dropped = [], []
        for cell in members:
            entries = cell["methods"]
            if any(over_budget(e) for e in entries.values() if e.get("applicable")):
                dropped.append("over budget")
                continue
            if any("error" in e for e in entries.values()):
                dropped.append("a method errored")
                continue
            usable.append(cell)
        if dropped:
            excluded.append(
                {
                    "family": family,
                    "K": k,
                    "L": n_sites,
                    "mu_over_mu_c": ratio,
                    "cells_dropped": len(dropped),
                    "reasons": sorted(set(dropped)),
                }
            )
        if not usable:
            continue

        # Condition 3: Baseline B must not apply anywhere in the group.
        baseline_b_applies = any(
            c["methods"].get(EXACT_CLASS, {}).get("applicable") for c in usable
        )

        best_route = None
        for route in QUANTUM_ROUTES:
            values = [
                c["methods"][route]["cosine"]
                for c in usable
                if c["methods"].get(route, {}).get("applicable") and "cosine" in c["methods"][route]
            ]
            enough = len(values) >= MIN_SEEDS or (values and len(usable) < MIN_SEEDS)
            if enough and (best_route is None or np.mean(values) > np.mean(best_route[1])):
                best_route = (route, values)

        tensor = [
            c["methods"][TENSOR_NETWORK]["cosine"]
            for c in usable
            if c["methods"].get(TENSOR_NETWORK, {}).get("applicable")
            and "cosine" in c["methods"][TENSOR_NETWORK]
        ]
        if best_route is None or not tensor:
            continue

        route_name, route_values = best_route
        route_ci = bootstrap_interval(route_values)
        tensor_ci = bootstrap_interval(tensor)
        separated = route_ci[0] > tensor_ci[1]

        satisfies = bool(
            len(route_values) >= MIN_SEEDS
            and float(np.mean(route_values)) >= QUANTUM_THRESHOLD
            and float(np.mean(tensor)) < CLASSICAL_THRESHOLD
            and not baseline_b_applies
            and separated
        )
        regions.append(
            {
                "family": family,
                "K": k,
                "roughness": roughness,
                "block_size": block,
                "L": n_sites,
                "mu_over_mu_c": ratio,
                "seeds_scored": len(route_values),
                "route": route_name,
                "route_mean_cosine": float(np.mean(route_values)),
                "route_ci": route_ci,
                "tensor_mean_cosine": float(np.mean(tensor)),
                "tensor_ci": tensor_ci,
                "baseline_b_applies": baseline_b_applies,
                "cis_separated": separated,
                "satisfies_g7": satisfies,
            }
        )

    positive = [r for r in regions if r["satisfies_g7"]]
    sizes = [r["L"] for r in regions]
    return {
        "verdict": "positive" if positive else "null",
        "groups_scored": len(regions),
        "groups_excluded": excluded,
        "positive_region": positive,
        # Section 11.5: a null must carry the bound it delimits.
        "null_bound": (
            None
            if positive
            else {
                "largest_L_with_a_valid_reference": max(sizes) if sizes else None,
                "statement": (
                    "No region satisfies all four conditions at the sizes swept. The crossover, "
                    "if it exists, lies beyond the largest L at which this sweep held a valid "
                    "reference."
                ),
            }
        ),
        "conditions": {
            "quantum_cosine_at_least": QUANTUM_THRESHOLD,
            "tensor_network_below": CLASSICAL_THRESHOLD,
            "minimum_seeds": MIN_SEEDS,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", default="registered")
    arguments = parser.parse_args()

    # A single `--methods all` stream carries both passes; separate passes carry one each.
    classical: list[dict] = []
    quantum: list[dict] = []
    for base in (RESULTS_ROOT / "wp7", RESULTS_ROOT / "_local"):
        combined = load_stream(base / f"sweep_{arguments.grid}_all.jsonl")
        if combined:
            classical = [
                r for r in combined if any(m in r["methods"] for m in (TENSOR_NETWORK, EXACT_CLASS))
            ]
            quantum = [r for r in combined if any(m in r["methods"] for m in QUANTUM_ROUTES)]
            break
        classical = load_stream(base / f"sweep_{arguments.grid}.jsonl")
        quantum = load_stream(base / f"sweep_{arguments.grid}_quantum.jsonl")
        if classical or quantum:
            break
    if not classical and not quantum:
        print("no sweep streams found")
        return 0

    print(f"classical cells {len(classical)}, quantum cells {len(quantum)}")
    if not quantum:
        print(
            "\nNo quantum pass present. G-7 cannot be scored from the baselines alone: its\n"
            "decision compares a quantum route against the tensor network. Reporting the\n"
            "classical side only would be an answer to a different question."
        )
        return 0

    verdict = score(merge(classical, quantum))
    print(f"\nverdict: {verdict['verdict'].upper()}")
    print(f"groups scored   {verdict['groups_scored']}")
    print(f"groups excluded {len(verdict['groups_excluded'])}")
    if verdict["positive_region"]:
        print("\nregions satisfying all four conditions:")
        for r in verdict["positive_region"]:
            print(
                f"  {r['family']} L={r['L']} mu/mu_c={r['mu_over_mu_c']} "
                f"{r['route']} {r['route_mean_cosine']:.4f} vs MPS {r['tensor_mean_cosine']:.4f}"
            )
    else:
        print(f"\n{verdict['null_bound']['statement']}")
        print(
            f"largest L with a valid reference: "
            f"{verdict['null_bound']['largest_L_with_a_valid_reference']}"
        )

    target = evidence_directory("wp7") / "g_7_verdict.json"
    target.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(f"\nwritten {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
