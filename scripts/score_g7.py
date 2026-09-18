"""Score the G-7 advantage criterion over the WP7 sweep.

A positive result is a non-empty region, reproducible across at least five seeds, in which a
quantum route reaches cosine >= 0.90, the compute-matched tensor-network baseline is below 0.80,
Baseline B does not apply, and the bootstrap confidence intervals of the two methods do not
overlap. Otherwise the result is a null, reported with the conditions each group failed and the
largest size at which the sweep held a valid reference.

A cell in which any applicable method exceeded its time allocation is excluded, with the count
reported per size. `over_budget` is recomputed from the recorded seconds.

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

from quasarstack.io.store import RESULTS_ROOT, write_gate_record  # noqa: E402

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
    """Whether a method used more than its allocated seconds."""
    used = entry.get("seconds_used")
    allotted = entry.get("seconds_allotted")
    if used is None or allotted is None:
        return False
    return float(used) > float(allotted)


def bootstrap_interval(values: list[float]) -> tuple[float, float]:
    """Percentile bootstrap of the mean."""
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
    # Cells removed by the budget rule, per size.
    seen_by_size: dict[Any, int] = {}
    dropped_by_size: dict[Any, dict[str, int]] = {}
    for key, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        family, k, roughness, block, n_sites, ratio = key
        seen_by_size[n_sites] = seen_by_size.get(n_sites, 0) + len(members)

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
        for reason in dropped:
            dropped_by_size.setdefault(n_sites, {})
            dropped_by_size[n_sites][reason] = dropped_by_size[n_sites].get(reason, 0) + 1
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

        # The conditions this group fails.
        failed = []
        if len(route_values) < MIN_SEEDS:
            failed.append(f"fewer than {MIN_SEEDS} seeds")
        if float(np.mean(route_values)) < QUANTUM_THRESHOLD:
            failed.append(f"no quantum route reaches {QUANTUM_THRESHOLD}")
        if float(np.mean(tensor)) >= CLASSICAL_THRESHOLD:
            failed.append(f"the tensor network stays at or above {CLASSICAL_THRESHOLD}")
        if baseline_b_applies:
            failed.append("baseline B applies")
        if not separated:
            failed.append("the bootstrap intervals overlap")
        satisfies = not failed
        regions.append(
            {
                "failed_conditions": failed,
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
    why_not: dict[str, int] = {}
    for region in regions:
        for reason in region["failed_conditions"]:
            why_not[reason] = why_not.get(reason, 0) + 1
    exclusion_summary = [
        {
            "L": size,
            "cells": seen_by_size[size],
            "cells_excluded": sum(dropped_by_size.get(size, {}).values()),
            "share_excluded": sum(dropped_by_size.get(size, {}).values()) / seen_by_size[size],
            "by_reason": dropped_by_size.get(size, {}),
        }
        for size in sorted(seen_by_size, key=lambda v: (v is None, v))
    ]
    return {
        "verdict": "positive" if positive else "null",
        "groups_scored": len(regions),
        "groups_excluded": excluded,
        "excluded_cells_by_size": exclusion_summary,
        "conditions_failed_by_group_count": dict(sorted(why_not.items(), key=lambda kv: -kv[1])),
        "positive_region": positive,
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
    parser.add_argument("--grid", default="full")
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
        print("\nno quantum pass present; G-7 compares a quantum route with the tensor network")
        return 0

    verdict = score(merge(classical, quantum))
    print(f"\nverdict: {verdict['verdict'].upper()}")
    print(f"groups scored   {verdict['groups_scored']}")
    print(f"groups excluded {len(verdict['groups_excluded'])}")
    print("\ncells excluded by size:")
    for row in verdict["excluded_cells_by_size"]:
        print(
            f"   L={row['L']}  {row['cells_excluded']} of {row['cells']} "
            f"({row['share_excluded']:.1%})  {row['by_reason'] or ''}"
        )
    if verdict["positive_region"]:
        print("\nregions satisfying all four conditions:")
        for r in verdict["positive_region"]:
            print(
                f"  {r['family']} L={r['L']} mu/mu_c={r['mu_over_mu_c']} "
                f"{r['route']} {r['route_mean_cosine']:.4f} vs MPS {r['tensor_mean_cosine']:.4f}"
            )
    else:
        print("\nwhich condition each group failed, by group count:")
        for reason, count in verdict["conditions_failed_by_group_count"].items():
            print(f"   {count:>5}  {reason}")
        print(f"\n{verdict['null_bound']['statement']}")
        print(
            f"largest L with a valid reference: "
            f"{verdict['null_bound']['largest_L_with_a_valid_reference']}"
        )

    # `passed` records that the criterion was evaluated; the outcome is `measured.verdict`.
    path = write_gate_record(
        gate="G-7",
        work_package="wp7",
        threshold={
            "criteria": (
                f"a non-empty region, contiguous and reproducible across at least {MIN_SEEDS} "
                f"seeds, in which a quantum route reaches cosine >= {QUANTUM_THRESHOLD}, the "
                f"compute-matched tensor network is below {CLASSICAL_THRESHOLD}, Baseline B "
                f"does not apply, and the bootstrap confidence intervals do not overlap"
            ),
            "defined_in": "docs/protocol.md, G-7",
        },
        measured=verdict,
        passed=True,
        cases=verdict.get("positive_region") or [],
        notes=(
            f"Verdict: {verdict['verdict']}. passed=true records that the criterion was "
            f"evaluated. measured.conditions_failed_by_group_count lists the conditions each "
            f"group failed."
        ),
    )
    print(f"\nwritten {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
