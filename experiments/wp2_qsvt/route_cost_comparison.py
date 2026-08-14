"""Route A against Route B in two currencies: simulator wall clock and quantum resources.

WP2, claim C19's second half. Reads the head-to-head cells the WP7 quantum sweep already
produced and adds no new evolution, so nothing here can disagree with the sweep it draws on.

**Why two currencies, and why reporting one is misleading.** On the simulator Route B beats
Route A by three orders of magnitude, and quoting that alone would be close to dishonest. The
two methods are expensive in different places and a simulator prices only one of them:

* **Route A, varQITE**, runs a shallow parameterised circuit many times and does its work in a
  classical optimisation loop. Its cost is circuit *repetitions*: McLachlan's condition needs
  the `A` matrix, `n(n+1)/2` distinct entries for `n` parameters, and the `C` vector, `n` more,
  measured at every imaginary-time step. Depth stays constant in imaginary time, which is the
  entire near-term argument for it, and it needs no ancillas.
* **Route B, QSVT eigenstate filtering**, runs one deep coherent circuit and does no
  optimisation at all. Its cost is *query depth*: the Chebyshev degree is the number of
  qubitisation walk operators, each of which is two block-encoding calls plus a reflection. It
  needs `ceil(log2(terms))` ancillas for the encoding and its accuracy is governed by the
  normalisation `alpha`, which inflates the effective gap the filter has to resolve.

A simulator charges for `2^L` state-vector arithmetic and is blind to both distinctions. It
therefore flatters whichever method touches the state vector fewer times, which is Route B by
construction. The wall-clock ratio is real and worth reporting; it is not a statement about
what either method would cost on hardware.

The repetition and depth figures are **models, not measurements**, computed from the recorded
`reps`, `steps` and `degree`. They are labelled as such in the record.

    python experiments/wp2_qsvt/route_cost_comparison.py
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

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
from quasarstack.hamiltonian.builder import diagonal_hamiltonian, pauli_term_count
from quasarstack.io.store import RESULTS_ROOT, write_gate_record

ROUTE_A = "route_a_varqite"
ROUTE_B = "route_b_qsvt_filter"


def fitness_for(cell: dict) -> np.ndarray | None:
    """Rebuild the landscape a sweep cell names, so its Pauli term count can be counted."""
    family, n_sites, seed = cell["family"], cell["L"], cell.get("seed") or 0
    if family == "single_peak":
        return class_fitness(single_peak_classes(n_sites, 1.0))
    if family == "additive_pairwise":
        return class_fitness(pairwise_uniform_classes(n_sites, 1.0, 0.1))
    if family == "additive":
        rng = np.random.default_rng(9000 + n_sites)
        return additive_fitness(rng.uniform(0.3, 1.5, size=n_sites))
    if family == "nk":
        return nk_fitness(n_sites, cell["K"], seed=seed)
    if family == "spin_glass":
        return spin_glass_fitness(n_sites, seed=seed)
    if family == "house_of_cards":
        return house_of_cards_fitness(n_sites, seed=seed)
    if family == "rough_mount_fuji":
        return rough_mount_fuji_fitness(n_sites, seed=seed, roughness=cell.get("roughness") or 0.5)
    if family == "block":
        return block_fitness(n_sites, cell.get("block_size") or 2, seed=seed)
    return None


def route_a_resources(detail: dict, n_sites: int) -> dict:
    """Circuit repetitions and depth for varQITE, from the recorded reps and steps.

    Parameter count is `L * (reps + 1)`, the rule the ansatz uses. McLachlan needs the upper
    triangle of a symmetric `n` by `n` Gram matrix plus an `n` vector each step, so the
    repetition count is `steps * (n(n+1)/2 + n)`. Depth is constant in imaginary time, which
    is the property G-R.6 exists to check.
    """
    reps, steps = detail.get("reps"), detail.get("steps")
    if reps is None or steps is None:
        return {}
    n_parameters = n_sites * (reps + 1)
    per_step = n_parameters * (n_parameters + 1) // 2 + n_parameters
    return {
        "n_parameters": n_parameters,
        "reps": reps,
        "steps": steps,
        "circuit_repetitions_model": steps * per_step,
        "two_qubit_depth_model": reps * (2 * n_sites - 1),
        "depth_constant_in_imaginary_time": True,
        "ancillas": 0,
    }


def route_b_resources(detail: dict, fitness: np.ndarray | None, mu: float) -> dict:
    """Query depth and ancillas for the QSVT filter, from the recorded degree and alpha."""
    degree, alpha = detail.get("degree"), detail.get("alpha")
    if degree is None:
        return {}
    resources = {
        "chebyshev_degree": degree,
        "walk_operator_queries": degree,
        "block_encoding_calls_model": 2 * degree,
        "alpha": alpha,
        "gap": detail.get("gap"),
        "effective_gap_over_alpha": (detail.get("gap") / alpha) if alpha else None,
        "depth_constant_in_imaginary_time": False,
    }
    if fitness is not None:
        operator = diagonal_hamiltonian(fitness, mu)
        terms = pauli_term_count(operator)
        resources["pauli_terms"] = terms
        resources["ancillas"] = max(1, int(np.ceil(np.log2(max(terms, 1)))))
    return resources


def run() -> tuple[bool, dict, list[dict]]:
    started = time.monotonic()
    stream = RESULTS_ROOT / "wp7" / "sweep_registered_quantum.jsonl"
    if not stream.is_file():
        raise SystemExit(f"no quantum sweep at {stream}; run the WP7 quantum pass first")

    cases: list[dict] = []
    for line in stream.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        a, b = row["methods"].get(ROUTE_A, {}), row["methods"].get(ROUTE_B, {})
        if not (a.get("applicable") and b.get("applicable")):
            continue
        if "cosine" not in a or "cosine" not in b:
            continue
        fitness = fitness_for(row)
        mu_used = row.get("mu")
        if mu_used is None and fitness is not None:
            n_sites = row["L"]
            mu_c = float((fitness.max() - fitness.mean()) / n_sites)
            mu_used = row["mu_over_mu_c"] * mu_c
        cases.append(
            {
                "family": row["family"],
                "K": row.get("K"),
                "L": row["L"],
                "seed": row.get("seed"),
                "mu_over_mu_c": row["mu_over_mu_c"],
                "route_a": {
                    "cosine": a["cosine"],
                    "seconds": a.get("seconds_used"),
                    "over_budget": a.get("over_budget"),
                    **route_a_resources(a.get("detail", {}), row["L"]),
                },
                "route_b": {
                    "cosine": b["cosine"],
                    "seconds": b.get("seconds_used"),
                    "over_budget": b.get("over_budget"),
                    **route_b_resources(b.get("detail", {}), fitness, mu_used or 0.0),
                },
            }
        )

    n = len(cases)
    a_sec = [c["route_a"]["seconds"] for c in cases]
    b_sec = [c["route_b"]["seconds"] for c in cases]
    a_reps = [c["route_a"].get("circuit_repetitions_model", 0) for c in cases]
    b_queries = [c["route_b"].get("walk_operator_queries", 0) for c in cases]
    b_anc = [c["route_b"].get("ancillas") for c in cases if c["route_b"].get("ancillas")]

    measured = {
        "cells": n,
        "simulator_currency": {
            "route_a_mean_seconds": float(np.mean(a_sec)),
            "route_b_mean_seconds": float(np.mean(b_sec)),
            "speed_ratio_a_over_b": float(np.mean(a_sec) / np.mean(b_sec)),
            "route_a_cells_over_budget": sum(1 for c in cases if c["route_a"]["over_budget"]),
            "route_b_cells_over_budget": sum(1 for c in cases if c["route_b"]["over_budget"]),
        },
        "quantum_currency": {
            "route_a_mean_circuit_repetitions": float(np.mean(a_reps)),
            "route_a_ancillas": 0,
            "route_a_depth_constant_in_imaginary_time": True,
            "route_b_mean_walk_operator_queries": float(np.mean(b_queries)),
            "route_b_mean_block_encoding_calls": 2.0 * float(np.mean(b_queries)),
            "route_b_ancillas_min": min(b_anc) if b_anc else None,
            "route_b_ancillas_max": max(b_anc) if b_anc else None,
            "route_b_mean_alpha": float(
                np.mean([c["route_b"]["alpha"] for c in cases if c["route_b"].get("alpha")])
            ),
        },
        "accuracy": {
            "route_a_min_cosine": float(min(c["route_a"]["cosine"] for c in cases)),
            "route_b_min_cosine": float(min(c["route_b"]["cosine"] for c in cases)),
            "cells_where_b_is_more_accurate": sum(
                1 for c in cases if c["route_b"]["cosine"] > c["route_a"]["cosine"]
            ),
        },
        "how_to_read_this": (
            "The simulator ratio and the quantum resource counts answer different questions "
            "and must be quoted together. On the simulator Route B wins by three orders of "
            "magnitude, because a state-vector simulator charges for 2^L arithmetic and Route "
            "B touches the state far fewer times. That says nothing about hardware. Route A "
            "buys its accuracy with a very large number of shallow, ancilla-free circuit "
            "repetitions driven by a classical optimiser, which is the near-term shape. Route "
            "B buys its accuracy with one deep coherent circuit of walk-operator queries plus "
            "encoding ancillas, which is the fault-tolerant shape. Neither number dominates "
            "the other; the repetition and depth figures are models computed from recorded "
            "reps, steps and degree, not measurements."
        ),
        "seconds": round(time.monotonic() - started, 2),
    }
    return bool(n > 0), measured, cases


def main() -> int:
    passed, measured, cases = run()
    path = write_gate_record(
        gate="WP2-ROUTE-COST",
        work_package="wp2",
        threshold={
            "statistic": "Route A against Route B in simulator wall clock and in quantum "
            "resources, on the cells where both ran",
            "registered_in": "claim C19, and the review note of 13 August asking for two "
            "currencies rather than one",
        },
        measured=measured,
        passed=passed,
        cases=cases,
        notes=measured["how_to_read_this"],
    )

    sim, qc, acc = (
        measured["simulator_currency"],
        measured["quantum_currency"],
        measured["accuracy"],
    )
    print(f"Route A against Route B on {measured['cells']} cells\n")
    print(f"  {'':32s} {'Route A':>18} {'Route B':>18}")
    print(
        f"  {'simulator seconds, mean':32s} {sim['route_a_mean_seconds']:>18.2f} "
        f"{sim['route_b_mean_seconds']:>18.2f}"
    )
    print(
        f"  {'cells over the 300 s allotment':32s} "
        f"{sim['route_a_cells_over_budget']:>18} {sim['route_b_cells_over_budget']:>18}"
    )
    print(
        f"  {'minimum cosine':32s} {acc['route_a_min_cosine']:>18.5f} "
        f"{acc['route_b_min_cosine']:>18.5f}"
    )
    print(
        f"  {'circuit repetitions, model':32s} "
        f"{qc['route_a_mean_circuit_repetitions']:>18.3e} {'n/a':>18}"
    )
    print(
        f"  {'walk-operator queries, model':32s} {'n/a':>18} "
        f"{qc['route_b_mean_walk_operator_queries']:>18.1f}"
    )
    print(
        f"  {'ancillas':32s} {qc['route_a_ancillas']:>18} "
        f"{str(qc['route_b_ancillas_min']) + ' to ' + str(qc['route_b_ancillas_max']):>18}"
    )
    print(f"  {'depth constant in imaginary time':32s} {'yes':>18} {'no':>18}")
    print(f"\n  simulator speed ratio A/B: {sim['speed_ratio_a_over_b']:.0f}x")
    print(f"  B more accurate on {acc['cells_where_b_is_more_accurate']} of {measured['cells']}")
    print(f"\n  record  {path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
