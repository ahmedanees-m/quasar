"""Resumable grid sweep for WP7's boundary map.

The sweep is the most expensive thing the project runs and the one whose integrity is
easiest to lose, so the design is driven by four failure modes rather than by convenience.

**A cell that was never run must not look like a cell that scored badly.** Every cell in the
declared grid ends up in the manifest either scored or excluded with a reason, which is claim
C27. There is no third state and no silent gap.

**A method that ran out of time must not look like a method that was wrong.** Section 11.3
denominates the budget in wall-clock seconds, and Amendment 19's second addendum measured
Baseline C spending ten to twenty times more of it on the dense-operator families than on the
additive one. So allotted and used seconds are recorded per cell per method, and a method that
hit its ceiling is marked `budget_exhausted` rather than scored as inaccurate.

**Interruption must not corrupt what came before.** Results append to a JSONL as they are
produced and a restart skips what is already there. The alternative, holding everything in
memory and writing at the end, loses a multi-hour run to one exception and quietly tempts
whoever restarts it to narrow the grid.

**The order parameter must mean the same thing in every cell.** ADR-0017 and Amendment 18:
localisation is measured from each instance's own fittest genotype, not from genotype 0,
because no rugged family keeps its optimum there. The reference genotype and its Hamming
weight are recorded per cell so the choice is auditable rather than implicit.

    python scripts/sweep_runner.py --wp 7 --grid smoke      # a few cells, for wiring
    python scripts/sweep_runner.py --wp 7 --grid registered # the section 11.1 grid
    python scripts/sweep_runner.py --wp 7 --list            # what would run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quasarstack.analytic.exact_diag import perron_vector  # noqa: E402
from quasarstack.classical.exact_class import applicability, solve  # noqa: E402
from quasarstack.classical.landscapes import (  # noqa: E402
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
from quasarstack.classical.mps_ite import evolve as mps_evolve  # noqa: E402
from quasarstack.classical.wright_fisher import sample_stationary  # noqa: E402
from quasarstack.io.store import RESULTS_ROOT, environment  # noqa: E402
from quasarstack.spectral.order_parameter import localisation  # noqa: E402

# Registered in GATES.md section 11.1, with Amendment 18's split of the ruggedness axis.
REGISTERED_GRID = {
    "sizes": [8, 10, 12],
    "mu_ratios": [0.4, 0.7, 1.0, 1.3, 1.6],
    "seeds": [0, 1, 2, 3, 4],
    "families": [
        {"family": "single_peak", "axis": "control"},
        {"family": "additive_pairwise", "axis": "control"},
        {"family": "nk", "K": 1, "axis": "biological"},
        {"family": "nk", "K": 2, "axis": "biological"},
        {"family": "nk", "K": 4, "axis": "biological"},
        {"family": "rough_mount_fuji", "roughness": 0.5, "axis": "biological"},
        {"family": "spin_glass", "axis": "compilation"},
        {"family": "block", "block_size": 2, "axis": "compilation"},
        {"family": "house_of_cards", "axis": "biological"},
    ],
}
SMOKE_GRID = {
    "sizes": [6],
    "mu_ratios": [1.0],
    "seeds": [0],
    "families": [
        {"family": "single_peak", "axis": "control"},
        {"family": "nk", "K": 2, "axis": "biological"},
    ],
}

# Section 11.3. Wall-clock seconds per cell per method.
BUDGET_SECONDS = {8: 300.0, 10: 300.0, 12: 300.0, 14: 900.0}
REFERENCE_DENSE_LIMIT = 10


def build_fitness(spec: dict[str, Any], n_sites: int, seed: int) -> np.ndarray:
    family = spec["family"]
    if family == "single_peak":
        return class_fitness(single_peak_classes(n_sites, 1.0))
    if family == "additive_pairwise":
        return class_fitness(pairwise_uniform_classes(n_sites, 1.0, 0.1))
    if family == "nk":
        return nk_fitness(n_sites, spec["K"], seed=seed)
    if family == "rough_mount_fuji":
        return rough_mount_fuji_fitness(n_sites, seed=seed, roughness=spec["roughness"])
    if family == "spin_glass":
        return spin_glass_fitness(n_sites, seed=seed)
    if family == "block":
        return block_fitness(n_sites, spec["block_size"], seed=seed)
    if family == "house_of_cards":
        return house_of_cards_fitness(n_sites, seed=seed)
    if family == "additive":
        return additive_fitness(np.random.default_rng(seed).uniform(0.3, 1.5, size=n_sites))
    raise ValueError(f"unregistered family {family!r}")


def threshold_for(spec: dict[str, Any], fitness: np.ndarray, n_sites: int) -> float:
    """mu_c per instance. Amendment 12's addendum defines the non-peak case."""
    if spec["family"] == "single_peak":
        return 1.0 / n_sites
    return float((fitness.max() - fitness.mean()) / n_sites)


def cells(grid: dict[str, Any]) -> Iterator[dict[str, Any]]:
    for n_sites in grid["sizes"]:
        for spec in grid["families"]:
            seeded = spec["family"] not in {"single_peak", "additive_pairwise"}
            for seed in grid["seeds"] if seeded else [0]:
                for ratio in grid["mu_ratios"]:
                    yield {
                        **spec,
                        "L": n_sites,
                        "seed": seed,
                        "mu_over_mu_c": ratio,
                    }


def cell_key(cell: dict[str, Any]) -> str:
    parts = [
        str(cell.get(k))
        for k in ("family", "K", "roughness", "block_size", "L", "seed", "mu_over_mu_c")
    ]
    return "|".join(parts)


# --------------------------------------------------------------------------------------
# Methods. Each returns a distribution or declares itself inapplicable, and each reports
# the seconds it used against the seconds it was allotted.


def method_baseline_b(fitness: np.ndarray, mu: float, budget: float) -> dict[str, Any]:
    """Baseline B applies only inside the polynomial-time class and refuses outside it."""
    verdict = applicability(fitness)
    if not verdict["applies"]:
        return {"applicable": False, "reason": "outside the polynomial-time class"}
    began = time.monotonic()
    result = solve(fitness, mu)
    return {
        "applicable": True,
        "distribution": np.asarray(result["distribution"]),
        "seconds": time.monotonic() - began,
        "detail": {"class": verdict["class"]},
    }


def method_baseline_c(fitness: np.ndarray, mu: float, budget: float) -> dict[str, Any]:
    """Baseline C, spending its budget on bond dimension until the clock runs out."""
    began = time.monotonic()
    best: dict[str, Any] | None = None
    exhausted = False
    for chi in (2, 4, 8, 16, 32, 64):
        if time.monotonic() - began > budget:
            exhausted = True
            break
        result = mps_evolve(fitness, mu, chi, dtau=0.05, max_steps=3000)
        best = {
            "distribution": np.asarray(result["distribution"]),
            "chi": chi,
            "discarded": float(result["total_discarded_weight"]),
        }
        if result["total_discarded_weight"] < 1e-12:
            break
    if best is None:
        return {"applicable": True, "budget_exhausted": True, "seconds": time.monotonic() - began}
    return {
        "applicable": True,
        "distribution": best["distribution"],
        "seconds": time.monotonic() - began,
        "budget_exhausted": exhausted,
        "detail": {"chi": best["chi"], "discarded_weight": best["discarded"]},
    }


def method_baseline_a(fitness: np.ndarray, mu: float, budget: float) -> dict[str, Any]:
    """Baseline A, spending its budget on population size."""
    began = time.monotonic()
    best: dict[str, Any] | None = None
    exhausted = False
    for population in (10**3, 10**4, 10**5, 10**6):
        if time.monotonic() - began > budget:
            exhausted = True
            break
        result = sample_stationary(fitness, mu, population, 3000, [0, 1, 2], dt=0.01)
        best = {"distribution": np.asarray(result["distribution"]), "population": population}
    if best is None:
        return {"applicable": True, "budget_exhausted": True, "seconds": time.monotonic() - began}
    return {
        "applicable": True,
        "distribution": best["distribution"],
        "seconds": time.monotonic() - began,
        "budget_exhausted": exhausted,
        "detail": {"population": best["population"]},
    }


METHODS: dict[str, Callable[[np.ndarray, float, float], dict[str, Any]]] = {
    "baseline_a_wright_fisher": method_baseline_a,
    "baseline_b_exact_class": method_baseline_b,
    "baseline_c_tensor_network": method_baseline_c,
}


# --------------------------------------------------------------------------------------


def score(distribution: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Cosine and total variation, both, as section 11.4 requires.

    Section 11.4 also makes total variation the deciding metric where the two disagree,
    because it is the less flattering one. Both are stored so that disagreement is visible.
    """
    cosine = float(
        distribution @ reference / (np.linalg.norm(distribution) * np.linalg.norm(reference))
    )
    return {
        "cosine": cosine,
        "total_variation": 0.5 * float(np.abs(distribution - reference).sum()),
    }


def run_cell(cell: dict[str, Any]) -> dict[str, Any]:
    n_sites = cell["L"]
    fitness = build_fitness(cell, n_sites, cell["seed"])
    mu_c = threshold_for(cell, fitness, n_sites)
    mu = cell["mu_over_mu_c"] * mu_c

    reference = np.abs(perron_vector(fitness, mu, dense_limit=REFERENCE_DENSE_LIMIT)[0])
    reference = reference / reference.sum()

    # ADR-0017: measured from this instance's own optimum, not from genotype 0.
    optimum = int(np.argmax(fitness))
    budget = BUDGET_SECONDS.get(n_sites, 300.0)

    record: dict[str, Any] = {
        **cell,
        "mu": mu,
        "mu_c": mu_c,
        "reference_genotype": optimum,
        "reference_genotype_hamming_weight": int(optimum.bit_count()),
        "order_parameter_of_reference": localisation(reference, optimum),
        "budget_seconds": budget,
        "methods": {},
        "excluded": False,
    }

    for name, method in METHODS.items():
        try:
            outcome = method(fitness, mu, budget)
        except Exception as error:  # a method failing must not lose the cell
            record["methods"][name] = {"applicable": True, "error": repr(error)}
            continue
        if not outcome.get("applicable", True):
            record["methods"][name] = {"applicable": False, "reason": outcome["reason"]}
            continue
        entry: dict[str, Any] = {
            "applicable": True,
            "seconds_used": round(float(outcome.get("seconds", 0.0)), 3),
            "seconds_allotted": budget,
            "budget_exhausted": bool(outcome.get("budget_exhausted", False)),
            "detail": outcome.get("detail", {}),
        }
        if "distribution" in outcome:
            entry.update(score(outcome["distribution"], reference))
            entry["order_parameter"] = localisation(outcome["distribution"], optimum)
        record["methods"][name] = entry

    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wp", default="7")
    parser.add_argument("--grid", choices=["smoke", "registered"], default="smoke")
    parser.add_argument("--list", action="store_true", help="show the cells and stop")
    arguments = parser.parse_args()

    grid = SMOKE_GRID if arguments.grid == "smoke" else REGISTERED_GRID
    planned = list(cells(grid))

    if arguments.list:
        for cell in planned:
            print(cell_key(cell))
        print(f"{len(planned)} cells")
        return 0

    # ADR-0012: a run outside the pinned image must not write where committed evidence
    # lives. write_gate_record enforces this for gates; the sweep writes its own stream
    # and so has to enforce it too. This was reintroduced here once and caught before it
    # produced anything, which is the only reason the check is duplicated rather than
    # trusted to live in one place.
    env = environment()
    in_pinned_image = env["image"] != "unknown" and str(env["platform"]).startswith("Linux")
    directory = RESULTS_ROOT / f"wp{arguments.wp}" if in_pinned_image else RESULTS_ROOT / "_local"
    directory.mkdir(parents=True, exist_ok=True)
    if not in_pinned_image:
        print(
            f"NOTE: not running in the pinned image, so this sweep writes to "
            f"{directory} and is not evidence."
        )
    stream = directory / f"sweep_{arguments.grid}.jsonl"

    done = set()
    if stream.exists():
        for line in stream.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(cell_key(json.loads(line)))
        print(f"resuming: {len(done)} cells already recorded in {stream.name}")

    started = time.monotonic()
    with stream.open("a", encoding="utf-8") as handle:
        for index, cell in enumerate(planned, start=1):
            if cell_key(cell) in done:
                continue
            record = run_cell(cell)
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(f"[{index}/{len(planned)}] {cell_key(cell)}", flush=True)

    records = [
        json.loads(line) for line in stream.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    manifest = {
        "grid": arguments.grid,
        "cells_planned": len(planned),
        "cells_recorded": len(records),
        # C27: every cell is scored or excluded with a reason. Neither state is silent.
        "cells_excluded": [r for r in records if r.get("excluded")],
        "methods_inapplicable_by_name": {
            name: sum(1 for r in records if not r["methods"].get(name, {}).get("applicable", True))
            for name in METHODS
        },
        "methods_budget_exhausted_by_name": {
            name: sum(1 for r in records if r["methods"].get(name, {}).get("budget_exhausted"))
            for name in METHODS
        },
        "seconds": round(time.monotonic() - started, 2),
        "env": env,
    }
    (directory / f"sweep_manifest_{arguments.grid}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"\n{manifest['cells_recorded']} of {manifest['cells_planned']} cells recorded")
    print(f"inapplicable per method: {manifest['methods_inapplicable_by_name']}")
    print(f"budget exhausted per method: {manifest['methods_budget_exhausted_by_name']}")
    print(f"manifest  {directory / f'sweep_manifest_{arguments.grid}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
