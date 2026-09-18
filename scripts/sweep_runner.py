"""Resumable grid sweep for the WP7 comparison.

Every cell of the grid is recorded, scored or marked inapplicable with a reason. For each method
the allocated and used wall-clock seconds are stored, and a method stopped by its allocation is
marked `budget_exhausted`. Records are appended to a JSONL stream as they are produced, and a
restart skips cells already present.

The order parameter is measured from each instance's fittest genotype, which is recorded per
cell with its Hamming weight.

    python scripts/sweep_runner.py --wp 7 --grid smoke      # a few cells
    python scripts/sweep_runner.py --wp 7 --grid full       # the full grid
    python scripts/sweep_runner.py --wp 7 --grid full --workers 8
    python scripts/sweep_runner.py --wp 7 --list            # what would run

Cells are independent, so `--workers` runs them in separate single-threaded processes. The
per-cell allocation is unchanged and the worker count is recorded in the manifest.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quasarstack.analytic.exact_diag import (  # noqa: E402
    mutation_selection_generator,
    perron_vector,
)
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
from quasarstack.classical.mps_dmrg import dominant_state  # noqa: E402
from quasarstack.classical.wright_fisher import sample_stationary  # noqa: E402
from quasarstack.hamiltonian.builder import diagonal_hamiltonian  # noqa: E402
from quasarstack.io.store import environment, output_directory  # noqa: E402
from quasarstack.ite.varqite import Ansatz  # noqa: E402
from quasarstack.ite.varqite import evolve as varqite_evolve  # noqa: E402
from quasarstack.qsvt.block_encoding import one_norm  # noqa: E402
from quasarstack.qsvt.filter import filtered_state  # noqa: E402
from quasarstack.spectral.order_parameter import localisation  # noqa: E402

# The WP7 grid (docs/protocol.md, G-7): 777 cells.
FULL_GRID = {
    "sizes": [8, 10, 12],
    "mu_ratios": [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
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

# One size and seed across the mutation axis, one family per axis.
LOCAL_GRID = {
    "sizes": [8],
    "mu_ratios": [0.4, 0.7, 1.0, 1.3, 1.6],
    "seeds": [0],
    "families": [
        {"family": "single_peak", "axis": "control"},
        {"family": "nk", "K": 2, "axis": "biological"},
        {"family": "spin_glass", "axis": "compilation"},
    ],
}

# Wall-clock seconds per cell per method.
BUDGET_SECONDS = {8: 300.0, 10: 300.0, 12: 300.0, 14: 900.0}
REFERENCE_DENSE_LIMIT = 10

# Route A runs on a subset of cells. Its cost scales as n_parameters^2 * 2^L with
# n_parameters = L(L + 3); at L = 6 a cell takes 198 to 235 s of the 300 s allocation.
ROUTE_A_PROBE_SIZE = 8
ROUTE_A_PROBE_MU = (0.4, 1.0, 1.6)
ROUTE_A_PROBE_SEED = 0


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
    raise ValueError(f"unknown family {family!r}")


def threshold_for(spec: dict[str, Any], fitness: np.ndarray, n_sites: int) -> float:
    """mu_c per instance: 1/L for the single peak, (max f - mean f)/L otherwise."""
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
    """Baseline C, two-site DMRG, with the bond dimension capped at its exact value.

    At ``L`` sites no bond needs more than ``2**(L // 2)``, so the cap never truncates a state
    the chain can hold. The weight discarded at each update is bounded by the cutoff instead.
    """
    began = time.monotonic()
    n_sites = fitness.size.bit_length() - 1
    result = dominant_state(fitness, mu, max_bond=1 << (n_sites // 2), budget=budget)
    return {
        "applicable": True,
        "distribution": np.asarray(result["distribution"]),
        "seconds": time.monotonic() - began,
        "budget_exhausted": bool(result["budget_exhausted"]),
        "detail": {
            "method": "dmrg2",
            "max_bond": result["max_bond"],
            "bond_reached": result["bond_reached"],
            "operator_bond_dimension": result["operator_bond_dimension"],
            "complement_symmetric": result["complement_symmetric"],
            "sweeps": result["sweeps"],
            "converged": bool(result["converged"]),
            "eigenvalue": result["eigenvalue"],
            "cutoff": result["cutoff"],
        },
    }


def method_baseline_a(fitness: np.ndarray, mu: float, budget: float) -> dict[str, Any]:
    """Baseline A, one Wright-Fisher run at N = 1e6 for 3000 generations.

    A generation in genotype-count space costs ``O(L 2^L)`` independent of N, and drift is
    injected once per generation, so neither a population ladder nor a longer chain improves
    the estimate. Measured at L = 10 on an NK K = 2 cell:

    | | seconds | cosine | total variation |
    |---|---|---|---|
    | ladder over N = 1e3 to 1e6, 3000 generations each | 59.1 | 0.999830 | 9.35e-3 |
    | N = 1e6 once, 3000 generations | 23.0 | 0.999830 | 9.35e-3 |
    | N = 1e6 once, 12000 generations | 105.8 | 0.999735 | 1.06e-2 |

    Accuracy is limited by drift rather than by the time allocation.
    """
    began = time.monotonic()
    population = 10**6
    generations = 3000
    result = sample_stationary(fitness, mu, population, generations, [0, 1, 2], dt=0.01)
    return {
        "applicable": True,
        "distribution": np.asarray(result["distribution"]),
        "seconds": time.monotonic() - began,
        "budget_exhausted": False,
        "detail": {
            "population": population,
            "generations": generations,
            "seed_spread": float(result["max_pairwise_tv_between_seeds"]),
            "burn_in_drift": float(result["max_burn_in_drift"]),
        },
    }


def in_route_a_probe(cell: dict[str, Any]) -> bool:
    """Whether Route A runs on this cell."""
    return (
        cell["L"] == ROUTE_A_PROBE_SIZE
        and cell.get("seed", 0) == ROUTE_A_PROBE_SEED
        and any(abs(cell["mu_over_mu_c"] - r) < 1e-9 for r in ROUTE_A_PROBE_MU)
    )


def method_route_a(fitness: np.ndarray, mu: float, budget: float) -> dict[str, Any]:
    """Route A, varQITE, spending its allocation on imaginary time.

    Each step solves the McLachlan system, whose geometric tensor costs
    `O(n_parameters^2 * 2^L)`, with `L * (reps + 1)` parameters. A cell that cannot finish
    inside its allocation is recorded as `budget_exhausted`.
    """
    began = time.monotonic()
    n_sites = fitness.size.bit_length() - 1
    matrix = np.asarray(diagonal_hamiltonian(fitness, mu).to_matrix()).real
    ansatz = Ansatz(n_sites, reps=n_sites + 2)

    best: dict[str, Any] | None = None
    exhausted = False
    for tau in (5.0, 15.0, 40.0):
        if time.monotonic() - began > budget:
            exhausted = True
            break
        result = varqite_evolve(ansatz, matrix, tau=tau, dtau=0.05)
        # Evolution exposes probs, already a distribution over genotypes.
        probabilities = np.abs(np.asarray(result.probs))
        best = {
            "distribution": probabilities / probabilities.sum(),
            "tau": tau,
            "tau_used": float(result.tau_used),
            "steps": int(result.steps),
            "converged": bool(result.converged),
        }
        if best["converged"]:
            break
    if best is None:
        return {"applicable": True, "budget_exhausted": True, "seconds": time.monotonic() - began}
    return {
        "applicable": True,
        "distribution": best["distribution"],
        "seconds": time.monotonic() - began,
        "budget_exhausted": exhausted,
        "detail": {
            "tau_ceiling": best["tau"],
            "tau_used": best["tau_used"],
            "steps": best["steps"],
            "converged": best["converged"],
            "reps": ansatz.reps,
        },
    }


def method_route_b(fitness: np.ndarray, mu: float, budget: float) -> dict[str, Any]:
    """Route B, QSVT eigenstate filtering, spending its allocation on polynomial degree.

    Scored as the filter applied to the operator, which is what the circuit's block encoding
    implements and what G-2 verifies against the circuit.
    """
    began = time.monotonic()
    n_sites = fitness.size.bit_length() - 1
    generator = np.asarray(mutation_selection_generator(fitness, mu).todense())
    values = np.linalg.eigvalsh(generator)
    lambda_1, lambda_2 = float(values[-1]), float(values[-2])
    alpha = one_norm(diagonal_hamiltonian(fitness, mu))
    initial = np.full(1 << n_sites, 1.0 / np.sqrt(1 << n_sites))

    best: dict[str, Any] | None = None
    exhausted = False
    for degree in (16, 64, 256, 1024):
        if time.monotonic() - began > budget:
            exhausted = True
            break
        state = filtered_state(generator, alpha, degree, lambda_1, lambda_2, initial=initial)
        amplitudes = np.abs(np.asarray(state))
        best = {"distribution": amplitudes / amplitudes.sum(), "degree": degree}
    if best is None:
        return {"applicable": True, "budget_exhausted": True, "seconds": time.monotonic() - began}
    return {
        "applicable": True,
        "distribution": best["distribution"],
        "seconds": time.monotonic() - began,
        "budget_exhausted": exhausted,
        "detail": {"degree": best["degree"], "alpha": alpha, "gap": lambda_1 - lambda_2},
    }


CLASSICAL_METHODS: dict[str, Callable[[np.ndarray, float, float], dict[str, Any]]] = {
    "baseline_a_wright_fisher": method_baseline_a,
    "baseline_b_exact_class": method_baseline_b,
    "baseline_c_tensor_network": method_baseline_c,
}

QUANTUM_METHODS: dict[str, Callable[[np.ndarray, float, float], dict[str, Any]]] = {
    "route_a_varqite": method_route_a,
    "route_b_qsvt_filter": method_route_b,
}

METHOD_SETS: dict[str, dict[str, Callable[[np.ndarray, float, float], dict[str, Any]]]] = {
    "classical": CLASSICAL_METHODS,
    "quantum": QUANTUM_METHODS,
    "all": {**CLASSICAL_METHODS, **QUANTUM_METHODS},
}

# Selected per invocation.
METHODS: dict[str, Callable[[np.ndarray, float, float], dict[str, Any]]] = dict(CLASSICAL_METHODS)


# --------------------------------------------------------------------------------------


def score(distribution: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Cosine similarity and total-variation distance."""
    cosine = float(
        distribution @ reference / (np.linalg.norm(distribution) * np.linalg.norm(reference))
    )
    return {
        "cosine": cosine,
        "total_variation": 0.5 * float(np.abs(distribution - reference).sum()),
    }


def prepare_worker(method_set: str) -> None:
    """Select the method set, and compile quimb's kernels before any cell is timed."""
    global METHODS
    METHODS = METHOD_SETS[method_set]
    if "baseline_c_tensor_network" in METHODS:
        dominant_state(np.linspace(0.0, 1.0, 64), 0.1, max_bond=4)


def run_cell(cell: dict[str, Any]) -> dict[str, Any]:
    n_sites = cell["L"]
    fitness = build_fitness(cell, n_sites, cell["seed"])
    mu_c = threshold_for(cell, fitness, n_sites)
    mu = cell["mu_over_mu_c"] * mu_c

    reference = np.abs(perron_vector(fitness, mu, dense_limit=REFERENCE_DENSE_LIMIT)[0])
    reference = reference / reference.sum()

    # Order parameter measured from this instance's fittest genotype.
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
        if name == "route_a_varqite" and not in_route_a_probe(cell):
            record["methods"][name] = {
                "applicable": False,
                "reason": "Route A runs at L = 8, seed 0, mu/mu_c in {0.4, 1.0, 1.6}",
            }
            continue
        try:
            outcome = method(fitness, mu, budget)
        except Exception as error:  # recorded, so one failing method does not lose the cell
            record["methods"][name] = {"applicable": True, "error": repr(error)}
            continue
        if not outcome.get("applicable", True):
            record["methods"][name] = {"applicable": False, "reason": outcome["reason"]}
            continue
        used = float(outcome.get("seconds", 0.0))
        entry: dict[str, Any] = {
            "applicable": True,
            "seconds_used": round(used, 3),
            "seconds_allotted": budget,
            # Stopped early by the allocation.
            "budget_exhausted": bool(outcome.get("budget_exhausted", False)),
            # Methods check the clock between units of work, so a single unit can overrun.
            "over_budget": bool(used > budget),
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
    parser.add_argument("--grid", choices=["smoke", "local", "full"], default="smoke")
    parser.add_argument(
        "--methods",
        choices=["classical", "quantum", "all"],
        default="classical",
        help="method set to run; each set writes its own stream",
    )
    parser.add_argument("--list", action="store_true", help="show the cells and stop")
    parser.add_argument("--workers", type=int, default=1, help="cells run in parallel")
    arguments = parser.parse_args()
    if arguments.workers < 1:
        parser.error("--workers must be at least 1")

    grid = {"smoke": SMOKE_GRID, "local": LOCAL_GRID, "full": FULL_GRID}[arguments.grid]
    planned = list(cells(grid))

    if arguments.list:
        for cell in planned:
            print(cell_key(cell))
        print(f"{len(planned)} cells")
        return 0

    env = environment()
    directory = output_directory(f"wp{arguments.wp}")
    suffix = "" if arguments.methods == "classical" else f"_{arguments.methods}"
    stream = directory / f"sweep_{arguments.grid}{suffix}.jsonl"

    done = set()
    if stream.exists():
        for line in stream.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(cell_key(json.loads(line)))
        print(f"resuming: {len(done)} cells already recorded in {stream.name}")

    started = time.monotonic()
    pending = [cell for cell in planned if cell_key(cell) not in done]
    prepare_worker(arguments.methods)
    with stream.open("a", encoding="utf-8") as handle:
        if arguments.workers == 1:
            pool = None
            outcomes: Iterator[dict[str, Any]] = map(run_cell, pending)
        else:
            context = multiprocessing.get_context("fork")
            pool = context.Pool(
                arguments.workers, initializer=prepare_worker, initargs=(arguments.methods,)
            )
            outcomes = pool.imap_unordered(run_cell, pending)
        try:
            for count, record in enumerate(outcomes, start=len(done) + 1):
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                print(f"[{count}/{len(planned)}] {cell_key(record)}", flush=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()

    records = [
        json.loads(line) for line in stream.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    manifest = {
        "grid": arguments.grid,
        "methods": arguments.methods,
        "cells_planned": len(planned),
        "cells_recorded": len(records),
        "cells_excluded": [r for r in records if r.get("excluded")],
        "methods_inapplicable_by_name": {
            name: sum(1 for r in records if not r["methods"].get(name, {}).get("applicable", True))
            for name in METHODS
        },
        "methods_budget_exhausted_by_name": {
            name: sum(1 for r in records if r["methods"].get(name, {}).get("budget_exhausted"))
            for name in METHODS
        },
        "methods_over_budget_by_name": {
            name: sum(1 for r in records if r["methods"].get(name, {}).get("over_budget"))
            for name in METHODS
        },
        "methods_errored_by_name": {
            name: sum(1 for r in records if "error" in r["methods"].get(name, {}))
            for name in METHODS
        },
        "seconds": round(time.monotonic() - started, 2),
        "workers": arguments.workers,
        "env": env,
    }
    (directory / f"sweep_manifest_{arguments.grid}{suffix}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"\n{manifest['cells_recorded']} of {manifest['cells_planned']} cells recorded")
    print(f"inapplicable per method: {manifest['methods_inapplicable_by_name']}")
    over = {k: v for k, v in manifest["methods_over_budget_by_name"].items() if v}
    if over:
        print(f"over budget per method: {over}")
    errored = {k: v for k, v in manifest["methods_errored_by_name"].items() if v}
    if errored:
        print(f"errored per method: {errored}")
    print(f"budget exhausted per method: {manifest['methods_budget_exhausted_by_name']}")
    print(f"manifest  {directory / f'sweep_manifest_{arguments.grid}{suffix}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
