"""Resumable grid sweep for WP7's boundary map.

The sweep is the most expensive thing the project runs and the one whose integrity is
easiest to lose, so the design is driven by four failure modes rather than by convenience.

**A cell that was never run must not look like a cell that scored badly.** Every cell in the
declared grid ends up in the manifest either scored or excluded with a reason, which is claim
C27. There is no third state and no silent gap.

**A method that ran out of time must not look like a method that was wrong.** Section 11.3
denominates the budget in wall-clock seconds, and revision 19's second addendum measured
Baseline C spending ten to twenty times more of it on the dense-operator families than on the
additive one. So allotted and used seconds are recorded per cell per method, and a method that
hit its ceiling is marked `budget_exhausted` rather than scored as inaccurate.

**Interruption must not corrupt what came before.** Results append to a JSONL as they are
produced and a restart skips what is already there. The alternative, holding everything in
memory and writing at the end, loses a multi-hour run to one exception and quietly tempts
whoever restarts it to narrow the grid.

**The order parameter must mean the same thing in every cell.** ADR-0017 and revision 18:
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
from quasarstack.classical.mps_ite import evolve as mps_evolve  # noqa: E402
from quasarstack.classical.wright_fisher import sample_stationary  # noqa: E402
from quasarstack.hamiltonian.builder import diagonal_hamiltonian  # noqa: E402
from quasarstack.io.store import environment, evidence_directory  # noqa: E402
from quasarstack.ite.varqite import Ansatz  # noqa: E402
from quasarstack.ite.varqite import evolve as varqite_evolve  # noqa: E402
from quasarstack.qsvt.block_encoding import one_norm  # noqa: E402
from quasarstack.qsvt.filter import filtered_state  # noqa: E402
from quasarstack.spectral.order_parameter import localisation  # noqa: E402

# Registered in GATES.md section 11.1, with revision 18's split of the ruggedness axis.
REGISTERED_GRID = {
    # revision 20: section 11.1's grid is 3108 cells and about 294 hours at measured cost.
    # This covers the same families and mutation range at 7 points instead of 21, and defers
    # L = 14, for 777 cells and about 27 hours. The seed count meets the registered minimum.
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

# A grid big enough to exercise the boundary-map figure without being the registered sweep.
# One size, one seed, the full mutation axis, and one family from each of the three axes
# revision 18 separates, so the figure has something with structure in it to draw.
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

# Section 11.3. Wall-clock seconds per cell per method.
BUDGET_SECONDS = {8: 300.0, 10: 300.0, 12: 300.0, 14: 900.0}
REFERENCE_DENSE_LIMIT = 10

# Registered in revision 21. Route A is measured, not assumed, to be unable to finish a
# cell within section 11.3's allotment anywhere on this grid: at L = 6 it used 198 to 235 s
# of 300, and its cost scales as n_parameters^2 * 2^L with n_parameters = L(L+3), so L = 8 is
# about ten times that. Running it on all 777 cells would spend 65 hours confirming
# budget exhaustion. It runs on a declared probe instead, and the exhaustion is the result.
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
    raise ValueError(f"unregistered family {family!r}")


def threshold_for(spec: dict[str, Any], fitness: np.ndarray, n_sites: int) -> float:
    """mu_c per instance. revision 12's addendum defines the non-peak case."""
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
    """Baseline A. One run at the largest declared population, and **it cannot use more
    budget than that**, which is a fairness fact WP7 has to carry.

    Section 11.3 says Wright-Fisher spends its budget on samples. Measured at L = 10 on an
    NK K = 2 cell, three ways of spending it:

    | | seconds | cosine | total variation |
    |---|---|---|---|
    | ladder over N = 1e3 to 1e6, 3000 generations each | 59.1 | 0.999830 | 9.35e-3 |
    | N = 1e6 once, 3000 generations | **23.0** | 0.999830 | 9.35e-3 |
    | N = 1e6 once, 12000 generations | 105.8 | 0.999735 | 1.06e-2 |

    The ladder costs 2.6 times more for a **bit-identical** answer, because a generation in
    genotype-count space is ``O(L 2^L)`` and independent of N, so the small-population runs
    are pure waste rather than a cheap approximation being refined.

    And the third row is the one that matters for the budget protocol: **more generations
    makes it worse.** Genetic drift is injected once per generation, so a longer chain
    accumulates noise faster than time-averaging removes it. This is finding 4.11 of the
    project record showing up again in a different place.

    So Baseline A's accuracy here is set by a drift floor and not by compute, and handing it
    a larger allotment cannot move it. Section 11.3's protocol assumes methods improve with
    budget; this one does not, and a WP7 cell where Baseline A looks weak is a cell where it
    is at its floor rather than one where it was starved.
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
            "budget_is_not_the_binding_constraint": True,
        },
    }


def in_route_a_probe(cell: dict[str, Any]) -> bool:
    """Is this cell in the declared Route A feasibility probe? See revision 21."""
    return (
        cell["L"] == ROUTE_A_PROBE_SIZE
        and cell.get("seed", 0) == ROUTE_A_PROBE_SEED
        and any(abs(cell["mu_over_mu_c"] - r) < 1e-9 for r in ROUTE_A_PROBE_MU)
    )


def method_route_a(fitness: np.ndarray, mu: float, budget: float) -> dict[str, Any]:
    """Route A, varQITE, spending its budget on imaginary time.

    The expensive route. Each step solves the McLachlan system, whose geometric tensor costs
    `O(n_parameters^2 * 2^L)`, and the ansatz carries `L * (reps + 1)` parameters. A cell
    that cannot finish inside its allotment is recorded as `budget_exhausted` rather than
    scored as inaccurate, which matters here more than for the baselines: G-7's positive
    result requires a quantum route to *succeed* where the tensor network fails, and a
    quantum route that merely ran out of time must not be read as either.
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
    """Route B, QSVT eigenstate filtering, spending its budget on polynomial degree.

    Scored as the filter applied to the operator, which is what the circuit's block encoding
    implements and what G-2 verified to 1.7e-12. Simulating the circuit itself at every cell
    is not affordable and would measure Qiskit rather than the method.
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

# Selected per invocation. G-7's decision compares a quantum route against the
# compute-matched tensor network, so a sweep holding only the baselines cannot answer it
# either way, which is what the first pass of this runner did.
METHODS: dict[str, Callable[[np.ndarray, float, float], dict[str, Any]]] = dict(CLASSICAL_METHODS)


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
        if name == "route_a_varqite" and not in_route_a_probe(cell):
            record["methods"][name] = {
                "applicable": False,
                "reason": "outside the declared Route A feasibility probe, revision 21",
            }
            continue
        try:
            outcome = method(fitness, mu, budget)
        except Exception as error:  # a method failing must not lose the cell
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
            # Stopped early because the clock ran out.
            "budget_exhausted": bool(outcome.get("budget_exhausted", False)),
            # Ran past the clock. Every method here checks its budget *between* units of
            # work, so a single unit that overruns is never caught by that check: Route A
            # converged on its first rung at L = 8 after 510 s of a 300 s allotment and
            # reported itself unexhausted. Section 11.3 calls the budget a fairness
            # firewall, and a cell where one method was allowed 1.7 times its allotment is
            # not a fair comparison. Recorded per cell so the sweep cannot be read as
            # compute-matched where it was not.
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
    parser.add_argument("--grid", choices=["smoke", "local", "registered"], default="smoke")
    parser.add_argument(
        "--methods",
        choices=["classical", "quantum", "all"],
        default="classical",
        help="which method set to run; streams are kept separate so a pass can be "
        "added without recomputing one already done",
    )
    parser.add_argument("--list", action="store_true", help="show the cells and stop")
    arguments = parser.parse_args()

    grid = {"smoke": SMOKE_GRID, "local": LOCAL_GRID, "registered": REGISTERED_GRID}[arguments.grid]
    global METHODS
    METHODS = {
        "classical": CLASSICAL_METHODS,
        "quantum": QUANTUM_METHODS,
        "all": {**CLASSICAL_METHODS, **QUANTUM_METHODS},
    }[arguments.methods]
    planned = list(cells(grid))

    if arguments.list:
        for cell in planned:
            print(cell_key(cell))
        print(f"{len(planned)} cells")
        return 0

    # ADR-0012, via the shared guard in quasarstack.io.store. This script had its own copy
    # and the G-7 scorer had none; the helper exists so a third writer cannot forget.
    env = environment()
    directory = evidence_directory(f"wp{arguments.wp}")
    suffix = "" if arguments.methods == "classical" else f"_{arguments.methods}"
    stream = directory / f"sweep_{arguments.grid}{suffix}.jsonl"

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
        "methods": arguments.methods,
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
        # A method that raises in every cell would otherwise be a silent column of nulls:
        # run_cell catches so one failure cannot lose a cell, and that same catch turns a
        # method broken everywhere into a sweep that looks complete. Counted, and the
        # printout calls it out.
        "methods_over_budget_by_name": {
            name: sum(1 for r in records if r["methods"].get(name, {}).get("over_budget"))
            for name in METHODS
        },
        "methods_errored_by_name": {
            name: sum(1 for r in records if "error" in r["methods"].get(name, {}))
            for name in METHODS
        },
        "seconds": round(time.monotonic() - started, 2),
        "env": env,
    }
    (directory / f"sweep_manifest_{arguments.grid}{suffix}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"\n{manifest['cells_recorded']} of {manifest['cells_planned']} cells recorded")
    print(f"inapplicable per method: {manifest['methods_inapplicable_by_name']}")
    over = {k: v for k, v in manifest["methods_over_budget_by_name"].items() if v}
    if over:
        print(f"OVER BUDGET per method: {over}  <- those cells are not compute-matched")
    errored = {k: v for k, v in manifest["methods_errored_by_name"].items() if v}
    if errored:
        print(f"ERRORED per method: {errored}  <- these cells carry no score")
    print(f"budget exhausted per method: {manifest['methods_budget_exhausted_by_name']}")
    print(f"manifest  {directory / f'sweep_manifest_{arguments.grid}{suffix}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
