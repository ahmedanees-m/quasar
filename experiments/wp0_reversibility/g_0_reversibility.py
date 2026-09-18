"""Conservation and reversibility of the mutation-selection generator.

Speedups for nonreversible Markov chains, such as that of Claudon, Piquemal and Monmarche
(arXiv:2501.05868, Nature Communications 16:10732, 2025), are stated for row-stochastic kernels
and rely on the absence of detailed balance. Two properties of the generator are therefore
classified:

1. Whether it is conservative, that is, a Markov generator.
2. Whether it is reversible.

Both are evaluated for the symmetric per-site mutation model, for asymmetric mutation, and for
context-dependent mutation, across four fitness landscapes.

    python experiments/wp0_reversibility/g_0_reversibility.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

from quasarstack.classical.landscapes import additive_fitness, class_fitness, single_peak_classes
from quasarstack.io.store import write_gate_record
from quasarstack.spectral.perron import (
    mutation_generator,
    reversibility_report,
    selection_generator,
)

N_SITES = 5
MU = 0.3


def _two_sided_context_mutation(n_sites: int, mu: float, strength: float) -> np.ndarray:
    """Context factor applied to both directions, a control that stays reversible."""
    dim = 1 << n_sites
    operator = np.zeros((dim, dim))
    for source in range(dim):
        for site in range(n_sites):
            rate = mu
            if source >> ((site - 1) % n_sites) & 1:
                rate *= 1.0 + strength
            target = source ^ (1 << site)
            operator[target, source] += rate
            operator[source, source] -= rate
    return operator


def _fitness_families(n_sites: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    coupling = np.zeros((n_sites, n_sites))
    coupling[0, 1] = 0.8
    coupling[1, 3] = -0.6
    return {
        "flat": np.zeros(1 << n_sites),
        "additive": additive_fitness(rng.uniform(0.25, 2.0, size=n_sites)),
        "additive_epistatic": additive_fitness(rng.uniform(0.25, 2.0, size=n_sites), coupling),
        "single_peak": class_fitness(single_peak_classes(n_sites, 2.0)),
    }


def run() -> tuple[bool, dict, list[dict]]:
    started = time.monotonic()
    cases: list[dict] = []
    fitness_families = _fitness_families(N_SITES)

    mutation_variants = {
        # The implemented model: symmetric, independent per-site mutation.
        "symmetric": mutation_generator(N_SITES, MU),
        # Different forward and backward rates.
        "asymmetric": mutation_generator(N_SITES, MU, mu_backward=0.1 * MU),
        # Context-dependent forward rate, as in CpG hypermutation or APOBEC motif preference:
        # the forward flip rate at a site depends on its neighbour.
        "context_dependent": mutation_generator(N_SITES, MU, context_strength=1.5),
        "asymmetric_context_dependent": mutation_generator(
            N_SITES, MU, mu_backward=0.1 * MU, context_strength=1.5
        ),
        # Control: the same context factor in both directions, which cancels from Kolmogorov's
        # cycle condition.
        "context_symmetric_control": _two_sided_context_mutation(N_SITES, MU, 1.5),
    }

    for mutation_name, mutation in mutation_variants.items():
        for fitness_name, fitness in fitness_families.items():
            generator = mutation + selection_generator(fitness)
            report = reversibility_report(generator)
            cases.append(
                {
                    "mutation": mutation_name,
                    "fitness": fitness_name,
                    "L": N_SITES,
                    "mu": MU,
                    **report,
                }
            )

    elapsed = time.monotonic() - started

    implemented = [c for c in cases if c["mutation"] == "symmetric" and c["fitness"] != "flat"]
    reachable_nonreversible = [c for c in cases if not c["is_reversible"]]

    measured = {
        "n_operators_tested": len(cases),
        "implemented_model_is_conservative": all(c["is_conservative"] for c in implemented),
        "implemented_model_is_symmetric": all(c["is_symmetric"] for c in implemented),
        "implemented_model_is_reversible": all(c["is_reversible"] for c in implemented),
        "max_reversibility_defect_over_independent_mutation": max(
            c["reversibility_defect"] for c in cases if c["mutation"] in {"symmetric", "asymmetric"}
        ),
        "nonreversible_variants_found": sorted({c["mutation"] for c in reachable_nonreversible}),
        "n_nonreversible_cases": len(reachable_nonreversible),
        "seconds": round(elapsed, 2),
    }

    # passed records that every operator was classified.
    decided = bool(len(cases) == 20 and len(implemented) == 3)
    return decided, measured, cases


def main() -> int:
    decided, measured, cases = run()

    path = write_gate_record(
        gate="WP0-REVERSIBILITY",
        work_package="wp0",
        threshold={
            "statistic": "classification, not a numerical threshold",
            "question": "whether the mutation-selection generator is conservative and "
            "whether it is reversible",
        },
        measured=measured,
        passed=decided,
        cases=cases,
        notes=(
            "Claudon, Piquemal and Monmarche (2025) state their results for row-stochastic "
            "Markov kernels, with the speedup coming from nonreversibility. Both properties "
            "are classified for the implemented generator and for asymmetric and "
            "context-dependent mutation."
        ),
    )

    print(f"Reversibility: {len(cases)} operators in {measured['seconds']} s\n")
    header = f"{'mutation':30s} {'fitness':20s} {'conserv':>8s} {'symm':>6s} {'revers':>7s} {'defect':>10s}"
    print(header)
    print("-" * len(header))
    for case in cases:
        print(
            f"{case['mutation']:30s} {case['fitness']:20s} "
            f"{str(case['is_conservative']):>8s} {str(case['is_symmetric']):>6s} "
            f"{str(case['is_reversible']):>7s} {case['reversibility_defect']:>10.2e}"
        )

    print()
    print(f"  implemented model conservative : {measured['implemented_model_is_conservative']}")
    print(f"  implemented model symmetric    : {measured['implemented_model_is_symmetric']}")
    print(f"  implemented model reversible   : {measured['implemented_model_is_reversible']}")
    print(f"  nonreversible variants         : {measured['nonreversible_variants_found']}")
    print(f"  record                         : {path.relative_to(path.parents[2])}")
    return 0 if decided else 1


if __name__ == "__main__":
    sys.exit(main())
