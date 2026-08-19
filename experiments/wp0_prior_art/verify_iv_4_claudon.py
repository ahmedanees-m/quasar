"""WP0 T0.3: does the Claudon-Piquemal-Monmarche construction apply to our generator?

Execution plan v4 exists because this reference was found. Its section 0 argues that "the
quasispecies is the Perron (dominant) eigenvector of a non-conservative linear operator.
Extracting a dominant eigenvector is precisely what QSVT eigenvalue transforms do", and
Route B, the novelty core of the whole paper, is built on that.

The reference (arXiv:2501.05868, Nature Communications 16:10732, 2025) states its results
for **row-stochastic Markov kernels**, defines reversibility by detailed balance, and gets
its beyond-quadratic speedup specifically from the *absence* of reversibility. So two
questions decide whether Route B can be built as planned, and they are independent:

1. Is the mutation-selection generator a Markov kernel? If not, their theorems do not apply
   to it directly.
2. Is it nonreversible? If it is reversible, then even after any conversion the speedup they
   exploit is not the one available.

This script answers both by measurement across the operator families that matter, including
two biologically motivated generalisations that the project has not implemented yet, to find
out whether nonreversibility is reachable at all within this problem class.

    python experiments/wp0_prior_art/verify_iv_4_claudon.py
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
    """Context factor applied to both directions: the control that stays reversible.

    Written here rather than in the package because it exists only to evidence a negative,
    and nothing in the project should be able to build it by accident.
    """
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
        # What QUASAR actually implements: symmetric, independent per-site mutation.
        "symmetric": mutation_generator(N_SITES, MU),
        # Biologically real: transition and transversion rates differ, strand asymmetry.
        "asymmetric": mutation_generator(N_SITES, MU, mu_backward=0.1 * MU),
        # Also biologically real: CpG hypermutation, APOBEC motif preference. The forward
        # flip rate at a site depends on its neighbour, so mutation is no longer a product
        # of independent per-site processes. Note the asymmetry is essential: a context
        # factor applied to both directions cancels out of Kolmogorov's condition and leaves
        # the chain reversible. That was measured before it was believed.
        "context_dependent": mutation_generator(N_SITES, MU, context_strength=1.5),
        "asymmetric_context_dependent": mutation_generator(
            N_SITES, MU, mu_backward=0.1 * MU, context_strength=1.5
        ),
        # Control: the same context factor applied to both directions, which is the version
        # that stays reversible. Kept in the record so the distinction is evidenced rather
        # than only described.
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

    # This is a verification, not a pass-or-fail gate. It "passes" when it has actually
    # decided the question in both directions: the implemented model is classified, and
    # whether nonreversibility is reachable at all within the problem class is classified.
    decided = bool(len(cases) == 20 and len(implemented) == 3)
    return decided, measured, cases


def main() -> int:
    decided, measured, cases = run()

    path = write_gate_record(
        gate="PRIOR-ART-IV.4",
        work_package="wp0",
        threshold={
            "statistic": "classification, not a numerical threshold",
            "question": "does the Claudon-Piquemal-Monmarche (2025) construction apply to "
            "the mutation-selection generator as execution plan v4 assumes",
            "registered_in": "docs/protocol.md section 4 (G-0), docs/references.md entry IV.4",
        },
        measured=measured,
        passed=decided,
        cases=cases,
        notes=(
            "The reference states its results for row-stochastic Markov kernels and gets "
            "its beyond-quadratic speedup from nonreversibility. This run measures both "
            "properties for the generator QUASAR implements and for two biologically "
            "motivated generalisations, to establish whether nonreversibility is reachable "
            "within this problem class at all."
        ),
    )

    print(f"Prior art IV.4 verification: {len(cases)} operators in {measured['seconds']} s\n")
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
