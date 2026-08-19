"""G-6.3: MPO bond dimension per landscape family. WP6 task T6.3, claim C25.

`docs/protocol.md` section 10 criterion 3 asks that MPO bond dimension be reported per family, that
at least two site orderings be tested on the non-local families with the better one used, and
that **where MPS is structurally disadvantaged the disadvantage is stated in the results, not
exploited**. This answers all three exactly and cheaply, because for a diagonal operator the
bond dimension across a cut is the rank of the fitness matricised across it.

**This is criterion 3 only. G-6 as a whole is not claimed**: criteria 1, 2 and 4 need actual
matrix-product imaginary-time evolution, which is not built yet.

One caveat travels with every number here. This is the bond dimension of the **operator**,
which sets the cost of applying one step. It is not the bond dimension of the **state**, which
sets whether the quasispecies can be represented at all. A family with a low-rank operator can
still have a state that is hard to hold, and the converse. The second half is what criteria 1
and 2 measure.

    python experiments/wp6_mps/mpo_analysis.py
"""

from __future__ import annotations

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
from quasarstack.classical.mpo_analysis import compare_orderings, mpo_bond_dimensions
from quasarstack.hamiltonian.builder import diagonal_hamiltonian, pauli_term_count
from quasarstack.io.store import write_gate_record

SIZES = [8, 10, 12, 14]
SEEDS = list(range(5))
MU = 0.20
PAULI_UP_TO = 12
NON_LOCAL = {"nk", "spin_glass", "block", "house_of_cards"}


def families(n_sites: int):
    rng = np.random.default_rng(7000 + n_sites)
    yield {"family": "additive"}, additive_fitness(rng.uniform(0.3, 1.5, size=n_sites))
    yield {"family": "single_peak"}, class_fitness(single_peak_classes(n_sites, 1.0))
    yield (
        {"family": "additive_pairwise"},
        class_fitness(pairwise_uniform_classes(n_sites, 1.0, 0.1)),
    )
    for k in (1, 2, 4):
        if k <= n_sites - 1:
            for seed in SEEDS:
                yield {"family": "nk", "K": k, "seed": seed}, nk_fitness(n_sites, k, seed=seed)
    for seed in SEEDS:
        yield {"family": "spin_glass", "seed": seed}, spin_glass_fitness(n_sites, seed=seed)
        yield {"family": "house_of_cards", "seed": seed}, house_of_cards_fitness(n_sites, seed=seed)
        for size in (2, 4):
            yield (
                {"family": "block", "block_size": size, "seed": seed},
                block_fitness(n_sites, size, seed=seed),
            )
    for roughness in (0.1, 0.5, 1.0):
        for seed in SEEDS:
            yield (
                {"family": "rough_mount_fuji", "roughness": roughness, "seed": seed},
                rough_mount_fuji_fitness(n_sites, seed=seed, roughness=roughness),
            )


def run() -> tuple[bool, dict, list[dict]]:
    """Measure, separately from reporting, so the summary can be replayed in a test.

    Split out after G-5 passed its science and then died formatting the result. The
    replay in tests/regression/test_gate_reporting.py can only reach gates shaped this
    way, and this one was one of the two it could not cover.
    """
    started = time.monotonic()
    cases: list[dict] = []

    for n_sites in SIZES:
        orderings = {
            "identity": list(range(n_sites)),
            "even_then_odd": list(range(0, n_sites, 2)) + list(range(1, n_sites, 2)),
        }
        for label, fitness in families(n_sites):
            report = mpo_bond_dimensions(fitness)
            row = {
                **label,
                "L": n_sites,
                "mpo_bond_dimension": report["middle_cut_bond_dimension"],
                "middle_cut_ceiling": report["middle_cut_ceiling"],
                "fraction_of_ceiling": report["middle_cut_fraction_of_ceiling"],
                "saturates_the_ceiling": report["saturates_the_ceiling"],
                # Pauli count only where building the operator is cheap.
                "pauli_terms": (
                    pauli_term_count(diagonal_hamiltonian(fitness, MU))
                    if n_sites <= PAULI_UP_TO
                    else None
                ),
            }
            if label["family"] in NON_LOCAL:
                comparison = compare_orderings(fitness, orderings)
                row["by_ordering"] = comparison["max_bond_dimension_by_ordering"]
                row["best_ordering"] = comparison["best_ordering"]
                row["ordering_ratio"] = comparison["best_over_worst_ratio"]
            cases.append(row)

    saturating = sorted({c["family"] for c in cases if c["saturates_the_ceiling"]})
    ordering_effects = [c["ordering_ratio"] for c in cases if "ordering_ratio" in c]

    # Is there any family cheap for the circuit and expensive for MPS? That is the only
    # shape in which a quantum advantage over this baseline could appear at all.
    at_pauli_size = [c for c in cases if c["L"] == PAULI_UP_TO and c["pauli_terms"]]
    cheap_circuit_hard_mps = [
        {k: c[k] for k in ("family", "seed", "pauli_terms", "mpo_bond_dimension") if k in c}
        for c in at_pauli_size
        if c["pauli_terms"] < (1 << PAULI_UP_TO) // 16 and c["saturates_the_ceiling"]
    ]

    measured = {
        "criterion_3_only": True,
        "gate_6_claimed": False,
        "n_cases": len(cases),
        "families_saturating_the_ceiling": saturating,
        "max_ordering_ratio": max(ordering_effects) if ordering_effects else 1.0,
        "mean_ordering_ratio": float(np.mean(ordering_effects)) if ordering_effects else 1.0,
        "families_cheap_for_the_circuit_and_hard_for_mps": cheap_circuit_hard_mps,
        "caveat": (
            "This is the bond dimension of the operator, which sets the cost of one step, "
            "not of the state, which sets whether the quasispecies is representable. "
            "Criteria 1, 2 and 4 need matrix-product imaginary-time evolution and are not "
            "built yet, so G-6 as a whole is not claimed."
        ),
        "seconds": round(time.monotonic() - started, 2),
    }

    return True, measured, cases


def main() -> int:
    passed, measured, cases = run()

    path = write_gate_record(
        gate="G-6.3",
        work_package="wp6",
        threshold={
            "statistic": "MPO bond dimension reported per family, at least two site "
            "orderings on the non-local families, structural disadvantage stated",
            "registered_in": "docs/protocol.md section 10 criterion 3",
        },
        measured=measured,
        passed=passed,
        cases=cases,
        notes=measured["caveat"],
    )

    print(f"G-6.3: {len(cases)} cases in {measured['seconds']} s\n")
    header = f"  {'family':24s} {'L':>3} {'chi_MPO':>8} {'% ceil':>8} {'pauli':>7}"
    print(header)
    seen = set()
    for case in cases:
        key = (
            case["family"],
            case.get("K"),
            case.get("roughness"),
            case.get("block_size"),
            case["L"],
        )
        if key in seen:
            continue
        seen.add(key)
        suffix = "".join(f" {k}={case[k]}" for k in ("K", "roughness", "block_size") if k in case)
        name = case["family"] + suffix
        print(
            f"  {name:24s} {case['L']:>3} {case['mpo_bond_dimension']:>8} "
            f"{100 * case['fraction_of_ceiling']:>7.1f}% {str(case['pauli_terms']):>7}"
        )

    # Read back out of measured and cases rather than kept as locals, so this block works on
    # a record loaded from disk exactly as it does on one just computed.
    at_pauli_size = [c for c in cases if c["L"] == PAULI_UP_TO and c["pauli_terms"]]
    cheap_circuit_hard_mps = measured["families_cheap_for_the_circuit_and_hard_for_mps"]
    print(f"\n  families saturating the ceiling: {measured['families_saturating_the_ceiling']}")
    print(f"  worst site-ordering penalty:     {measured['max_ordering_ratio']:.2f}x")
    print(
        f"  cheap for the circuit and hard for MPS: "
        f"{len(cheap_circuit_hard_mps)} of {len(at_pauli_size)} at L = {PAULI_UP_TO}"
    )
    print(f"\n  record  {path}")
    print(
        "  G-6.3 recorded. G-6 as a whole NOT claimed: criteria 1, 2 and 4 need MPS "
        "evolution, which is not built."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
