"""G-R.10: the sparse representation costs far fewer Pauli terms than the projector.

The planning documents record this as the finding that the wall for dense landscapes is the
Pauli-term count rather than the circuit depth, and put the ratio at 152 times at L = 12.

What is actually being compared needs saying precisely, because the two forms describe
*different landscapes*, not two encodings of one. The single-peak projector puts fitness on
one genotype and nothing anywhere else, and having no structure it needs every Z subset:
2^L terms. The sparse additive-plus-epistasis form is what a real biological landscape looks
like, a few site effects and a few interactions, and it needs a number of terms linear in L
plus one per coupling. The claim is that the realistic case is exponentially cheaper than
the textbook one, not that a compiler found a clever encoding.

Thresholds are in docs/protocol.md section 3 and revision 9, committed before this ran.

    python experiments/wp_r_rebuild/g_r_10_pauli_count.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

from quasarstack.classical.landscapes import (
    class_fitness,
    nk_fitness,
    single_peak_classes,
)
from quasarstack.hamiltonian.builder import (
    additive_hamiltonian,
    diagonal_hamiltonian,
    pauli_term_count,
)
from quasarstack.io.store import write_gate_record

# Registered in docs/protocol.md section 3.
RATIO_THRESHOLD = 50.0
GATE_SIZE = 12

# Registered in docs/protocol.md revision 9.
MU = 0.20
SIZES = [4, 6, 8, 10, 12]
N_COUPLINGS = 2
NK_SIZES = [4, 6, 8]
NK_CONNECTIVITIES = [0, 1, 2, 4]


def _sparse_coefficients(n_sites: int) -> tuple[np.ndarray, np.ndarray]:
    """Additive fitness with a small fixed number of pairwise couplings.

    Deterministic, so the term count is a property of the family and not of a draw.
    """
    a = np.linspace(0.5, 1.5, n_sites)
    b = np.zeros((n_sites, n_sites))
    for index in range(min(N_COUPLINGS, n_sites - 1)):
        b[index, index + 1] = 0.4
    return a, b


def run() -> tuple[bool, dict, list[dict]]:
    started = time.monotonic()
    cases: list[dict] = []

    for n_sites in SIZES:
        a, b = _sparse_coefficients(n_sites)
        sparse_terms = pauli_term_count(additive_hamiltonian(a, MU, b))

        # The projector's decomposition is known exactly, so it is counted rather than
        # built: every Z subset, 2^L of them, with the identity absorbed into the mutation
        # identity, plus one transverse term per site. Building it at L = 12 would mean
        # materialising 4096 Pauli terms to count something arithmetic.
        projector_terms = 2**n_sites + n_sites
        if n_sites <= 8:
            built = pauli_term_count(
                diagonal_hamiltonian(class_fitness(single_peak_classes(n_sites, 1.0)), MU)
            )
            if built != projector_terms:
                raise AssertionError(
                    f"the projector term count formula is wrong at L = {n_sites}: "
                    f"predicted {projector_terms}, built {built}"
                )

        cases.append(
            {
                "L": n_sites,
                "mu": MU,
                "sparse_terms": int(sparse_terms),
                "projector_terms": int(projector_terms),
                "ratio": projector_terms / sparse_terms,
                "projector_count_verified_by_construction": n_sites <= 8,
            }
        )

    # Diagnostic: term count against ruggedness. The sparse form is cheap because the
    # landscape has structure, so the honest question is what happens when it does not.
    nk_terms = []
    for n_sites in NK_SIZES:
        for k in NK_CONNECTIVITIES:
            if k > n_sites - 1:
                continue
            terms = pauli_term_count(diagonal_hamiltonian(nk_fitness(n_sites, k, seed=0), MU))
            nk_terms.append(
                {
                    "L": n_sites,
                    "K": k,
                    "terms": int(terms),
                    "fraction_of_projector": terms / (2**n_sites + n_sites),
                }
            )

    elapsed = time.monotonic() - started
    gate_case = next(c for c in cases if c["L"] == GATE_SIZE)

    measured = {
        "gate_size": GATE_SIZE,
        "sparse_terms_at_gate_size": gate_case["sparse_terms"],
        "projector_terms_at_gate_size": gate_case["projector_terms"],
        "ratio_at_gate_size": gate_case["ratio"],
        "ratio_by_size": {c["L"]: c["ratio"] for c in cases},
        "nk_term_counts": nk_terms,
        "seconds": round(elapsed, 2),
    }
    return bool(gate_case["ratio"] >= RATIO_THRESHOLD), measured, cases


def main() -> int:
    passed, measured, cases = run()

    path = write_gate_record(
        gate="G-R.10",
        work_package="wp_r",
        threshold={
            "statistic": "ratio of Pauli terms, single-peak projector against the sparse "
            f"additive-plus-epistasis form, at L = {GATE_SIZE}",
            "value": RATIO_THRESHOLD,
            "registered_in": "docs/protocol.md section 3, families in revision 9",
        },
        measured=measured,
        passed=passed,
        cases=cases,
        notes=(
            "The two forms describe different landscapes rather than two encodings of one. "
            "The projector has no structure and needs every Z subset; the sparse form is "
            "what a real biological landscape looks like. The claim is that the realistic "
            "case is exponentially cheaper than the textbook one. The NK diagnostic shows "
            "what happens as structure is removed, which is the honest counterweight."
        ),
    )

    print(f"G-R.10: {len(cases)} sizes in {measured['seconds']} s\n")
    print(f"  {'L':>3s} {'sparse':>7s} {'projector':>10s} {'ratio':>9s} {'verified':>9s}")
    for case in cases:
        print(
            f"  {case['L']:>3d} {case['sparse_terms']:>7d} {case['projector_terms']:>10d} "
            f"{case['ratio']:>9.1f} "
            f"{str(case['projector_count_verified_by_construction']):>9s}"
        )

    print("\nterm count against ruggedness, as a fraction of the dense projector:")
    for row in measured["nk_term_counts"]:
        print(
            f"  L={row['L']:>2d} K={row['K']} terms={row['terms']:>5d} "
            f"fraction={row['fraction_of_projector']:.3f}"
        )

    print(
        f"\n  ratio at L = {GATE_SIZE}      {measured['ratio_at_gate_size']:.1f}  "
        f"(threshold {RATIO_THRESHOLD:.0f})"
    )
    print(f"  record                {path.relative_to(path.parents[2])}")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
