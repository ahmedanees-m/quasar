"""G-R.5: rugged epistatic landscapes, against brute-force exact diagonalisation.

Every earlier gate used a landscape with structure to exploit: additive, permutation
symmetric, or a sharp peak. This one uses NK landscapes, where the Pauli decomposition is
dense and there is nothing to exploit, which is the case the compiler has not yet had to
survive.

Two routes are compared for the gate and they share no code: the compiler goes through a
Walsh-Hadamard decomposition into Pauli terms and back through Qiskit, while the reference
assembles the sparse generator directly from the fitness vector.

A third route, Trotterised imaginary-time evolution, runs on the L = 8 instances as a
diagnostic. It is the first test of whether imaginary time actually converges on a rugged
landscape at a fixed budget, and it is deliberately not a pass condition: an instance with a
small gap may fail to converge, and that is a finding for G-R.6, G-R.7 and WP7, not a defect.

Thresholds and instances are in GATES.md section 3 and Amendment 5, committed before this
ran.

    python experiments/wp_r_rebuild/g_r_5_rugged.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

from quasarstack.analytic.exact_diag import perron_vector
from quasarstack.circuit.trotter_ite import evolve
from quasarstack.classical.landscapes import nk_fitness, ruggedness_statistics
from quasarstack.hamiltonian.builder import diagonal_hamiltonian, ground_state, pauli_term_count
from quasarstack.io.store import write_gate_record
from quasarstack.scoring.metrics import score

# Registered in GATES.md section 3.
THRESHOLD = 0.99999

# Registered in GATES.md Amendment 5.
MU = 0.25
SEEDS = list(range(10))
CELLS = [(6, 1), (6, 2), (6, 4), (8, 1), (8, 2), (8, 4), (8, 7), (10, 1), (10, 2), (10, 4)]
TROTTER_SIZE = 8
TAU = 60.0
DTAU = 0.01


def run() -> tuple[bool, dict, list[dict]]:
    started = time.monotonic()
    cases: list[dict] = []

    for n_sites, k in CELLS:
        for seed in SEEDS:
            fitness = nk_fitness(n_sites, k, seed=seed)

            reference, _, gap = perron_vector(fitness, MU)
            hamiltonian = diagonal_hamiltonian(fitness, MU)
            compiled, _ = ground_state(hamiltonian)
            scores = score(compiled, reference)

            stats = ruggedness_statistics(fitness)

            trotter: dict[str, float] | None = None
            if n_sites == TROTTER_SIZE:
                probs, _ = evolve(fitness, MU, TAU, DTAU)
                trotter = score(probs, reference)

            cases.append(
                {
                    "L": n_sites,
                    "K": k,
                    "seed": seed,
                    "mu": MU,
                    "cosine": scores["cosine"],
                    "tv": scores["tv"],
                    "spectral_gap": gap,
                    "pauli_terms": pauli_term_count(hamiltonian),
                    "trotter_cosine": None if trotter is None else trotter["cosine"],
                    "trotter_tv": None if trotter is None else trotter["tv"],
                    **stats,
                }
            )

    elapsed = time.monotonic() - started
    cosines = np.array([c["cosine"] for c in cases])
    worst = int(np.argmin(cosines))

    # WP3 task T3.3, pulled forward: does ruggedness rise monotonically with K?
    by_k: dict[int, dict[str, float]] = {}
    for k in sorted({c["K"] for c in cases}):
        subset = [c for c in cases if c["K"] == k and c["L"] == TROTTER_SIZE]
        if not subset:
            continue
        by_k[k] = {
            "mean_local_optima": float(np.mean([c["n_local_optima"] for c in subset])),
            "mean_autocorrelation": float(np.mean([c["autocorrelation"] for c in subset])),
            "mean_pauli_terms": float(np.mean([c["pauli_terms"] for c in subset])),
            "mean_optimum_hamming_weight": float(
                np.mean([c["optimum_hamming_weight"] for c in subset])
            ),
            "mean_spectral_gap": float(np.mean([c["spectral_gap"] for c in subset])),
        }
    ks = sorted(by_k)
    optima_monotone = all(
        by_k[a]["mean_local_optima"] <= by_k[b]["mean_local_optima"]
        for a, b in zip(ks, ks[1:], strict=False)
    )
    autocorrelation_monotone = all(
        by_k[a]["mean_autocorrelation"] >= by_k[b]["mean_autocorrelation"]
        for a, b in zip(ks, ks[1:], strict=False)
    )

    trotter_cases = [c for c in cases if c["trotter_cosine"] is not None]
    trotter_cosines = np.array([c["trotter_cosine"] for c in trotter_cases])

    measured = {
        "n_instances": len(cases),
        "min_cosine": float(cosines.min()),
        "max_one_minus_cosine": float((1.0 - cosines).max()),
        "max_tv": float(max(c["tv"] for c in cases)),
        "n_below_threshold": int((cosines < THRESHOLD).sum()),
        "worst_case": cases[worst],
        "min_spectral_gap": float(min(c["spectral_gap"] for c in cases)),
        "ruggedness_by_k_at_L8": by_k,
        "local_optima_monotone_in_k": optima_monotone,
        "autocorrelation_monotone_in_k": autocorrelation_monotone,
        "trotter": {
            "n_instances": len(trotter_cases),
            "min_cosine": float(trotter_cosines.min()),
            "mean_cosine": float(trotter_cosines.mean()),
            "n_below_gate_threshold": int((trotter_cosines < THRESHOLD).sum()),
            "worst": min(trotter_cases, key=lambda c: c["trotter_cosine"]),
        },
        "seconds": round(elapsed, 2),
    }
    return bool(cosines.min() >= THRESHOLD), measured, cases


def main() -> int:
    passed, measured, cases = run()

    path = write_gate_record(
        gate="G-R.5",
        work_package="wp_r",
        threshold={
            "statistic": "cosine between the compiled Pauli Hamiltonian's ground state and "
            "brute-force exact diagonalisation, on every NK instance",
            "value": THRESHOLD,
            "registered_in": "GATES.md section 3, instance set in Amendment 5",
        },
        measured=measured,
        passed=passed,
        cases=cases,
        notes=(
            "NK landscapes have no master sequence: the global optimum sits at a random "
            "genotype near Hamming weight L/2, which is recorded per instance as ADR-0011 "
            "requires. Statements about the error threshold do not carry over to this "
            "family unchanged. The Trotterised route is a diagnostic and not a pass "
            "condition, because a rugged instance with a small gap may legitimately fail to "
            "converge at a fixed imaginary-time budget."
        ),
    )

    print(f"G-R.5: {len(cases)} NK instances in {measured['seconds']} s\n")
    print(f"  min cosine        {measured['min_cosine']:.15f}  (threshold {THRESHOLD})")
    print(f"  max 1 - cosine    {measured['max_one_minus_cosine']:.3e}")
    print(f"  max TV            {measured['max_tv']:.3e}")
    print(f"  instances failing {measured['n_below_threshold']}")
    print(f"  min spectral gap  {measured['min_spectral_gap']:.3e}")

    print("\nruggedness against K at L = 8, mean over 10 seeds:")
    header = f"  {'K':>2s} {'local optima':>13s} {'autocorr':>9s} {'pauli terms':>12s} {'opt weight':>11s} {'gap':>8s}"
    print(header)
    for k, row in measured["ruggedness_by_k_at_L8"].items():
        print(
            f"  {k:>2d} {row['mean_local_optima']:>13.1f} {row['mean_autocorrelation']:>9.3f} "
            f"{row['mean_pauli_terms']:>12.1f} {row['mean_optimum_hamming_weight']:>11.1f} "
            f"{row['mean_spectral_gap']:>8.4f}"
        )
    print(f"  local optima rise with K:      {measured['local_optima_monotone_in_k']}")
    print(f"  autocorrelation falls with K:  {measured['autocorrelation_monotone_in_k']}")

    trotter = measured["trotter"]
    print(f"\nTrotterised imaginary time on {trotter['n_instances']} instances (diagnostic):")
    print(f"  mean cosine {trotter['mean_cosine']:.7f}, min {trotter['min_cosine']:.7f}")
    print(
        f"  below the gate threshold: {trotter['n_below_gate_threshold']} "
        f"of {trotter['n_instances']}"
    )
    worst = trotter["worst"]
    print(
        f"  worst: L={worst['L']} K={worst['K']} seed={worst['seed']} "
        f"cosine={worst['trotter_cosine']:.7f} gap={worst['spectral_gap']:.4f}"
    )

    print(f"\n  record  {path.relative_to(path.parents[2])}")
    print(f"  {'PASS' if passed else 'FAIL'}")
    if not passed:
        print(f"  worst case: {measured['worst_case']}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
