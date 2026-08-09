"""G-2: Route B, QSVT eigenstate filtering. WP2 tasks T2.2, T2.3, T2.4.

Built under ADR-0015's working assumption that Route B is ADR-0010 option C: eigenstate
filtering for a Hermitian stoquastic operator, not the nonreversible-Markov-chain
construction execution plan v4 originally cited. ADR-0010 records that the G-2 thresholds do
not depend on that choice.

Criteria, registered in `GATES.md` section 6 with configurations in Amendment 13:

1. Route B reproduces the analytic quasispecies at cosine >= 0.95 for L = 2..6.
2. The block encoding satisfies its defining property to 1e-10.
3. The derived degree agrees with the empirically sufficient degree within a factor of 2,
   using the WP1 gap map as input.

    python experiments/wp2_qsvt/g_2_route_b.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

from quasarstack.analytic.exact_diag import mutation_selection_generator
from quasarstack.classical.landscapes import (
    additive_fitness,
    class_fitness,
    pairwise_uniform_classes,
    single_peak_classes,
)
from quasarstack.hamiltonian.builder import additive_hamiltonian, diagonal_hamiltonian
from quasarstack.io.store import write_gate_record
from quasarstack.qsvt.block_encoding import (
    circuit_is_unitary,
    encoding_qubit_count,
    lcu_block_encoding,
    one_norm,
    verify_block_encoding,
)
from quasarstack.qsvt.filter import (
    filtered_state,
    predicted_degree,
    smallest_sufficient_degree,
)
from quasarstack.qsvt.qubitisation import verify_chebyshev

# Registered in GATES.md section 6 and Amendment 13.
COSINE_THRESHOLD = 0.95
BLOCK_ENCODING_TOLERANCE = 1e-10
DEGREE_AGREEMENT_FACTOR = 2.0
SIZES = [2, 3, 4, 5, 6]
MU = 0.20
ADDITIVE_SEEDS = list(range(10))
PEAK_HEIGHTS = [1.0, 2.5]
EPISTASIS_B = [0.1]
CHEBYSHEV_DEGREES = [0, 1, 2, 3, 5, 8]
UNITARITY_CHECK_UP_TO_QUBITS = 11
# Registered in Amendment 17. Verification costs 2**n statevector simulations of an
# (m + n)-qubit circuit and the cliff is steep: measured at 15 s for 9 qubits, 129 s for
# 11, and past twenty minutes at 13, which is what stalled the first G-2 run.
VERIFICATION_QUBIT_BUDGET = 12
# The block-encoding property is a statement about the construction, not about the
# coefficient values, so verifying all ten additive seeds at every size is redundant
# cost. Two are kept rather than one so that a sign-handling bug still has varied
# coefficients to show up in.
VERIFICATION_SEEDS_PER_FAMILY = 2
MAX_DEGREE = 4096
EPSILON = 1.0 - COSINE_THRESHOLD**2

ASSUMPTION = (
    "Route B is built as ADR-0010 option C under ADR-0015, a working assumption pending "
    "confirmation by both PIs: QSVT eigenstate filtering for a Hermitian stoquastic "
    "operator. ADR-0010 records that the G-2 thresholds do not depend on this choice."
)


def configurations():
    """Every (label, hamiltonian, fitness) the gate runs on. Seeded per case."""
    for n_sites in SIZES:
        for seed in ADDITIVE_SEEDS:
            rng = np.random.default_rng(10_000 * n_sites + seed)
            a = rng.uniform(0.3, 1.5, size=n_sites)
            yield (
                {"family": "additive", "L": n_sites, "seed": seed},
                additive_hamiltonian(a, MU),
                additive_fitness(a),
            )
        for height in PEAK_HEIGHTS:
            fitness = class_fitness(single_peak_classes(n_sites, height))
            yield (
                {"family": "single_peak", "L": n_sites, "height": height},
                diagonal_hamiltonian(fitness, MU),
                fitness,
            )
        for b in EPISTASIS_B:
            fitness = class_fitness(pairwise_uniform_classes(n_sites, 1.0, b))
            yield (
                {"family": "additive_pairwise", "L": n_sites, "b": b},
                diagonal_hamiltonian(fitness, MU),
                fitness,
            )


def run() -> tuple[bool, dict, list[dict]]:
    started = time.monotonic()
    cases: list[dict] = []

    worst_encoding_error = 0.0
    worst_cosine = 1.0
    worst_degree_ratio = 1.0
    unitarity_failures = 0
    degree_failures = 0
    unreached = 0
    verified = 0
    skipped_verification: list[dict] = []

    for label, operator, fitness in configurations():
        generator = np.asarray(mutation_selection_generator(fitness, MU).todense())
        values, vectors = np.linalg.eigh(generator)
        lambda_1, lambda_2 = float(values[-1]), float(values[-2])
        perron = np.abs(vectors[:, -1])
        perron = perron / np.linalg.norm(perron)

        dimension = generator.shape[0]
        initial = np.full(dimension, 1.0 / np.sqrt(dimension))
        overlap = float(abs(np.vdot(perron, initial)))

        # alpha is a sum over Pauli coefficients and needs no circuit. Criterion 1 needs
        # it at every configuration; criterion 2's verification is the expensive part and is
        # budgeted separately.
        alpha = one_norm(operator)
        qubits = encoding_qubit_count(operator)

        encodings = {}
        within_seed_cap = label.get("seed", 0) < VERIFICATION_SEEDS_PER_FAMILY
        if qubits <= VERIFICATION_QUBIT_BUDGET and within_seed_cap:
            for form in ("asymmetric", "symmetric"):
                encoding = lcu_block_encoding(operator, symmetric=(form == "symmetric"))
                report = verify_block_encoding(encoding, operator)
                worst_encoding_error = max(worst_encoding_error, report["max_abs_error"])
                assert abs(report["alpha"] - alpha) < 1e-9 * max(
                    alpha, 1.0
                ), "one_norm disagrees with the built encoding's alpha"
                unitary = None
                if encoding.n_ancilla + encoding.n_system <= UNITARITY_CHECK_UP_TO_QUBITS:
                    unitary = circuit_is_unitary(encoding, BLOCK_ENCODING_TOLERANCE)
                    if not unitary:
                        unitarity_failures += 1
                encodings[form] = {**report, "circuit_unitary": unitary}
            verified += 1
        elif qubits > VERIFICATION_QUBIT_BUDGET:
            skipped_verification.append(
                {**label, "encoding_qubits": qubits, "reason": "over the qubit budget"}
            )

        # Criterion 3: the smallest degree that works, against the derived one.
        found = smallest_sufficient_degree(
            generator,
            alpha,
            lambda_1,
            lambda_2,
            COSINE_THRESHOLD,
            perron,
            max_degree=MAX_DEGREE,
            initial=initial,
        )
        empirical = found["sufficient_degree"]
        predicted = predicted_degree(lambda_1 - lambda_2, alpha, overlap, EPSILON)

        if empirical is None:
            unreached += 1
            ratio = float("inf")
            cosine = float(found["reached"])
        else:
            state = filtered_state(generator, alpha, empirical, lambda_1, lambda_2, initial=initial)
            cosine = float(abs(np.vdot(perron, state)))
            ratio = predicted / empirical
            if not (1.0 / DEGREE_AGREEMENT_FACTOR <= ratio <= DEGREE_AGREEMENT_FACTOR):
                degree_failures += 1

        worst_cosine = min(worst_cosine, cosine)
        if np.isfinite(ratio):
            worst_degree_ratio = max(worst_degree_ratio, max(ratio, 1.0 / ratio))

        cases.append(
            {
                **label,
                "alpha": alpha,
                "gap": lambda_1 - lambda_2,
                "initial_overlap": overlap,
                "empirical_degree": empirical,
                "predicted_degree": predicted,
                "degree_ratio": ratio if np.isfinite(ratio) else None,
                "cosine": cosine,
                "encoding_qubits": qubits,
                "encoding_verified": bool(encodings),
                "encoding": encodings,
            }
        )

    # Supporting check: does the walk give Chebyshev polynomials?
    chebyshev = [
        verify_chebyshev(operator, CHEBYSHEV_DEGREES)
        for label, operator, _ in configurations()
        if label["L"] <= 3 and label.get("seed", 0) == 0
    ]
    worst_chebyshev = max(float(c["worst_max_abs_error"]) for c in chebyshev)

    criterion_1 = bool(worst_cosine >= COSINE_THRESHOLD and unreached == 0)
    criterion_2 = bool(worst_encoding_error < BLOCK_ENCODING_TOLERANCE and unitarity_failures == 0)
    criterion_3 = bool(degree_failures == 0 and unreached == 0)

    measured = {
        "assumption": ASSUMPTION,
        "n_configurations": len(cases),
        "criterion_1_accuracy": {
            "passed": criterion_1,
            "worst_cosine": worst_cosine,
            "threshold": COSINE_THRESHOLD,
            "configurations_never_reaching_threshold": unreached,
        },
        "criterion_2_block_encoding": {
            "passed": criterion_2,
            "worst_max_abs_error": worst_encoding_error,
            "tolerance": BLOCK_ENCODING_TOLERANCE,
            "unitarity_failures": unitarity_failures,
            "configurations_verified": verified,
            "configurations_over_the_qubit_budget": skipped_verification,
            "qubit_budget": VERIFICATION_QUBIT_BUDGET,
        },
        "criterion_3_resource_scaling": {
            "passed": criterion_3,
            "worst_degree_ratio": worst_degree_ratio,
            "allowed_factor": DEGREE_AGREEMENT_FACTOR,
            "configurations_outside_factor": degree_failures,
        },
        "supporting_chebyshev_check": {
            "worst_max_abs_error": worst_chebyshev,
            "degrees": CHEBYSHEV_DEGREES,
            "instances": chebyshev,
        },
        "seconds": round(time.monotonic() - started, 2),
    }
    return bool(criterion_1 and criterion_2 and criterion_3), measured, cases


def main() -> int:
    passed, measured, cases = run()

    path = write_gate_record(
        gate="G-2",
        work_package="wp2",
        threshold={
            "criterion_1": f"cosine >= {COSINE_THRESHOLD} against the analytic quasispecies, "
            f"L = {SIZES}",
            "criterion_2": f"block encoding defining property to {BLOCK_ENCODING_TOLERANCE}",
            "criterion_3": f"derived degree within a factor of {DEGREE_AGREEMENT_FACTOR} of "
            f"the empirically sufficient degree",
            "registered_in": "GATES.md section 6, configurations in Amendment 13",
        },
        measured=measured,
        passed=passed,
        cases=cases,
        notes=ASSUMPTION
        + " Amendment 13 discloses that the first derivation of the degree omitted the "
        "initial-overlap term and overshot by factors of 3.3 to 7.2; the corrected form is "
        "the standard two-factor eigenstate-filtering cost and contains no fitted constant. "
        "The degree is linear in alpha over the gap, not square root: Chebyshev acceleration "
        "needs the target eigenvalue outside the encoded spectrum and here it is inside by "
        "construction.",
    )

    print(f"G-2: {len(cases)} configurations in {measured['seconds']} s\n")
    print(
        f"  {'family':12s} {'L':>2} {'alpha':>7} {'gap':>8} {'overlap':>8} "
        f"{'emp d':>6} {'pred d':>7} {'ratio':>6} {'cosine':>9}"
    )
    for case in cases:
        if case["family"] == "additive" and case.get("seed", 0) != 0:
            continue
        ratio = case["degree_ratio"]
        print(
            f"  {case['family']:12s} {case['L']:>2} {case['alpha']:>7.3f} "
            f"{case['gap']:>8.5f} {case['initial_overlap']:>8.4f} "
            f"{str(case['empirical_degree']):>6} {case['predicted_degree']:>7.1f} "
            f"{(f'{ratio:.2f}' if ratio else 'n/a'):>6} {case['cosine']:>9.6f}"
        )

    for key, title in (
        ("criterion_1_accuracy", "Criterion 1, accuracy"),
        ("criterion_2_block_encoding", "Criterion 2, block encoding"),
        ("criterion_3_resource_scaling", "Criterion 3, resource scaling"),
    ):
        block = measured[key]
        print(f"\n  {title}: {'PASS' if block['passed'] else 'FAIL'}")
        for field, value in block.items():
            if field != "passed":
                print(f"    {field:42s} {value}")

    print(
        f"\n  Chebyshev support check, worst error "
        f"{measured['supporting_chebyshev_check']['worst_max_abs_error']:.3e}"
    )
    print(f"  record  {path}")
    print(f"  G-2: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
