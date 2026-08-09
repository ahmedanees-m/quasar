"""G-R.2: the compiled qubit Hamiltonian against the analytic quasispecies.

G-R.1 established that the oracle is trustworthy. This gate uses it: the biology-to-qubit
compiler is judged by whether the ground state of the Pauli operator it emits is the
quasispecies the oracle predicts.

The eigenvector comparison is the registered criterion, but it is not the sharpest tool
here. An endianness error permutes the basis and can leave the spectrum untouched, so the
run also compares the compiled operator entry by entry against the generator assembled
independently in `exact_diag`. That comparison is recorded as a diagnostic rather than
promoted to the gate, because the threshold was registered in cosine and thresholds do not
move.

Threshold and configuration set are in GATES.md section 3 and Amendment 2, both committed
before this script was run.

    python experiments/wp_r_rebuild/g_r_2_hamiltonian_vs_oracle.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

from quasarstack.analytic.crow_kimura import (
    additive_mean_fitness,
    additive_quasispecies,
    class_quasispecies,
)
from quasarstack.analytic.exact_diag import mutation_selection_generator
from quasarstack.classical.landscapes import (
    additive_fitness,
    class_fitness,
    single_peak_classes,
)
from quasarstack.hamiltonian.builder import (
    additive_hamiltonian,
    diagonal_hamiltonian,
    ground_state,
    pauli_term_count,
)
from quasarstack.io.store import write_gate_record
from quasarstack.scoring.metrics import score

# Registered in GATES.md section 3.
THRESHOLD = 0.999999
REQUIRED_CONFIGURATIONS = 40

# Registered in GATES.md Amendment 2.
MUS = [0.10, 0.30, 0.60, 1.00]
CONFIGURATIONS = [
    {"family": "additive_random", "L": 2, "seed": 0},
    {"family": "additive_random", "L": 4, "seed": 0},
    {"family": "additive_random", "L": 6, "seed": 1},
    {"family": "additive_random", "L": 8, "seed": 2},
    {"family": "additive_uniform", "L": 3, "a": 0.5},
    {"family": "additive_uniform", "L": 7, "a": 1.5},
    {"family": "single_peak", "L": 4, "height": 2.0},
    {"family": "single_peak", "L": 8, "height": 3.0},
    {"family": "class_quadratic", "L": 6, "height": 2.0},
    {"family": "class_exponential", "L": 5, "height": 2.0},
]


def _build(config: dict, mu: float) -> tuple:
    """Return (hamiltonian, fitness_vector, oracle_probs, oracle_mean_fitness, route)."""
    family = config["family"]
    n_sites = config["L"]

    if family == "additive_random":
        rng = np.random.default_rng(config["seed"])
        a = rng.uniform(0.25, 2.00, size=n_sites)
        return (
            additive_hamiltonian(a, mu),
            additive_fitness(a),
            additive_quasispecies(a, mu),
            additive_mean_fitness(a, mu),
            "structured",
        )

    if family == "additive_uniform":
        a = np.full(n_sites, config["a"])
        return (
            additive_hamiltonian(a, mu),
            additive_fitness(a),
            additive_quasispecies(a, mu),
            additive_mean_fitness(a, mu),
            "structured",
        )

    d = np.arange(n_sites + 1, dtype=np.float64)
    height = config["height"]
    if family == "single_peak":
        f_by_class = single_peak_classes(n_sites, height)
    elif family == "class_quadratic":
        f_by_class = height * (1.0 - d / n_sites) ** 2
    elif family == "class_exponential":
        f_by_class = height * np.exp(-2.0 * d / n_sites)
    else:
        raise ValueError(f"unregistered family {family!r}")

    fitness = class_fitness(f_by_class)
    oracle, _, mean_fitness = class_quasispecies(f_by_class, mu)
    return diagonal_hamiltonian(fitness, mu), fitness, oracle, mean_fitness, "walsh_hadamard"


def run() -> tuple[bool, dict, list[dict]]:
    cases: list[dict] = []
    started = time.monotonic()

    for config in CONFIGURATIONS:
        for mu in MUS:
            hamiltonian, fitness, oracle, oracle_mean_fitness, route = _build(config, mu)
            probs, energy = ground_state(hamiltonian)
            scores = score(probs, oracle)

            # Operator-level identity. H is -W by construction, so the two must cancel.
            compiled = np.asarray(hamiltonian.to_matrix()).real
            generator = mutation_selection_generator(fitness, mu).toarray()
            operator_error = float(np.max(np.abs(compiled + generator)))

            # Where a structured build exists, the Walsh-Hadamard route must reproduce it.
            route_error = None
            if route == "structured":
                by_transform = np.asarray(diagonal_hamiltonian(fitness, mu).to_matrix()).real
                route_error = float(np.max(np.abs(compiled - by_transform)))

            cases.append(
                {
                    **config,
                    "mu": mu,
                    "route": route,
                    "cosine": scores["cosine"],
                    "tv": scores["tv"],
                    "operator_max_abs_error": operator_error,
                    "structured_vs_transform_error": route_error,
                    "energy_error": abs(-energy - oracle_mean_fitness),
                    "pauli_terms": pauli_term_count(hamiltonian),
                }
            )

    elapsed = time.monotonic() - started
    cosines = np.array([c["cosine"] for c in cases])
    worst = int(np.argmin(cosines))

    passed = bool(len(cases) == REQUIRED_CONFIGURATIONS and (cosines >= THRESHOLD).all())

    measured = {
        "n_configurations": len(cases),
        "n_at_or_above_threshold": int((cosines >= THRESHOLD).sum()),
        "min_cosine": float(cosines.min()),
        "max_one_minus_cosine": float((1.0 - cosines).max()),
        "max_tv": float(max(c["tv"] for c in cases)),
        "max_operator_error": float(max(c["operator_max_abs_error"] for c in cases)),
        "max_structured_vs_transform_error": float(
            max(
                c["structured_vs_transform_error"]
                for c in cases
                if c["structured_vs_transform_error"] is not None
            )
        ),
        "max_energy_error": float(max(c["energy_error"] for c in cases)),
        "worst_case": cases[worst],
        "seconds": round(elapsed, 2),
    }
    return passed, measured, cases


def main() -> int:
    passed, measured, cases = run()

    path = write_gate_record(
        gate="G-R.2",
        work_package="wp_r",
        threshold={
            "statistic": "cosine similarity between the ground state of the compiled Pauli "
            "operator and the analytic quasispecies, on every configuration",
            "value": THRESHOLD,
            "required_configurations": REQUIRED_CONFIGURATIONS,
            "registered_in": "GATES.md section 3, configuration set in Amendment 2",
        },
        measured=measured,
        passed=passed,
        cases=cases,
        notes=(
            "The registered criterion is the cosine, but the operator-level comparison "
            "against the independently assembled generator is the stricter check and is "
            "recorded per configuration: an endianness error permutes the basis and can "
            "leave the spectrum intact, so a spectral check alone would not catch it. "
            "Pauli term counts are recorded here and feed G-R.10."
        ),
    )

    counts = {c["family"]: c["pauli_terms"] for c in cases if c["mu"] == MUS[0]}
    print(f"G-R.2: {len(cases)} configurations in {measured['seconds']} s")
    print(f"  min cosine            {measured['min_cosine']:.15f}  (threshold {THRESHOLD})")
    print(f"  max 1 - cosine        {measured['max_one_minus_cosine']:.3e}")
    print(f"  max total variation   {measured['max_tv']:.3e}")
    print(f"  max operator error    {measured['max_operator_error']:.3e}")
    print(f"  max route disagreement{measured['max_structured_vs_transform_error']:.3e}")
    print(f"  max energy error      {measured['max_energy_error']:.3e}")
    print(f"  pauli terms by family {counts}")
    print(f"  record                {path.relative_to(path.parents[2])}")
    print(f"  {'PASS' if passed else 'FAIL'}")

    if not passed:
        print(f"  worst case: {measured['worst_case']}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
