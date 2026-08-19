"""G-R.3: the Trotterised propagator converges, and its splitting error is second order.

Two things are measured, and they need different references.

**A. The splitting exponent.** Trotter against `exp(-H tau)` computed without splitting, at
the same total time from the same initial state. That isolates the splitting error. Fitting
against the analytic quasispecies instead would fold in the residual from tau being finite,
and that floor flattens the exponent at small step sizes, giving a number that describes the
choice of tau rather than the method.

**B. Convergence.** The finest step size run to a long total time, scored against the
analytic oracle, which is what the biology actually asks for.

Thresholds and configurations are in docs/protocol.md section 3 and revision 3, committed before
this ran.

    python experiments/wp_r_rebuild/g_r_3_trotter_scaling.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

from quasarstack.analytic.crow_kimura import additive_quasispecies, class_quasispecies
from quasarstack.circuit.trotter_ite import evolve, evolve_exact, trotter_circuit
from quasarstack.classical.landscapes import (
    additive_fitness,
    class_fitness,
    single_peak_classes,
)
from quasarstack.io.store import write_gate_record
from quasarstack.scoring.metrics import score

# Registered in docs/protocol.md section 3.
COSINE_THRESHOLD = 0.999
EXPONENT_BOUNDS = (1.8, 2.2)
R_SQUARED_THRESHOLD = 0.99

# Registered in docs/protocol.md revision 3.
MU = 0.30
TAU_SCALING = 2.0
DTAUS = [0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
TAU_CONVERGE = 60.0
DTAU_CONVERGE = 0.01

CONFIGURATIONS = [
    {"family": "additive_random", "L": 3, "seed": 0},
    {"family": "additive_random", "L": 6, "seed": 1},
    {"family": "additive_uniform", "L": 4, "a": 1.0},
    {"family": "additive_epistatic", "L": 4, "seed": 3},
    {"family": "single_peak", "L": 5, "height": 2.0},
    {"family": "class_quadratic", "L": 6, "height": 2.0},
]


def _build(config: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (fitness_vector, oracle_probs, circuit_parameters_or_empty)."""
    family = config["family"]
    n_sites = config["L"]

    if family == "additive_random":
        rng = np.random.default_rng(config["seed"])
        a = rng.uniform(0.25, 2.00, size=n_sites)
        return additive_fitness(a), additive_quasispecies(a, MU), {"a": a, "b": None}

    if family == "additive_uniform":
        a = np.full(n_sites, config["a"])
        return additive_fitness(a), additive_quasispecies(a, MU), {"a": a, "b": None}

    if family == "additive_epistatic":
        rng = np.random.default_rng(config["seed"])
        a = rng.uniform(0.25, 2.00, size=n_sites)
        b = np.triu(rng.normal(scale=0.5, size=(n_sites, n_sites)), k=1)
        fitness = additive_fitness(a, b)
        # No closed form with epistasis, so the oracle here is sparse exact diagonalisation.
        from quasarstack.analytic.exact_diag import perron_vector

        oracle, _, _ = perron_vector(fitness, MU)
        return fitness, oracle, {"a": a, "b": b}

    d = np.arange(n_sites + 1, dtype=np.float64)
    height = config["height"]
    if family == "single_peak":
        f_by_class = single_peak_classes(n_sites, height)
    elif family == "class_quadratic":
        f_by_class = height * (1.0 - d / n_sites) ** 2
    else:
        raise ValueError(f"unregistered family {family!r}")

    fitness = class_fitness(f_by_class)
    oracle, _, _ = class_quasispecies(f_by_class, MU)
    return fitness, oracle, {}


def _fit_power_law(dtaus: list[float], errors: list[float]) -> tuple[float, float, float]:
    """Least-squares fit of log(error) against log(dtau). Returns slope, intercept, R^2."""
    x = np.log(np.asarray(dtaus))
    y = np.log(np.asarray(errors))
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - residual / total if total > 0 else 0.0
    return float(slope), float(intercept), float(r_squared)


def run() -> tuple[bool, dict, list[dict]]:
    started = time.monotonic()
    cases: list[dict] = []

    for config in CONFIGURATIONS:
        fitness, oracle, circuit_params = _build(config)

        # A. splitting-error exponent, against the un-split propagator at the same tau
        reference = evolve_exact(fitness, MU, TAU_SCALING)
        errors = []
        for dtau in DTAUS:
            trotter, _ = evolve(fitness, MU, TAU_SCALING, dtau)
            errors.append(float(np.max(np.abs(trotter - reference))))
        slope, intercept, r_squared = _fit_power_law(DTAUS, errors)

        # B. convergence to the quasispecies at the finest registered step
        converged, n_steps = evolve(fitness, MU, TAU_CONVERGE, DTAU_CONVERGE)
        scores = score(converged, oracle)

        resources = {}
        if circuit_params:
            circuit = trotter_circuit(
                config["L"], circuit_params["a"], MU, DTAU_CONVERGE, circuit_params["b"]
            )
            resources = {
                "structural_circuit_depth": int(circuit.depth()),
                "structural_circuit_two_qubit_gates": int(
                    sum(1 for instruction in circuit.data if len(instruction.qubits) == 2)
                ),
            }

        cases.append(
            {
                **config,
                "mu": MU,
                "tau_scaling": TAU_SCALING,
                "dtaus": DTAUS,
                "splitting_errors": errors,
                "fitted_exponent": slope,
                "fitted_intercept": intercept,
                "r_squared": r_squared,
                "tau_converge": TAU_CONVERGE,
                "dtau_converge": DTAU_CONVERGE,
                "n_steps_converge": n_steps,
                "cosine": scores["cosine"],
                "tv": scores["tv"],
                **resources,
            }
        )

    elapsed = time.monotonic() - started

    exponents = np.array([c["fitted_exponent"] for c in cases])
    r_squareds = np.array([c["r_squared"] for c in cases])
    cosines = np.array([c["cosine"] for c in cases])

    exponents_ok = bool(
        (exponents >= EXPONENT_BOUNDS[0]).all() and (exponents <= EXPONENT_BOUNDS[1]).all()
    )
    fits_ok = bool((r_squareds >= R_SQUARED_THRESHOLD).all())
    convergence_ok = bool((cosines >= COSINE_THRESHOLD).all())

    measured = {
        "n_configurations": len(cases),
        "min_fitted_exponent": float(exponents.min()),
        "max_fitted_exponent": float(exponents.max()),
        "min_r_squared": float(r_squareds.min()),
        "min_cosine": float(cosines.min()),
        "max_tv": float(max(c["tv"] for c in cases)),
        "exponents_within_bounds": exponents_ok,
        "fits_above_r_squared_threshold": fits_ok,
        "all_converged": convergence_ok,
        "seconds": round(elapsed, 2),
    }
    return bool(exponents_ok and fits_ok and convergence_ok), measured, cases


def main() -> int:
    passed, measured, cases = run()

    path = write_gate_record(
        gate="G-R.3",
        work_package="wp_r",
        threshold={
            "convergence": {
                "statistic": "cosine against the analytic oracle",
                "value": COSINE_THRESHOLD,
            },
            "exponent": {
                "statistic": "fitted slope of log error against log dtau",
                "bounds": EXPONENT_BOUNDS,
            },
            "fit_quality": {"statistic": "R squared of that fit", "value": R_SQUARED_THRESHOLD},
            "registered_in": "docs/protocol.md section 3, protocol in revision 3",
        },
        measured=measured,
        passed=passed,
        cases=cases,
        notes=(
            "The exponent is fitted against exp(-H tau) computed without splitting, not "
            "against the analytic quasispecies, so that the residual from tau being finite "
            "cannot flatten the fit. Circuit depth and two-qubit counts are the unitary "
            "real-time analogue of the same interaction pattern and are labelled structural: "
            "imaginary-time evolution is non-unitary and this propagator is not a "
            "hardware-runnable circuit. The hardware-faithful routes are G-R.6 and G-R.7."
        ),
    )

    print(f"G-R.3: {len(cases)} configurations in {measured['seconds']} s\n")
    header = f"{'family':20s} {'L':>2s} {'exponent':>9s} {'R^2':>8s} {'cosine':>10s} {'tv':>10s}"
    print(header)
    print("-" * len(header))
    for case in cases:
        print(
            f"{case['family']:20s} {case['L']:>2d} {case['fitted_exponent']:>9.3f} "
            f"{case['r_squared']:>8.5f} {case['cosine']:>10.7f} {case['tv']:>10.2e}"
        )

    print()
    print(
        f"  exponent range   {measured['min_fitted_exponent']:.3f} to "
        f"{measured['max_fitted_exponent']:.3f}  (bounds {EXPONENT_BOUNDS})"
    )
    print(f"  min R^2          {measured['min_r_squared']:.5f}  (threshold {R_SQUARED_THRESHOLD})")
    print(f"  min cosine       {measured['min_cosine']:.7f}  (threshold {COSINE_THRESHOLD})")
    print(f"  max TV           {measured['max_tv']:.2e}")
    print(f"  record           {path.relative_to(path.parents[2])}")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
