"""G-R.9: decay of the varQITE gradient variance with system size.

The variance, over random parameter draws, of a component of the McLachlan force
``C_i = -<d_i psi|H|psi>``, minus half the energy gradient, is fitted as ``base**L``. Motta QITE
has no parameter optimisation and so no such decay, at the cost of a generator support that
grows with L.

Three statistics are recorded: the first component, which sits at the circuit boundary and is
the least scrambled; a mid-circuit component, which is the tested statistic; and the mean over
all components.

    python experiments/wp_r/g_r_9_barren.py
"""

from __future__ import annotations

import sys
import time

import numpy as np

from quasarstack.classical.landscapes import class_fitness, nk_fitness, single_peak_classes
from quasarstack.hamiltonian.builder import diagonal_hamiltonian
from quasarstack.io.store import write_gate_record
from quasarstack.ite.varqite import Ansatz, force_components

DECAY_BASE_BOUNDS = (0.30, 0.55)
R_SQUARED_THRESHOLD = 0.95

# Configuration.
MU = 0.20
SIZES = [2, 3, 4, 5, 6, 7, 8]
SAMPLES = 400
SEED = 0
GATE_LANDSCAPE = "nk_k2"
GATE_STATISTIC = "middle"


def _fitness(name: str, n_sites: int) -> np.ndarray:
    if name == "nk_k2":
        return nk_fitness(n_sites, min(2, n_sites - 1), seed=0)
    if name == "single_peak":
        return class_fitness(single_peak_classes(n_sites, 1.0))
    raise ValueError(f"unknown landscape {name!r}")


def _fit_exponential(sizes: list[int], values: list[float]) -> tuple[float, float]:
    """Fit ``variance ~ base**L``. Returns the base and the fit's R squared."""
    slope, intercept = np.polyfit(sizes, np.log(values), 1)
    predicted = slope * np.array(sizes, dtype=float) + intercept
    residual = float(np.sum((np.log(values) - predicted) ** 2))
    total = float(np.sum((np.log(values) - np.mean(np.log(values))) ** 2))
    return float(np.exp(slope)), 1.0 - residual / total


def run() -> tuple[bool, dict, list[dict]]:
    started = time.monotonic()
    cases: list[dict] = []

    series: dict[str, dict[str, list[float]]] = {}
    for landscape in ("nk_k2", "single_peak"):
        collected = {"first": [], "middle": [], "mean_all": []}
        for n_sites in SIZES:
            matrix = np.asarray(
                diagonal_hamiltonian(_fitness(landscape, n_sites), MU).to_matrix()
            ).real
            ansatz = Ansatz(n_sites, reps=n_sites + 2)
            rng = np.random.default_rng(SEED)
            rows = [
                force_components(
                    ansatz, rng.uniform(0.0, 2.0 * np.pi, size=ansatz.n_parameters), matrix
                )
                for _ in range(SAMPLES)
            ]
            arr = np.array(rows)
            variances = np.var(arr, axis=0)
            middle = ansatz.n_parameters // 2

            collected["first"].append(float(variances[0]))
            collected["middle"].append(float(variances[middle]))
            collected["mean_all"].append(float(np.mean(variances)))

            cases.append(
                {
                    "landscape": landscape,
                    "L": n_sites,
                    "reps": ansatz.reps,
                    "n_parameters": ansatz.n_parameters,
                    "samples": SAMPLES,
                    "variance_first": float(variances[0]),
                    "variance_middle": float(variances[middle]),
                    "variance_mean_all": float(np.mean(variances)),
                }
            )
        series[landscape] = collected

    fits = {
        f"{landscape}_{statistic}": dict(
            zip(
                ("base", "r_squared"),
                _fit_exponential(SIZES, values),
                strict=True,
            )
        )
        for landscape, collected in series.items()
        for statistic, values in collected.items()
    }

    gate_key = f"{GATE_LANDSCAPE}_{GATE_STATISTIC}"
    gate_fit = fits[gate_key]
    within_bounds = bool(DECAY_BASE_BOUNDS[0] <= gate_fit["base"] <= DECAY_BASE_BOUNDS[1])
    fit_good = bool(gate_fit["r_squared"] >= R_SQUARED_THRESHOLD)

    elapsed = time.monotonic() - started
    measured = {
        "gate_statistic": gate_key,
        "decay_base": gate_fit["base"],
        "r_squared": gate_fit["r_squared"],
        "within_bounds": within_bounds,
        "fit_above_threshold": fit_good,
        "decays_exponentially": bool(gate_fit["base"] < 1.0 and fit_good),
        "all_fits": fits,
        "variance_series": series,
        "seconds": round(elapsed, 2),
    }
    return bool(within_bounds and fit_good), measured, cases


def main() -> int:
    passed, measured, cases = run()

    path = write_gate_record(
        gate="G-R.9",
        work_package="wp_r",
        threshold={
            "statistic": f"fitted decay base of the McLachlan force variance against system "
            f"size, {GATE_STATISTIC} component on the {GATE_LANDSCAPE} landscape",
            "bounds": DECAY_BASE_BOUNDS,
            "r_squared": R_SQUARED_THRESHOLD,
            "defined_in": "docs/protocol.md, G-R.9",
        },
        measured=measured,
        passed=passed,
        cases=cases,
        notes=(
            "All three gradient statistics are recorded in measured.all_fits; the tested "
            "one is the mid-circuit component on the NK K = 2 landscape."
        ),
    )

    print(f"G-R.9: {len(cases)} measurements in {measured['seconds']} s\n")
    print(f"  {'landscape and component':28s} {'base':>8s} {'R^2':>9s}")
    for key, fit in measured["all_fits"].items():
        marker = "  <- gate" if key == measured["gate_statistic"] else ""
        print(f"  {key:28s} {fit['base']:>8.4f} {fit['r_squared']:>9.5f}{marker}")

    print(f"\n  gate statistic          {measured['gate_statistic']}")
    print(
        f"  decay base              {measured['decay_base']:.4f}  " f"(bounds {DECAY_BASE_BOUNDS})"
    )
    print(
        f"  R squared               {measured['r_squared']:.5f}  (threshold {R_SQUARED_THRESHOLD})"
    )
    print(f"  decays exponentially    {measured['decays_exponentially']}")
    print(f"  record                  {path.relative_to(path.parents[2])}")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
