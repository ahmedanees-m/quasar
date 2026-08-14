"""G-R.4: the error threshold, as computed on the qubit representation.

The error catastrophe is a localisation-delocalisation transition, and the gate asks whether
the qubit route sees it where the analytic theory puts it. The surplus is computed two ways
at every point of a mutation-rate sweep: from the ground state of the compiled Pauli
Hamiltonian, and from the analytic Hamming-class reduction. The gate is the maximum
disagreement.

Everything about *where* the threshold sits is a diagnostic rather than a pass condition,
including the direction in which epistasis moves it. That is deliberate. The planning
documents state an expected direction, and a gate that required the expected answer would
not be a measurement.

Thresholds, sweep and landscapes are in GATES.md section 3 and revision 4, committed
before this ran.

    python experiments/wp_r_rebuild/g_r_4_error_threshold.py
"""

from __future__ import annotations

import sys
import time
from math import comb

import numpy as np

from quasarstack.analytic.crow_kimura import class_quasispecies
from quasarstack.analytic.exact_diag import perron_vector
from quasarstack.classical.landscapes import (
    class_fitness,
    pairwise_uniform_classes,
    single_peak_classes,
)
from quasarstack.hamiltonian.builder import diagonal_hamiltonian
from quasarstack.io.store import write_gate_record
from quasarstack.spectral.order_parameter import (
    locate_threshold,
    magnetisation,
    magnetisation_from_classes,
)

# Registered in GATES.md section 3.
THRESHOLD = 1e-3

# Registered in GATES.md revision 4.
SIZES = [4, 6, 8]
MUS = np.round(np.arange(0.01, 3.001, 0.01), 10)
LANDSCAPES = [
    {"name": "single_peak", "height": 1.0},
    {"name": "pairwise_additive", "a": 0.5, "B": 0.0},
    {"name": "pairwise_synergistic_1", "a": 0.5, "B": 1.0},
    {"name": "pairwise_synergistic_2", "a": 0.5, "B": 2.0},
    {"name": "pairwise_antagonistic", "a": 0.5, "B": -1.0},
]
ASYMPTOTIC_SIZES = [4, 6, 8, 10, 12, 14, 16, 18, 20]


def _classes(landscape: dict, n_sites: int) -> np.ndarray:
    if landscape["name"] == "single_peak":
        return single_peak_classes(n_sites, landscape["height"])
    return pairwise_uniform_classes(n_sites, landscape["a"], landscape["B"] / (n_sites - 1))


def _compiled_pieces(fitness: np.ndarray, n_sites: int) -> tuple[np.ndarray, np.ndarray]:
    """Compile the selection and mutation parts of H once each, as dense matrices.

    The compiled operator is linear in the mutation rate: the selection terms do not depend
    on it, and the mutation terms are exactly proportional to it. So the sweep needs two
    compilations rather than one per point, which at L = 8 is the difference between
    summing 256 Pauli terms 300 times and doing it twice.

    Assembling this way is exact rather than approximate, and the caller checks it against a
    directly compiled operator at one mutation rate so the shortcut cannot drift from the
    thing it is standing in for.
    """
    selection = np.asarray(diagonal_hamiltonian(fitness, 0.0).to_matrix()).real
    mutation_at_unit_rate = np.asarray(
        diagonal_hamiltonian(np.zeros_like(fitness), 1.0).to_matrix()
    ).real
    return selection, mutation_at_unit_rate


def _ground_state_and_gap(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Ground state as an L1-normalised distribution, plus the gap to the first excited state.

    One eigendecomposition gives both, where calling the library twice would do the work
    twice.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    probs = np.abs(eigenvectors[:, 0])
    return probs / probs.sum(), float(eigenvalues[1] - eigenvalues[0])


def run() -> tuple[bool, dict, list[dict]]:
    started = time.monotonic()
    cases: list[dict] = []

    for n_sites in SIZES:
        for landscape in LANDSCAPES:
            f_by_class = _classes(landscape, n_sites)
            fitness = class_fitness(f_by_class)
            selection, mutation = _compiled_pieces(fitness, n_sites)

            # Guard the shortcut: the assembled operator must equal a directly compiled one.
            check_mu = 0.37
            assembled_check = selection + check_mu * mutation
            direct_check = np.asarray(diagonal_hamiltonian(fitness, check_mu).to_matrix()).real
            assembly_error = float(np.max(np.abs(assembled_check - direct_check)))
            if assembly_error > 1e-12:
                raise AssertionError(
                    f"the mu-linear assembly diverged from the compiler by {assembly_error:.2e}"
                )

            analytic_m, qubit_m, gaps = [], [], []
            for mu in MUS:
                _, class_probs, _ = class_quasispecies(f_by_class, float(mu))
                analytic_m.append(magnetisation_from_classes(class_probs))

                probs, gap = _ground_state_and_gap(selection + float(mu) * mutation)
                qubit_m.append(magnetisation(probs, n_sites))
                gaps.append(gap)

            analytic_m = np.array(analytic_m)
            qubit_m = np.array(qubit_m)
            gaps = np.array(gaps)

            analytic_threshold = locate_threshold(MUS, analytic_m)
            qubit_threshold = locate_threshold(MUS, qubit_m)

            cases.append(
                {
                    **landscape,
                    "L": n_sites,
                    "max_abs_delta_m": float(np.max(np.abs(qubit_m - analytic_m))),
                    "assembly_error": assembly_error,
                    "analytic": analytic_threshold,
                    "qubit": qubit_threshold,
                    "min_spectral_gap": float(gaps.min()),
                    "mu_at_min_gap": float(MUS[int(np.argmin(gaps))]),
                    "m_at_smallest_mu": float(analytic_m[0]),
                    "m_at_largest_mu": float(analytic_m[-1]),
                }
            )

    # Diagnostic: does the sharp-peak threshold approach the infinite-size prediction?
    asymptotic = []
    for n_sites in ASYMPTOTIC_SIZES:
        f_by_class = single_peak_classes(n_sites, 1.0)
        m = np.array(
            [magnetisation_from_classes(class_quasispecies(f_by_class, float(mu))[1]) for mu in MUS]
        )
        found = locate_threshold(MUS, m)
        asymptotic.append(
            {
                "L": n_sites,
                "mu_c": found["mu_c"],
                "mu_c_times_L": found["mu_c"] * n_sites,
                "width": found["width"],
                "mu_half": found["mu_half"],
            }
        )

    # Diagnostic: how fast does the gap close at the sharp-peak threshold?
    # Three points from the sweep above hinted at a factor of two per two sites, which is too
    # thin to say anything with. This extends the range using sparse eigensolves in a narrow
    # window around each threshold, which is cheap because only the two lowest eigenvalues
    # are needed. The full map across landscapes and mutation rates is WP1's job, G-1.2.
    gap_scaling = []
    for n_sites in [4, 6, 8, 10, 12]:
        fitness = class_fitness(single_peak_classes(n_sites, 1.0))
        centre = 1.0 / n_sites  # the infinite-size prediction, mu_c = A / L
        window = np.round(np.linspace(0.4 * centre, 2.5 * centre, 40), 10)
        # Sparse, forced. perron_vector goes dense up to L = 12 because dense is faster and
        # more accurate at small sizes, but the image pins BLAS to a single thread so that
        # the compute-budget protocol means something, and a single-threaded dense solve at
        # 4096 by 4096 takes minutes. Forty of them takes an hour. Only two eigenvalues are
        # wanted here, so the sparse path is both correct and orders of magnitude cheaper.
        gaps = [perron_vector(fitness, float(mu), dense_limit=8)[2] for mu in window]
        best = int(np.argmin(gaps))
        gap_scaling.append(
            {"L": n_sites, "min_gap": float(gaps[best]), "mu_at_min_gap": float(window[best])}
        )

    sizes = np.array([row["L"] for row in gap_scaling], dtype=float)
    log_gaps = np.log(np.array([row["min_gap"] for row in gap_scaling]))
    decay_slope, _ = np.polyfit(sizes, log_gaps, 1)

    # Diagnostic: which way does epistasis move the crossover?
    #
    # Recorded alongside it: where each landscape's fitness optimum actually sits. A family
    # that relocates the optimum away from the master sequence is not varying ruggedness
    # alone, and the error-threshold question stops being well posed there, because there is
    # no master sequence left to delocalise from. See DECISIONS.md ADR-0011.
    epistasis = {}
    for n_sites in SIZES:
        row = {}
        for landscape in LANDSCAPES:
            if landscape["name"] == "single_peak":
                continue
            case = next(c for c in cases if c["L"] == n_sites and c["name"] == landscape["name"])
            f_by_class = _classes(landscape, n_sites)
            optimum_class = int(np.argmax(f_by_class))
            row[landscape["name"]] = {
                "mu_half": case["analytic"]["mu_half"],
                "optimum_hamming_class": optimum_class,
                "optimum_multiplicity": int(comb(n_sites, optimum_class)),
                "master_sequence_is_optimal": optimum_class == 0,
                "surplus_at_smallest_mu": case["m_at_smallest_mu"],
            }
        epistasis[f"L{n_sites}"] = row

    elapsed = time.monotonic() - started
    deltas = np.array([c["max_abs_delta_m"] for c in cases])
    worst = int(np.argmax(deltas))

    measured = {
        "n_cases": len(cases),
        "n_sweep_points": int(MUS.size),
        "max_abs_delta_m": float(deltas.max()),
        "median_abs_delta_m": float(np.median(deltas)),
        "worst_case": {k: v for k, v in cases[worst].items() if k != "analytic"},
        "sharp_peak_asymptotics": asymptotic,
        "sharp_peak_gap_scaling": gap_scaling,
        "gap_decay_per_site": float(np.exp(decay_slope)),
        "epistasis_half_surplus_crossover": epistasis,
        "min_spectral_gap_over_all_cases": float(min(c["min_spectral_gap"] for c in cases)),
        "seconds": round(elapsed, 2),
    }
    return bool(deltas.max() < THRESHOLD), measured, cases


def main() -> int:
    passed, measured, cases = run()

    path = write_gate_record(
        gate="G-R.4",
        work_package="wp_r",
        threshold={
            "statistic": "max absolute difference in surplus between the compiled "
            "Hamiltonian ground state and the analytic class reduction, over the whole sweep",
            "value": THRESHOLD,
            "registered_in": "GATES.md section 3, sweep and landscapes in revision 4",
        },
        measured=measured,
        passed=passed,
        cases=cases,
        notes=(
            "Threshold location is a diagnostic, not a pass condition, and so is the "
            "direction in which epistasis moves it. The planning documents state an "
            "expected direction; requiring it would not be a measurement. Two location "
            "measures are recorded because the susceptibility peak is undefined for "
            "landscapes additive in the surplus, which decay monotonically with steepest "
            "slope at zero mutation rate."
        ),
    )

    print(
        f"G-R.4: {len(cases)} cases over {measured['n_sweep_points']} sweep points "
        f"in {measured['seconds']} s\n"
    )
    header = f"{'landscape':24s} {'L':>2s} {'max|dm|':>10s} {'mu_c':>7s} {'interior':>9s} {'width':>7s} {'mu_half':>8s}"
    print(header)
    print("-" * len(header))
    for case in cases:
        a = case["analytic"]
        print(
            f"{case['name']:24s} {case['L']:>2d} {case['max_abs_delta_m']:>10.2e} "
            f"{a['mu_c']:>7.2f} {str(a['peak_is_interior']):>9s} {a['width']:>7.2f} "
            f"{a['mu_half']:>8.3f}"
        )

    print("\nsharp peak against system size, from the class reduction:")
    print(f"  {'L':>3s} {'mu_c':>7s} {'mu_c * L':>9s} {'width':>7s}")
    for row in measured["sharp_peak_asymptotics"]:
        print(
            f"  {row['L']:>3d} {row['mu_c']:>7.3f} {row['mu_c_times_L']:>9.3f} {row['width']:>7.3f}"
        )

    print("\nspectral gap at the sharp-peak threshold:")
    print(f"  {'L':>3s} {'min gap':>10s} {'at mu':>7s}")
    for row in measured["sharp_peak_gap_scaling"]:
        print(f"  {row['L']:>3d} {row['min_gap']:>10.4f} {row['mu_at_min_gap']:>7.3f}")
    print(f"  fitted decay per site: {measured['gap_decay_per_site']:.4f}")

    print("\nhalf-surplus crossover against epistasis, with where the optimum sits:")
    print(
        f"  {'L':>3s} {'landscape':16s} {'mu_half':>8s} {'d*':>3s} {'mult':>5s} {'master optimal':>15s}"
    )
    for size, row in measured["epistasis_half_surplus_crossover"].items():
        for name, entry in row.items():
            print(
                f"  {size[1:]:>3s} {name.replace('pairwise_', ''):16s} "
                f"{entry['mu_half']:>8.3f} {entry['optimum_hamming_class']:>3d} "
                f"{entry['optimum_multiplicity']:>5d} "
                f"{str(entry['master_sequence_is_optimal']):>15s}"
            )

    print(f"\n  max |delta m|    {measured['max_abs_delta_m']:.3e}  (threshold {THRESHOLD:.0e})")
    print(f"  min spectral gap {measured['min_spectral_gap_over_all_cases']:.3e}")
    print(f"  record           {path.relative_to(path.parents[2])}")
    print(f"  {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
