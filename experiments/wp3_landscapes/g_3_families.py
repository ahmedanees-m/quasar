"""G-3: the landscape families, their reproducibility, and the ruggedness axis. WP3.

Criteria, registered in `docs/protocol.md` section 7 with parameters in revision 14:

1. Every landscape reproduces exactly from its seed, byte-for-byte.
2. NK with K = 0 equals the additive family analytically, to 1e-12.
3. Ruggedness increases monotonically in K over seeds 0 to 9 at L = 10 and L = 12, of the
   seed mean, with per-seed values reported.

The gate also answers a question docs/notes.md left open. That ADR withdrew a claim because a
family had been varying ruggedness and silently relocating the fitness optimum at the same
time, and it required every family to report where its optimum sits. This gate reports that
for all seven, which turns "which family should the ruggedness axis use" from a preference
into a measurement.

    python experiments/wp3_landscapes/g_3_families.py
"""

from __future__ import annotations

import hashlib
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
    ruggedness_statistics,
    single_peak_classes,
    spin_glass_fitness,
)
from quasarstack.hamiltonian.builder import diagonal_hamiltonian, pauli_term_count
from quasarstack.io.store import write_gate_record

# Registered in docs/protocol.md section 7 and revision 14.
ADDITIVE_TOLERANCE = 1e-12
NK_K = [0, 1, 2, 3, 4, 6]
RMF_ROUGHNESS = [0.0, 0.1, 0.3, 1.0, 3.0]
BLOCK_SIZES = [1, 2, 4]
SEEDS = list(range(10))
MONOTONICITY_SIZES = [10, 12]
HASH_SIZES = [8, 10]
PEAK_HEIGHT = 1.0
EPISTASIS_A, EPISTASIS_B = 1.0, 0.1
MU = 0.20


def families(n_sites: int):
    """Every (label, builder) the gate covers, as registered."""
    yield (
        {"family": "single_peak"},
        lambda s: class_fitness(single_peak_classes(n_sites, PEAK_HEIGHT)),
    )
    yield (
        {"family": "additive_pairwise"},
        lambda s: class_fitness(pairwise_uniform_classes(n_sites, EPISTASIS_A, EPISTASIS_B)),
    )
    for k in NK_K:
        if k <= n_sites - 1:
            yield {"family": "nk", "K": k}, lambda s, k=k: nk_fitness(n_sites, k, seed=s)
    yield {"family": "spin_glass"}, lambda s: spin_glass_fitness(n_sites, seed=s)
    yield {"family": "house_of_cards"}, lambda s: house_of_cards_fitness(n_sites, seed=s)
    for roughness in RMF_ROUGHNESS:
        yield (
            {"family": "rough_mount_fuji", "roughness": roughness},
            lambda s, r=roughness: rough_mount_fuji_fitness(n_sites, seed=s, roughness=r),
        )
    for size in BLOCK_SIZES:
        if size <= n_sites:
            yield (
                {"family": "block", "block_size": size},
                lambda s, b=size: block_fitness(n_sites, b, seed=s),
            )


def correlation_length(autocorrelation: float) -> float:
    """Weinberger's ``ell = -1 / ln(rho)``. Zero where there is no correlation to speak of.

    revision 14 fixes this convention. A non-positive ``rho`` has no correlation length,
    and returning zero rather than a complex number keeps the monotonicity comparison
    well defined at the rugged end where ``rho`` crosses zero.
    """
    if autocorrelation <= 0.0 or autocorrelation >= 1.0:
        return 0.0
    return float(-1.0 / np.log(autocorrelation))


def strict_local_optima(fitness: np.ndarray) -> int:
    """Local optima counted with a strict inequality, which plateaus do not satisfy.

    `ruggedness_statistics` uses ``>=``, the usual definition, and on a landscape with large
    flat regions that counts every plateau point. The single peak is the extreme case: it is
    zero everywhere but one genotype, so the ``>=`` count reads 248 out of 256 at L = 8 and
    makes the least rugged landscape in the project look like the most rugged. Both counts
    are reported here so neither reading can mislead on its own.
    """
    fitness = np.asarray(fitness, dtype=np.float64)
    n_sites = fitness.size.bit_length() - 1
    index = np.arange(fitness.size, dtype=np.int64)
    strict = np.ones(fitness.size, dtype=bool)
    for site in range(n_sites):
        strict &= fitness > fitness[index ^ (1 << site)]
    return int(strict.sum())


def criterion_one() -> tuple[bool, dict, list[dict]]:
    """Byte-for-byte reproduction, within the run and, through the hashes, across runs."""
    cases: list[dict] = []
    mismatches = 0

    for n_sites in HASH_SIZES:
        for label, build in families(n_sites):
            for seed in SEEDS:
                first = build(seed)
                # Disturb the global state, which is what docs/notes.md showed can leak in.
                np.random.seed(seed + 7919)  # noqa: NPY002
                np.random.random(1000)  # noqa: NPY002
                second = build(seed)

                identical = bool(first.tobytes() == second.tobytes())
                if not identical:
                    mismatches += 1
                cases.append(
                    {
                        "criterion": 1,
                        **label,
                        "L": n_sites,
                        "seed": seed,
                        "identical_within_run": identical,
                        "sha256": hashlib.sha256(first.tobytes()).hexdigest(),
                    }
                )

    return (
        mismatches == 0,
        {"n_landscapes": len(cases), "mismatches": mismatches},
        cases,
    )


def criterion_two() -> tuple[bool, dict, list[dict]]:
    """NK at K = 0 is additive, so it must equal the additive builder to 1e-12.

    The comparison is up to an affine map, because `nk_fitness` standardises to zero mean
    and unit spread and `additive_fitness` does not. An affine map is exactly what
    standardisation is, so allowing it tests the structural claim, that K = 0 has no
    epistasis, rather than a normalisation convention. What would fail is any dependence of
    one site's contribution on another, which no affine map can remove.
    """
    cases: list[dict] = []
    worst = 0.0

    for n_sites in (4, 6, 8, 10):
        for seed in SEEDS:
            nk = nk_fitness(n_sites, 0, seed=seed)
            # Recover per-site coefficients from the K = 0 landscape and rebuild additively.
            spins = 1.0 - 2.0 * ((np.arange(1 << n_sites)[:, None] >> np.arange(n_sites)) & 1)
            coefficients, *_ = np.linalg.lstsq(
                np.column_stack([np.ones(1 << n_sites), spins]), nk, rcond=None
            )
            rebuilt = coefficients[0] + additive_fitness(coefficients[1:])
            error = float(np.max(np.abs(nk - rebuilt)))
            worst = max(worst, error)
            cases.append(
                {
                    "criterion": 2,
                    "L": n_sites,
                    "seed": seed,
                    "max_abs_error": error,
                }
            )

    return (
        bool(worst < ADDITIVE_TOLERANCE),
        {"worst_max_abs_error": worst, "tolerance": ADDITIVE_TOLERANCE},
        cases,
    )


def criterion_three() -> tuple[bool, dict, list[dict]]:
    """Does ruggedness rise monotonically with K, in the mean over seeds?"""
    cases: list[dict] = []
    summary = []
    monotone = True

    for n_sites in MONOTONICITY_SIZES:
        optima, lengths = [], []
        for k in NK_K:
            per_seed = []
            for seed in SEEDS:
                stats = ruggedness_statistics(nk_fitness(n_sites, k, seed=seed))
                row = {
                    "criterion": 3,
                    "family": "nk",
                    "K": k,
                    "L": n_sites,
                    "seed": seed,
                    "n_local_optima": stats["n_local_optima"],
                    "autocorrelation": stats["autocorrelation"],
                    "correlation_length": correlation_length(stats["autocorrelation"]),
                    "optimum_hamming_weight": stats["optimum_hamming_weight"],
                }
                cases.append(row)
                per_seed.append(row)
            optima.append(float(np.mean([r["n_local_optima"] for r in per_seed])))
            lengths.append(float(np.mean([r["correlation_length"] for r in per_seed])))

        optima_rise = all(b > a for a, b in zip(optima[:-1], optima[1:], strict=True))
        length_falls = all(b < a for a, b in zip(lengths[:-1], lengths[1:], strict=True))
        monotone = monotone and optima_rise and length_falls
        summary.append(
            {
                "L": n_sites,
                "K": NK_K,
                "mean_local_optima": optima,
                "mean_correlation_length": lengths,
                "local_optima_rise_with_K": optima_rise,
                "correlation_length_falls_with_K": length_falls,
            }
        )

    return monotone, {"by_size": summary}, cases


def optimum_survey() -> tuple[dict, list[dict]]:
    """Where does each family put its optimum, and what does it cost to compile?

    The docs/notes.md requirement, applied to every family at once. A family whose optimum stays
    at the master sequence while ruggedness varies is usable for error-threshold statements;
    one whose optimum wanders is not, whatever else it has to recommend it.
    """
    cases: list[dict] = []
    n_sites = 8

    for label, build in families(n_sites):
        weights, optima, strict_optima, autocorrelations, terms = [], [], [], [], []
        for seed in SEEDS:
            fitness = build(seed)
            stats = ruggedness_statistics(fitness)
            weights.append(stats["optimum_hamming_weight"])
            optima.append(stats["n_local_optima"])
            strict_optima.append(strict_local_optima(fitness))
            autocorrelations.append(stats["autocorrelation"])
            if seed < 3:  # the Pauli count is seed-independent in structure, not in value
                terms.append(pauli_term_count(diagonal_hamiltonian(fitness, MU)))
        cases.append(
            {
                "survey": True,
                **label,
                "L": n_sites,
                "mean_optimum_hamming_weight": float(np.mean(weights)),
                "max_optimum_hamming_weight": int(np.max(weights)),
                "mean_local_optima": float(np.mean(optima)),
                "mean_strict_local_optima": float(np.mean(strict_optima)),
                "mean_autocorrelation": float(np.mean(autocorrelations)),
                "mean_pauli_terms": float(np.mean(terms)),
                # The master sequence is Hamming weight zero. A family that keeps its
                # optimum there is the one an error-threshold sweep can use.
                "keeps_master_sequence": bool(np.mean(weights) < 0.5),
            }
        )

    usable = [c for c in cases if c["keeps_master_sequence"]]
    return (
        {
            "families_surveyed": len(cases),
            "families_keeping_the_master_sequence": [
                {k: v for k, v in c.items() if k in {"family", "roughness", "block_size"}}
                for c in usable
            ],
        },
        cases,
    )


def run() -> tuple[bool, dict, list[dict]]:
    """Compute the three criteria and the optimum survey, separately from reporting them.

    Kept apart from `main` so the summary printing can be replayed against a committed
    artefact in milliseconds. G-5 passed its science and then died formatting the result,
    and the test that should have caught it exercised only this half.
    """
    started = time.monotonic()

    one_ok, one, one_cases = criterion_one()
    two_ok, two, two_cases = criterion_two()
    three_ok, three, three_cases = criterion_three()
    survey, survey_cases = optimum_survey()

    passed = bool(one_ok and two_ok and three_ok)
    measured = {
        "criterion_1_reproduction": {"passed": one_ok, **one},
        "criterion_2_nk_k0_is_additive": {"passed": two_ok, **two},
        "criterion_3_monotone_ruggedness": {"passed": three_ok, **three},
        "optimum_survey": survey,
        "seconds": round(time.monotonic() - started, 2),
    }
    return passed, measured, one_cases + two_cases + three_cases + survey_cases


def main() -> int:
    passed, measured, cases = run()
    one = measured["criterion_1_reproduction"]
    two = measured["criterion_2_nk_k0_is_additive"]
    three = measured["criterion_3_monotone_ruggedness"]
    one_ok, two_ok, three_ok = one["passed"], two["passed"], three["passed"]

    path = write_gate_record(
        gate="G-3",
        work_package="wp3",
        threshold={
            "criterion_1": "every landscape reproduces exactly from its seed",
            "criterion_2": f"NK at K = 0 equals additive to {ADDITIVE_TOLERANCE}",
            "criterion_3": "local optima rise and correlation length falls with K, in the "
            "seed mean, at L = 10 and L = 12",
            "registered_in": "docs/protocol.md section 7, parameters in revision 14",
        },
        measured=measured,
        passed=passed,
        cases=cases,
        notes=(
            "The optimum survey is that requirement applied to all seven families. "
            "A family whose global optimum wanders away from the master sequence as "
            "ruggedness rises cannot carry an error-threshold statement, because the "
            "threshold is defined by delocalisation away from that sequence."
        ),
    )

    print(f"G-3: {len(cases)} cases in {measured['seconds']} s\n")
    print(
        f"  Criterion 1, reproduction: {one['n_landscapes']} landscapes, "
        f"{one['mismatches']} mismatches  {'PASS' if one_ok else 'FAIL'}"
    )
    print(
        f"  Criterion 2, NK K=0 additive: worst {two['worst_max_abs_error']:.3e}  "
        f"{'PASS' if two_ok else 'FAIL'}"
    )
    print(f"  Criterion 3, monotone ruggedness: {'PASS' if three_ok else 'FAIL'}")
    for row in three["by_size"]:
        print(f"    L={row['L']}  K={row['K']}")
        print(f"      local optima       {[round(v, 1) for v in row['mean_local_optima']]}")
        print(f"      correlation length {[round(v, 3) for v in row['mean_correlation_length']]}")

    print(f"\n  Optimum survey at L = 8, mean over {len(SEEDS)} seeds")
    print(
        f"    {'family':26s} {'opt wt':>7} {'optima':>7} {'strict':>7} "
        f"{'autocorr':>9} {'pauli':>6} {'keeps master':>13}"
    )
    # The survey rows are the ones carrying an optimum weight; the criteria rows do not.
    for case in [c for c in cases if "mean_optimum_hamming_weight" in c]:
        name = case["family"]
        for extra in ("roughness", "block_size", "K"):
            if extra in case:
                name = f"{name} {extra}={case[extra]}"
        print(
            f"    {name:26s} {case['mean_optimum_hamming_weight']:>7.2f} "
            f"{case['mean_local_optima']:>7.1f} {case['mean_strict_local_optima']:>7.1f} "
            f"{case['mean_autocorrelation']:>9.3f} "
            f"{case['mean_pauli_terms']:>6.0f} {str(case['keeps_master_sequence']):>13}"
        )

    print(f"\n  record  {path}")
    print(f"  G-3: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
