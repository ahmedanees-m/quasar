"""Wright-Fisher forward simulation: the finite-population baseline.

The simulation runs in count space
----------------------------------

Individuals are exchangeable, so the state is the vector of ``2^L`` genotype counts rather than
``N`` individuals. A generation then costs ``O(L 2^L)``, independent of ``N``, and the result is
exact rather than approximate.

- **Selection** is a multinomial draw of ``N`` from probabilities proportional to ``c_g w_g``.
- **Mutation** factorises over sites: the number of carriers of genotype ``g`` that flip site
  ``i`` is ``Binomial(c_g, u)``, drawn for all ``g`` at once, and sites are applied in turn
  because the flips commute.

Time step
---------

Wright-Fisher is discrete-generation and Crow-Kimura continuous-time; they agree in the limit of
small steps. Fitness weights are ``1 + f dt`` and the per-site mutation probability is
``mu dt``, so ``dt -> 0`` recovers the Crow-Kimura generator.

Sampling error falls as ``1 / sqrt(N)`` and with the sampling window; discretisation bias falls
with ``dt``. The two are not independent. Resampling once per generation injects drift ``1 / N``
per generation, or ``1 / (N dt)`` per unit time, so halving ``dt`` at fixed ``N`` doubles the
accumulated drift. At ``N = 10^5``, ``L = 8``, ``mu = 0.10`` and fixed simulated time, ``dt`` of
0.04, 0.02, 0.01 and 0.005 give distances to the analytic quasispecies of 0.024, 0.015, 0.016
and 0.015, while the equilibration drift rises 0.019, 0.030, 0.040 and 0.052. The
continuous-time limit therefore holds ``N dt`` constant, and `time_step_bias` scales the
population with ``1 / dt`` by default.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "mutation_step",
    "sample_stationary",
    "selection_step",
    "simulate",
    "time_step_bias",
]


def selection_step(
    counts: NDArray[np.int64], weights: NDArray[np.float64], rng: np.random.Generator
) -> NDArray[np.int64]:
    """One Wright-Fisher selection round: multinomial resampling by fitness weight."""
    total = int(counts.sum())
    unnormalised = counts * weights
    mass = float(unnormalised.sum())
    if mass <= 0.0:
        raise ValueError(
            "every surviving genotype has non-positive weight; dt is too large for this "
            "fitness range and the population has gone extinct"
        )
    return rng.multinomial(total, unnormalised / mass)


def mutation_step(
    counts: NDArray[np.int64],
    n_sites: int,
    probability: float,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """One mutation round, applied site by site in count space.

    For a single site each individual flips independently, so the number of carriers of each
    genotype that flip is binomial; sites commute, so applying them in sequence gives the same law.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"mutation probability must be in [0, 1], got {probability}")
    counts = counts.copy()
    for site in range(n_sites):
        flipped = rng.binomial(counts, probability)
        counts = counts - flipped
        # np.add.at accumulates correctly even if destination indices repeat.
        np.add.at(counts, np.arange(counts.size) ^ (1 << site), flipped)
    return counts


def simulate(
    fitness: NDArray[np.float64],
    mu: float,
    population: int,
    generations: int,
    seed: int,
    dt: float = 0.01,
    burn_in_fraction: float = 0.2,
    initial: int | None = None,
) -> dict[str, object]:
    """Run the chain and return the time-averaged genotype distribution.

    Parameters
    ----------
    initial
        Genotype the population starts on. Defaults to the fittest, which shortens burn-in;
        ``burn_in_drift`` reports whether the chain has equilibrated.

    Returns
    -------
    dict
        ``distribution`` is the post-burn-in time average, L1-normalised. ``burn_in_drift``
        compares the first and second halves of the sampling window; a large value means the
        chain had not equilibrated.
    """
    fitness = np.asarray(fitness, dtype=np.float64)
    size = fitness.size
    n_sites = size.bit_length() - 1
    if 1 << n_sites != size:
        raise ValueError(f"fitness length must be a power of two, got {size}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    weights = 1.0 + fitness * dt
    if weights.min() <= 0.0:
        raise ValueError(
            f"dt = {dt} makes the selection weight non-positive for the least fit genotype "
            f"(min 1 + f dt = {weights.min():.3f}); reduce dt"
        )
    probability = mu * dt
    if probability > 0.5:
        raise ValueError(
            f"mu dt = {probability:.3f} exceeds 0.5, so a site is more likely to flip than "
            f"not in one generation and the discrete chain no longer approximates the "
            f"continuous-time model at all; reduce dt"
        )

    rng = np.random.default_rng(seed)
    counts = np.zeros(size, dtype=np.int64)
    counts[int(np.argmax(fitness)) if initial is None else initial] = population

    burn_in = int(round(burn_in_fraction * generations))
    accumulated = np.zeros(size, dtype=np.float64)
    first_half = np.zeros(size, dtype=np.float64)
    second_half = np.zeros(size, dtype=np.float64)
    midpoint = burn_in + (generations - burn_in) // 2

    for generation in range(generations):
        counts = selection_step(counts, weights, rng)
        counts = mutation_step(counts, n_sites, probability, rng)
        if generation >= burn_in:
            accumulated += counts
            if generation < midpoint:
                first_half += counts
            else:
                second_half += counts

    def normalise(vector: NDArray[np.float64]) -> NDArray[np.float64]:
        total = vector.sum()
        return vector / total if total > 0 else vector

    distribution = normalise(accumulated)
    drift = 0.5 * float(np.abs(normalise(first_half) - normalise(second_half)).sum())

    return {
        "distribution": distribution,
        "burn_in_drift": drift,
        "population": population,
        "generations": generations,
        "burn_in": burn_in,
        "dt": dt,
        "effective_time": generations * dt,
    }


def sample_stationary(
    fitness: NDArray[np.float64],
    mu: float,
    population: int,
    generations: int,
    seeds: list[int],
    dt: float = 0.01,
    burn_in_fraction: float = 0.2,
) -> dict[str, object]:
    """Average over independent chains and report the spread between them.

    The spread across seeds is the error bar of the stochastic baseline.
    """
    runs = [
        simulate(fitness, mu, population, generations, seed, dt, burn_in_fraction) for seed in seeds
    ]
    stacked = np.array([r["distribution"] for r in runs])
    pooled = stacked.mean(axis=0)

    pairwise = [
        0.5 * float(np.abs(stacked[i] - stacked[j]).sum())
        for i in range(len(runs))
        for j in range(i + 1, len(runs))
    ]

    return {
        "distribution": pooled,
        "n_seeds": len(seeds),
        "max_burn_in_drift": max(float(cast(float, r["burn_in_drift"])) for r in runs),
        "mean_pairwise_tv_between_seeds": float(np.mean(pairwise)) if pairwise else 0.0,
        "max_pairwise_tv_between_seeds": float(np.max(pairwise)) if pairwise else 0.0,
        "population": population,
        "generations": generations,
        "dt": dt,
    }


def time_step_bias(
    fitness: NDArray[np.float64],
    mu: float,
    reference: NDArray[np.float64],
    steps: list[float],
    population: int,
    generations: int,
    seeds: list[int],
    scale_population: bool = True,
) -> list[dict[str, float]]:
    """Total variation against the analytic quasispecies as ``dt`` shrinks.

    The generation count scales as ``1 / dt``, so every row covers the same simulated time. Unless
    ``scale_population`` is False, the population also scales as ``1 / dt``, so every row carries the
    same genetic drift per unit time (see the module docstring).
    """
    reference = np.asarray(reference, dtype=np.float64)
    rows = []
    for dt in steps:
        factor = steps[0] / dt
        scaled_generations = int(round(generations * factor))
        scaled_population = int(round(population * factor)) if scale_population else population
        result = sample_stationary(fitness, mu, scaled_population, scaled_generations, seeds, dt=dt)
        distribution = np.asarray(result["distribution"])
        rows.append(
            {
                "dt": dt,
                "generations": scaled_generations,
                "population": scaled_population,
                "total_variation": 0.5 * float(np.abs(distribution - reference).sum()),
                "max_burn_in_drift": float(cast(float, result["max_burn_in_drift"])),
                "max_pairwise_tv_between_seeds": float(
                    cast(float, result["max_pairwise_tv_between_seeds"])
                ),
            }
        )
    return rows
