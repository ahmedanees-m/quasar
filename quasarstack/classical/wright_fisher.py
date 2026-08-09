"""Wright-Fisher forward simulation: the finite-population baseline. WP4.

The honest baseline. The quasispecies is a deterministic infinite-population object, and the
first question any reviewer asks is whether a quantum method is being compared against
something a population geneticist would actually run. This is that something, and the
execution plan is explicit that it must be well optimised so it cannot be called a strawman.

Why this runs in count space, and why that is not a shortcut
------------------------------------------------------------

The obvious implementation carries `N` individuals and touches each one every generation, so
a generation costs `O(N L)` and `N = 10^6` is slow enough to force a small sweep. But
individuals are exchangeable: nothing depends on which individual carries a genotype, only on
how many do. Carrying the `2^L` counts instead makes a generation cost `O(L 2^L)`,
**independent of N**, and the result is not an approximation.

- **Selection** is a multinomial draw of `N` from probabilities proportional to `c_g w_g`,
  which is exactly the Wright-Fisher definition.
- **Mutation** factorises over sites, and for one site every individual flips independently
  with probability `u`. So the number of carriers of genotype `g` that flip site `i` is
  `Binomial(c_g, u)`, drawn for all `g` at once, and sites are applied in turn because the
  flips commute.

At `L = 8` a generation is about two thousand operations whatever `N` is, which is what makes
the declared sweep to `N = 10^6` affordable at ten seeds.

The discrete-time correspondence, which is a real bias and is reported as one
------------------------------------------------------------------------------

Wright-Fisher is discrete-generation; Crow-Kimura is continuous-time. They agree only in the
limit of small steps. This module takes an explicit `dt`: fitness weights are `1 + f dt` and
the per-site mutation probability is `mu dt`, so `dt -> 0` recovers the Crow-Kimura generator
the analytic oracle solves.

That leaves two error sources that must not be conflated. **Sampling error** falls as
`1 / sqrt(N)` and as the sampling window grows. **Discretisation bias** falls with `dt`. A
convergence study that varies only `N` will therefore plateau at the discretisation bias and
look like a failure to converge. `time_step_bias` measures the second separately so the
plateau can be attributed rather than puzzled over.

`dt` cannot be taken to zero at fixed `N`
------------------------------------------

The two knobs are not independent, and the way they interact is the opposite of the
intuitive one. Wright-Fisher resamples the whole population once per **generation**, so the
genetic drift it injects is `1 / N` per generation, which is `1 / (N dt)` per unit of
simulated time. Halving `dt` at fixed `N` and fixed simulated time therefore **doubles** the
accumulated drift while it halves the discretisation bias.

Measured directly: at `N = 10^5`, `L = 8`, `mu = 0.10`, holding simulated time fixed while
`dt` goes 0.04, 0.02, 0.01, 0.005, the distance to the analytic quasispecies stays flat at
0.024, 0.015, 0.016, 0.015 while the equilibration drift climbs 0.019, 0.030, 0.040, 0.052.
The bias was already negligible by `dt = 0.02` and everything after that is added noise.

So the continuous-time limit needs `N dt` held constant, not `N` held constant.
`time_step_bias` scales the population with `1 / dt` by default for exactly this reason, and
the un-scaled behaviour is available for anyone who wants to see the trade-off directly.
"""

from __future__ import annotations

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

    Exact, not an approximation: for a single site each individual flips independently, so
    the number of carriers of each genotype that flip is binomial, and the sites commute so
    applying them in sequence gives the same law as applying them at once.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"mutation probability must be in [0, 1], got {probability}")
    counts = counts.copy()
    for site in range(n_sites):
        flipped = rng.binomial(counts, probability)
        counts = counts - flipped
        # np.add.at rather than fancy-index assignment: the destination indices are a
        # permutation here so it would happen to work, but only by accident of this
        # particular kernel, and the accumulating form stays correct if that changes.
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
        Genotype the population starts on. Defaults to the fittest, which is the least
        favourable choice for the baseline in one specific sense worth naming: starting on
        the eventual mode shortens burn-in and so flatters the method. It is used anyway
        because the alternative, starting uniformly, takes far longer to equilibrate at
        small mutation rates and the burn-in diagnostic below is what actually guards
        against a short run being mistaken for a converged one.

    Returns
    -------
    dict
        ``distribution`` is the post-burn-in time average, L1-normalised. ``burn_in_drift``
        compares the first and second halves of the sampling window; a value that is not
        small means the chain had not equilibrated and the distribution should not be used.
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
    """Average over independent chains, and report the spread between them.

    The spread across seeds is the honest error bar on a stochastic baseline. Reporting only
    the pooled mean would make the baseline look more precise than it is, and the comparison
    against a deterministic quantum method would inherit that.
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
        "max_burn_in_drift": max(float(r["burn_in_drift"]) for r in runs),
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

    Two things are held fixed so that only the discretisation changes. The generation count
    scales as ``1 / dt``, so every row covers the same **simulated time** rather than the
    same number of steps. And, unless ``scale_population`` is turned off, the population
    scales as ``1 / dt`` too, so every row carries the same **physical genetic drift**:
    Wright-Fisher resamples once per generation, so drift is ``1 / (N dt)`` per unit time and
    shrinking ``dt`` at fixed ``N`` adds noise faster than it removes bias. The module
    docstring has the measurement that establishes this.

    Set ``scale_population=False`` to see the trade-off rather than to correct for it.
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
                "max_burn_in_drift": float(result["max_burn_in_drift"]),
                "max_pairwise_tv_between_seeds": float(result["max_pairwise_tv_between_seeds"]),
            }
        )
    return rows
