"""Fitness landscapes, in the spin convention.

Families: additive with optional pairwise epistasis, permutation-symmetric (class-dependent)
fitness including the single peak, NK, spin glass, Rough Mount Fuji, House of Cards and block.

Fitness is written in the spin convention

    f(sigma) = sum_i a_i z_i + sum_{i<j} b_ij z_i z_j

where ``z_i = +1`` when site i is wild type and ``z_i = -1`` when it is mutated, the
eigenvalue of Pauli Z under the encoding in which ``|0>`` is wild type and ``|1>`` mutated.
The projector form ``a_i (I + Z_i) / 2`` is not used; mixing the two gives a wrong
distribution without any error.

Genotype indexing follows `quasarstack.io.conventions`: the fitness vector has length ``2**L``,
and entry j is the fitness of the genotype whose site i is mutated exactly when bit i of j is
set.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# A fitness vector has length 2**L; the guard catches argument errors well above the sizes used.
MAX_SITES = 24


def _check_sites(n_sites: int) -> None:
    if n_sites < 1:
        raise ValueError(f"n_sites must be at least 1, got {n_sites}")
    if n_sites > MAX_SITES:
        raise ValueError(
            f"n_sites = {n_sites} would need a fitness vector of 2**{n_sites} entries; "
            f"the guard is set at {MAX_SITES}"
        )


def spin_matrix(n_sites: int) -> NDArray[np.int8]:
    """Return the ``(n_sites, 2**n_sites)`` array of z values.

    Entry ``[i, j]`` is +1 when site i of genotype j is wild type and -1 when it is mutated. Held
    as int8 so it can be reused across the pairwise loop.
    """
    _check_sites(n_sites)
    index = np.arange(1 << n_sites, dtype=np.int64)
    z = np.empty((n_sites, 1 << n_sites), dtype=np.int8)
    for site in range(n_sites):
        z[site] = 1 - 2 * ((index >> site) & 1).astype(np.int8)
    return z


def additive_fitness(
    a: NDArray[np.float64], b: NDArray[np.float64] | None = None
) -> NDArray[np.float64]:
    """Fitness vector for an additive landscape with optional pairwise epistasis.

    Parameters
    ----------
    a
        Length-L array of per-site coefficients. Positive ``a_i`` makes wild type fitter at
        site i.
    b
        Optional ``(L, L)`` array of pairwise couplings. Only the strict upper triangle is
        read, so ``b[i, j]`` for ``i < j`` is the coupling and everything else is ignored.
        Passing a symmetric matrix therefore does not double count.

    Returns
    -------
    ndarray
        Length ``2**L`` fitness vector.
    """
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"a must be one-dimensional, got shape {a.shape}")
    n_sites = a.size
    z = spin_matrix(n_sites)

    fitness = np.zeros(1 << n_sites, dtype=np.float64)
    for site in range(n_sites):
        fitness += a[site] * z[site]

    if b is not None:
        b = np.asarray(b, dtype=np.float64)
        if b.shape != (n_sites, n_sites):
            raise ValueError(f"b must have shape ({n_sites}, {n_sites}), got {b.shape}")
        for i in range(n_sites):
            for j in range(i + 1, n_sites):
                if b[i, j] != 0.0:
                    fitness += b[i, j] * (z[i] * z[j])

    return fitness


def class_fitness(f_by_class: NDArray[np.float64]) -> NDArray[np.float64]:
    """Fitness vector for a permutation-symmetric landscape.

    Parameters
    ----------
    f_by_class
        Length ``L + 1`` array; entry d is the fitness shared by every genotype with exactly
        d mutated sites.

    Returns
    -------
    ndarray
        Length ``2**L`` fitness vector.
    """
    f_by_class = np.asarray(f_by_class, dtype=np.float64)
    if f_by_class.ndim != 1 or f_by_class.size < 2:
        raise ValueError(
            f"f_by_class must be one-dimensional of length L+1, got {f_by_class.shape}"
        )
    n_sites = f_by_class.size - 1
    _check_sites(n_sites)
    index = np.arange(1 << n_sites, dtype=np.uint64)
    weights = np.bitwise_count(index).astype(np.int64)
    return f_by_class[weights]


def single_peak_classes(n_sites: int, height: float) -> NDArray[np.float64]:
    """Class fitnesses for the sharp-peak landscape.

    The master sequence, with zero mutated sites, has fitness ``height``; every other genotype
    has fitness zero. Both the analytic solution and the exact class solver apply to it.
    """
    _check_sites(n_sites)
    f = np.zeros(n_sites + 1, dtype=np.float64)
    f[0] = float(height)
    return f


def epistatic_classes(n_sites: int, cost: float, exponent: float) -> NDArray[np.float64]:
    """Class fitnesses with tunable epistasis: ``f_d = -cost * d**exponent``.

    Parameters
    ----------
    cost
        Selective cost of the first mutation, since ``f_1 - f_0 = -cost`` for every exponent.
    exponent
        1.0 is additive, and the family then coincides with the additive landscape up to a
        constant. Above 1.0 is synergistic (negative) epistasis: the cost per additional mutation
        grows. Below 1.0 is antagonistic (positive) epistasis: later mutations cost less.

    Notes
    -----
    The first-mutation cost is fixed rather than the total range. Fixing the range would make the
    per-mutation cost near the master sequence scale as ``1/L**exponent``, so the exponent would
    change overall selection strength as well as curvature.

    Synergistic epistasis is expected to raise the error threshold and antagonistic epistasis to
    lower it; G-R.4 measures the direction.
    """
    _check_sites(n_sites)
    if exponent <= 0.0:
        raise ValueError(f"exponent must be positive, got {exponent}")
    d = np.arange(n_sites + 1, dtype=np.float64)
    return -float(cost) * d ** float(exponent)


def nk_fitness(
    n_sites: int,
    k: int,
    seed: int,
    amplitude: float = 1.0,
    neighbourhood: str = "adjacent",
) -> NDArray[np.float64]:
    """Kauffman NK landscape, standardised so that K varies ruggedness only.

    Each site contributes a term depending on its own state and on K others, drawn independently
    and uniformly for every configuration of that neighbourhood. K = 0 is additive and K = L - 1 is
    maximally rugged.

    Parameters
    ----------
    n_sites
        Number of loci, L.
    k
        Epistatic connectivity, from 0 to ``n_sites - 1``.
    seed
        Passed to ``default_rng``. The landscape reproduces exactly from it.
    amplitude
        Selection strength, as the standard deviation of the resulting fitness.
    neighbourhood
        ``"adjacent"`` uses sites i+1 to i+K with wrap-around. ``"random"`` draws K distinct
        partners per site from the same generator.

    Returns
    -------
    ndarray
        Length ``2**L`` fitness vector with zero mean and standard deviation ``amplitude``.

    Notes
    -----
    Raw NK fitness is a mean of L uniform draws, so its spread shrinks as ``1/sqrt(L)`` and grows
    with K; standardising keeps selection strength fixed while K changes the structure. The global
    optimum of an NK landscape sits at a random genotype rather than at all-wild-type, and
    `ruggedness_statistics` reports its location.
    """
    _check_sites(n_sites)
    if not 0 <= k <= n_sites - 1:
        raise ValueError(f"k must be between 0 and {n_sites - 1}, got {k}")
    if neighbourhood not in {"adjacent", "random"}:
        raise ValueError(f"neighbourhood must be 'adjacent' or 'random', got {neighbourhood!r}")

    rng = np.random.default_rng(seed)
    dim = 1 << n_sites
    index = np.arange(dim, dtype=np.int64)
    total = np.zeros(dim, dtype=np.float64)

    for site in range(n_sites):
        if neighbourhood == "adjacent":
            partners = [(site + offset) % n_sites for offset in range(1, k + 1)]
        else:
            others = [s for s in range(n_sites) if s != site]
            partners = list(rng.choice(others, size=k, replace=False)) if k else []

        # Address into this site's table: its own bit first, then its partners' bits.
        address = (index >> site) & 1
        for position, partner in enumerate(partners, start=1):
            address = address | (((index >> partner) & 1) << position)

        table = rng.random(1 << (k + 1))
        total += table[address]

    total /= n_sites
    spread = float(total.std())
    if spread <= 0.0:
        raise ValueError("degenerate NK draw: the landscape is flat")
    standardised: NDArray[np.float64] = amplitude * (total - total.mean()) / spread
    return standardised


def ruggedness_statistics(fitness: NDArray[np.float64]) -> dict[str, float | int]:
    """Structure of a landscape.

    Returns
    -------
    dict
        ``n_local_optima`` counts genotypes at least as fit as every single-mutation neighbour.
        ``autocorrelation`` is the correlation of fitness across single-mutation neighbour pairs,
        which falls toward zero as the landscape becomes rugged. ``optimum_index`` and
        ``optimum_hamming_weight`` give the location of the global optimum.
    """
    fitness = np.asarray(fitness, dtype=np.float64)
    size = fitness.size
    if size < 2 or (size & (size - 1)) != 0:
        raise ValueError(f"fitness length must be a power of two and at least 2, got {size}")
    n_sites = size.bit_length() - 1
    index = np.arange(size, dtype=np.int64)

    is_optimum = np.ones(size, dtype=bool)
    correlations = []
    for site in range(n_sites):
        neighbour = fitness[index ^ (1 << site)]
        is_optimum &= fitness >= neighbour
        correlations.append(float(np.corrcoef(fitness, neighbour)[0, 1]))

    optimum_index = int(np.argmax(fitness))
    return {
        "n_local_optima": int(is_optimum.sum()),
        "autocorrelation": float(np.mean(correlations)),
        "optimum_index": optimum_index,
        "optimum_hamming_weight": int(optimum_index.bit_count()),
        "fitness_range": float(fitness.max() - fitness.min()),
    }


def pairwise_uniform_classes(n_sites: int, a: float, b: float) -> NDArray[np.float64]:
    """Class fitnesses for uniform additive fitness plus uniform pairwise epistasis.

    The form ``f = a sum_i z_i + b sum_{i<j} z_i z_j``, compiled natively as ``a_i Z_i`` and
    ``b_ij Z_i Z_j``. With uniform coefficients it depends only on the total spin ``S = L - 2d``, so
    the class reduction applies:

        f_d = a S + b (S**2 - L) / 2

    Parameters
    ----------
    a
        Per-site fitness. Positive favours wild type.
    b
        Pairwise coupling. Positive is synergistic, negative antagonistic.

    Notes
    -----
    A landscape additive in the surplus has no error threshold: the surplus decays smoothly from
    one with its steepest slope at zero mutation rate. The transition needs either a peak or the
    interaction term here; G-R.4 includes the additive case for comparison.
    """
    _check_sites(n_sites)
    d = np.arange(n_sites + 1, dtype=np.float64)
    spin_sum = n_sites - 2.0 * d
    return a * spin_sum + b * (spin_sum**2 - n_sites) / 2.0


def uniform_additive_classes(n_sites: int, a: float) -> NDArray[np.float64]:
    """Class fitnesses equivalent to an additive landscape with every ``a_i`` equal to ``a``.

    With uniform coefficients ``sum_i z_i = L - 2d`` depends only on the number of mutated sites,
    so both analytic routes apply, which G-R.1 uses to compare the product solution with the
    Hamming-class reduction.
    """
    _check_sites(n_sites)
    d = np.arange(n_sites + 1, dtype=np.float64)
    return a * (n_sites - 2.0 * d)


def spin_glass_fitness(
    n_sites: int, seed: int, amplitude: float = 1.0, field: float = 0.0
) -> NDArray[np.float64]:
    """Sherrington-Kirkpatrick spin glass, ``f = sum_{i<j} J_ij z_i z_j + h sum_i z_i``.

    The couplings are the ``b_ij Z_i Z_j`` terms of the compiled Hamiltonian, so the Pauli expansion
    has ``L(L-1)/2`` weight-two terms plus ``L`` weight-one terms for any seed: a rugged landscape
    with polynomial compilation cost.

    Parameters
    ----------
    field
        Uniform longitudinal field. Zero gives the standard SK model, whose optimum is degenerate
        under global spin flip. A non-zero field breaks that symmetry.

    Notes
    -----
    Couplings are ``+/- 1``, the discrete SK convention. The fitness is standardised to standard
    deviation ``amplitude``, as in `nk_fitness`, so that its spread does not grow with L.
    """
    _check_sites(n_sites)
    rng = np.random.default_rng(seed)
    spins = spin_matrix(n_sites).astype(np.float64)

    total = np.zeros(1 << n_sites, dtype=np.float64)
    for i in range(n_sites):
        for j in range(i + 1, n_sites):
            total += rng.choice([-1.0, 1.0]) * spins[i] * spins[j]
    if field:
        total += field * spins.sum(axis=0)

    spread = float(total.std())
    if spread == 0.0:
        return total
    return np.asarray(amplitude * (total - total.mean()) / spread, dtype=np.float64)


def house_of_cards_fitness(n_sites: int, seed: int, amplitude: float = 1.0) -> NDArray[np.float64]:
    """Every genotype's fitness drawn independently: the maximally rugged reference.

    The ``K = L - 1`` limit of NK. Its Pauli expansion is dense, with all ``2**L`` subsets.
    """
    _check_sites(n_sites)
    rng = np.random.default_rng(seed)
    draws = rng.normal(size=1 << n_sites)
    return np.asarray(amplitude * (draws - draws.mean()) / float(draws.std()), dtype=np.float64)


def rough_mount_fuji_fitness(
    n_sites: int, seed: int, slope: float = 1.0, roughness: float = 0.5
) -> NDArray[np.float64]:
    """Additive gradient plus independent noise: ``f = -slope * d + roughness * eta``.

    The Rough Mount Fuji model varies ruggedness without moving the optimum: the additive part
    points at the all-wild-type genotype at every roughness, so only local structure changes. In
    NK landscapes, by contrast, the optimum sits at a random genotype.

    Parameters
    ----------
    slope
        Fitness lost per mutation.
    roughness
        Standard deviation of the independent noise, in the same units. ``0`` is exactly
        additive; large values approach House of Cards.

    Notes
    -----
    Not standardised: ``roughness / slope`` is the ruggedness parameter, and the caller sets the
    overall scale through ``slope``.
    """
    _check_sites(n_sites)
    if slope < 0.0 or roughness < 0.0:
        raise ValueError(f"slope and roughness must be non-negative, got {slope}, {roughness}")

    rng = np.random.default_rng(seed)
    weights = np.bitwise_count(np.arange(1 << n_sites, dtype=np.uint64)).astype(np.float64)
    noise = rng.normal(size=1 << n_sites)
    return -slope * weights + roughness * noise


def block_fitness(
    n_sites: int, block_size: int, seed: int, amplitude: float = 1.0
) -> NDArray[np.float64]:
    """Genome split into independent blocks, each contributing an arbitrary function.

    ``block_size`` 1 is additive and ``block_size`` L is House of Cards; in between the landscape is
    rugged within blocks and additive across them. Epistasis is bounded in range, so the Pauli
    expansion is dense within a block and empty across blocks, with at most
    ``ceil(L / b) * 2**b`` terms.

    When ``block_size`` does not divide ``n_sites`` the last block is shorter.
    """
    _check_sites(n_sites)
    if not 1 <= block_size <= n_sites:
        raise ValueError(f"block_size must be between 1 and {n_sites}, got {block_size}")

    rng = np.random.default_rng(seed)
    index = np.arange(1 << n_sites, dtype=np.int64)
    total = np.zeros(1 << n_sites, dtype=np.float64)

    for start in range(0, n_sites, block_size):
        width = min(block_size, n_sites - start)
        table = rng.normal(size=1 << width)
        total += table[(index >> start) & ((1 << width) - 1)]

    spread = float(total.std())
    if spread == 0.0:
        return total
    return np.asarray(amplitude * (total - total.mean()) / spread, dtype=np.float64)
