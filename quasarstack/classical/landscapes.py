"""Fitness landscapes, in the spin convention.

Only the families that work package WP-R needs are here: additive with optional pairwise
epistasis, and permutation-symmetric (class-dependent) fitness, of which the single peak is
a special case. WP3 extends this module with NK, spin glass, Rough Mount Fuji,
House of Cards and Block families, and gate G-3 judges those.

**Convention, binding project-wide.** Fitness is written in the spin convention

    f(sigma) = sum_i a_i z_i + sum_{i<j} b_ij z_i z_j

where ``z_i = +1`` when site i is wild type and ``z_i = -1`` when it is mutated. This is
the eigenvalue of the Pauli Z operator under the project's encoding, in which the qubit
state ``|0>`` is wild type and ``|1>`` is mutated.

The projector form ``a_i (I + Z_i) / 2`` is *not* used. Mixing the two is silent: the
resulting quasispecies is a plausible-looking distribution that is simply wrong. See
`DECISIONS.md` ADR-0002.

Genotype indexing follows `quasarstack.io.conventions`: the fitness vector returned by
these functions has length ``2**L``, and entry j is the fitness of the genotype whose site
i is mutated exactly when bit i of j is set.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# A fitness vector is a dense array of length 2**L. At L = 24 that is 134 MB, which is well
# past anything this project sweeps, so the guard catches an argument mistake rather than
# imposing a real limit.
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

    Entry ``[i, j]`` is +1 when site i of genotype j is wild type and -1 when it is
    mutated. Held as int8 so that the array stays small enough to reuse across the pairwise
    loop instead of being recomputed per term.
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

    The master sequence, meaning zero mutated sites, has fitness ``height``; every other
    genotype has fitness zero. This is the control landscape: the analytic oracle and the
    Dixit-Srivastava-Vishnoi baseline both apply here, and no quantum method is expected to
    offer anything.
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
        Selective cost of the *first* mutation, since ``f_1 - f_0 = -cost`` for every
        exponent.
    exponent
        1.0 is additive: every mutation costs the same, and this family then coincides with
        the additive landscape up to a constant.
        Above 1.0 is **synergistic** (negative) epistasis: the cost per additional mutation
        grows, so mutations are punished harder as they accumulate.
        Below 1.0 is **antagonistic** (positive) epistasis: diminishing returns, where the
        first mutations cost most and later ones cost less.

    Notes
    -----
    The normalisation is deliberate and was corrected once. Fixing the *total* range instead,
    so that the all-mutant genotype sits at a common depth, makes the per-mutation cost near
    the master scale as ``1/L**exponent``. Selection near the master then becomes vanishingly
    weak, the population delocalises at any mutation rate worth sweeping, and the exponent
    ends up varying overall selection strength rather than epistasis. Fixing the
    first-mutation cost is what isolates curvature, which is the thing the exponent is
    supposed to control.

    The expected effect on the error threshold is that synergistic epistasis raises it and
    antagonistic epistasis lowers it, because selection that punishes accumulation more
    steeply holds the population together against a higher mutation rate. That expectation
    is stated in the planning documents. It is *measured* in gate G-R.4, not assumed here,
    and the measured direction is reported whichever way it falls.
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
    """Kauffman NK landscape, standardised so that K varies ruggedness and nothing else.

    Each site contributes a fitness term depending on its own state and on K others. The
    contributions are drawn independently and uniformly for every configuration of that
    neighbourhood, so K tunes how much the effect of one site depends on the rest: K = 0 is
    additive, K = L - 1 is maximally rugged.

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
        ``"adjacent"`` uses sites i+1 to i+K wrapping around, which is deterministic given
        the seed and is the usual choice for a one-dimensional genome. ``"random"`` draws K
        distinct partners per site from the same generator.

    Returns
    -------
    ndarray
        Length ``2**L`` fitness vector, standardised to zero mean and standard deviation
        ``amplitude``.

    Notes
    -----
    The standardisation is deliberate and follows the lesson of `DECISIONS.md` ADR-0011.
    Raw NK fitness is a mean of L uniform draws, so its spread shrinks as ``1/sqrt(L)`` and
    grows with K. Sweeping K on the raw scale would therefore vary selection strength at the
    same time as ruggedness, and any result would be a mixture of the two. Fixing the spread
    leaves K varying only the structure.

    Note also what this family does *not* have: a master sequence. The global optimum of an
    NK landscape sits at a random genotype, not at all-wild-type. Statements about the error
    threshold, which is defined by delocalisation away from a master sequence, do not carry
    over unchanged, and `ruggedness_statistics` reports where the optimum actually is.
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
    """Structure of a landscape, as WP3 task T3.3 requires and ADR-0011 now insists on.

    Returns
    -------
    dict
        ``n_local_optima`` counts genotypes at least as fit as every single-mutation
        neighbour. ``autocorrelation`` is the correlation of fitness across single-mutation
        neighbour pairs, which falls toward zero as the landscape becomes rugged.
        ``optimum_index`` and ``optimum_hamming_weight`` say where the global optimum sits,
        which is the ADR-0011 requirement: a family that silently relocates its optimum is
        not varying ruggedness alone.
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

    The pairwise-epistatic form ``f = a sum_i z_i + b sum_{i<j} z_i z_j``, which the
    Hamiltonian compiler builds natively as ``a_i Z_i`` and ``b_ij Z_i Z_j``. With uniform
    coefficients it depends only on the total spin ``S = L - 2d``, so it is permutation
    symmetric and the analytic class reduction applies:

        f_d = a S + b (S**2 - L) / 2

    Parameters
    ----------
    a
        Per-site fitness. Positive favours wild type.
    b
        Pairwise coupling. **Positive is synergistic**: aligned sites reinforce each other,
        so breaking alignment costs more as mutations accumulate. **Negative is
        antagonistic**, giving diminishing cost.

    Notes
    -----
    Unlike a purely additive landscape, this family has an error threshold to find. A
    landscape additive in the surplus has no catastrophe at all: the surplus decays smoothly
    from one with its steepest slope at zero mutation rate, so there is no interior
    susceptibility peak to locate. The transition needs either a peak, as in the sharp-peak
    landscape, or the interaction term here. That distinction is measured in gate G-R.4 and
    the additive case is kept in the sweep as the control that shows it.
    """
    _check_sites(n_sites)
    d = np.arange(n_sites + 1, dtype=np.float64)
    spin_sum = n_sites - 2.0 * d
    return a * spin_sum + b * (spin_sum**2 - n_sites) / 2.0


def uniform_additive_classes(n_sites: int, a: float) -> NDArray[np.float64]:
    """Class fitnesses equivalent to an additive landscape with every ``a_i`` equal to ``a``.

    With uniform coefficients the additive landscape is permutation symmetric, because
    ``sum_i z_i = L - 2d`` depends only on the number of mutated sites d. That makes it the
    one family reachable by both analytic routes, which is what lets gate G-R.1 cross-check
    the closed-form product solution against the Hamming-class reduction as well as against
    exact diagonalisation.
    """
    _check_sites(n_sites)
    d = np.arange(n_sites + 1, dtype=np.float64)
    return a * (n_sites - 2.0 * d)


def spin_glass_fitness(
    n_sites: int, seed: int, amplitude: float = 1.0, field: float = 0.0
) -> NDArray[np.float64]:
    """Sherrington-Kirkpatrick spin glass, ``f = sum_{i<j} J_ij z_i z_j + h sum_i z_i``.

    The fourth family WP3 task T3.1 asks for, and the one that is already written in the
    project's own convention: the couplings *are* the ``b_ij Z_i Z_j`` terms the Hamiltonian
    compiler builds, so the Pauli expansion is exactly ``L(L-1)/2`` weight-two terms plus
    ``L`` weight-one terms, with no dependence on the seed. That makes it the family where
    ruggedness is high and the compilation cost is still polynomial, which is the opposite
    corner from the single peak and worth having.

    Parameters
    ----------
    field
        Uniform longitudinal field. Zero gives the standard SK model, whose ground state sits
        at a random genotype and whose optimum is degenerate under global spin flip: every
        configuration and its complement have the same fitness. A non-zero field breaks that
        symmetry. `ruggedness_statistics` reports where the optimum lands either way, which
        ADR-0011 requires of every family.

    Notes
    -----
    Couplings are ``+/- 1`` rather than Gaussian, which is the discrete SK convention and
    keeps the coefficient one-norm exactly ``L(L-1)/2`` before standardisation. Standardised
    to standard deviation ``amplitude`` for the same reason `nk_fitness` is: otherwise the
    spread grows with L and a sweep would vary selection strength alongside structure.
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
    """Every genotype's fitness drawn independently. The maximally rugged reference.

    One of the standard models WP3 task T3.2 asks for, so results are comparable with the
    population-genetics literature. It is the ``K = L - 1`` limit of NK and it is the case
    where no compilation structure exists at all: the Pauli expansion is dense, all ``2**L``
    subsets, which makes it the honest worst case for the resource-scaling analysis.
    """
    _check_sites(n_sites)
    rng = np.random.default_rng(seed)
    draws = rng.normal(size=1 << n_sites)
    return np.asarray(amplitude * (draws - draws.mean()) / float(draws.std()), dtype=np.float64)


def rough_mount_fuji_fitness(
    n_sites: int, seed: int, slope: float = 1.0, roughness: float = 0.5
) -> NDArray[np.float64]:
    """Additive gradient plus independent noise: ``f = -slope * d + roughness * eta``.

    The Rough Mount Fuji model, and the family WP3 should prefer for the ruggedness axis,
    because it is the one that varies ruggedness **without moving the optimum**, which is
    exactly what ADR-0011 was written about. The additive part points at the all-wild-type
    genotype at every roughness, so the master sequence stays where the error threshold is
    defined relative to, and only the amount of local structure changes.

    Contrast NK, where the optimum sits at a random genotype whose Hamming weight was
    measured at 3.4 to 4.4 out of 8, and where error-threshold statements therefore do not
    carry over unchanged.

    Parameters
    ----------
    slope
        Fitness lost per mutation, the deterministic gradient.
    roughness
        Standard deviation of the independent noise, in the same units. ``0`` is exactly
        additive; large values approach house-of-cards.

    Notes
    -----
    Not standardised, unlike `nk_fitness` and `spin_glass_fitness`. Here the two components
    are the axis: ``roughness / slope`` is the ruggedness parameter and rescaling the total
    would destroy it. The caller sets the overall scale through ``slope``.
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

    The block model from WP3 task T3.2. Ruggedness is tuned by ``block_size``: size 1 is
    additive, size L is house-of-cards, and in between the landscape is rugged within blocks
    and additive across them. It separates cleanly from NK in one respect worth having, that
    epistasis here is *bounded in range* rather than spread over a neighbourhood, so the
    Pauli expansion is dense within a block and empty across blocks: at most
    ``ceil(L / b) * 2**b`` terms rather than ``2**L``.

    The last block is short when ``block_size`` does not divide ``n_sites``, which is
    recorded rather than padded, because padding would silently make one block less rugged
    than the rest.
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
