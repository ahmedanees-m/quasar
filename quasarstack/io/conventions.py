"""The single source of truth for index, ordering, and normalisation conventions.

Endianness errors are silent. A wrong bitstring-to-integer convention produces a
distribution that is a permutation of the right one, which is non-negative, sums to one,
and looks entirely reasonable. So every conversion in the project goes through this module
and nowhere else, and the round trip is a unit test.

Two conventions are in play and must never be confused.

**Genotype order (biology).** Site 0 is the leftmost locus of the genotype string, which is
how a biologist writes a sequence and how the landscape generators index sites.

**Qiskit order (measurement).** Qiskit is little-endian: in a bitstring returned by a
measurement, the rightmost character is qubit 0. A Qiskit counts key must therefore be
reversed before it is read as a genotype.

The mapping from qubit index to site index is the identity: qubit i carries site i. The
only thing that differs is the direction in which a *string* is read.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def genotype_to_index(genotype: str) -> int:
    """Convert a genotype string to its integer index in the state vector.

    Parameters
    ----------
    genotype
        Genotype in biology order: character 0 is site 0. `"0"` is wild type at that site,
        `"1"` is mutated.

    Returns
    -------
    int
        Index into a length-2^L vector, using the Qiskit little-endian layout in which site
        i contributes ``2**i``.

    Examples
    --------
    >>> genotype_to_index("000")
    0
    >>> genotype_to_index("100")   # site 0 mutated
    1
    >>> genotype_to_index("001")   # site 2 mutated
    4
    """
    if not genotype or any(c not in "01" for c in genotype):
        raise ValueError(f"genotype must be a non-empty string of 0 and 1, got {genotype!r}")
    return sum(1 << site for site, char in enumerate(genotype) if char == "1")


def index_to_genotype(index: int, n_sites: int) -> str:
    """Convert a state-vector index back to a genotype string in biology order.

    Parameters
    ----------
    index
        Index into a length-2^L vector.
    n_sites
        Number of loci, L.

    Returns
    -------
    str
        Genotype string of length ``n_sites``, character 0 being site 0.

    Examples
    --------
    >>> index_to_genotype(1, 3)
    '100'
    >>> index_to_genotype(4, 3)
    '001'
    """
    if n_sites < 1:
        raise ValueError(f"n_sites must be at least 1, got {n_sites}")
    if not 0 <= index < (1 << n_sites):
        raise ValueError(f"index {index} out of range for {n_sites} sites")
    return "".join("1" if index >> site & 1 else "0" for site in range(n_sites))


def qiskit_bitstring_to_genotype(bitstring: str) -> str:
    """Convert a Qiskit measurement bitstring to a genotype string in biology order.

    Qiskit counts keys are little-endian: the rightmost character is qubit 0. Reading such
    a key directly as a genotype silently reverses every sequence.

    Examples
    --------
    >>> qiskit_bitstring_to_genotype("001")   # qubit 0 is the rightmost character
    '100'
    """
    cleaned = bitstring.replace(" ", "")
    if not cleaned or any(c not in "01" for c in cleaned):
        raise ValueError(f"expected a binary string, got {bitstring!r}")
    return cleaned[::-1]


def hamming_weight(index: int) -> int:
    """Number of mutated sites in the genotype at this state-vector index."""
    return int(index).bit_count()


def hamming_class_collapse(distribution: NDArray[np.float64], n_sites: int) -> NDArray[np.float64]:
    """Collapse a full 2^L distribution onto its L+1 Hamming classes.

    The storage policy stores this collapsed form for every cell and the full distribution
    only for L <= 12 and designated representative cells, which is what keeps the result set
    small enough to archive. See `DECISIONS.md` ADR-0008.

    Parameters
    ----------
    distribution
        Length-2^L array of probabilities.
    n_sites
        Number of loci, L.

    Returns
    -------
    ndarray
        Length ``n_sites + 1`` array; entry k is the total probability of all genotypes with
        exactly k mutated sites.
    """
    expected = 1 << n_sites
    if distribution.shape != (expected,):
        raise ValueError(
            f"expected shape ({expected},) for {n_sites} sites, got {distribution.shape}"
        )
    weights = np.array([hamming_weight(i) for i in range(expected)])
    collapsed = np.zeros(n_sites + 1, dtype=np.float64)
    np.add.at(collapsed, weights, distribution)
    return collapsed


def normalise_l1(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the L1-normalised, non-negative distribution corresponding to a state vector.

    This is the decode boundary. Everything upstream of it works in L2; everything
    downstream is a biological probability distribution. The absolute value is safe here
    only because the target is the ground state of the stoquastic operator
    ``-(H_sel + H_mut)``, whose Perron vector is sign-definite, so taking the modulus
    recovers the intended ray rather than destroying sign information. See `DECISIONS.md`
    ADR-0003.
    """
    magnitude = np.abs(vector).astype(np.float64)
    total = float(magnitude.sum())
    if total <= 0.0:
        raise ValueError("cannot L1-normalise a vector whose entries sum to zero")
    normalised: NDArray[np.float64] = magnitude / total
    return normalised


def decode_from_measurement(
    frequencies: NDArray[np.float64], floor: float = 0.0
) -> NDArray[np.float64]:
    """Recover the quasispecies from measured frequencies, by taking the square root.

    **This step is not optional and its absence is silent.** The Perron vector of the
    mutation-selection generator *is* the quasispecies, and the ground state of the
    stoquastic Hamiltonian *is* that vector, so the circuit holds the distribution in its
    **amplitudes**. A computational-basis measurement returns ``|psi|^2``. Those are
    different distributions, and the difference is not small: measured at L = 3 on an
    additive landscape, the squared distribution sits at total-variation distance **0.224**
    from the quasispecies.

    It is also exactly the kind of error this project is built to catch, because the wrong
    answer looks right. The squared distribution is non-negative, normalised, and peaked on
    the same genotype, and its cosine similarity to the quasispecies is 0.987, which passes
    an eyeball check and most thresholds. Total variation is what exposes it, which is why
    `quasarstack.scoring.metrics` reports both and why `GATES.md` section 11.4 lets total
    variation decide.

    Taking the square root inverts it exactly in the infinite-shot limit, and improves the
    statistics rather than degrading them: a relative error ``eps`` on a frequency becomes
    ``eps / 2`` on its square root. The one hazard is a bin that received no counts, which
    returns exactly zero and asserts a genotype is impossible on finite evidence; ``floor``
    exists to soften that when the caller wants it to.

    Parameters
    ----------
    frequencies
        Measured relative frequencies over the ``2**L`` computational basis states.
    floor
        Optional value substituted for empty bins before taking the root.

    Returns
    -------
    ndarray
        The L1-normalised quasispecies distribution.
    """
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if np.any(frequencies < 0.0):
        raise ValueError("measured frequencies cannot be negative")
    working = np.where(frequencies <= 0.0, floor, frequencies)
    amplitudes = np.sqrt(working)
    total = float(amplitudes.sum())
    if total <= 0.0:
        raise ValueError("every bin was empty; there is nothing to decode")
    decoded: NDArray[np.float64] = amplitudes / total
    return decoded


def assert_dense_allowed(n_sites: int, limit: int = 12) -> None:
    """Guard against building a dense 2^L by 2^L operator at a size that will not fit.

    Dense float64 at L = 14 is about 2.1 GB and at L = 16 about 34 GB. The compute VM has
    62 GB of RAM shared with other work, so this is a guard rail rather than a suggestion.
    See `GATES.md` section 1 and `DECISIONS.md` ADR-0004.
    """
    if n_sites > limit:
        size_gb = (1 << (2 * n_sites)) * 8 / 1024**3
        raise RuntimeError(
            f"dense construction forbidden above L = {limit}; L = {n_sites} would need about "
            f"{size_gb:.1f} GB. Use scipy.sparse.linalg.eigsh instead."
        )
