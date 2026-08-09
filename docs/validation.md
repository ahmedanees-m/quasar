# Validation: the gate system, and what each gate is defending against

## Why the ruler is built first

Every downstream method in this project is judged against the analytic oracle. So the
oracle itself cannot be judged against anything downstream. It is checked against
brute-force exact diagonalisation, which shares none of its assumptions: the oracle never
forms the 2^L generator, and exact diagonalisation never uses the structure the oracle
exploits. Agreement between them is therefore evidence about both. Agreement between two
implementations of the same idea would be evidence about neither.

This ordering is not a preference. The planning documents record three implementation bugs
in the earlier, now lost, codebase, and every one of them produced output that looked
entirely reasonable:

1. **L2 in place of L1 normalisation** in the imaginary-time iteration, converging to a
   parity-locked distribution that was non-negative, summed to one, and was not the
   biological steady state.
2. **A fitness-convention mismatch**, projector `a_i (I + Z_i)/2` against spin `a_i Z_i`,
   which dropped cosine similarity to 0.47 on rugged landscapes.
3. **A wrong Motta generator equation**, an element-wise gradient that vanishes for real
   states, making the energy ascend rather than descend.

None of these is visible by inspection. All three are visible against a number that theory
fixes in advance. Those bugs belong to code that no longer exists, so they are reported here
as methodological history rather than claimed as results of this repository. What this
repository does is lock each of them out by construction, so that a recurrence is a test
failure rather than a plausible-looking plot.

## The conventions each historical failure maps to

| Failure mode | What locks it now |
|---|---|
| L1 against L2 normalisation | One decode boundary, `quasarstack.io.conventions.normalise_l1`, with the sign-definiteness argument written into its docstring. Everything upstream is L2, everything downstream is a probability distribution. |
| Projector against spin fitness | The spin convention is the only representation in `quasarstack.classical.landscapes`, checked against hand-computed L = 2 values in `tests/unit/test_landscapes.py`. |
| Endianness | Every bitstring conversion routes through `quasarstack.io.conventions`, with the Qiskit little-endian reversal isolated in one function and round-tripped by property test. |
| Dense diagonalisation at a size that will not fit | `assert_dense_allowed` raises immediately rather than letting the machine swap. |

## The gates

Each gate in `GATES.md` is an executable test under `tests/gates/` and writes a JSON record
under `results/`. A gate with no artefact has not passed. Records carry the commit, the
image tag, whether the tree was dirty, the interpreter version, and the SHA-256 of
`GATES.md` itself, so the claim that a threshold was registered before the run is checkable
rather than asserted.

### G-R.1, the oracle against exact diagonalisation

The first gate, and the one everything else leans on.

**Two analytic routes, one brute-force reference.** Additive fitness makes the generator a
sum of commuting single-site operators, so the Perron eigenvector is a product state with a
per-site closed form and no eigensolver at any L. Permutation-symmetric fitness reduces to
an (L+1)-dimensional tridiagonal problem on Hamming classes. The two overlap exactly when
every `a_i` is equal, and on that overlap the gate checks the closed form against the class
reduction as well as against exact diagonalisation, which makes it a three-way agreement
rather than a pairwise one.

**Diagnostics that are kept rather than discarded.** The spectral gap is recorded for every
case, because the Perron eigenvector is only well conditioned while that gap is open. No
case is excluded on the basis of its gap: a small-gap failure would be a finding that feeds
WP1, not a case to drop.

**One numerical detail worth naming.** The per-site ratio
`(sqrt(a^2 + mu^2) - a) / mu` subtracts two nearly equal numbers when selection is strong,
which is precisely the regime the error-threshold sweep spends most of its time in. It is
rationalised to `mu / (sqrt(a^2 + mu^2) + a)` on that branch. The direct form is kept for
`a < 0`, where both terms of the numerator are positive and it is the stable one.

### G-R.2, the compiled Hamiltonian against the oracle

G-R.1 established that the oracle can be trusted. G-R.2 spends that trust: the
biology-to-qubit compiler is judged by whether the ground state of the Pauli operator it
emits is the quasispecies the oracle predicts.

**The registered criterion is not the sharpest check, and the run says so.** Cosine
similarity between eigenvectors cannot see an endianness error, because permuting the
computational basis leaves the spectrum untouched and permutes both vectors the same way.
So the run also compares the compiled operator entry by entry against the generator
assembled independently in `analytic/exact_diag.py`. That comparison is recorded as a
diagnostic rather than promoted to the gate, because the threshold was registered in cosine
and thresholds do not move. It agreed to 3.6e-15.

**Why the identity term is carried.** The compiler emits `H = -W` including the `mu L I`
term. Dropping it would shift every eigenvalue without touching any eigenvector, so every
spectral check would still pass and the operator-level comparison above would fail. Keeping
it is what makes that comparison exact rather than approximate.

**Two compiler routes, cross-checked.** The structured build writes `a_i Z_i` and
`b_ij Z_i Z_j` directly. The Walsh-Hadamard build recovers the exact Pauli decomposition of
any diagonal operator from its values alone. On additive landscapes both must produce the
same operator, and disagreement would localise a bug to one of them. They agree to 1.8e-15.

**An observation from the term counts, not claimed as a result.** The Walsh-Hadamard route
finds that a class-dependent landscape quadratic in Hamming distance has Pauli support only
up to weight two, 28 terms at L = 6 rather than the 70 a dense weight-two expansion would
need, while an exponential class function is dense at all 2^L. Sparsity in Pauli space
tracks polynomial degree in Hamming distance, not permutation symmetry. That distinction
belongs in the WP1 resource-scaling analysis, where it can be stated properly.

## What reproduction actually showed

Both gates were run more than once, at different commits, inside the same image. Every
scientific field came back bit-identical: the same max absolute error to every digit, the
same cosines, the same term counts. Only `git_sha`, `timestamp`, and the elapsed seconds
moved.

That is the reproducibility claim in its checkable form, and it is why `make gates` is a
verification step rather than a step that produces commits. A rerun that changes a
scientific field is a finding, not noise, and gets committed and explained. See
`DECISIONS.md` ADR-0009.

## Running them

```bash
make gates
```

runs every gate inside the pinned image. A single gate:

```bash
python experiments/wp_r_rebuild/g_r_1_oracle_vs_ed.py
```

`pytest -m gate` asserts the thresholds. That suite runs nightly and at release rather than
on every push, because the gates are slow by nature: G-R.1 alone is about 1700 comparisons,
each of which diagonalises a full generator.

## When a gate fails

Open a `validation_failure` issue. The template asks what failed, the measured value against
the registered threshold, the hypothesis, and whether the threshold was lowered. The answer
to the last question is always no. If the reasoning behind a threshold turns out to have
been wrong, that is an amendment appended to `GATES.md` with a dated rationale, and it is a
separate discussion from the failure that exposed it.
