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

### G-R.4, the error threshold

The gate asks whether the qubit route sees the localisation-delocalisation transition where
the analytic theory puts it. The surplus is computed at every point of a mutation-rate sweep
from the ground state of the compiled Pauli Hamiltonian and from the analytic Hamming-class
reduction, and the gate is the maximum disagreement.

**Where the threshold sits is a diagnostic, not a pass condition.** So is the direction in
which epistasis moves it. The planning documents state an expected direction, and a gate
that required the expected answer would not be a measurement. That distinction earned its
keep: one of the two epistasis directions came out disagreeing with the documents, and
because it was never a pass condition there was no pressure on it.

**Two location measures, because one is not always defined.** The susceptibility peak,
`chi = -dm/dmu`, is the natural definition and is used wherever the peak is interior to the
sweep. A landscape additive in the surplus has no interior peak at all: it decays
monotonically with its steepest slope at zero mutation rate, so the peak sits on the sweep
boundary and carries no information. `locate_threshold` reports whether the peak was
interior rather than returning a boundary index as though it were an answer. The
half-surplus crossover is defined for any monotone decay and is what makes families
comparable.

**A shortcut, and the guard on it.** The compiled operator is linear in the mutation rate,
so the sweep compiles the selection and mutation parts once each and assembles per point.
At L = 8 that is the difference between summing 256 Pauli terms 300 times and doing it
twice. The assembly is checked against a directly compiled operator at one mutation rate per
case, and the check is recorded in the artefact; it came back at exactly zero. A speed
optimisation that quietly stopped matching the thing it stands in for would invalidate the
gate, so it is verified rather than trusted.

**Two normalisation mistakes, both found before the recorded run and both disclosed in
`GATES.md` revision 4 rather than quietly fixed.** An epistatic family that fixed the total
fitness range made the per-mutation cost near the master scale as 1/L, so selection vanished
and the exponent varied overall selection strength instead of epistasis. And a pairwise
coupling held at fixed strength across sizes made the total interaction grow as L squared,
which reversed the apparent direction of the epistasis effect between L = 6 and L = 8.

### G-R.6, varQITE at constant depth

Accuracy alone would be satisfied by any competent solver. What makes varQITE a near-term
method is that the circuit never changes: only its parameters move, so evolving ten times
longer costs no extra depth. That is the registered criterion, and the run measures it by
evolving each configuration twice and comparing the circuits, before and after
transpilation. Comparing only the written circuit would be the weaker check, because a run
that left an angle near zero could have that rotation optimised away.

**Two design choices that are easy to get wrong.**

*Convergence is judged on the state, not the parameters.* The ansatz has gauge directions,
combinations of parameters that move without changing the state at all, so the parameter
update norm keeps fluctuating long after the state has settled. Measured on a rugged L = 4
instance: at step 400 the state was already at cosine 0.99991 against its reference while
the largest parameter update was still 5.1e-2. A parameter-space criterion would trigger at
an arbitrary moment determined by where the gauge drift happened to be, or never trigger.

*The ansatz depth was chosen by a scan, and the scan is disclosed.* varQITE holds a fixed
circuit, so its accuracy is capped by what that circuit can represent. An ansatz too shallow
to hold the answer fails for reasons unrelated to the method. Depth is a method parameter
rather than an acceptance threshold, so choosing it adequately is legitimate; choosing it
invisibly is not. The numbers are in `GATES.md` revision 6.

**What the scan found, which is more interesting than the rule it produced.** The ansatz
depth needed grows faster than the system does. At L = 6, reps = L reaches only 0.998137 on
K = 2 instances and reps = L + 2 is required. The gate keeps that as a diagnostic, measuring
accuracy at reps 4, 6 and 8, rather than letting it vanish into a configuration constant. It
prefigures the barren-plateau ceiling that G-R.9 measures and is directly relevant to how far
Route A can be pushed.

**The check that licenses the word "hardware-faithful".** varQITE is computed here by
differentiating a state vector, which no quantum computer can do. That is only a legitimate
stand-in because the same two objects come from circuit measurements: the McLachlan force is
the energy gradient via parameter shift, and the metric is the Fubini-Study tensor via
fidelity shift. `verify_hardware_route` recomputes both that way, touching no derivative
state, and the gate records the comparison. Without it, the claim would rest on the
literature rather than on this code.

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
