# Notes

Decisions I'd otherwise re-argue with myself in three months. Newest at the bottom. Thresholds
and grids are in `protocol.md`.

## 2026-08-09, starting from the spec

The planning documents describe an implementation with specific numbers (oracle to 3.85e-13,
cosine 1.0 across 40 configs) but I can't find the source or the result files anywhere: not on
the Drive archive, not on the compute host. Rebuilding from the spec rather than trusting those
numbers. Everything gets re-derived and re-measured; the old figures stay unverified until
something reproduces them.

Their values go into `protocol.md` as targets, set at or slightly looser than reported. If the
original turns up later, diff it against this rather than swapping it in.

## 2026-08-09, two representation traps

Additive fitness can be written `a_i (I + Z_i)/2` or `a_i Z_i`. They differ by a constant and a
factor, and mixing them gives a distribution that looks entirely reasonable and is wrong. Cosine
0.47 on rugged landscapes. Spin convention everywhere now; the projector form survives in one
conversion helper, unit-tested against a hand-computed L = 2 case.

The second one cost about the same. A quasispecies distribution is L1-normalised and
non-negative, a quantum state is L2. Imposing L2 on the imaginary-time iteration converges to
the wrong eigenvector and also passes an eyeball check. Target the ground state of the
stoquastic operator `-(H_sel + H_mut)` instead: its Perron vector is sign-definite, so L1 and L2
pick the same ray. Work in L2 internally, convert at the decode boundary in `io/`.

## 2026-08-09, sparse eigensolver above twelve loci

Dense 2^L float64 is ~2.1 GB at L = 14 and ~34 GB at L = 16, on a host with 62 GB shared with
other people's jobs. `eigsh` for L >= 12, and dense above 12 raises rather than being merely
discouraged.

Found later, related: the image pins BLAS and OpenMP to one thread so thread count can't quietly
undermine the time allocation. Dense eigendecomposition inside the image is therefore much
slower than outside. A 4096x4096 solve that takes seconds on the host takes minutes in the
image, so anything wanting a few extreme eigenvalues should ask for the sparse path.

## 2026-08-09, Route B is eigenstate filtering

Route B was planned on Claudon, Piquemal & Monmarché (2025). Read it properly before building on
it, and it doesn't apply, for two independent reasons.

Their results are for row-stochastic Markov kernels. The mutation-selection generator isn't one;
that is what non-conservative means. The natural conversion is a Doob h-transform built from the
Perron vector we're trying to compute, so it's circular. Separately, their speedup is bought by
*non*reversibility, and ours is reversible: with symmetric mutation it's outright symmetric,
defect exactly zero. Measured over twenty operators, `results/wp0/prior_art_iv_4.json`.

Two things worth keeping. Asymmetric per-site mutation makes the generator non-symmetric but
still reversible, because independent site flips are a product of two-state birth-death
processes. And selection can't affect reversibility at all, since detailed balance constrains
off-diagonal entries and selection is diagonal.

So: QSVT eigenstate filtering for a Hermitian stoquastic operator. Cost depends on the spectral
gap and initial overlap, which the WP1 gap map produces anyway. The context-dependent-mutation
extension is written up as the route by which the nonreversible machinery would matter, and not
implemented. Narrower claim, accurate one.

## 2026-08-10, provenance on records

Records carry image tag, platform, protocol hash and commit. A run outside the container goes to
`results/_local/` and isn't used for paper numbers.

This was policy with nothing enforcing it and broke the first time it was tested. A G-R.4 run
outside the image wrote a record, `git add -A` swept it into a commit, and it got pushed. The
record said so itself (`image: unknown`, `platform: Windows-10`, `git_dirty: true`) and nothing
was reading it. The test suite reads it now.

Worse, in the same afternoon: the Makefile's docker invocation set neither `QUASAR_IMAGE` nor
`PYTHONPATH`, so `make gates`, the documented one-command reproduction, would have filed every
record under `_local` and the imports would have failed outright. Every check that passed was
launched with both set by hand, so the results are fine; the path the README tells everyone else
to use had never been run. Both live in the `DOCKER` variable now, and a test parses the
Makefile and fails if either is dropped. Asserting on Makefile contents from a unit test is
ugly. Baking them into the image is worse, because then the tag can't distinguish which image a
record came from.

## 2026-08-10, `eigsh` needs a fixed start vector

Nine of ten records reproduced bit-identically. G-R.4 didn't: gap decay per site moved from
`0.7167436421588269` to `0.7167436421588261`, with no code change.

`eigsh` takes an optional `v0`, the vector ARPACK starts from. Unset, SciPy draws it from NumPy's
global random state, seeded from OS entropy at import, so every process starts somewhere
different and ARPACK returns as soon as the residual is under tolerance. Three call sites did
this. All pass `v0=deterministic_start(n)` now.

Fixed pseudo-random vector, not all-ones: all-ones overlaps heavily with the Perron vector of
*this* operator, so it would work here and hide the problem the day these functions point at
something else. `test_numerics.py` parses the package and fails if any `eigsh` call omits `v0`.
That's the check that matters in six months.

Difference was ~8e-16 relative and changes nothing. Records from before this should be read as
good to ~1e-14 rather than bit-reproducible.

## 2026-08-10, truncated SVD in the Motta generator

Energy rose at one step by 2.28e-3 against a bound of 1e-10. Gram matrix condition number
2.95e32, and the absolute regularisation term was picking an arbitrary null-space vector out of
it. Truncated SVD at 1e-8 instead, cutoff from the measured conditioning rather than guessed.

While in there: the generator has to be real and antisymmetric, so it's restricted to Pauli
strings with an odd number of Y factors. Corrected version has even-Y contribution exactly zero
against an odd-Y norm of 0.128.

## 2026-08-10, order parameter from the instance's own optimum

`magnetisation` measures concentration on genotype 0. Fine for the single peak and for additive
landscapes, wrong for a rugged one, because a rugged landscape's optimum is somewhere random.

Rough Mount Fuji was the best candidate for a ruggedness axis that leaves the master sequence
alone, and it does, but only while barely rugged. Fraction of 40 seeds whose global optimum is
still genotype 0, at L = 12:

| roughness | 0.1 | 0.3 | 0.5 | 0.7 | 1.0 | 2.0 |
|---|---|---|---|---|---|---|
| retained | 1.00 | 0.97 | 0.62 | 0.40 | 0.25 | 0.05 |
| local optima | 1.0 | 1.4 | 14.9 | 53.9 | 121.2 | 246.1 |

Retention gets worse with L, not better. Every other family is worse still; NK sits at Hamming
weight 3.2 to 4.2 out of 8 at every K including K = 0.

So a sweep holding `magnetisation` while raising ruggedness stops measuring localisation partway
along the axis and starts measuring how far the optimum has wandered. That also falls with
ruggedness and also looks like a threshold crossing, and WP7's central figure sweeps this axis.

Measured from the instance's own fittest genotype now.
`order_parameter.localisation(probs, reference)` reduces to `magnetisation` exactly when the
optimum is genotype 0, so nothing already measured moves.

The cost, stated plainly: "error threshold" in the classical literature means delocalisation from
a master sequence, and a rugged landscape hasn't got one. Concentration on the fittest genotype
is the natural generalisation, not the same object. Say "localisation transition" for rugged
landscapes and don't quote a rugged threshold beside the sharp-peak value.

Rejected two alternatives. Capping roughness at 0.3 keeps 97% retention but leaves 1.4 local
optima, so there's no ruggedness left to map. Conditioning on instances that keep the master
sequence is selection on the outcome and biases the threshold toward looking sharp.

## 2026-08-11, time allocation and overruns

Equal wall-clock per cell, 300 s at L <= 12. Two things came out of running it.

An equal-wall-clock budget disadvantages imaginary time in exactly the rugged small-gap cells
this project is about. ITE suppresses the leading contaminant like `exp(-gap*tau)`, so time
needed goes as `1/gap`, and the gap shrinks with both ruggedness and size. A null in those cells
would be an artefact of the protocol; handing ITE more time is the same error in reverse. The
sweep reports both, accuracy at the fixed allocation and the budget needed to reach a fixed
accuracy.

The matrix-product baseline is the only method that overruns, and it does it more as size grows:
0 of 259 at L = 8, 5 of 259 at L = 10, a third of the grid at L = 12. Worst case 3.28x the
allocation. `evolve` stops on convergence or `max_steps` and never looks at the clock, so the
budget is enforced after the fact by exclusion rather than during the run by truncation.

That matters, because exclusion removes precisely the cells where the classical reference is most
strained, which is where a crossover would live. It doesn't change this verdict: over-budget
cells have minimum cosine 0.999519, so "tensor network below 0.80" misses by four orders of
magnitude either way. The verdict reports exclusion counts per size anyway, so a reader can see
how much of the grid the rule took out. `over_budget` is recomputed from `seconds_used` rather
than trusted, after Route A reported `budget_exhausted=False` having spent 510 s of 300.

For any future run, L >= 14 especially: give `evolve` a deadline, return the state it holds when
the deadline arrives, mark the record `budget_limited`. A degraded comparable answer beats a
dropped cell.

## 2026-08-11, locus ordering is not free

Even-then-odd relabelling costs 2.17x more bond dimension on the NK family at K = 2, and 2.67x
worst case on the block family. Measure it, don't assume it. Better ordering used throughout.

## 2026-08-13, hardware runs outside the image

The pinned image has no `qiskit-ibm-runtime`, so there's no route from inside it to a QPU at all.
Adding the package changes the image tag, and the tag is what eleven reproduced records are
pinned to; buying provenance for one check by unpinning eleven is a bad trade. Preparing circuits
inside and submitting outside fails too, because qpy doesn't deserialise forward from a newer
writer to an older reader.

So WP8 runs outside the image, the record lands in `results/_local/`, and the guard stays as it
is. Committed with its status visible rather than laundered.

What makes it still worth having: client environment recorded in full, device provenance recorded
(backend, calibration timestamp, basis gates, job ids, transpiled depth, two-qubit counts,
shots), transpiler seeded so the circuits inspected are the circuits submitted, and state
preparation is varQITE with G-R.8's settings, which does have a reproduced record from inside the
image. What's added here is network access, not physics.

G-8 is feasibility, not accuracy, and nothing in `results-index.md` cites it as reproduced
evidence.

## 2026-08-15, attribution of the class-invariant reduction

The `L + 1` tridiagonal reduction for fitness depending only on mutation number is Swetina &
Schuster (1982), not Dixit, Srivastava & Vishnoi (2012). Dixit et al. cite Swetina & Schuster for
it explicitly; their own contribution is a finite-population model and a steady-state algorithm
for it, at a cost that isn't polynomial in L. Different model, different cost.

Carried the wrong way round from the planning documents, and the kind of thing a referee in this
field notices. Same pass: Jain & Krug is an introductory review and doesn't contain the
pairwise-epistatic transverse-field Ising Hamiltonian it had been credited with. That term is
standard Ising notation, validated against exact diagonalisation rather than taken on authority.
