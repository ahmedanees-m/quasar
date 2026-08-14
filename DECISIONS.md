# DECISIONS.md: architecture and design decision log

Short ADR-style entries for consequential choices. Each entry: context, decision,
consequences. This is what lets a reviewer, or the author in eight months, understand why
something is the way it is.

---

## ADR-0001: Rebuild Phases 1–3 rather than inherit reported results

**Date:** 2026-08-09
**Status:** accepted

**Context.** The planning documents (`QUASAR_FINAL_execution_plan_v4.md`,
`QUASAR_project_explainer.md`, `QuBiS-HiQ_to_QUASAR.md`) report a working `quasarstack`
implementation with seven passed gates and specific measured values, among them an oracle
agreeing with exact diagonalisation to 3.85e-13 and a Hamiltonian ground state matching the
analytic quasispecies at cosine 1.000000 across 40 configurations.

At repository creation the implementation could not be located. Searches covered the Drive
archive and the compute host. No source and no result
artefacts were found.

**Decision.** Treat the reported values as specified targets for a fresh
implementation rather than as inherited results. Register them as thresholds in `GATES.md`
section 3 (WP-R). No number from the planning documents enters the manuscript unless a run
in this repository reproduces it and writes an artefact under `results/`.

**Consequences.**

- Roughly three to four weeks of rebuild work before WP1 begins.
- The rebuild is done inside the repository skeleton with `GATES.md` already committed, so
  the one-command reproducibility requirement holds from the first gate rather than being
  retrofitted.
- Thresholds in WP-R are set at or slightly looser than the reported values. A threshold
  reverse-engineered from a number one is trying to match is not a threshold.
- If the original implementation later surfaces, it is diffed against the rebuild rather
  than replacing it. Two independent implementations agreeing is stronger evidence than
  either alone.

---

## ADR-0002: Spin convention for fitness, everywhere

**Date:** 2026-08-09
**Status:** accepted

**Context.** Additive fitness can be written as a projector `a_i (I + Z_i)/2` or in the spin
convention `a_i Z_i`. The two differ by a constant and a factor. Mixing them is silent: the
resulting distribution looks plausible. The planning documents record this as Bug 2, which
dropped cosine similarity to 0.47 on rugged landscapes.

**Decision.** The spin convention `a_i Z_i`, `b_ij Z_i Z_j` is the project-wide
representation. The projector form appears only inside one documented conversion helper,
which is unit-tested against a hand-computed L = 2 case.

**Consequences.** Every function that consumes fitness parameters documents the convention
in its docstring. A convention mismatch becomes a test failure rather than a wrong number.

---

## ADR-0003: Target the stoquastic operator, and normalise L1 at the decode boundary only

**Date:** 2026-08-09
**Status:** accepted

**Context.** A biological quasispecies distribution is L1-normalised and non-negative. A
quantum state is L2-normalised. Imposing L2 normalisation on the imaginary-time iteration
converges to the wrong eigenvector, recorded in the planning documents as Bug 1: a
parity-locked distribution that passes an eyeball check.

**Decision.** Target the ground state of the stoquastic operator `-(H_sel + H_mut)`. Its
Perron vector is sign-definite, so L1 and L2 normalisation select the same ray. Work in L2
internally, convert to L1 only at the decode boundary in `quasarstack/io/`.

**Consequences.** The method is simultaneously hardware-faithful and biologically correct.
The two normalisation regimes stay explicit rather than blending, and the boundary between
them is one function that can be tested.

---

## ADR-0004: Sparse eigensolvers only above L = 12

**Date:** 2026-08-09
**Status:** accepted

**Context.** A dense 2^L x 2^L float64 array is about 2.1 GB at L = 14 and about 34 GB at
L = 16. The compute host has 62 GB of RAM shared with other running work.

**Decision.** `scipy.sparse.linalg.eigsh` for L >= 12. Dense diagonalisation is forbidden
above L = 12 and is guarded by an assertion in the code, not only by convention.

**Consequences.** Exact diagonalisation stays available as a reference to L = 16, which is
what the WP7 grid needs. The guard turns a machine-killing mistake into an immediate error.

---

## ADR-0005: Apache-2.0

**Date:** 2026-08-09
**Status:** accepted

**Context.** The project may touch patentable algorithmic content, and it must be reusable
by other researchers without friction.

**Decision.** Apache-2.0, as specified in `QUASAR_engineering_standards.md` section 2.1.
Permissive, with an explicit patent grant.

**Consequences.** Compatible with the deposit requirements of every target venue. No
copyleft obligation on downstream users.

---

## ADR-0006: All computation runs in the pinned container image

**Date:** 2026-08-09
**Status:** accepted

**Context.** The authoring machine is memory constrained. The compute host has 32 cores, 62 GB of RAM, and an
RTX A4000, and its administrators require that models and tools run from Docker images
rather than being installed onto the host.

**Decision.** One pinned image, `quasar:v1`, built from the `Dockerfile` in this
repository, is the only execution environment for anything that produces a result record.
The authoring machine runs fast unit tests, figure scripts over already-computed JSON, and the
manuscript. Nothing is installed onto the host directly.

**Consequences.** The image tag is part of every result record, so any number traces to an
exact environment. Reproduction by a third party is a `docker build` away. The authoring machine never
becomes a source of results that cannot be reproduced.

---

## ADR-0007: Code moves by git, artefacts move by authenticated transfer

**Date:** 2026-08-09
**Status:** accepted

**Context.** Three locations hold project state: the authoring machine, the long-term archive,
and the compute host. The archive has ample space. The compute host has limited free space and
is shared with other work that must not be disturbed.

**Decision.** GitHub is the single source of truth for code and documents; each machine holds a clone. Result artefacts, figures, and image tarballs move between the compute host and the archive over an authenticated channel using `infra/sync.py`.

**Consequences.** Code never travels as a file copy, so the two clones cannot silently
diverge. Large artefacts never enter git history. The compute host keeps only what a currently
running sweep needs; the Drive archive keeps everything.

---

## ADR-0008: Working storage is treated as a hard 40 GB ceiling

**Date:** 2026-08-09
**Status:** accepted

**Context.** The compute host root filesystem is 468 GB and 91% full, leaving about 43 GB. About
127 GB is reclaimable from unused Docker images, but those are another project's tagged
backups and are not ours to remove.

**Decision.** QUASAR stays under 40 GB on the compute host at all times: image about 2.5 GB, working
tree under 100 MB, and a rolling results window. `scripts/sweep_runner.py` ships completed
result records to the archive and prunes the local copy once the transfer is
verified by checksum. A pre-run disk check aborts a sweep that would breach the ceiling.

**Consequences.** No other work on the host is affected. The Drive archive, not the host, is
where the complete result set lives. Sweeps are resumable from the archive.

---

## ADR-0009: A gate record is committed at the commit that produced it, and reruns that change nothing scientific are discarded

**Date:** 2026-08-09
**Status:** accepted

**Context.** Committing a gate record changes the tree, so the next run of the same gate
records a different `git_sha` and a different timestamp than the record already committed.
Running `make gates` on a clean checkout therefore always leaves the tree dirty, even when
the run reproduced the committed result exactly.

Measured on the first two gates: rerunning G-R.1 and G-R.2 at a later commit changed
`git_sha`, `timestamp`, and the elapsed seconds. Every scientific field was bit-identical.

**Decision.** A record is committed once, at the commit whose code produced it. A rerun
whose only differences are provenance and timing is discarded with `git checkout --
results/`. A rerun that changes any scientific field is a finding: it is committed, and the
change is explained, because a gate that does not reproduce is either non-deterministic or
newly broken and both need saying out loud.

**Consequences.** `make gates` is a verification step, not a step that produces commits.
The repository does not accumulate churn from re-verification. The one-command
reproducibility claim stays checkable, because the check is "did the scientific fields come
back the same", which is exactly what a reviewer would want to run.

---

## ADR-0010: Route B cannot rest on the nonreversible-Markov-chain result, and needs a new foundation

**Date:** 2026-08-09
**Status:** finding accepted; the redesign choice is open and needs both PIs

**Context.** Execution plan v4 exists because Claudon, Piquemal and Monmarche (2025) was
found, and Route B is the novelty core of the paper. The plan's section 0 argues that the
quasispecies is the Perron eigenvector of a non-conservative operator and that extracting a
dominant eigenvector is what QSVT eigenvalue transforms do.

The reference was read and its applicability measured, ahead of schedule, precisely because
everything in WP2 depends on it. Artefact: `results/wp0/prior_art_iv_4.json`, twenty
operators. Full write-up in `PRIOR_ART.md` entry IV.4.

**What was found.** The construction does not apply, for two independent reasons.

1. The paper's results are stated for row-stochastic Markov kernels. The mutation-selection
   generator is not one; that is what non-conservative means. The natural conversion, a
   Doob h-transform, is built from the Perron vector being computed, so it is circular.
2. The paper's beyond-quadratic speedup is bought by *non*reversibility. The
   mutation-selection generator is reversible. With the symmetric mutation this project
   implements it is outright symmetric, defect exactly zero, symmetrising measure uniform.

Two measurements sharpen the scope. Asymmetric per-site mutation makes the generator
non-symmetric but leaves it reversible, because independent per-site flips are a product of
two-state birth-death processes. And selection cannot affect reversibility at all, since
detailed balance constrains off-diagonal entries and selection is diagonal, so no amount of
ruggedness changes the answer.

**What this does not mean.** Route B is not dead. Nothing above says QSVT is inapplicable
to this problem; it says the *nonreversible Markov chain* framing is the wrong justification
for it.

**Options, in the order they should be considered.**

- **A. Reframe Route B as QSVT eigenstate filtering for a Hermitian stoquastic operator.**
  This works for any Hermitian operator, which ours is. It connects to the ground-state
  preparation literature rather than the nonreversible-chain literature, and its cost
  depends on the spectral gap and the initial overlap, which is exactly what the WP1 gap map
  produces. Cheapest change, keeps the schedule, and remains unpublished for this problem.
  The novelty claim becomes narrower and more accurate.
- **B. Extend the biology to direction-specific context-dependent mutation.** Measured
  reversibility defect 0.60, so this genuinely enters the paper's class. Biologically
  faithful, since CpG and APOBEC effects are one-directional. This is a scope increase: it
  changes the model being simulated, and every analytic result and gate in WP-R would need
  its counterpart re-derived for the new generator.
- **C. Both.** Build Route B under option A, and treat option B as the stated route by which
  the nonreversible machinery would become relevant, without implementing it. Costs a
  paragraph, buys a defensible answer to the reviewer who asks why the nonreversible
  literature was cited and then not used.

**Recommendation.** C. Option A is the buildable route on the current schedule, and option B
written up as an outlook is what makes the prior-art engagement complete rather than
decorative.

**Consequences either way.** The plan's section 0 wording, which treats non-conservative and
nonreversible as though one implied the other, must be corrected before drafting. The
correction is cheap now and would have been expensive after WP2 was built on it. The
`GATES.md` G-2 thresholds are unaffected: they speak of reproducing the analytic
quasispecies and deriving resource scaling, neither of which depends on which QSVT
construction is used.

---

## ADR-0011: A landscape family must not move the fitness optimum while it varies ruggedness

**Date:** 2026-08-09
**Status:** accepted

**Context.** Gate G-R.4 swept a uniform pairwise-epistasis family, `f = a sum_i z_i + b
sum_{i<j} z_i z_j`, to measure how epistasis moves the error threshold. The planning
documents state that synergistic epistasis raises the threshold and antagonistic epistasis
lowers it.

Synergistic coupling behaved cleanly and convergently. Antagonistic coupling did not: the
half-surplus crossover came out at 0.301, 0.704 and 0.424 for L = 4, 6 and 8, which is not
monotone and disagrees with the stated direction at two of the three sizes.

The cause is not noise, and it is not frustration, which was the first guess. Measured
directly: with negative uniform coupling the fitness optimum **moves off the master
sequence** to an interior Hamming class, at d* = 1, 2, 2, 3, 3 for L = 4, 6, 8, 10, 12, with
multiplicities up to 220. The surplus at zero mutation rate is correspondingly 0.500, 0.333,
0.500, 0.400 and 0.500 rather than 1, jumping as d* jumps.

So there is no master sequence to delocalise from, and the error-threshold question is not
merely noisy in that family, it is **ill-posed**. The apparent non-monotonicity is the
optimum relocating, not a threshold shifting.

**Decision.** Any landscape family used as a ruggedness axis must be checked, and reported,
for where its fitness optimum sits. A family that silently relocates the optimum is not
varying ruggedness alone, and comparisons across it are comparisons between different
problems.

Concretely:

- `experiments` that sweep a landscape parameter record the optimum's Hamming class and its
  multiplicity alongside the result.
- WP3 gate G-3 gains this as a requirement on the NK, spin-glass, Rough Mount Fuji,
  House-of-Cards and Block families: report where the optimum sits across the ruggedness
  parameter, and state plainly where it moves.
- G-R.4 keeps the antagonistic case in its record rather than dropping it. It is the
  evidence for this decision, and dropping the case that disagreed with the expected
  direction would be exactly the wrong instinct.

**Consequences.** The claim "antagonistic epistasis lowers the error threshold" is not
supported by this family and is not made. Testing it needs a family that keeps the master
sequence optimal while varying the curvature of the cost of accumulating mutations. That is
WP3 work, not a G-R.4 fix.

The wider risk this catches early: the WP7 boundary map sweeps ruggedness as its main axis.
If ruggedness were confounded with which genotype is optimal, cells of that map would not be
comparable, and the central figure of the paper would be comparing different problems
against each other.

---

## ADR-0012: A committed result must prove it came from the pinned image

**Date:** 2026-08-09
**Status:** accepted

**Context.** ADR-0006 says every result record is produced inside `quasar:v1`.
That was policy with nothing enforcing it, and it broke the first time it was tested.

A G-R.4 run outside the pinned image wrote a record. A `git add -A` swept it into a
commit, it was pushed, and it propagated. The record itself said exactly what had
happened, in the provenance block that exists for this purpose: `image: unknown`,
`platform: Windows-10`, `git_dirty: true`. The information was there and nothing was reading
it.

**Decision.** `scripts/check_results_provenance.py` reads it, and CI runs it on every push.
A committed record must carry a non-placeholder image tag, a Linux platform, a recorded
commit, and a clean tree. A record failing any of those is rejected as evidence regardless
of the numbers inside it.

**Consequences.**

- The provenance block stops being decoration and becomes a gate.
- Running a gate outside the image for a quick look stays fine. Committing what it produced does
  not.
- The rejected record is replaced by a rerun in the image rather than patched, because the
  problem was never the numbers, it was that nothing could vouch for where they came from.

**Related finding, recorded because it will matter for WP7.** The image pins BLAS and OpenMP
to a single thread on purpose, so the compute-budget protocol is not undermined by threads
was never declared. A consequence is that dense eigendecomposition is far slower inside the
image than outside it, and the practical dense-to-sparse crossover sits well below the
L = 12 limit in `GATES.md` section 1. A dense 4096 by 4096 solve that takes seconds on a
multi-threaded host takes minutes in the image, and forty of them takes an hour. Code that
only needs a few extreme eigenvalues should ask for the sparse path explicitly rather than
inherit the default.

---

## ADR-0013: An equal-wall-clock budget systematically disadvantages imaginary time in exactly the regime WP7 is about

**Date:** 2026-08-09
**Status:** accepted as a finding; the protocol amendment is drafted below and needs both PIs

**Context.** Gate G-R.5 ran the Trotterised imaginary-time route as a diagnostic across 40
rugged NK instances at a fixed budget of tau = 60 and dtau = 0.01. Three fell below the
gate's accuracy threshold, and they were the three instances with the smallest spectral
gaps:

| K | seed | gap | shortfall in cosine | local optima |
|---|---|---|---|---|
| 7 | 4 | 0.0276 | 1.5e-3 | 31 |
| 7 | 3 | 0.0476 | 1.8e-4 | 24 |
| 4 | 3 | 0.0835 | 1.3e-5 | 15 |

Mean shortfall by connectivity runs 8.6e-12 at K = 1, 2.4e-11 at K = 2, 1.3e-6 at K = 4 and
1.7e-4 at K = 7. Gate G-R.4 separately measured the gap at the sharp-peak error threshold
closing at 0.717 per site.

This is not a defect. Imaginary-time evolution suppresses the leading contaminant by roughly
`exp(-gap * tau)`, so the time it needs scales as `1 / gap`, and a gap that shrinks with
ruggedness and with system size means a budget that must grow the same way.

**Why it is a problem for the paper.** `GATES.md` section 11.3 gives every method equal
wall-clock per cell: 300 seconds at L <= 12, 900 seconds at L >= 14. Under that protocol, in
the rugged small-gap cells, an imaginary-time route would score badly **because it was
under-budgeted, not because the method is unsuited**. Those cells are not incidental. They
are the candidate quantum-relevant regime: rugged, near the error threshold, where
Dixit-Vishnoi does not apply. The boundary map would report a null in precisely the region
the paper exists to examine, and the null would be an artefact of the protocol.

The mirror risk is just as real. Simply giving imaginary time more time would be handing the
quantum route a budget no other method receives, which is the strawman objection in reverse.

**Decision.** The fairness protocol must be stated in terms that are gap-aware and applied
to *every* method equally, rather than in raw wall-clock. Three candidate forms, to settle
before WP7 and record as a `GATES.md` amendment:

1. **Accuracy-targeted budget.** Fix a target accuracy and measure the wall-clock each method
   needs to reach it, reporting time-to-accuracy rather than accuracy-at-fixed-time. Cells
   where a method never reaches the target within a stated ceiling are reported as such.
   This is the fairest and the most expensive.
2. **Gap-scaled budget.** Keep equal wall-clock but scale the per-cell allotment as
   `1 / gap`, using the WP1 gap map, so every method gets more time in the hard cells. Cheap,
   and it uses a map the project is producing anyway.
3. **Report both.** Accuracy at the fixed budget, and the budget needed for accuracy, as two
   panels of the boundary map.

**Recommendation.** Option 3. It costs one extra column in the sweep and it makes the
protocol's effect visible instead of buried, which is the difference between a benchmark a
reviewer trusts and one they interrogate.

**Consequences.** `GATES.md` section 11.3 will need an appended amendment before WP7 runs.
Gates G-R.6 and G-R.7 should set their imaginary-time budget from the measured gap rather
than from a constant, and should report the budget used. Nothing already registered is
lowered by this; the change makes the comparison harder to game in either direction.
to a single thread on purpose, so the compute-budget protocol is not undermined by threads
was never declared. A consequence is that dense eigendecomposition is far slower inside the
image than outside it, and the practical dense-to-sparse crossover sits well below the
L = 12 limit in `GATES.md` section 1. A dense 4096 by 4096 solve that takes seconds on a
multi-threaded host takes minutes in the image, and forty of them takes an hour. Code that
only needs a few extreme eigenvalues should ask for the sparse path explicitly rather than
inherit the default.

---

## ADR-0014: The one-command reproduction did not actually reproduce anything

**Status.** Accepted. **Date.** 10 August 2026.

**Context.** `README.md` and `CONTRIBUTING.md` both point a reader at `make gates` as the way
to reproduce every specified gate from a clean checkout. `Makefile` built its container
invocation as

    docker run --rm -v "$(CURDIR)":/work -w /work -u $(UID):$(GID) quasar:v1

which sets neither `QUASAR_IMAGE` nor `PYTHONPATH`.

Two separate consequences, both silent:

1. `quasarstack.io.store.environment()` reads `QUASAR_IMAGE` to decide whether a run counts
   as evidence. Unset, every record produced by `make gates` would be filed under the
   gitignored `results/_local/` with a console note, and the tree would gain nothing
   committable. ADR-0012 built that redirection deliberately to stop out-of-image runs
   masquerading as evidence; the same mechanism silently disarmed the official entry point.
2. `quasarstack` is mounted at `/work`, not installed into the image, and running
   `python experiments/.../g_r_9_barren.py` puts the *script's* directory on `sys.path`
   rather than the working tree. Without `PYTHONPATH=/work` the import fails outright.

**How it went unnoticed.** Every gate that has passed so far was launched with both variables
supplied by hand on the command line, so the gates are sound and their artefacts are real.
What was never exercised is the path the documentation tells everyone else to use. The two
failure modes also mask each other in a reader's mind: the import error is loud enough to
look like the only problem, and fixing it by hand hides the quiet one behind it.

**Decision.** Set both in the Makefile's `DOCKER` variable, so every target that produces
records inherits them. Add `tests/unit/test_store.py::test_make_gates_runs_the_container_so_that_records_count_as_evidence`,
which parses the Makefile and fails if either is dropped.

**Consequences.** The project's first engineering principle is that every claim maps to a
re-runnable artefact. A reproduction command that has never been run end to end does not
satisfy it, whatever the artefacts say. `make gates` is therefore run in full immediately
after this change, and the resulting records are compared field by field against the
committed ones under the ADR-0009 rule: a rerun differing only in provenance is discarded, a
rerun that changes any scientific field is a finding.

**A note on where the test lives.** Asserting on the contents of a Makefile from a unit test
is ugly, and the alternative considered was to move the two settings into the image itself.
That was rejected because `QUASAR_IMAGE` baked into the image can no longer distinguish which
image a record came from, which is the entire question ADR-0012 asks. The ugliness is the
cheaper of the two.

---

## ADR-0015: Proceeding under the recommendations of ADR-0010 and ADR-0013, pending PI confirmation

**Status.** Accepted as a working assumption, reversible, flagged. **Date.** 10 August 2026.

**Context.** Two decisions were referred to both PIs and neither has been answered. Both sit
on the critical path: ADR-0010 gates WP2, which is the novelty core, and ADR-0013 gates WP7,
which is the boundary map. Waiting stops the project; guessing silently would be worse than
either. This ADR takes the third option, which is to proceed on the recommendation already
written down, say so in the artefacts, and keep the choice cheap to reverse.

**Decision.**

1. **Route B is built as ADR-0010 option C.** QSVT eigenstate filtering for a Hermitian
   stoquastic operator, with direction-specific context-dependent mutation written up as the
   route by which the nonreversible machinery would become relevant, and not implemented.
2. **WP7 reports both budget protocols, ADR-0013 option 3.** Accuracy at a fixed budget, and
   the budget needed to reach a fixed accuracy, as two panels rather than one.

**Why these are safe to assume rather than block on.**

- ADR-0010 records that "the `GATES.md` G-2 thresholds are unaffected" by the choice: they
  speak of reproducing the analytic quasispecies and deriving resource scaling, neither of
  which depends on which QSVT construction is used. So the specification does not have to
  be rewritten if the PIs pick A or B instead.
- Option C differs from option A only by a paragraph of write-up, so choosing C and being
  told A costs nothing. Being told B is the expensive branch, and B is a scope increase that
  would need its own re-derivation of every WP-R gate; that cost exists whenever it is
  decided and is not made worse by starting A now.
- ADR-0013 option 3 is a superset of options 1 and 2. Whichever the PIs pick, the data will
  already have been collected, because reporting both means measuring both.

**What is not assumed.** Nothing in the manuscript's novelty claim. The narrowing that
option A implies, from "beyond-quadratic speedup for nonreversible chains" to "eigenstate
filtering for a Hermitian stoquastic operator", is a claim about what the paper says and
stays with the PIs.

**How this is surfaced.** Every WP2 artefact carries the assumption in its notes field, and
`CLAIMS.md` marks the affected claims as resting on an unconfirmed decision. If either PI
decides otherwise, the search term is ADR-0015.

---

## ADR-0016: A gate result moved in its fifteenth digit, and the cause was the eigensolver's starting vector

**Status.** Accepted. **Date.** 10 August 2026.

**Context.** The full `make gates` run that closed WP-R reproduced nine of ten records
bit-identically in every scientific field. G-R.4 did not. Its measured gap decay per site
moved from `0.7167436421588269` to `0.7167436421588261`, and two entries of its sharp-peak
gap scaling moved in their fourteenth digit. No code had changed between the runs.

**Cause.** `scipy.sparse.linalg.eigsh` takes an optional `v0`, the vector ARPACK starts its
Krylov iteration from. Left unset, SciPy draws one from **NumPy's global random state**, which
is seeded from OS entropy at import. Every process therefore starts somewhere different, and
ARPACK returns as soon as its residual falls under tolerance, so it returns from a slightly
different place. Three call sites in `quasarstack` did this: `analytic/exact_diag.py`,
`hamiltonian/builder.py`, and the newly written `spectral/gap.py`.

**Why this is worse than an ordinary rounding difference.** The results are correct to about
`1e-14` throughout, and they usually agree to the last bit, so the disagreement surfaces
rarely and looks like a real change when it does. ADR-0009 draws the line between a rerun
that differs only in provenance, which is discarded, and one that changes a scientific field,
which is a finding to be committed and explained. That rule cannot work if the scientific
fields move on their own. The project's first engineering principle, that every claim maps to
a re-runnable artefact, is not satisfied by an artefact that re-runs to a different number.

It also would have got worse rather than better. The WP1 gap map calls the sparse path
thousands of times, and WP7's boundary map more again.

**Decision.** Every `eigsh` call in `quasarstack` passes `v0=deterministic_start(n)` from the
new `quasarstack/numerics.py`. The start is a fixed pseudo-random vector rather than a
constant such as all-ones: all-ones has large overlap with the Perron vector of *this*
operator, so it would work here and hide the problem the day these functions are pointed at
an operator whose target eigenvector is orthogonal to it.

`tests/unit/test_numerics.py` checks three things: that the start vector does not depend on
global state, that `sparse_gap` and `perron_vector` return bit-identical results across
deliberately disturbed global RNG states, and, by parsing the package, that no `eigsh` call
anywhere omits `v0`. The last of those is the one that matters in six months.

**Consequences.** G-R.4's committed record is superseded by the rerun, which is the first
result in the project to change on reproduction. The difference is at the level of `8e-16`
relative and changes no conclusion; what changes is that the number is now stable. Records
produced before this ADR carry a git SHA from before the fix and should be read as accurate
to about `1e-14` rather than exactly reproducible.

---

## ADR-0017: Ruggedness and the master sequence are in conflict, and the order parameter has to change

**Status.** Accepted for WP3 and WP7. The narrowed claim it implies needs both PIs.
**Date.** 10 August 2026.

**Context.** ADR-0011 withdrew a claim because a landscape family had been varying ruggedness
and relocating the fitness optimum at the same time, and it required every family to report
where its optimum sits. WP3 now reports that for all seven families, and the picture is worse
than ADR-0011 assumed.

The error threshold is a localisation-delocalisation transition, and the order parameter
`magnetisation` measures concentration on genotype 0, the master sequence. That is correct
for the single peak and for any additive landscape. It is not correct for a rugged one,
because a rugged landscape's optimum is somewhere random.

**Measurement.** Rough Mount Fuji was the best candidate for a ruggedness axis that leaves
the master sequence in place, and it does, but only while it is barely rugged. Fraction of 40
seeds whose global optimum is still genotype 0, against mean local-optima count:

| roughness | 0.1 | 0.3 | 0.5 | 0.7 | 1.0 | 2.0 |
|---|---|---|---|---|---|---|
| retained, L = 12 | 1.00 | 0.97 | 0.62 | 0.40 | 0.25 | 0.05 |
| local optima, L = 12 | 1.0 | 1.4 | 14.9 | 53.9 | 121.2 | 246.1 |

Retention gets **worse** with L, not better: at roughness 1.0 it falls from 0.38 at L = 6 to
0.25 at L = 12. Every other family is worse still. NK sits at Hamming weight 3.2 to 4.2 out
of 8 at every K including K = 0, the spin glass at 2.2, house-of-cards at 4.8, and the block
model at 4.0 to 4.2.

**The problem this creates.** There is no family that is both meaningfully rugged and keeps
its optimum at a fixed reference. A sweep that holds `magnetisation` as its order parameter
while raising ruggedness therefore stops measuring localisation partway along the axis and
starts measuring how far the optimum has wandered from genotype 0. That quantity also falls
with ruggedness, and it also looks like a threshold crossing. WP7's central figure is a
boundary map over exactly this axis, so the confusion would land in the paper's main result.

**Decision.** The order parameter is measured from **the instance's own fittest genotype**,
not from genotype 0. `quasarstack.spectral.order_parameter.localisation(probs, reference)`
does this and reduces to `magnetisation` exactly when the optimum is genotype 0, so nothing
already measured changes. Every sweep cell records which reference it used and the Hamming
weight of that reference.

**What this costs, stated plainly.** "Error threshold" in the classical quasispecies
literature means delocalisation from a master sequence, and on a rugged landscape there is no
master sequence. Measuring concentration on the fittest genotype is the natural
generalisation and it is not the same object. The project should say "localisation
transition" rather than "error threshold" wherever the landscape is rugged, and should not
quote a rugged-landscape threshold as though it were comparable to the sharp-peak value. That
is a narrowing of what WP7 can claim and it belongs to the PIs.

**Options considered and rejected.**

- *Restrict the axis to roughness at most 0.3, where retention is 97%.* Defensible, but at that
  roughness the landscape has 1.4 local optima, so the ruggedness axis would span almost no
  ruggedness and WP7 would have nothing to map.
- *Condition on instances that retain the master sequence.* Selection on the outcome. At
  roughness 1.0 it would keep a quarter of the instances, chosen for having an unusually
  dominant wild type, and the resulting threshold would be biased toward looking sharp.
- *Keep `magnetisation` and note the caveat.* The caveat would be a sentence and the figure
  would be the result.

**Consequences.** `GATES.md` section 11's WP7 order parameter needs an appended amendment
before WP7 runs. G-R.4 is unaffected: it uses the single peak, where the two definitions
coincide exactly. The ruggedness axis for WP7 should be Rough Mount Fuji for the biology and
the spin glass for the compilation question, and those are different axes because the spin
glass is the only rugged family whose Pauli count stays polynomial, at `L(L-1)/2 + L + 1`
terms against `2**L` for Rough Mount Fuji at any non-zero roughness.

---

## ADR-0018: G-4's throughput criterion compares two different complexity classes and would pass for the wrong reason

**Status.** Finding accepted; the replacement criterion needs both PIs.
**Date.** 10 August 2026.

**Context.** `GATES.md` section 8 requires that the Wright-Fisher baseline's "throughput is
within 5x of the reference community implementation on a matched configuration", and adds
that "falling outside 5x is a fail and the implementation is optimised, not excused". The
intent is clear and right: stop the project from comparing a quantum method against a
deliberately weak baseline.

**What the implementation turned out to be.** Individuals in Wright-Fisher are exchangeable,
so nothing depends on which individual carries a genotype, only on how many do. Carrying the
`2^L` genotype counts instead of `N` individuals makes a generation cost `O(L 2^L)`
**independent of N**, with no approximation: selection is the same multinomial draw, and
mutation factorises over sites into binomials because the site flips commute. Measured, five
seeds and 4000 generations at L = 8 take about ten seconds at `N = 10^3` and about ten
seconds at `N = 10^6`.

**Why the criterion cannot be evaluated as written.** A community forward simulator is
individual-based and costs `O(N L)` per generation. At the top of the declared sweep,
`N = 10^6`, the two implementations differ by roughly three orders of magnitude *by
construction*. The gate would pass by a factor of a thousand and would have established
nothing about whether the baseline is well built, which is the only thing it was there to
establish. A criterion that cannot fail is not a criterion.

The comparison also has a crossover that the criterion does not anticipate. Count space wins
while `2^L` is smaller than `N`, and loses beyond roughly `L = 20` where `2^L` overtakes any
population a forward simulator would run. The project's grid stops well below that, so count
space is the right choice here and the wrong one for a general-purpose tool.

**A second obstacle, separate from the first.** No reference community implementation is
present in the pinned image, and ADR-0006 forbids installing software outside
Docker. Adding one means rebuilding the image, which is a disk-budget decision on a machine
with 42 GB free that also hosts other people's projects.

**Decision.** G-4 criterion 1 is run and reported. **Criterion 2 is reported as blocked, not
as passed**, with the reason above, and the gate records absolute throughput and the measured
`O(L 2^L)` scaling so the comparison can be completed later without rerunning anything.

**Recommended replacement, for the PIs.** Replace "throughput within 5x" with
**time-to-accuracy at matched total-variation distance**: the wall-clock each implementation
needs to reach TV 0.02 against the analytic quasispecies at L = 8. That is invariant to the
representation, it is the quantity the WP7 boundary map actually consumes, and it can fail.
It also composes with ADR-0013's recommendation that WP7 report budget-to-accuracy alongside
accuracy-at-budget, so the two decisions want the same measurement.

**Consequences.** `GATES.md` section 8 needs an appended amendment before G-4 can be called
passed. Nothing else depends on criterion 2. Criterion 1 is unaffected and is met: total
variation `0.0051` at `N = 10^6`, against a threshold of `0.02`.

---

## ADR-0019: The tensor-network baseline overruns its allotment instead of stopping at it

**Status.** Finding accepted. The reporting change is made now; the implementation change is
recommended for any future run and is not applied to this one.
**Date.** 11 August 2026.

**Context.** Section 11.3 calls the per-cell per-method allotment a fairness firewall: 300 s
at `L <= 12`, 900 s at `L >= 14`. ADR-0013 and the amendment that followed it made the runner
record `seconds_used` beside `seconds_allotted` and set `over_budget` from the measurement
rather than from what a method believes about itself, after Route A reported
`budget_exhausted=False` having spent 510 s of 300. `score_g7.py` excludes an over-budget cell
from the decision, on the reasoning that a method which won on 1.7 times the allotted time has
not won.

**What the sweep measured.** Baseline C, the matrix-product reference, is the only method that
overruns, and it does so at a rate that grows sharply with size:

| L | cells with a method over budget | share |
|---|---|---|
| 8 | 0 of 259 | 0.0% |
| 10 | 5 of 259 | 1.9% |
| 12 | 33 of 89 | 37.1% |

Worst case is 3.28 times the allotment. Baseline A peaks at 0.57 of its allotment and Baseline
B at effectively zero.

**Why it happens.** `evolve` stops on convergence or on `max_steps`. It never looks at the
clock. The budget is therefore enforced *after* the fact by exclusion, and not *during* the
run by truncation. Nothing in the method is aware there is a deadline, so it cannot stop at
one and hand back what it has.

**Why this is a real defect in the protocol and not only in the method.** Exclusion removes
precisely the cells where the classical reference is most strained, which is the subset most
likely to contain a crossover. A rule that drops the hardest cells for the reference can
manufacture a null. That the rule is conservative in intent does not make it neutral in
effect, and at `L = 12` it is discarding more than a third of the grid.

**Why it does not change the present verdict.** The 38 over-budget cells have a minimum cosine
of 0.999519. Including every one of them leaves G-7's second condition, a tensor network below
0.80, unmet by four orders of magnitude. The null does not depend on the exclusion rule, and
saying so is part of reporting it completely rather than a reason to leave the rule unexamined.

**Decision.** The verdict reports the exclusion counts per size, so a reader can see how much
of the grid the rule removed and at which sizes. The grid is not rerun for this: the cells in
question are measured, their answers are recorded, and rerunning 777 cells to change a flag
that does not change the conclusion would spend a day of compute to learn nothing.

**Recommended for any future run, and for `L >= 14` in particular.** Make the method
budget-aware. `evolve` should take a deadline, return the state it holds when the deadline
arrives, and mark the record `budget_limited`. A cell then yields a comparable if degraded
answer instead of being dropped, which is what a compute-matched comparison is supposed to
mean: not that every method finished, but that every method got the same time and is judged on
what it produced in it. The trend above says this matters more at every size step, and the
sweep does not currently go past `L = 12`.


---

## ADR-0020: The hardware run cannot happen inside the pinned image, and says so rather than hiding it

**Status.** Accepted. Registered before any job is submitted.
**Date.** 13 August 2026.

**Context.** ADR-0012 makes the rule that a gate record produced outside the pinned image is
written to `results/_local/` and is not evidence. Every gate so far has satisfied it. G-8
cannot, for a reason that is structural rather than careless: **the pinned image does not
contain `qiskit-ibm-runtime`.** There is no `QiskitRuntimeService` in it, no `fake_provider`,
and therefore no route from inside the image to a real QPU or to the fake backend the dry run
needs. The dry run above and the read-only transpile against `ibm_marrakesh` both ran outside the image for that reason.

**Options considered.**

1. *Add the package to the pinned image.* This changes the image tag, and the image tag is the
   thing eleven reproduced gate records are pinned to. Buying provenance for G-8 by unpinning
   the other eleven is a bad trade.
2. *Split the work: prepare circuits in the image, submit from outside it.* Attractive, and it
   fails on a detail. The image carries qiskit 2.5.1 and the authoring environment 2.1.2, and qpy does not
   deserialise forward from a newer writer to an older reader. The split would need the image
   downgraded, which is option 1 wearing a different hat.
3. *Run WP8 outside the image and record the exception.* Chosen.

**Decision.** WP8 runs outside the pinned image. The G-8 record lands in `results/_local/`, unchanged, and
the evidence guard is left exactly as it is: it is doing its job, and switching it off for one
work package would remove the only mechanism that makes the other eleven records mean anything.
The record is committed with its non-evidence status visible rather than laundered.

What compensates, and it is not nothing:

- The **client environment is recorded in full** in the record itself, both qiskit and
  qiskit-ibm-runtime versions, so the run is reconstructible even though it is not replayable
  in the image.
- The **device provenance is recorded** as section 12 requires: backend name, calibration
  timestamp, processor type, basis gates, job ids, transpiled depth, two-qubit counts, shots.
- The **transpiler is seeded**, so the circuits inspected free by `--mode isa` are the circuits
  submitted, and a reader can regenerate them.
- The **scientific content is already evidence-grade elsewhere.** The state preparation is
  varQITE with G-R.8's settings, and G-R.8 has a reproduced record from inside the image. What
  is added is network access to the provider, not physics.

**Consequence for the claim.** G-8 is registered as feasibility, not accuracy, and this ADR
narrows it further: the hardware result is reported as an observation with recorded provenance
and stated non-evidence status under ADR-0012, and no claim in `CLAIMS.md` may cite it as
reproduced evidence. A claim citing it must carry the qualifier in its own text.

**What would change this.** If the pinned image is ever rebuilt for an unrelated reason, adding
`qiskit-ibm-runtime` at that point costs nothing and this ADR should be revisited. Rebuilding
it *for* this is what the decision declines.
