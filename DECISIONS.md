# DECISIONS.md — architecture and design decision log

Short ADR-style entries for consequential choices. Each entry: context, decision,
consequences. This is what lets a reviewer, or the author in eight months, understand why
something is the way it is.

---

## ADR-0001 — Rebuild Phases 1–3 rather than inherit reported results

**Date:** 2026-08-09
**Status:** accepted

**Context.** The planning documents (`QUASAR_FINAL_execution_plan_v4.md`,
`QUASAR_project_explainer.md`, `QuBiS-HiQ_to_QUASAR.md`) report a working `quasarstack`
implementation with seven passed gates and specific measured values, among them an oracle
agreeing with exact diagonalisation to 3.85e-13 and a Hamiltonian ground state matching the
analytic quasispecies at cosine 1.000000 across 40 configurations.

At repository creation the implementation could not be located. Searches covered the Drive
archive, the compute VM filesystem, and the GitHub account. No source and no result
artefacts were found.

**Decision.** Treat the reported values as pre-registered targets for a fresh
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

## ADR-0002 — Spin convention for fitness, everywhere

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

## ADR-0003 — Target the stoquastic operator, and normalise L1 at the decode boundary only

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

## ADR-0004 — Sparse eigensolvers only above L = 12

**Date:** 2026-08-09
**Status:** accepted

**Context.** A dense 2^L x 2^L float64 array is about 2.1 GB at L = 14 and about 34 GB at
L = 16. The compute VM has 62 GB of RAM shared with other running work.

**Decision.** `scipy.sparse.linalg.eigsh` for L >= 12. Dense diagonalisation is forbidden
above L = 12 and is guarded by an assertion in the code, not only by convention.

**Consequences.** Exact diagonalisation stays available as a reference to L = 16, which is
what the WP7 grid needs. The guard turns a machine-killing mistake into an immediate error.

---

## ADR-0005 — Apache-2.0

**Date:** 2026-08-09
**Status:** accepted

**Context.** The project may touch patentable algorithmic content, and it must be reusable
by other researchers without friction.

**Decision.** Apache-2.0, as specified in `QUASAR_engineering_standards.md` section 2.1.
Permissive, with an explicit patent grant.

**Consequences.** Compatible with the deposit requirements of every target venue. No
copyleft obligation on downstream users.

---

## ADR-0006 — All computation runs in Docker on the VM; the laptop only authors

**Date:** 2026-08-09
**Status:** accepted

**Context.** The laptop has 8 GB of RAM. The compute VM has 32 cores, 62 GB of RAM, and an
RTX A4000, and its administrators require that models and tools run from Docker images
rather than being installed onto the host.

**Decision.** One pinned image, `quasar:v1`, built on the VM from the `Dockerfile` in this
repository, is the only execution environment for anything that produces a result record.
The laptop runs fast unit tests, figure scripts over already-computed JSON, and the
manuscript. Nothing is installed onto the VM host.

**Consequences.** The image tag is part of every result record, so any number traces to an
exact environment. Reproduction by a third party is a `docker build` away. The laptop never
becomes a source of results that cannot be reproduced.

---

## ADR-0007 — Code moves by git, data moves by SFTP, no rclone

**Date:** 2026-08-09
**Status:** accepted

**Context.** Three locations hold project state: the laptop, the Google Drive archive at
`G:\My Drive\Qubis_HiQ\QUASAR`, and the VM. Drive is the long-term archive and has ample
space. The VM has about 43 GB free and is shared with other projects, none of which may be
disturbed.

**Decision.** GitHub is the single source of truth for code and documents; laptop and VM
each hold a clone. Result artefacts, figures, and image tarballs move between the VM and
the Drive archive over SFTP using `infra/sync.py`. rclone is not used.

**Consequences.** Code never travels as a file copy, so the two clones cannot silently
diverge. Large artefacts never enter git history. The VM keeps only what a currently
running sweep needs; the Drive archive keeps everything.

---

## ADR-0008 — VM storage is treated as a hard 40 GB ceiling

**Date:** 2026-08-09
**Status:** accepted

**Context.** The VM root filesystem is 468 GB and 91% full, leaving about 43 GB. About
127 GB is reclaimable from unused Docker images, but those are another project's tagged
backups and are not ours to remove.

**Decision.** QUASAR stays under 40 GB on the VM at all times: image about 2.5 GB, working
tree under 100 MB, and a rolling results window. `scripts/sweep_runner.py` ships completed
result records to the Drive archive over SFTP and prunes the local copy once the transfer is
verified by checksum. A pre-run disk check aborts a sweep that would breach the ceiling.

**Consequences.** No other project on the VM is affected. The Drive archive, not the VM, is
where the complete result set lives. Sweeps are resumable from the archive.

---

## ADR-0009 — A gate record is committed at the commit that produced it, and reruns that change nothing scientific are discarded

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

## ADR-0010 — Route B cannot rest on the nonreversible-Markov-chain result, and needs a new foundation

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
written up as an outlook is what makes the prior-art engagement honest rather than
decorative.

**Consequences either way.** The plan's section 0 wording, which treats non-conservative and
nonreversible as though one implied the other, must be corrected before drafting. The
correction is cheap now and would have been expensive after WP2 was built on it. The
`GATES.md` G-2 thresholds are unaffected: they speak of reproducing the analytic
quasispecies and deriving resource scaling, neither of which depends on which QSVT
construction is used.
