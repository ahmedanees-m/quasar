# GATES.md — Pre-registration

**This file is append-only.** Thresholds are never lowered. Amendments are appended with a
date and a rationale; nothing above an amendment line is ever edited in place. The commit
hash of this file at the time of each run is recorded in that run's result record and cited
in the paper's methods section.

Project: QUASAR — quantum algorithms for mutation–selection dynamics
Execution plan: `QUASAR_FINAL_execution_plan_v4.md` (v4.0)
Maintainer: Anees Ahmed Mahaboob Ali

---

## 0. Standing rules

1. A gate threshold is fixed before the run it judges. If a method fails a gate, the method
   is fixed or the failure is reported. The threshold does not move.
2. Every gate is executable as a test under `tests/gates/` and writes a JSON record to
   `results/`. A gate with no artefact has not passed.
3. Every stochastic component takes an explicit seed. Seeds are listed here and recorded in
   the result record.
4. Reference validity is declared per cell. A cell whose reference is not trustworthy is
   excluded and reported as excluded, never scored against a weak reference.
5. `numpy.random.default_rng(seed)` only. No legacy global RNG.
6. float64 throughout. Tolerances are declared per gate here, never chosen at run time.

---

## 1. Numerical conventions (binding, project-wide)

These are not style choices. Each one corresponds to a class of silent, plausible-looking
error, and two of them correspond to bugs this project has already been bitten by.

| Convention | Rule |
|---|---|
| Fitness encoding | Spin convention `a_i Z_i`, `b_ij Z_i Z_j`. Projector form `a_i (I + Z_i)/2` is forbidden outside a documented conversion helper. |
| Biological normalisation | Quasispecies distributions are L1-normalised and non-negative. |
| Quantum-state normalisation | L2 internally. Conversion to L1 happens only at the decode boundary. |
| Target operator | Ground state of the stoquastic operator `-(H_sel + H_mut)`, whose Perron vector is sign-definite, so L1 and L2 normalisation select the same ray. |
| Qubit ordering | Qiskit little-endian. Every bitstring/integer conversion goes through one helper in `quasarstack/io/`. |
| Eigensolvers | `scipy.sparse.linalg.eigsh` for L >= 12. Dense diagonalisation is forbidden above L = 12 (dense at L = 14 is about 2.1 GB, at L = 16 about 34 GB). |
| Hamiltonian storage | Sparse CSR or Qiskit `SparsePauliOp`. Never a dense 2^L x 2^L array above L = 12. |

---

## 2. Registration note on provenance (2026-08-09)

An earlier implementation of Phases 1–3 was reported in the planning documents with
specific measured values. **That implementation could not be located** on the laptop, the
Drive archive, the compute VM, or the GitHub account at the time this repository was
created. No code and no result artefacts survive.

Consequently the values reported in the planning documents are treated here as
**pre-registered targets to be re-hit by a fresh implementation**, not as inherited
results. No number from those documents enters the manuscript unless a run in this
repository reproduces it and writes an artefact. This is recorded in `DECISIONS.md` as
ADR-0001.

---

## 3. Work package WP-R — rebuild and re-validate Phases 1–3

Objective: reconstruct the validated stack (analytic oracle, Hamiltonian compiler,
Trotter circuit, both imaginary-time routes, noise backends) and re-hit the seven gates
that the planning documents record.

Thresholds below are set at or slightly looser than the previously reported values, with
the reported value noted. Setting them looser is deliberate: a threshold must be
defensible on its own terms, not reverse-engineered from a number we are trying to match.
Where a run beats the threshold, the measured value is reported.

| Gate | Statement | Threshold | Previously reported |
|---|---|---|---|
| G-R.1 | Analytic Crow–Kimura oracle agrees with brute-force exact diagonalisation, L = 2..10, all landscape configurations in the WP-R set | max abs error < 1e-9 | 3.85e-13 |
| G-R.2 | Qubit Hamiltonian ground state matches the analytic quasispecies | cosine >= 0.999999 on 40/40 configurations | cosine 1.000000, 40/40 |
| G-R.3 | Trotterised circuit converges to the oracle, and the error scales as O(dtau^2) | cosine >= 0.999 at dtau = 0.01; fitted exponent in [1.8, 2.2] with R^2 >= 0.99 | cosine 1.0, O(dtau^2) |
| G-R.4 | Error-threshold transition location matches the analytic prediction, L = 4, 6, 8 | max abs(delta m) < 1e-3 | 0.0000 |
| G-R.5 | Rugged epistatic landscapes reproduce brute-force ED | cosine >= 0.99999 across all 10 seeded NK instances | cosine 1.000000 |
| G-R.6 | varQITE reproduces the oracle at depth constant in imaginary time | cosine >= 0.999; circuit depth identical at tau = 2.5 and tau = 20 | cosine >= 0.99993, depth 10 / 6 CX at L = 3 |
| G-R.7 | Motta-QITE reproduces the oracle and the energy descends monotonically | cosine >= 0.95; no energy increase over any step beyond 1e-10 | cosine >= 0.97 |
| G-R.8 | Noise models plus mitigation, L = 2..4, IBM-Heron-like and trapped-ion | mitigated cosine >= 0.98 | 0.991 to 0.9998 |
| G-R.9 | Barren-plateau diagnostic: gradient variance decays exponentially in L | fitted decay base in [0.30, 0.55], R^2 >= 0.95, L = 2..8 | var(C) ~ 0.42^L |
| G-R.10 | Sparse additive+epistasis representation uses fewer Pauli terms than the single-peak projector at L = 12 | ratio >= 50x | 152x |

**Gate G-R (composite).** All of G-R.1 through G-R.10 pass, each with a committed JSON
artefact under `results/wp_r/`. Binary. No WP1+ run is judged before G-R passes.

**Seeds for WP-R.** Landscape seeds `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`. Optimiser seeds
`[11, 12, 13, 14, 15]`. Shot seeds `[101, 102, 103]`.

---

## 4. WP0 — pre-registration and prior art

**Gate G-0.** `GATES.md` and `PRIOR_ART.md` both complete and committed before any WP4+
run. Binary. `PRIOR_ART.md` must cover all four literatures named in the execution plan,
with at least the works listed in T0.2, each carrying a one-line statement of what it
establishes and what it leaves open.

---

## 5. WP1 — spectral and structural analysis

**Gate G-1.**

1. The computed spectral gap reproduces the closed form where one exists (single-peak,
   permutation-symmetric) to relative error < 1e-6.
2. The gap closes as mu approaches the analytically located error threshold: the minimum of
   the computed gap over the mu sweep lies within 5% of the analytic mu_c, for L = 6, 8, 10.
3. Every stated property of the operator (Perron–Frobenius structure, stoquasticity,
   reversibility or its absence) is derived in `docs/theory.md` with the derivation
   referenced from the code docstring. Asserted-but-underived claims fail this gate.

**Grid for the WP1 gap map.** Families: single-peak, additive, NK with K in {0,1,2,3,4,6},
spin-glass. L in {4, 6, 8, 10, 12, 14}. mu on a 41-point grid spanning
[0.2 mu_c, 2.0 mu_c] per instance. Seeds `[0..9]` per (family, K, L).

---

## 6. WP2 — Route B, QSVT Perron-vector extraction

**Gate G-2.**

1. Route B reproduces the analytic quasispecies at cosine >= 0.95 for L = 2..6 in noiseless
   simulation.
2. The block encoding satisfies its defining property to 1e-10: the top-left block of the
   unitary equals `A / alpha` for the declared normalisation alpha.
3. Resource scaling (query complexity, ancilla count, polynomial degree) is derived
   analytically and the derived degree agrees with the empirically sufficient degree to
   within a factor of 2, using the WP1 gap map as input.

**Pre-registered acceptable outcome.** If a full circuit-level implementation proves
infeasible within the work package, a rigorous resource-estimation-only treatment of
Route B, with the limitation stated plainly, satisfies the work package. This is recorded
in advance so that the fallback cannot be presented later as a planned success.

---

## 7. WP3 — landscape families

**Gate G-3.**

1. Every landscape reproduces exactly from its seed, byte-for-byte, on the declared image.
2. NK with K = 0 equals the additive family analytically, to 1e-12.
3. Ruggedness statistics (number of local optima, fitness-correlation length) increase
   monotonically in K, verified over seeds `[0..9]` at L = 10 and L = 12, with monotonicity
   required of the seed mean and reported per seed.

Families: single-peak, additive + weak pairwise epistasis, NK(K), spin-glass (random +/-J),
Rough Mount Fuji, House-of-Cards, Block.

---

## 8. WP4 — Baseline A, Wright–Fisher

**Gate G-4.**

1. Reproduces the analytic single-peak quasispecies as population size N and sample budget
   grow: total-variation distance < 0.02 at the largest declared budget, L = 8.
2. Throughput is within 5x of the reference community implementation on a matched
   configuration, measured and reported. Falling outside 5x is a fail and the
   implementation is optimised, not excused.

Declared N sweep: `[1e3, 1e4, 1e5, 1e6]`. Burn-in 20% of generations. Seeds `[0..9]`.

---

## 9. WP5 — Baseline B, Dixit–Srivastava–Vishnoi

**Gate G-5.**

1. Matches the analytic oracle to <= 1e-6 on every landscape in its declared applicability
   class.
2. The applicability boundary is documented as an explicit predicate in code, and the set
   of WP7 grid cells it covers is emitted as a machine-readable map before the sweep runs.

---

## 10. WP6 — Baseline C, tensor-network imaginary time

**Gate G-6.**

1. Converges to sparse ED where both run: cosine >= 0.999 at sufficient chi, for
   L in {8, 10, 12, 14} across all families.
2. Bond-dimension growth required to hold cosine >= 0.999 is mapped across
   (family, K, mu, L). This map is a primary deliverable.
3. MPO bond dimension is reported per family. At least two site-ordering strategies are
   tested for the non-local families (NK with K >= 2, spin-glass) and the better one is
   used. Where MPS is structurally disadvantaged, the disadvantage is stated in the results,
   not exploited.
4. Truncation error is tracked and recorded at every step, not only at the end.

chi sweep: `[16, 32, 64, 128, 256, 512, 1024]`. dtau: `[0.1, 0.05, 0.02]`.

---

## 11. WP7 — the grid sweep and the decision gate

### 11.1 The grid

- Ruggedness axis: single-peak, additive+weak epistasis, NK with K in {1, 2, 3, 4, 6},
  spin-glass, Rough Mount Fuji, House-of-Cards.
- Mutation axis: 21 points spanning [0.4 mu_c, 1.6 mu_c] per instance, mu_c located
  analytically per family where a closed form exists and numerically otherwise.
- Size axis: L in {8, 10, 12, 14}, extended to {16, 18, 20} only in cells where the
  high-chi MPS reference is demonstrably converged.
- Seeds: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`, minimum 5 used per cell, all 10 where budget allows.
- Methods per cell: Route A (varQITE, Motta-QITE), Route B where feasible, Baseline A,
  Baseline B where applicable, Baseline C.

### 11.2 Reference

Sparse ED (`eigsh`) for L <= 16. Converged high-chi MPS beyond, used as reference only
where a chi-doubling changes the result by less than 1e-4 in total-variation distance.
Cells with no trustworthy reference are excluded and counted in `sweep_manifest.json`.

### 11.3 Compute-budget protocol (the fairness firewall)

- Budget unit: **wall-clock seconds per cell per method**, on the declared hardware, single
  declared worker count, inside the declared Docker image.
- Declared hardware: VM `sjt418scope025`, 32 vCPU, 62 GB RAM, NVIDIA RTX A4000 16 GB,
  Ubuntu 22.04. Image tag recorded per run.
- Allotment per cell per method: 300 s at L <= 12, 900 s at L >= 14. Fixed here, not tuned
  later.
- Wright–Fisher spends its budget on samples. MPS spends its budget on chi and smaller
  dtau. Route A spends its budget on optimiser iterations.
- **QUASAR's classical optimisation time counts against its own budget.** varQITE's
  parameter-update solve is a real cost and is not excluded.
- Allotted and used seconds are recorded per cell in the result record.

### 11.4 Scoring

Cosine similarity **and** total-variation distance, both reported. TV is the less
flattering and more conservative metric and is the one used for the decision gate where the
two disagree. Bootstrap 95% confidence intervals across seeds, 10000 resamples. Seed-to-seed
spread reported alongside the mean, never only the mean.

### 11.5 Gate G-7 — the decision gate

**Positive result.** A non-empty region of the (ruggedness, mu, L) grid, contiguous and
reproducible across at least 5 of the seeds run, in which:

- a quantum route achieves cosine >= 0.90 against the reference, and
- the compute-matched tensor-network baseline achieves cosine < 0.80, and
- Baseline B does not apply, and
- the effect survives the bootstrap CI, meaning the CIs of the two methods do not overlap.

**Null result.** No such region exists at accessible L. Reported as a delimitation with an
explicit bound: the crossover, if it exists, lies beyond L = X and chi = Y, where X and Y
are the largest values at which the sweep held a valid reference.

Both outcomes are recorded here in advance as publishable. The null is the more probable
outcome given the tensor-network literature, and reporting it is not a failure of the
experiment.

---

## 12. WP8 — live QPU

**Gate G-8.** Results are reported as measured. Job IDs, backend name, calibration date,
transpiled depth, two-qubit gate count, shots, and both raw and mitigated distributions are
recorded. No threshold is set on accuracy, because the purpose is feasibility, not
performance. Framing the result as evidence of advantage fails this gate.

---

## Amendments

*(Append below this line only. Never edit above it.)*

- 2026-08-09 — Initial registration. Anees Ahmed Mahaboob Ali.
