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

- 2026-08-09 — **Amendment 1: the G-R.1 case set, registered before the run.**
  Anees Ahmed Mahaboob Ali.

  Section 3 fixed the G-R.1 threshold but named the configurations only as "the WP-R set".
  That set is now fixed, below, before the gate is executed. This amendment adds detail; it
  does not change the threshold, which remains max absolute error < 1e-9.

  **What is compared.** For every case, the analytic oracle
  (`quasarstack/analytic/crow_kimura.py`, which never forms the 2^L generator) against
  brute-force exact diagonalisation (`quasarstack/analytic/exact_diag.py`, which builds the
  full sparse generator and knows nothing about the structure the oracle exploits). The
  statistic is the maximum absolute difference between the two L1-normalised genotype
  distributions, over all 2^L entries. The gate statistic is the maximum of that over every
  case.

  **Sizes.** L = 2, 3, 4, 5, 6, 7, 8, 9, 10.

  **Mutation rates.** mu in {0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00} for every case.

  **Families.**

  1. `additive_random` — a_i drawn i.i.d. from Uniform(0.25, 2.00) with
     `default_rng(seed)`, seeds 0 through 9. Solved by the closed-form product route.
  2. `additive_uniform` — every a_i equal to a, for a in {0.25, 0.50, 1.00, 2.00}. This
     family is reachable by *both* analytic routes, so it is additionally checked
     closed-form against Hamming-class reduction, making it a three-way agreement. Both
     comparisons count toward the gate statistic.
  3. `single_peak` — fitness `height` on the master sequence and zero elsewhere, for
     height in {1.0, 2.0, 5.0}. Solved by the class reduction.
  4. `class_quadratic` — f_d = height * (1 - d/L)^2, height in {1.0, 2.0, 5.0}.
  5. `class_exponential` — f_d = height * exp(-2d/L), height in {1.0, 2.0, 5.0}.

  **Diagnostics recorded per case.** The spectral gap lambda_1 - lambda_2 from the exact
  diagonalisation, and the mean-fitness difference between the two routes. The gap is
  recorded because the Perron eigenvector is only well conditioned while the gap is
  comfortably non-zero. **No case is excluded on the basis of its gap.** If a small-gap case
  fails, that is reported as a finding and feeds WP1, not quietly dropped.

  **Expected case count.** 9 sizes x 7 mutation rates x (10 + 4 + 3 + 3 + 3) configurations
  = 1449 comparisons, plus 252 closed-form-versus-class cross-checks on family 2.

- 2026-08-09 — **Amendment 2: the G-R.2 configuration set, registered before the run.**
  Anees Ahmed Mahaboob Ali.

  Section 3 fixed the G-R.2 threshold as cosine >= 0.999999 on 40 out of 40 configurations
  but did not say which forty. They are fixed here, before the gate is executed. The
  threshold is unchanged.

  **The forty.** Ten landscape configurations, each at four mutation rates.

  | # | Family | L | Parameters | Compiler route |
  |---|---|---|---|---|
  | 1 | additive_random | 2 | seed 0 | structured |
  | 2 | additive_random | 4 | seed 0 | structured |
  | 3 | additive_random | 6 | seed 1 | structured |
  | 4 | additive_random | 8 | seed 2 | structured |
  | 5 | additive_uniform | 3 | a = 0.5 | structured |
  | 6 | additive_uniform | 7 | a = 1.5 | structured |
  | 7 | single_peak | 4 | height = 2.0 | Walsh-Hadamard |
  | 8 | single_peak | 8 | height = 3.0 | Walsh-Hadamard |
  | 9 | class_quadratic | 6 | height = 2.0 | Walsh-Hadamard |
  | 10 | class_exponential | 5 | height = 2.0 | Walsh-Hadamard |

  Mutation rates: mu in {0.10, 0.30, 0.60, 1.00}. Random coefficients are drawn from
  Uniform(0.25, 2.00) with `default_rng(seed)`, matching Amendment 1.

  **Gate statistic.** The minimum, over all forty, of the cosine similarity between the
  ground state of the compiled Pauli operator and the analytic quasispecies. Pass requires
  every one of the forty at or above 0.999999.

  **Diagnostics recorded per configuration, not gating.**

  - Total-variation distance as well as cosine. Cosine is the flattering metric on a
    concentrated distribution and the threshold is written in it, so the conservative
    number is recorded alongside rather than left out.
  - The operator-level maximum absolute difference between the compiled Pauli operator and
    the negated generator assembled independently in `exact_diag`. This is a stricter check
    than the eigenvector comparison and is the one that would catch an endianness error
    that happened to leave the spectrum intact.
  - Ground-state energy against the analytic mean fitness.
  - Pauli term count, which feeds gate G-R.10.
  - For the additive families, the difference between the structured build and the
    Walsh-Hadamard build of the same operator, since both routes must produce it.

- 2026-08-09 — **Amendment 3: the G-R.3 configuration set and protocol, registered before
  the run.** Anees Ahmed Mahaboob Ali.

  Section 3 fixed the G-R.3 thresholds as cosine >= 0.999 at dtau = 0.01 and a fitted
  exponent in [1.8, 2.2] with R^2 >= 0.99. The configurations and the two sub-experiments
  are fixed here, before execution. The thresholds are unchanged.

  **What is being evolved.** The Trotterised imaginary-time propagator of
  `quasarstack/circuit/trotter_ite.py`, symmetric second-order splitting
  S(dtau/2) M(dtau) S(dtau/2), started from the uniform superposition and renormalised each
  step. This is not a hardware-runnable circuit; imaginary-time evolution is non-unitary,
  and the hardware-faithful routes are gates G-R.6 and G-R.7.

  **Sub-experiment A, the splitting-error exponent.** Total time tau = 2.0. Step sizes
  dtau in {0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125}, so the step count runs 8 to
  256 and every step size divides tau exactly. The error statistic is the maximum absolute
  difference between the Trotter distribution and `exp(-H tau)` applied to the *same*
  initial state and computed without splitting, by eigendecomposition.

  Measuring against the un-split propagator rather than against the analytic quasispecies is
  deliberate. Comparing to the quasispecies would fold in the residual from tau being finite,
  and that floor would flatten the fitted exponent at small dtau, producing a number that
  says more about the choice of tau than about the splitting.

  The exponent is the slope of a least-squares fit of log(error) against log(dtau), with
  R^2 reported. Both must be inside the registered bounds for every configuration.

  **Sub-experiment B, convergence to the quasispecies.** dtau = 0.01, tau = 60.0, so 6000
  steps. tau is set from the smallest spectral gap observed in G-R.1, which was 0.1197: at
  tau = 60 the leading contaminating amplitude is suppressed by exp(-7.2), around 7e-4, so
  the residual is far below the 0.999 cosine threshold and the gate is measuring convergence
  rather than the choice of tau. Scored by cosine and total variation against the analytic
  oracle.

  **Configurations.** Six, each run through both sub-experiments.

  | # | Family | L | Parameters |
  |---|---|---|---|
  | 1 | additive_random | 3 | seed 0 |
  | 2 | additive_random | 6 | seed 1 |
  | 3 | additive_uniform | 4 | a = 1.0 |
  | 4 | additive_epistatic | 4 | seed 3, couplings from the same generator |
  | 5 | single_peak | 5 | height = 2.0 |
  | 6 | class_quadratic | 6 | height = 2.0 |

  Mutation rate mu = 0.30 throughout, which sits inside the range swept in G-R.1.

  **Diagnostics recorded, not gating.** Per-configuration error at each step size; the
  fitted intercept as well as the slope; total-variation distance alongside cosine; step
  counts; and, for the additive families, the depth and two-qubit gate count of the
  structural circuit analogue, labelled as such, since that analogue is unitary and does not
  itself perform imaginary-time evolution.
