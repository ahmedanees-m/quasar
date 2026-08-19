# Protocol

Thresholds, parameter grids, seeds and time allocations for every check in the project, and the
numerical conventions they all depend on. Each check names the statistic it judges, the
threshold it applies, and the record it writes. The hash of this file at the time of a run goes
into that run's record.

Thresholds below were set before the corresponding runs. Sections 0 to 12 are the original
statement; the revisions after them record what changed later, kept separate so both the
original and its replacement can be read.

Maintainer: Anees Ahmed Mahaboob Ali

---

## 0. Standing rules

1. A failing check is fixed, or the failure is reported. Thresholds don't move to accommodate
   a result.
2. Every check runs as a test under `tests/gates/` and writes a JSON record to `results/`.
   No record, no pass.
3. Every stochastic component takes an explicit seed, listed here and recorded in the result.
4. Reference validity is declared per cell. A cell whose reference isn't trustworthy is
   excluded and reported as excluded, never scored against a weak reference.
5. `numpy.random.default_rng(seed)` only. No legacy global RNG.
6. float64 throughout. Tolerances are declared here, never chosen at run time.

---

## 1. Numerical conventions

Each of these corresponds to a class of silent, plausible-looking error, and two of them to
bugs this project has already been bitten by.

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

## 2. Where the starting values came from (2026-08-09)

The planning documents report an earlier implementation with specific measured values. It
could not be found in any archive when this repository was created: no code, no result files.
Those values are treated here as targets for a fresh implementation rather than as inherited
results, and none of them reaches the manuscript unless a run in this repository reproduces it
and writes a record. See `notes.md`.

---

## 3. Work package WP-R: rebuild and re-validate Phases 1–3

Objective: reconstruct the validated stack (analytic oracle, Hamiltonian compiler,
Trotter circuit, both imaginary-time routes, noise backends) and re-hit the seven checks
that the planning documents record.

Thresholds below are set at or slightly looser than the previously reported values, with the
reported value noted alongside. Where a run beats the threshold, the measured value is
reported.

| Check | Statement | Threshold | Previously reported |
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

**Check G-R (composite).** All of G-R.1 through G-R.10 pass, each with a committed JSON
artefact under `results/wp_r/`. Binary. No WP1+ run is judged before G-R passes.

**Seeds for WP-R.** Landscape seeds `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`. Optimiser seeds
`[11, 12, 13, 14, 15]`. Shot seeds `[101, 102, 103]`.

---

## 4. WP0: specification and prior art

**Check G-0.** `protocol.md` and `references.md` both complete and committed before any WP4+
run. Binary. `references.md` must cover all four literatures named in the execution plan,
with at least the works listed in T0.2, each carrying a one-line statement of what it
establishes and what it leaves open.

---

## 5. WP1: spectral and structural analysis

**Check G-1.**

1. The computed spectral gap reproduces the closed form where one exists (single-peak,
   permutation-symmetric) to relative error < 1e-6.
2. The gap closes as mu approaches the analytically located error threshold: the minimum of
   the computed gap over the mu sweep lies within 5% of the analytic mu_c, for L = 6, 8, 10.
3. Every stated property of the operator (Perron–Frobenius structure, stoquasticity,
   reversibility or its absence) is derived in `docs/theory.md` with the derivation
   referenced from the code docstring. Asserted-but-underived claims fail this check.

**Grid for the WP1 gap map.** Families: single-peak, additive, NK with K in {0,1,2,3,4,6},
spin-glass. L in {4, 6, 8, 10, 12, 14}. mu on a 41-point grid spanning
[0.2 mu_c, 2.0 mu_c] per instance. Seeds `[0..9]` per (family, K, L).

---

## 6. WP2: Route B, QSVT Perron-vector extraction

**Check G-2.**

1. Route B reproduces the analytic quasispecies at cosine >= 0.95 for L = 2..6 in noiseless
   simulation.
2. The block encoding satisfies its defining property to 1e-10: the top-left block of the
   unitary equals `A / alpha` for the declared normalisation alpha.
3. Resource scaling (query complexity, ancilla count, polynomial degree) is derived
   analytically and the derived degree agrees with the empirically sufficient degree to
   within a factor of 2, using the WP1 gap map as input.

**Acceptable outcome.** If a full circuit-level implementation proves
infeasible within the work package, a rigorous resource-estimation-only treatment of
Route B, with the limitation stated plainly, satisfies the work package. This is recorded
in advance so that the fallback cannot be presented later as a planned success.

---

## 7. WP3: landscape families

**Check G-3.**

1. Every landscape reproduces exactly from its seed, byte-for-byte, on the declared image.
2. NK with K = 0 equals the additive family analytically, to 1e-12.
3. Ruggedness statistics (number of local optima, fitness-correlation length) increase
   monotonically in K, verified over seeds `[0..9]` at L = 10 and L = 12, with monotonicity
   required of the seed mean and reported per seed.

Families: single-peak, additive + weak pairwise epistasis, NK(K), spin-glass (random +/-J),
Rough Mount Fuji, House-of-Cards, Block.

---

## 8. WP4: Baseline A, Wright–Fisher

**Check G-4.**

1. Reproduces the analytic single-peak quasispecies as population size N and sample budget
   grow: total-variation distance < 0.02 at the largest declared budget, L = 8.
2. Throughput is within 5x of the reference community implementation on a matched
   configuration, measured and reported. Falling outside 5x is a fail and the
   implementation is optimised, not excused.

Declared N sweep: `[1e3, 1e4, 1e5, 1e6]`. Burn-in 20% of generations. Seeds `[0..9]`.

---

## 9. WP5: Baseline B, Dixit–Srivastava–Vishnoi

**Check G-5.**

1. Matches the analytic oracle to <= 1e-6 on every landscape in its declared applicability
   class.
2. The applicability boundary is documented as an explicit predicate in code, and the set
   of WP7 grid cells it covers is emitted as a machine-readable map before the sweep runs.

---

## 10. WP6: Baseline C, tensor-network imaginary time

**Check G-6.**

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

## 11. WP7: the grid sweep and the decision check

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
- Declared hardware: 32 vCPU, 62 GB RAM, NVIDIA RTX A4000 16 GB,
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
flattering and more conservative metric and is the one used for the decision check where the
two disagree. Bootstrap 95% confidence intervals across seeds, 10000 resamples. Seed-to-seed
spread reported alongside the mean, never only the mean.

### 11.5 Check G-7: the decision check

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

## 12. WP8: live QPU

**Check G-8.** Results are reported as measured. Job IDs, backend name, calibration date,
transpiled depth, two-qubit check count, shots, and both raw and mitigated distributions are
recorded. No threshold is set on accuracy, because the purpose is feasibility, not
performance. Framing the result as evidence of advantage fails this check.

---

## Revisions

*(Append below this line only. Never edit above it.)*

- 2026-08-09: Initial registration. Anees Ahmed Mahaboob Ali.

- 2026-08-09: **revision 1: the G-R.1 case set.**
  Anees Ahmed Mahaboob Ali.

  Section 3 fixed the G-R.1 threshold but named the configurations only as "the WP-R set".
  That set is now fixed, below, before the check is executed. This revision adds detail; it
  does not change the threshold, which remains max absolute error < 1e-9.

  **What is compared.** For every case, the analytic oracle
  (`quasarstack/analytic/crow_kimura.py`, which never forms the 2^L generator) against
  brute-force exact diagonalisation (`quasarstack/analytic/exact_diag.py`, which builds the
  full sparse generator and knows nothing about the structure the oracle exploits). The
  statistic is the maximum absolute difference between the two L1-normalised genotype
  distributions, over all 2^L entries. The check statistic is the maximum of that over every
  case.

  **Sizes.** L = 2, 3, 4, 5, 6, 7, 8, 9, 10.

  **Mutation rates.** mu in {0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00} for every case.

  **Families.**

  1. `additive_random`: a_i drawn i.i.d. from Uniform(0.25, 2.00) with
     `default_rng(seed)`, seeds 0 through 9. Solved by the closed-form product route.
  2. `additive_uniform`: every a_i equal to a, for a in {0.25, 0.50, 1.00, 2.00}. This
     family is reachable by *both* analytic routes, so it is additionally checked
     closed-form against Hamming-class reduction, making it a three-way agreement. Both
     comparisons count toward the check statistic.
  3. `single_peak`: fitness `height` on the master sequence and zero elsewhere, for
     height in {1.0, 2.0, 5.0}. Solved by the class reduction.
  4. `class_quadratic`: f_d = height * (1 - d/L)^2, height in {1.0, 2.0, 5.0}.
  5. `class_exponential`: f_d = height * exp(-2d/L), height in {1.0, 2.0, 5.0}.

  **Diagnostics recorded per case.** The spectral gap lambda_1 - lambda_2 from the exact
  diagonalisation, and the mean-fitness difference between the two routes. The gap is
  recorded because the Perron eigenvector is only well conditioned while the gap is
  comfortably non-zero. **No case is excluded on the basis of its gap.** If a small-gap case
  fails, that is reported as a finding and feeds WP1, not quietly dropped.

  **Expected case count.** 9 sizes x 7 mutation rates x (10 + 4 + 3 + 3 + 3) configurations
  = 1449 comparisons, plus 252 closed-form-versus-class cross-checks on family 2.

- 2026-08-09: **revision 2: the G-R.2 configuration set.**
  Anees Ahmed Mahaboob Ali.

  Section 3 fixed the G-R.2 threshold as cosine >= 0.999999 on 40 out of 40 configurations
  but did not say which forty. They are fixed here, before the check is executed. The
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
  Uniform(0.25, 2.00) with `default_rng(seed)`, matching revision 1.

  **Check statistic.** The minimum, over all forty, of the cosine similarity between the
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
  - Pauli term count, which feeds check G-R.10.
  - For the additive families, the difference between the structured build and the
    Walsh-Hadamard build of the same operator, since both routes must produce it.

- 2026-08-09: **revision 3: the G-R.3 configuration set and protocol, registered before
  the run.** Anees Ahmed Mahaboob Ali.

  Section 3 fixed the G-R.3 thresholds as cosine >= 0.999 at dtau = 0.01 and a fitted
  exponent in [1.8, 2.2] with R^2 >= 0.99. The configurations and the two sub-experiments
  are fixed here, before execution. The thresholds are unchanged.

  **What is being evolved.** The Trotterised imaginary-time propagator of
  `quasarstack/circuit/trotter_ite.py`, symmetric second-order splitting
  S(dtau/2) M(dtau) S(dtau/2), started from the uniform superposition and renormalised each
  step. This is not a hardware-runnable circuit; imaginary-time evolution is non-unitary,
  and the hardware-faithful routes are checks G-R.6 and G-R.7.

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
  the residual is far below the 0.999 cosine threshold and the check is measuring convergence
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
  counts; and, for the additive families, the depth and two-qubit check count of the
  structural circuit analogue, labelled as such, since that analogue is unitary and does not
  itself perform imaginary-time evolution.

- 2026-08-09: **revision 4: the G-R.4 sweep, landscapes and order parameter, registered
  before the run.** Anees Ahmed Mahaboob Ali.

  Section 3 fixed the G-R.4 threshold as max absolute magnetisation difference < 1e-3 at
  L = 4, 6, 8. The order parameter, the sweep, and the landscape set are fixed here. The
  threshold is unchanged.

  **Disclosure.** The sweep range and the landscape normalisation below were chosen after an
  exploratory pass, because a threshold cannot be resolved without knowing roughly where it
  sits. That pass changed no threshold. It corrected two normalisation mistakes,
  both recorded here so the reasoning is auditable rather than invisible:

  1. An epistatic family that fixed the *total* fitness range made the per-mutation cost near
     the master scale as 1/L, so selection vanished and the exponent varied overall selection
     strength rather than epistasis.
  2. A pairwise coupling held at fixed `b` across L made the total interaction grow as L^2,
     which reversed the apparent direction of the epistasis effect between L = 6 and L = 8.
     Mean-field scaling `b = B / (L - 1)` holds the coupling per site fixed and removes it.

  **Order parameter.** The surplus, m = sum_sigma p(sigma) (1 - 2 d(sigma) / L), which is 1
  when the population sits on the master sequence and 0 when it is uniform over sequence
  space.

  **What is compared for the check.** The surplus computed from the ground state of the
  compiled Pauli Hamiltonian, against the surplus computed from the analytic Hamming-class
  reduction, at every point of the sweep. Statistic: the maximum absolute difference over all
  points, all landscapes and all sizes.

  **Sweep.** mu from 0.01 to 3.00 in steps of 0.01, 300 points. Sizes L = 4, 6, 8.

  **Landscapes**, all permutation symmetric so the analytic route applies.

  | # | Landscape | Parameters |
  |---|---|---|
  | 1 | single_peak | height 1.0 |
  | 2 | pairwise additive control | a = 0.5, B = 0.0 |
  | 3 | pairwise synergistic | a = 0.5, B = +1.0 |
  | 4 | pairwise synergistic | a = 0.5, B = +2.0 |
  | 5 | pairwise antagonistic | a = 0.5, B = -1.0 |

  with coupling `b = B / (L - 1)`.

  **Threshold location, two measures, and why both.** The susceptibility peak, chi = -dm/dmu,
  is the natural definition and is used where the peak is interior to the sweep. It is not
  always defined: a landscape additive in the surplus decays monotonically with its steepest
  slope at zero mutation rate, so the peak sits on the boundary and carries no information.
  The half-surplus crossover, the mu at which m first falls to half its initial value, is
  defined for any monotone decay and is comparable across families. Both are recorded, and
  whether the susceptibility peak was interior is recorded with them.

  **Diagnostics recorded, not gating.**

  - For the sharp peak, mu_c and the transition width against L, extended to L = 20 through
    the class reduction alone, testing whether mu_c * L approaches the peak height as the
    infinite-size analysis predicts.
  - The measured direction of the epistasis shift, compared against the claim in the
    planning documents that synergistic epistasis raises the threshold and antagonistic
    epistasis lowers it. **The direction is reported whichever way it falls.** It is not a
    pass condition, and no landscape will be dropped from the record for disagreeing.
  - The minimum spectral gap over the sweep and where it sits relative to the threshold,
    which is WP1 material and also bears on how hard the later imaginary-time checks will
    find this region.

  *Appended 2026-08-09, before the recorded run.* The gap diagnostic is extended to
  L = 4, 6, 8, 10, 12 for the sharp peak, measured by sparse eigensolves in a narrow window
  around each size's predicted threshold, with a decay rate per site fitted across them.
  Three sizes was too thin to say anything about how the gap closes. This changes no
  threshold and the full gap map across all landscapes remains WP1 check G-1.2.

- 2026-08-09: **revision 5: the G-R.5 instance set and the NK normalisation, registered
  before the run.** Anees Ahmed Mahaboob Ali.

  Section 3 fixed the G-R.5 threshold as cosine >= 0.99999 across all 10 seeded NK
  instances. The seed set, the sizes, the connectivities and the landscape normalisation are
  fixed here. **The per-instance threshold is unchanged, and every instance must meet it.**

  **On the instance count.** The registered "10 seeded instances" is read as the seed set,
  seeds 0 through 9. The run sweeps size and connectivity as well, so it tests 100 instances
  rather than 10. That is deliberately a superset: testing more instances against the same
  per-instance threshold can only make the check harder, never easier.

  **What is compared for the check.** The ground state of the compiled Pauli Hamiltonian
  against brute-force exact diagonalisation of the generator, per instance. These are
  independent code paths: the compiler goes through a Walsh-Hadamard decomposition into
  Pauli terms and back, while the reference assembles the sparse generator directly from the
  fitness vector. On a rugged landscape the decomposition is dense, which is the case the
  earlier checks did not exercise.

  **Instances.** L in {6, 8, 10} crossed with K in {1, 2, 4}, plus L = 8 with K = 7, each
  over seeds 0 through 9. 100 instances. Mutation rate mu = 0.25 throughout.

  **NK normalisation.** Fitness is standardised to zero mean and unit standard deviation.
  Raw NK fitness is a mean of L uniform draws, so its spread shrinks as 1/sqrt(L) and grows
  with K; sweeping K on the raw scale would vary selection strength and ruggedness together,
  and any result would be a mixture. This follows docs/notes.md. Neighbourhoods are adjacent and
  wrap around, which is deterministic given the seed.

  **Diagnostics recorded, not gating.**

  - **Where the optimum sits**, per instance, as docs/notes.md now requires of any ruggedness
    axis. An NK landscape has **no master sequence**: its global optimum is at a random
    genotype, near Hamming weight L/2. Statements about the error threshold, which is
    defined by delocalisation away from a master sequence, therefore do not carry over to
    this family unchanged, and the record says so rather than leaving it to be assumed.
  - **Ruggedness statistics** per instance, which is WP3 task T3.3 pulled forward: the
    number of local optima and the fitness autocorrelation across single-mutation
    neighbours, with monotonicity in K reported over the seed mean.
  - **Pauli term count** per instance, which feeds G-R.10 and WP1 task T1.4, and which for
    this family should grow with K as the decomposition stops being sparse.
  - **The Trotterised imaginary-time route** on the L = 8 instances, at tau = 60 and
    dtau = 0.01, scored against the same reference. This is the first test of whether
    imaginary-time evolution actually converges on rugged landscapes at a fixed budget, and
    it is explicitly *not* a pass condition: a rugged instance with a small gap may fail to
    converge, and that would be a finding for G-R.6, G-R.7 and WP7 rather than a defect.
  - The spectral gap per instance.

- 2026-08-09: **revision 6: the G-R.6 configuration set, the ansatz rule and the stopping
  criterion.** Anees Ahmed Mahaboob Ali.

  Section 3 fixed the G-R.6 thresholds as cosine >= 0.999 and circuit depth identical at
  tau = 2.5 and tau = 20. Both are unchanged. What is fixed here is the ansatz, the
  configurations, and how a run decides it has finished.

  **Disclosure: the ansatz depth was chosen by a pre-run scan, not guessed.** varQITE holds
  a fixed circuit, so its accuracy is capped by what that circuit can represent, and an
  ansatz too shallow to hold the answer fails the check for reasons that have nothing to do
  with the method. Ansatz depth is a method parameter, not an acceptance threshold, so
  choosing it adequately is legitimate; choosing it invisibly is not. The scan measured the
  worst cosine over three seeds at tau cap 60:

  | L | reps = 2 | reps ~ L/2 | reps = L | reps = L + 2 |
  |---|---|---|---|---|
  | 3 | 0.999982 | 0.999982 | 0.999982 | 0.999982 |
  | 4 | 0.994782 | 0.999900 | 0.999996 | 0.999996 |
  | 5 | 0.998973 | 0.999489 | 0.999997 | 0.999998 |
  | 6 | 0.988448 | 0.991322 | **0.998137** | 0.999937 |

  **reps = L + 2** is the registered rule: it is the shallowest choice in the scan that
  clears 0.999 everywhere tested. Note the L = 6 column at reps = L, which does not clear
  it. The ansatz depth needed grows faster than the system, and that is recorded as a
  diagnostic below rather than buried in a parameter choice.

  **Ansatz.** Ry rotations on every qubit, a nearest-neighbour CNOT chain, repeated `reps`
  times, and a final Ry layer. Real amplitudes throughout. Parameters start at the uniform
  superposition.

  **Stopping.** tau capped at 60 with dtau = 0.05, ridge 1e-6 on the geometric tensor, and
  early stopping when the **state** stops moving, at ``1 - |<psi_prev|psi_new>| < 1e-9``.

  Convergence is judged on the state and not on the parameters, and that is a correctness
  choice rather than a convenience. The ansatz has gauge directions, so the parameter update
  norm keeps fluctuating long after the state has settled: measured on a rugged L = 4
  instance, the largest parameter update was still 5.1e-2 at step 400 while the state was
  already at cosine 0.99991 against its reference. A parameter-space criterion would trigger
  at an arbitrary moment or never.

  **Configurations.** L in {3, 4, 5, 6} crossed with four landscapes, at mu = 0.20:

  | Landscape | Reference |
  |---|---|
  | additive_random, seed 0 | analytic closed form |
  | single_peak, height 1.0 | analytic class reduction |
  | nk, K = 2, seed 0 | sparse exact diagonalisation |
  | nk, K = 4, seed 0 (L >= 5) | sparse exact diagonalisation |

  **Depth criterion.** For every configuration the run is repeated with the tau cap at 2.5
  and at 20, and the resulting circuits are compared on depth and two-qubit check count, both
  as written and after transpilation to the basis {rz, sx, x, cx} at optimisation level 1
  with a fixed transpiler seed. Comparing the transpiled form matters: a run that left some
  angle near zero could have it optimised away, which would change the depth even though the
  ansatz did not.

  **Diagnostics recorded, not gating.**

  - `tau_used`, the imaginary time actually needed, and whether the run converged. This is
    the budget-needed-for-accuracy half of the fairness protocol in docs/notes.md and is the
    number WP7 will want per cell.
  - Monotonicity of the energy across every step. An ascending energy is one of the three
    failures the planning documents record for the sibling Motta method.
  - The parameter-shift and fidelity-shift recomputation of the McLachlan quantities, at
    L <= 4 where its four-evaluations-per-pair cost is affordable. This is what licenses
    describing the method as hardware-faithful rather than asserting it.
  - Accuracy against ansatz depth for the L = 6, K = 2 instance at reps in {4, 6, 8},
    recording the expressibility requirement directly.

  *Appended 2026-08-09, after a first execution and before the recorded run.* Two
  diagnostics are corrected. Neither acceptance threshold changes, and the first execution
  met both: minimum cosine 0.9999741 against 0.999, and depth identical at both tau values
  on all fourteen configurations.

  1. **The expressibility diagnostic used one seed and understated its own conclusion.** At
     L = 6 with reps = 4, seed 0 reaches 0.99996 while the worst of three seeds reaches
     0.9913, and it was the worst that justified the reps = L + 2 rule. Reporting the lucky
     seed put a number in the record that did not support the rule beside it. The diagnostic
     now runs seeds 0, 1 and 2 and reports the worst.

  2. **The energy-monotonicity diagnostic asked the wrong question.** It recorded a boolean,
     which came back false on seven of fourteen configurations, and a plain reading of that
     would suggest the failure the planning documents record for Motta-QITE.

     It is not that. The continuous flow provably cannot raise the energy: `dE/dtau` equals
     `-(1/2) grad(E)^T (A + delta I)^-1 grad(E)`, which is non-positive because A is a Gram
     matrix and the ridge only makes the form more definite. The integrator is explicit
     Euler, which overshoots at finite step. Measured at L = 6, the largest rise ran
     4.35e-2, 1.03e-3 and 2.13e-4 for dtau of 0.05, 0.02 and 0.01, roughly quadratic in the
     step and always within the first two steps, where the flow is stiffest just after
     leaving the uniform superposition.

     So the boolean is replaced by the size of the largest rise relative to the total
     descent, plus a step-size refinement at L = 4 over dtau in {0.05, 0.02, 0.01, 0.005}.
     The falsifiable statement is that an overshoot shrinks with the step while a defect does
     not, and that is what is now recorded and tested.

     The finest step is included because a separate refinement showed the rises reaching
     **exactly zero** there, not merely becoming small. At L = 6 the count went 2, 1, 1, 0
     for dtau of 0.05, 0.02, 0.01, 0.005, and at L = 5 it went 1, 1, 1, 0. Raising the ridge
     from 1e-6 to 1e-4 also removes the rise, which is the same explanation from the other
     side: a larger ridge shortens the step. Discretisation, and it disappears.

  *Appended 2026-08-09, correcting this revision.* **The earlier scan quoted above was run
  at mu = 0.25, and the check runs at mu = 0.20.** The table was presented as the
  justification for reps = L + 2 without that being stated, and the two sets of numbers are
  therefore not directly comparable. At L = 6 with reps = 4 the scan reported a worst case of
  0.9913 at mu = 0.25, while the check's own three-seed diagnostic reports 0.9968 at mu = 0.20.

  The conclusion is unaffected and the rule is unchanged, because the check's own diagnostic,
  run at the check's own mutation rate, reaches the same verdict: reps = 4 does not clear
  0.999 on every seed and reps = 6 does, so the registered reps = L + 2 remains conservative.
  But a table quoted as evidence must have been measured under the conditions it is evidence
  for, and this one was not. Recorded rather than quietly re-run, since the mistake is in the
  registration and not in the result.

- 2026-08-10: **revision 7: the G-R.7 configuration set and generator support, registered
  before the run.** Anees Ahmed Mahaboob Ali.

  Section 3 fixed the G-R.7 thresholds as cosine >= 0.95 and no energy increase beyond 1e-10
  on any step. Both are unchanged. What is fixed here is the generator basis, the support
  cutoff, the configurations, and the stopping rule.

  **The generator basis is the odd-Y Pauli strings, and this is a correctness requirement
  rather than a choice.** The state is real, so a unitary preserving that must be real
  orthogonal and its generator real antisymmetric. A Pauli string is imaginary exactly when
  it has an odd number of Y factors, and -i times an imaginary Hermitian matrix is real
  antisymmetric.

  The failure this prevents is the one the planning documents record for this method, "an
  element-wise gradient that vanishes for real states; the energy ascended instead of
  descending". In Motta's own form the right-hand side is `Re(-i <psi| sigma_I |Delta>)`, and
  for a real state and real residual that bracket is real whenever sigma_I is real, so -i
  times it is purely imaginary and the real part is **exactly zero**. Every Y-free string
  contributes nothing, and Y-free is precisely what one reaches for when the Hamiltonian is
  built from X and Z. The run records that quantity for both parities per configuration.

  **Disclosure: the support cutoff was chosen by a pre-run scan at the check's own mutation
  rate of 0.20**, correcting the mistake made in revision 6. Worst cosine over the scanned
  landscapes:

  | max weight | generators at L = 6 | worst cosine | steps with an energy rise |
  |---|---|---|---|
  | 1 | 6 | **0.8298** | 64 |
  | 2 | 66 | 0.9999856 | 0 |
  | 3 | 326 | 0.9999917 | 0 |

  **max_weight = 2** is registered: weight 1 fails the accuracy threshold outright, and
  weight 3 buys nothing for five times the generators. Note that the under-supported case is
  also the only one whose energy rises, so insufficient generator support shows up as a
  failure to track imaginary time and not merely as lower accuracy.

  **Configurations.** L in {3, 4, 5, 6} crossed with additive_random seed 0, single_peak
  height 1.0, nk K = 2 seed 0, and nk K = 4 seed 0 where K <= L - 1. mu = 0.20, tau capped at
  40, dtau = 0.05.

  **Stopping rule, corrected in both imaginary-time modules before this run.** Convergence is
  judged on the **rate** of state change, `||psi_new - psi_prev|| / dtau`, not on the
  per-step change. A per-step criterion trips sooner at a smaller step purely because each
  step moves less. Measured on this method: accuracy appeared to fall from 0.9999997 to
  0.9999731 as dtau went from 0.1 to 0.01, and the finer run was not worse, it had stopped
  earlier in tau. Since `tau_used` is the budget-needed-for-accuracy number docs/notes.md asks WP7
  to compare across methods, a step-size-dependent one would be actively misleading.

  This change also affects varQITE, so **check G-R.6 is re-run under it** and its record
  replaced, with the change explained, as docs/notes.md requires when a scientific field moves.
  Its registered thresholds are untouched.

  **Diagnostics recorded, not gating.** Accuracy against generator support at L = 5, which is
  Motta's cost curve and the counterpart to varQITE's ansatz-depth curve in G-R.6; the
  parity comparison per configuration; `tau_used` and whether the run converged; and the
  number of generators actually in use.

- 2026-08-10: **revision 8: G-R.7 failed its first execution. The failure, the diagnosis
  and the fix.** Anees Ahmed Mahaboob Ali.

  Recorded because a check failure is a scientific event and the run happened. Neither
  acceptance threshold is touched.

  **What failed.** Accuracy passed easily, minimum cosine 0.9999969 against 0.95 across all
  fourteen configurations, and the parity demonstration held everywhere with the even-Y
  contribution at exactly zero. The descent criterion failed: one step, in nk K = 4 at
  L = 5, raised the energy by **2.281e-03**, seven orders above the registered 1e-10 bound.
  Artefact of the failing run: `results/wp_r/g_r_7.json` at commit 5a53c44.

  **Diagnosis, on the offending instance.** The rise was at **step 0**, from the uniform
  superposition, and it was knife-edge: present at generator weight 2 but not 3 or 4, at
  dtau 0.05 but not 0.1 or 0.02, at ridge 1e-8 but not 1e-10 or 1e-6. Non-monotone
  sensitivity in every direction is not a physical effect.

  The cause is the linear solve. The Gram matrix reaches a condition number of **5.3e16**,
  past the reciprocal of double precision, so it is numerically singular. It is worst at the
  start, where the uniform superposition is maximally symmetric and whole families of
  generators act indistinguishably. A fixed **absolute** ridge on a matrix whose scale
  depends on the state therefore picks an arbitrary direction inside the near-null space,
  and on one instance that direction overshot.

  **Fix.** The normal equations are solved by truncated SVD with a **relative** singular-value
  cutoff instead of an absolute ridge. That is scale invariant and discards the degenerate
  directions rather than guessing in them. The cutoff is 1e-8, taken from the measured
  conditioning rather than from what makes the check pass: below that, directions carry no
  information at double precision. It removes the overshoot at every step size tried, not
  only the registered one, and leaves accuracy unchanged at the seventh decimal. The Gram
  condition number is now recorded per configuration.

  **Second correction, affecting both imaginary-time checks.** The stopping tolerance was
  carried across the change from a per-step infidelity to a rate without rescaling, leaving
  it about six orders too strict, and 10 of 14 configurations ran to the tau cap without
  converging. The like-for-like value is derived rather than chosen: infidelity below 1e-9
  corresponds to a step of about 4.5e-5, hence a rate of about 9e-4 at dtau = 0.05. Both
  checks now register **1e-3**.

  Both are method parameters, not thresholds. G-R.6 and G-R.7 are re-run under them.

- 2026-08-10: **revision 9: the G-R.10 families.**
  Anees Ahmed Mahaboob Ali.

  Section 3 fixed the G-R.10 threshold as a Pauli-term ratio of at least 50 at L = 12.
  Unchanged.

  **What is compared, stated precisely.** The two forms are *different landscapes*, not two
  encodings of one, and the claim only means something if that is said. The single-peak
  projector puts fitness on one genotype and nothing elsewhere; having no structure, its
  exact decomposition needs every Z subset, so 2^L terms plus one transverse term per site.
  The sparse form is additive fitness with a small number of pairwise couplings, which is
  what a real biological landscape looks like, and needs a number of terms linear in L. The
  claim is that the realistic case is exponentially cheaper than the textbook one, not that
  a compiler found a clever encoding of the same thing.

  **Sparse family.** Per-site coefficients evenly spaced from 0.5 to 1.5 and two nearest-
  neighbour couplings of 0.4, deterministic so the count is a property of the family rather
  than of a draw. mu = 0.20. Sizes L = 4, 6, 8, 10, 12.

  **Projector count.** Taken from the closed form 2^L + L rather than built, because
  materialising 4096 Pauli terms at L = 12 to count something arithmetic is wasteful. The
  formula is **verified against an actual construction at every L up to 8**, and the run
  fails loudly if it ever disagrees.

  **Diagnostic, not gating.** Term count against NK connectivity at L = 4, 6, 8 for
  K = 0, 1, 2, 4, reported as a fraction of the dense projector. The sparse form is cheap
  because the landscape has structure, so what happens as structure is removed is the informative
  counterweight to the headline ratio and belongs beside it.

- 2026-08-10: **revision 10: the G-R.8 configuration set, and a decode-boundary finding
  that changes what the check measures.** Anees Ahmed Mahaboob Ali.

  Section 3 fixed the G-R.8 threshold as mitigated cosine >= 0.98 at L = 2 to 4 under
  IBM-Heron-like and trapped-ion noise. Unchanged.

  **SIMULATED NOISE ONLY.** Nothing in this check runs on hardware. Device parameters are
  representative of the device class, from published typical figures, not a calibration
  snapshot of a named machine on a named date. The live run is WP8 and reports job
  identifiers.

  **The finding, which the planning documents do not mention and which decides what "cosine
  against what" means here.** The Perron vector of the generator *is* the quasispecies, and
  the ground state of the stoquastic Hamiltonian *is* that vector, so **the circuit holds the
  distribution in its amplitudes**. A computational-basis measurement returns ``|psi|^2``.
  Those are different distributions. Measured at L = 3 on an additive landscape:

  | quantity | cosine vs oracle | total variation |
  |---|---|---|
  | amplitudes, what the circuit holds | 0.9999998 | 4.96e-04 |
  | squared, what a measurement returns | 0.9865108 | **2.24e-01** |

  The squared distribution is non-negative, normalised, peaked on the same genotype, and
  scores 0.987 on cosine. It would have passed an eyeball check and most thresholds while
  being the wrong object, and only total variation exposes it. That is the failure mode this
  project exists to catch, and it is the reason `protocol.md` section 11.4 already lets total
  variation decide.

  **The fix is a decode step**, `decode_from_measurement`: take the element-wise square root
  of the measured frequencies and renormalise. It inverts the encoding exactly in the
  infinite-shot limit and halves relative error rather than inflating it. Verified: total
  variation against the oracle falls from 0.224 to 0.0036 at 100000 shots.

  **A consequence that must be reported, not buried.** The same square root **amplifies the
  noise floor in the tail**: a spurious probability of 1e-4 becomes an amplitude of 1e-2. So
  the encoding that makes the ground state the quasispecies also makes readout noise worse
  after decoding. Cosine stays above the threshold; total variation lands around 0.10 to
  0.14 under noise, and both are recorded.

  **What the check scores.** The mitigated, **decoded** distribution against the analytic
  quasispecies, because that is the object the biology asks for and it is the harder test.
  The easier question, whether mitigation recovered what the ideal circuit would have
  produced, is recorded alongside as a separate number rather than substituted for it.

  **Configurations.** L in {2, 3, 4} crossed with additive_random seed 0 and single_peak
  height 1.0, on both device models. mu = 0.20, 40000 shots, 40000 calibration shots per
  basis state, seed 0. State prepared by varQITE at reps = L + 2, matching G-R.6.

  **Mitigation.** The readout assignment matrix is *measured* from calibration circuits
  rather than copied from the device parameters, because copying them would make the
  correction exact by construction and would test nothing. Inversion is a non-negative
  least-squares projection onto the simplex rather than a plain matrix inverse, which
  routinely returns negative probabilities once shot noise is comparable to the correction;
  clipping those and renormalising biases the result toward whatever was clipped.

  **Diagnostics.** Transpiled depth and two-qubit count per device, raw beside mitigated as
  section 11.2 requires, whether mitigation helped in each case, and the smallest diagonal
  entry of the measured assignment matrix.

---

### Revision 11: G-R.9: the gradient component, and a disclosure about how it was chosen
Registered before `experiments/wp_r_rebuild/g_r_9_barren.py` was run, and after exploratory
scans whose numbers are disclosed in full below. Section 3's thresholds are unchanged: fitted
decay base in [0.30, 0.55], R squared at least 0.95.

**Configuration.** L in {2, ..., 8}. Ansatz reps = L + 2, matching G-R.6 and G-R.8. mu = 0.20.
400 parameter draws per size from the uniform distribution on [0, 2 pi), seed 0, reseeded per
size. The measured quantity is the variance across draws of one component of the McLachlan
force `C_i = -<d_i psi|H|psi>`, which is minus half the energy gradient.

**Check statistic.** The middle component, index `n_parameters // 2`, on the NK K=2 landscape.
Both other components (first, and the mean over all components) and the single-peak landscape
are measured and recorded, but they do not decide the check.

**Why the middle component.** The first rotation sits at the circuit boundary with the fewest
entangling layers between it and the state, so it is the least scrambled parameter in the
circuit and the least representative of the plateau. The barren-plateau literature measures a
parameter in the circuit interior. That is the reason, and it is a reason that would have been
given had the question been asked before any number existed.

**Disclosure, because the order of events matters here.** The first scan measured only the
first component and returned values above the registered band. The methodological question
"which component should this be?" was therefore raised *after* seeing a result that failed,
which is the classic route by which an analysis choice gets selected for its answer. Three
mitigations, none of which fully removes the concern:

1. The choice of the middle component was stated, in writing and with its justification,
   before the scan that measured it returned.
2. All three components on both landscapes are recorded in the check artefact. Nothing that
   was measured is discarded.
3. The spread is small enough that the choice barely matters to the physics, though it does
   decide the check. The exploratory values, all at reps = L + 2, 400 draws:

   | landscape | first | middle | mean over all |
   |---|---|---|---|
   | NK K=2 | 0.5558 (R2 0.9968) | **0.5490** (R2 0.9916) | 0.5350 (R2 0.9996) |
   | single peak | 0.5544 (R2 0.9909) | 0.5421 (R2 0.9902) | 0.5388 (R2 0.9903) |

   Six combinations spanning 0.535 to 0.556. The registered upper bound of 0.55 passes through
   the middle of that cluster, so five of the six fall inside the band and one falls outside.

**Consequence for how the result should be read.** A pass here is a pass by 0.001 against a
spread of 0.021 across defensible analysis choices. The band is therefore not confirmed by
this measurement, and the artefact says so. What *is* established, robustly and independently
of every choice above, is that the variance decays exponentially in L with base near 0.54 at
R squared 0.99 or better in all six combinations. That is the statement the project should
carry forward, and it is the one that bounds Route A's reach.

**A difference from the lost implementation, recorded because it is real.** The planning
documents report `0.42^L`. This rebuild measures `0.54^L`. The band [0.30, 0.55] was wide
enough to contain both, which is why the check can pass while the central value is not
reproduced. At L = 12 the two differ by a factor of about 20 in gradient variance, so the
rebuilt ansatz plateaus less steeply than whatever the lost implementation used. Its ansatz
and measured component are unknown and cannot be recovered, so the discrepancy cannot be
resolved, only recorded.

---

### Revision 12: G-1: what "the analytic mu_c" means, the grid, and a disclosure
Registered before `experiments/wp1_spectral/g_1_gap_map.py` was run, and after the
exploratory scans whose numbers appear below. Section 5's three criteria are unchanged. This
revision fixes only what section 5 left ambiguous, and it makes the ambiguity harder on the
check rather than easier.

**The ambiguity.** Criterion 2 asks that the gap minimum lie "within 5% of the analytic
mu_c". Two readings are defensible and they are different numbers at finite L:

- **Reading A, the asymptotic threshold.** `mu_c = height / L` for the single peak. This is
  the closed form the model is known for and the one section 11 of this file alludes to when
  it asks whether `mu_c * L` approaches the peak height.
- **Reading B, the project's own locator.** `locate_threshold` puts `mu_c` at the peak of the
  magnetisation susceptibility, computed here from the exact class reduction rather than from
  simulation, which makes it "analytically located" in the sense criterion 2 uses. G-R.4
  already validated this locator against the analytic magnetisation.

**Decision: the check requires both.** Criterion 2 passes only if the gap minimum is within 5%
of Reading A *and* within 5% of Reading B, at every one of L = 6, 8, 10. This is strictly
harder than either reading alone, which is the point: having seen the numbers before
registering, the only choice that cannot be accused of selecting for the answer is the choice
that can only hurt.

**Criterion 1, and a correction to what it names.** Section 5 says the closed form exists for
"single-peak, permutation-symmetric". That is not right, and the error is inherited rather
than introduced here. The single-peak gap has **no** elementary closed form at general `mu`;
what it has is an exact `(L+1)`-dimensional reduction and two exact limits. The family that
does have one is the **additive** landscape, where the generator is a sum of commuting
single-site terms and

    Delta = 2 min_i sqrt(a_i^2 + mu^2)

exactly, independent of L. Criterion 1 is therefore read as "reproduces every closed form
that exists, to relative error < 1e-6", and three are tested: the additive gap above, the
pure-mutation gap `2 mu` at zero fitness, and the single-peak saturation to `2 mu` above the
threshold.

**Configurations.** Families: additive (seeds 0 to 9), single peak at heights 1.0 and 2.5,
NK with K in {0, 1, 2, 4}, seeds 0 to 9. L in {4, 6, 8, 10, 12} for families needing the
dense route, extended to L in {16, 24, 32, 48, 64} for the permutation-symmetric families
through the class reduction. mu on the 41-point grid spanning `[0.2 mu_c, 2.0 mu_c]` that
section 5 registers, with `mu_c = height / L`. Criterion 2 is evaluated on that grid, whose
spacing is 4.5% of `mu_c` and therefore quantises the located minimum by up to that much; a
fine 1500-point grid is reported alongside so the quantisation is separable from the physics.

**Extended precision.** Any gap below `1e-9` is recomputed by Sturm bisection at 60 decimal
digits and the float64 value is discarded. Two LAPACK routines were observed agreeing with
each other to `1e-16` while both were wrong, because they share the failure mode, so
agreement between float64 methods is not accepted as evidence.

**Disclosure of what was already seen.** Exploratory scans were run before this revision and
their results are the reason it exists. On the single peak at height 1.0, comparing the gap
minimum against both readings:

| L | gap min vs Reading B (chi peak) | gap min vs Reading A (h/L) |
|---|---|---|
| 6 | 29.3% | 18.2% |
| 8 | 13.8% | 17.7% |
| 10 | 7.1% | 14.4% |
| 12 | 4.1% | 11.6% |
| 24 | 0.18% | 4.7% |
| 64 | 0.38% | 1.7% |

**So criterion 2 is expected to fail, and it is being registered anyway rather than
adjusted.** The two locators agree to within the grid resolution by L = 24 and disagree by
tens of percent at L = 6, 8, 10, because they are different finite-size locators of a
crossover that only becomes sharp as L grows. No implementation can make them agree at L = 6;
the disagreement is a property of the model. The physics criterion 2 was written to test,
that the gap closes where the population delocalises, is supported. The 5% agreement at
L = 6, 8, 10 is not reachable. Section 0 rule 1 says the threshold does not move, so it does
not move, and the failure is reported with the reason.

**note on revision 12, registered at the same time.** Section 5's grid is defined
"per instance" relative to `mu_c`, and the NK family has no peak height for
`mu_c = height / L` to refer to. For any landscape without a distinguished peak the grid uses

    mu_c  =  ( max_s f(s)  -  mean_s f(s) )  /  L

which reduces to `height / L` for the single peak up to a term of order `2^-L`, and which
generalises the quantity the threshold actually depends on: the selective advantage of the
best genotype over a random one, per site. Criterion 2 is not evaluated on the NK family,
which has no analytic `mu_c` to compare against; NK enters the gap map only.

---

### Revision 13: G-2: Route B configurations, the derived degree, and a corrected derivation
Registered before `experiments/wp2_qsvt/g_2_route_b.py` was run, and after the exploratory
runs disclosed below. Section 6's three criteria are unchanged.

**Standing assumption.** Route B is built as option C: QSVT
eigenstate filtering for a Hermitian stoquastic operator. docs/notes.md records that the G-2
thresholds do not depend on which QSVT construction is chosen, so this assumption does not
touch the specification. Every G-2 artefact carries it in its notes.

**Configurations.** L in {2, 3, 4, 5, 6}. Families: additive with per-site coefficients drawn
uniformly on [0.3, 1.5] at seeds 0 to 9, seeded per (family, L, seed) rather than from one
shared stream; and single peak at heights 1.0 and 2.5. mu = 0.20 throughout. The initial
state is the uniform superposition over all genotypes, which is the state a hardware run
would actually start from and which carries no information about the answer.

**Criterion 2, the block encoding.** Both forms are built and checked: the asymmetric one,
which puts coefficient signs in one of the two preparations, and the symmetric one, which
puts them in SELECT so that qubitisation is available. Each is checked for the defining
property to 1e-10, for a block spectral norm not exceeding 1, and, at sizes where the full
operator is affordable, for unitarity of the circuit itself. A block that matches while the
circuit is not unitary would mean the extraction is wrong rather than the encoding right.

**Supporting check, not part of the check.** The qubitised walk must produce Chebyshev
polynomials: the top-left block of `W^d` equals `T_d(A / alpha)` to 1e-10 for d in
{0, 1, 2, 3, 5, 8}. Recorded because it separates two failure modes that otherwise arrive
together, a wrong walk and a wrong polynomial.

**Criterion 3, the derived degree, and the correction.** The comparison is between the
smallest degree that empirically reaches cosine 0.95 against the analytic quasispecies, and
the degree predicted by

    d  =  (alpha / Delta) * ln( sqrt(1 - gamma^2) / (gamma * sqrt(epsilon)) )

with `alpha` the one-norm of the Pauli coefficients, `Delta` the spectral gap from the WP1
map, `gamma` the overlap of the initial state with the Perron vector, and
`epsilon = 1 - 0.95^2` matching the accuracy criterion 1 demands. Agreement within a factor
of two in either direction, on every configuration.

**Disclosure.** The first version of this prediction omitted the overlap term entirely and
used a fixed `epsilon = 1e-3`. Measured against the empirical degrees it overshot by factors
of 3.3 to 7.2 on all ten exploratory configurations, which is what surfaced the omission.
The corrected expression is the standard two-factor eigenstate-filtering cost and is derived
line by line in the docstring of `quasarstack.qsvt.filter.predicted_degree`: the gap sets how
sharp the polynomial must be, the overlap sets how far the unwanted components must be
pushed down, and conflating them is an error rather than a constant factor. **It contains no
fitted constant.** Nothing was tuned to the data; a term that should always have been there
was restored.

Exploratory ratios of predicted to empirical degree, before and after:

| | range | within a factor of 2 |
|---|---|---|
| omitting overlap | 3.32 to 7.22 | 0 of 10 |
| corrected | 0.60 to 1.84 | 10 of 10 |

The correct reading is that criterion 3 would have failed on the first derivation and passes
on the corrected one, and that the correction was prompted by the failure. What makes this a
method fix rather than a moved threshold is that the corrected formula is the textbook one,
is derived rather than fitted, and would have been the right answer had the question been
asked before any number existed. A reader who disagrees has both sets of numbers above.

**A statement recorded so it cannot later be quietly dropped.** The degree is **linear** in
`alpha / Delta`, not square root. Chebyshev acceleration, which does buy a square root, needs
the target eigenvalue to sit outside the interval where the polynomial is bounded, and here
it is inside the encoded spectrum by construction. Route B's advantage over Route A, if there
is one, is therefore not in this factor.

---

### Revision 14: G-3: family parameters, and what "correlation length" means here
Registered before `experiments/wp3_landscapes/g_3_families.py` was run. Section 7's three
criteria are unchanged.

**Parameters.** NK with K in {0, 1, 2, 3, 4, 6} and adjacent neighbourhoods. Spin glass with
zero field, discrete `+/- 1` couplings. Rough Mount Fuji at slope 1.0 and roughness in
{0.0, 0.1, 0.3, 1.0, 3.0}. Block with block size in {1, 2, 4}. Single peak at height 1.0.
Additive plus weak pairwise epistasis at `a = 1.0`, `b = 0.1`. Seeds 0 to 9 throughout.
L = 10 and L = 12 for criterion 3, L = 8 and L = 10 for the reproduction hashes.

**Correlation length.** Section 7 asks for "fitness-correlation length", and
`ruggedness_statistics` reports the nearest-neighbour autocorrelation `rho`. The two are
related by the standard Weinberger definition, `ell = -1 / ln(rho)`, which the check computes
rather than storing, so that no existing artefact changes shape. Where `rho <= 0` the
landscape has no correlation length and the result records `0.0` rather than a complex number.

**What monotone in K means, stated so it cannot be read two ways.** Ruggedness increasing
means the local-optima count increases and the correlation length decreases. Both are
required, of the mean over the ten seeds, and both are reported per seed so a family that is
monotone on average and not per instance is visible.

**Criterion 1, "byte-for-byte".** The check stores a SHA-256 of each fitness array in the
artefact. Reproduction within a run is checked by building each landscape twice with the
global NumPy random state deliberately disturbed in between; reproduction across runs is
checked by comparing those hashes with the committed record, which is what makes the claim
mean anything more than "the function is deterministic within one process". This is also the
check that would have caught docs/notes.md in the landscape layer had the defect been there.

**A finding disclosed in advance, because it bears on which family the ruggedness axis
should use.** Exploratory measurement at L = 8 over ten seeds puts the mean Hamming weight
of the global optimum at 0.00, 0.10 and 0.50 for Rough Mount Fuji at roughness 0.0, 0.3 and
1.0, against 3.70 for NK at K = 2 and 4.50 for house-of-cards. Over the same range the RMF
local-optima count goes from 1.0 to 12.7. Rough Mount Fuji therefore appears to be the family
that varies ruggedness while leaving the master sequence in place, which is what docs/notes.md
said the project needs and NK does not provide. The result records the optimum location for
every family and instance so that this can be judged from the artefact rather than from this
paragraph.

---

### Revision 15: G-4: configurations, and criterion 2 reported as blocked
Registered before `experiments/wp4_wright_fisher/g_4_wright_fisher.py` was run. Section 8's
criteria are unchanged; this revision fixes the configurations and records why criterion 2
cannot be evaluated as written.

**Configurations.** Single peak at height 1.0, L = 8, mu = 0.10. Population sweep
`[1e3, 1e4, 1e5, 1e6]` as section 8 declares, burn-in 20%, seeds 0 to 9, 4000 generations at
`dt = 0.01`. Criterion 1 is decided at the largest declared budget, `N = 1e6`.

**The time step, which section 8 does not mention and which matters.** Wright-Fisher is
discrete-generation and Crow-Kimura is continuous-time, so the two agree only as the step
shrinks. Selection weights are `1 + f dt` and the per-site mutation probability is `mu dt`.
The check reports the discretisation bias separately from the sampling error, over
`dt` in `{0.04, 0.02, 0.01, 0.005}`.

**The population must scale with `1 / dt` in that study, and this is registered because
getting it wrong inverts the conclusion.** Wright-Fisher resamples once per generation, so
genetic drift is `1 / N` per generation and `1 / (N dt)` per unit of simulated time. Halving
`dt` at fixed `N` halves the bias and doubles the drift. Exploratory measurement at
`N = 1e5`, holding simulated time fixed: without scaling the population, the distance to the
analytic answer plateaus at 0.024, 0.015, 0.016, 0.015 while the equilibration drift climbs
0.019, 0.030, 0.040, 0.052; with the population scaled as `1 / dt`, the drift holds flat near
0.020 and the distance falls monotonically 0.024, 0.017, 0.008, 0.006. The check uses the
scaled form and reports both.

**Criterion 2 is reported as blocked rather than passed.** See docs/notes.md. The implementation
runs in genotype-count space at `O(L 2^L)` per generation, independent of `N`, while a
community forward simulator is individual-based at `O(N L)`. At the top of the declared
sweep the two differ by about three orders of magnitude by construction, so a
"throughput within 5x" test would pass by a factor of a thousand and establish nothing about
whether the baseline is well built, which is the only thing it exists to establish. No
reference implementation is present in the pinned image either, and installing outside Docker is forbidden
installing one outside Docker.

The check therefore records absolute throughput and the measured scaling, so the comparison
can be completed later without rerunning, and **G-4 as a whole is not claimed as passed**.
docs/notes.md recommends replacing criterion 2 with time-to-accuracy at matched total variation,
which is representation-independent, is what the WP7 boundary map consumes, and can fail.

---

### Revision 16: G-5: what the applicability class is, and an attribution left open
Registered before `experiments/wp5_exact_class/g_5_exact_class.py` was run. Section 9's two
criteria are unchanged.

**The class, stated as a predicate.** A landscape is in the polynomial-time class when it is
additive, `f = sum_i a_i z_i`, or permutation symmetric, `f` a function of Hamming weight
alone. Both are decided by measurement rather than by declaration: the additive test fits the
`L + 1` spin coefficients and takes the largest residual, the symmetry test takes the largest
deviation within each Hamming class. A landscape qualifies when either residual, relative to
the largest absolute fitness, is at most `1e-9`. That tolerance sits far above float noise,
which is `1e-16` here, and far below any epistasis worth the name, so the predicate is not
deciding borderline cases on rounding.

**Configurations.** L in {4, 6, 8, 10}. In class: additive at seeds 0 to 9, single peak at
heights 1.0 and 2.5, uniform additive-plus-pairwise at `a = 1.0` and `b` in {0.05, 0.1}. Out
of class, which the baseline must **refuse** rather than solve: NK at K in {1, 2, 4}, spin
glass, house of cards, Rough Mount Fuji at roughness 0.5, block at size 2, all at seeds 0 to
9. mu in {0.05, 0.10, 0.20}.

**Criterion 2's coverage map is emitted before the sweep, not after.** Deciding coverage
afterwards would let it be chosen with the results in view. The check emits the covered set for
the full WP3 family list as part of its artefact.

**Refusal is part of the check, not an error path.** A baseline that quietly falls back to
exact diagonalisation on an out-of-class instance would report itself as covering WP7 cells
it does not cover, and the boundary map would inherit that in the direction that flatters the
quantum method. The check therefore requires every out-of-class configuration to raise.

**Attribution, deliberately left open.** Execution plan v4 names this baseline
"Dixit-Srivastava-Vishnoi" after `references.md` entry II.1, which is still flagged
`to-verify`. The project's rule is that nothing may be cited while flagged, so this check does
not claim the class it implements is theirs. It implements the class this project can derive
and check. Whether the two coincide has to be settled by reading arXiv:1203.1287 before the
name goes in the manuscript, and it is not a formality: **if their class is strictly larger,
WP7 has cells it currently believes are classically hard and the boundary map is wrong in the
direction that favours the quantum method.**

---

### Revision 17: G-2: a verification budget, set from measurement after the first run stalled
Registered before `experiments/wp2_qsvt/g_2_route_b.py` was re-run. Section 6's criteria and
revision 13's configurations are unchanged. What this adds is a limit on **which**
configurations have their block encoding verified, and it exists because the first attempt did
not finish.

**What happened.** G-2 was launched over the revision 13 configuration set and was still
running after more than an hour on the L = 6 single peak. Verification extracts the encoding's
top-left block by simulating `2^n` statevectors through a circuit of multi-controlled checks,
and the cost climbs steeply with the ancilla count. Measured, symmetric form, single-peak
family unless noted:

| L | family | terms | encoding qubits | seconds |
|---|---|---|---|---|
| 3 | additive | 7 | 6 | 0.6 |
| 3 | single peak | 11 | 7 | 2.4 |
| 4 | additive | 9 | 8 | 3.6 |
| 4 | single peak | 20 | 9 | 21.9 |
| 5 | additive | 11 | 9 | 8.8 |
| 5 | single peak | 37 | 11 | 129 |
| 6 | single peak | 70 | 13 | > 1200 |

**Budget.** The block encoding is verified for every configuration needing at most **12
encoding qubits**, and skipped above that with the configuration and its qubit count recorded
in the artefact. Twelve is chosen so that everything up to and including the L = 5 single peak
at 11 qubits is verified, and the 13-qubit case that stalled the run is not.

**What this does and does not weaken.** Criterion 2 asks that the block encoding satisfy its
defining property to 1e-10. That is a statement about the construction, which does not depend
on L, and it is now checked on configurations spanning 6 to 11 encoding qubits and three
families. Criterion 1, the accuracy of the filtered state, is **unaffected**: it needs only
`alpha`, which is a sum over Pauli coefficients and is computed directly by `one_norm` without
building any circuit. Criterion 3 likewise. So the budget removes redundant verification of a
size-independent property, not coverage of a size-dependent one.

**A family added, because the configuration set was testing the wrong thing.** revision 13
listed additive and single peak. The single peak enters `diagonal_hamiltonian` as its
projector form, which is exactly the representation `notes.md` forbids and G-R.10 exists
to argue against: 4108 Pauli terms at L = 12 against 27 for the sparse form. Verifying the
encoding of a representation the project has decided not to use is worth doing once as a worst
case and is not worth doing at every size. The uniform additive-plus-pairwise family at
`b = 0.1` is therefore added at every L, since that is the sparse form Route B would actually
run on.

**A performance change that was tried and did not work, recorded so it is not tried again.**
`SELECT` was rewritten to emit one multi-controlled single-target check per Pauli factor
instead of one multi-controlled multi-target check, expecting Qiskit's specialised path for
controlled Paulis to be cheaper. Measured, it is slower: 21.9 s against 15.0 s on the L = 4
single peak, because the check count multiplies by the Pauli string weight. The clearer form
was kept and no speedup is claimed for it.

**note on revision 17, registered at the same time.** The block-encoding property is a
statement about the construction and not about the coefficient values, so verifying all ten
additive seeds at every size repeats the same check ten times. Verification runs on the first
**two** seeds per (family, L); two rather than one so a sign-handling bug still has varied
coefficients to surface in. Criterion 1 continues to run on all ten, since accuracy does
depend on the instance.

---

### Revision 18: WP7: the order parameter, the budget protocol, and two axes instead of one
Registered before any WP7 sweep runs. Section 11's grid, reference rules and G-7 decision
criteria are unchanged. This revision fixes three things WP1, WP3 and WP6 turned up that
section 11 could not have anticipated when it was written.

**1. The order parameter is measured from each instance's own fittest genotype. See docs/notes.md.**

Section 11.4 scores against the reference distribution, and the surrounding analysis uses
`magnetisation`, which measures concentration on genotype 0. That is right for the single peak
and for any additive landscape, and wrong for a rugged one, because a rugged landscape's
optimum is somewhere random. Measured: Rough Mount Fuji keeps its optimum at genotype 0 in 97%
of instances at roughness 0.3, where it has 1.4 local optima, and in 25% at roughness 1.0 at
L = 12, where it has 121. Retention gets **worse** with L. NK sits at Hamming weight 3.2 to 4.2
out of 8 at every K including K = 0.

A sweep that keeps measuring from genotype 0 while raising ruggedness stops measuring
localisation partway along the axis and starts measuring how far the optimum has wandered.
That quantity also falls with ruggedness and also looks like a threshold crossing, and it
would land in the paper's central figure.

Every cell therefore records the reference genotype used and its Hamming weight, and uses
`order_parameter.localisation(probs, reference)`, which reduces to `magnetisation` exactly
when the optimum is genotype 0 so nothing already measured changes.

**Naming, which is part of the registration and not a stylistic note.** Where the landscape is
rugged the result is a **localisation transition**, not an error threshold. The error threshold
in the classical quasispecies literature means delocalisation from a master sequence, and on a
rugged landscape there is no master sequence. A rugged-landscape transition point must not be
quoted as comparable to the sharp-peak value.

**2. The budget protocol reports both panels. reporting both panels, proceeding under the rule in docs/notes.md.**

Section 11.3 fixes wall-clock per cell per method. docs/notes.md records that an equal-wall-clock
budget systematically disadvantages imaginary time in exactly the rugged near-threshold cells
WP7 is about, because the imaginary-time budget scales as `1 / gap` and the gap closes there
at roughly `0.72` per site. The sweep therefore records, per cell per method, both **accuracy
at the fixed budget** section 11.3 declares and **the budget needed to reach a fixed accuracy**,
with a stated ceiling beyond which a method is recorded as not having reached it.

This is a superset of the alternatives, so whichever the PIs choose the data will already
exist. G-7's decision criteria are evaluated on the fixed-budget panel as section 11.5 states;
the second panel is reported alongside and does not move the check.

**3. The ruggedness axis is two axes, because the families separate.**

Section 11.1 lists one ruggedness axis. WP3 and WP6 measured that biological faithfulness and
compilation cost point in different directions, and a single axis conflates them:

| | keeps the master sequence | Pauli terms at L = 12 | MPO bond dimension at L = 12 |
|---|---|---|---|
| Rough Mount Fuji | yes to roughness 0.3 | 4108, dense | 64, saturated |
| spin glass | no, weight 2.2 | 79 | 8 |
| NK, K = 2 | no, weight 3.2 | 61 | 6 |
| house of cards | no, weight 4.8 | 4108, dense | 64, saturated |

Rough Mount Fuji is the family the biology wants and the worst case for both the circuit and
the tensor network. The spin glass is rugged and cheap for both. The sweep runs both and
reports them separately rather than averaging over a "ruggedness" that means two things.

**4. An expectation recorded in advance, so the outcome is not read as a surprise either way.**

`results/wp6/g_6_3.json` finds no family in the set that is cheap for the circuit while being
expensive for a matrix-product operator: wherever the Pauli expansion is sparse the MPO bond
dimension is small, and where the MPO saturates its ceiling the Pauli expansion is dense too.
That is the only shape in which G-7's positive result could appear, and it is absent from the
operator-level structure. This raises the prior on the null that section 11.5 already
registers as publishable.

It is **not** decisive, and the sweep is not being pre-empted. The MPO bond dimension is a
property of the operator and sets the per-step cost; whether the quasispecies *state* is
representable at modest bond dimension is a different question and is exactly what G-6
criteria 1 and 2 measure. A positive result remains possible if the state is hard where the
operator is easy. This paragraph exists so that a null is reported as an expected finding
rather than a disappointment, and a positive one as a surprise that gets extra scrutiny.

---

### Revision 19: G-6: the chi sweep is extended downward, and how dtau is swept
Registered before `experiments/wp6_mps/g_6_tensor_network.py` was run. Section 10's four
criteria are unchanged.

**The chi sweep is extended downward to {1, 2, 4, 8}.** Section 10 registers
`[16, 32, 64, 128, 256, 512, 1024]`. Exploratory measurement, disclosed below, finds that the
smallest registered value already suffices on every family tested, so the registered sweep
would report "16" in every cell and the map criterion 2 calls a primary deliverable would be
a constant. The extension only adds resolution below the registered floor; every registered
value is still run, and criterion 1 is unaffected because a family that reaches cosine 0.999
at chi = 2 also reaches it at chi = 16.

**dtau is swept as a convergence study, not crossed with everything.** Section 10 registers
`[0.1, 0.05, 0.02]`. Crossing three time steps with every family, size and chi triples the
run for a quantity that does not interact with them: the Trotter error is a property of the
step size and the operator, not of the truncation. The chi map is therefore built at
`dtau = 0.05`, and all three steps are run on a fixed subset (single peak and NK K=2, at
L = 8 and L = 12) to characterise and report the Trotter floor separately. That floor is real
and is around 2 to 3 parts in ten thousand in total variation at `dtau = 0.05`, which is why
criterion 1's threshold is on cosine and not on total variation.

**Reference.** Sparse `eigsh` through `analytic.exact_diag.perron_vector`, which is section
11.2's rule and is exact at every size this check runs.

**Disclosure of what was already seen.** The implementation was validated before this
revision and the numbers are the reason for the downward extension. Bond dimension needed to
reach cosine 0.999 at `mu = 0.2`, one seed:

| family | L = 8 | L = 10 | L = 12 | operator chi at L = 12 |
|---|---|---|---|---|
| additive | 2 | 2 | 2 | 2 |
| single peak | 2 | 2 | 2 | 1 |
| NK K = 2 | 4 | 4 | 4 | 6 |
| spin glass | 8 | 8 | 16 | 8 |
| Rough Mount Fuji, roughness 0.5 | 2 | 2 | 4 | 64, saturated |
| house of cards | 2 | 4 | - | 64, saturated |

And across the error threshold, `mu / mu_c` from 0.4 to 1.6: the single peak needs chi = 2 at
every point at both L = 10 and L = 12, and NK K = 2 needs chi = 4 at every point at both.

**Two things this says that the project should not lose.** The **operator's** bond dimension
is a poor predictor of the **state's**: Rough Mount Fuji and house-of-cards saturate the
operator ceiling at 64 and their states need 2 to 4. That is the caveat attached to
`results/wp6/g_6_3.json`, now measured, and it cuts against the earlier reading rather than
for it. And the requirement is **flat across the error threshold**, where it was expected to
peak; both deep phases are low rank and so is the crossover between them.

Together these raise the prior on the G-7 null further than revision 18 recorded, and that
revision's reasoning stands: the sweep is not being pre-empted, and a positive result would
now need extra scrutiny rather than less.

**note on revision 19, registered at the same time.** Section 10 does not fix a seed
count, so one is chosen here and disclosed: **two seeds** per seeded family, at every size and
every mutation rate. One would risk a rugged family being represented by an atypical instance;
ten would put the check past four hours in the image for a quantity that exploratory runs show
is stable across seeds. The reference uses the sparse `eigsh` route from `L = 11` upward
rather than the default dense cutoff at 12, because a dense 4096 by 4096 solve in the
single-threaded image costs 37 s against 0.2 s for the sparse path, which would make the
reference more expensive than the evolution it exists to check. The two agree to machine
precision; that is what G-R.1 established and what `tests/unit/test_numerics.py` keeps true.

**Second note on revision 19.** The grid uses **one seed at
L = 14** and two below it, and the reason is a measurement that qualifies the finding above.

The state's bond dimension is small on every family, but the **wall-clock cost of a step is
set by the bond dimension of `exp(dtau f)`**, which appears in the Hadamard product before
rounding pulls it back. Measured at L = 12, chi = 4, one evolution to convergence:

| family | operator chi | seconds |
|---|---|---|
| additive | 1 | 2.6 |
| single peak | 2 | 1.8 |
| block, size 2 | 2 | 2.9 |
| NK K = 2 | 16 | 16.6 |
| Rough Mount Fuji, roughness 0.5 | 64 | 15.9 |
| spin glass | 60 | 26.9 |
| house of cards | 64 | 30.6 |

So the earlier reading, that a tensor network handles every family cheaply, is **true of
memory and not of time**. Rough Mount Fuji and house-of-cards need a state of bond dimension
two to four and still cost ten to twenty times more per run than the additive family, because
each step contracts against a saturated operator. That distinction matters directly for WP7,
whose budget protocol in section 11.3 is denominated in wall-clock seconds: Baseline C will
spend its budget very unevenly across the ruggedness axis, and a cell where it looks weak may
be a cell where it ran out of time rather than out of bond dimension.

The result records the per-cell wall-clock alongside the bond dimension so the two costs are
separable in the artefact rather than conflated in a single "MPS is cheap" claim.

---

### Revision 20: WP7: the registered grid costs 294 hours, and what is run instead
Registered before the sweep runs. Section 11.1's grid and section 11.5's decision criteria
are unchanged in kind; what changes is how much of the grid is covered, and the reason is a
cost measurement that section 11.1 was written without.

**The registered grid is not runnable.** Section 11.1 declares L in {8, 10, 12, 14}, a
21-point mutation axis, and at least 5 seeds across nine landscape families. That is **3108
cells**. Measured per-cell cost on the declared hardware, all three baselines, after the
Baseline A correction below:

| L | seconds per cell |
|---|---|
| 8 | about 25 |
| 10 | about 100 |
| 12 | about 250 |
| 14 | budget-capped, about 900 |

which totals **about 294 hours**, twelve days of exclusive use of a shared machine. The grid
was specified without a cost model and cannot be executed as written. That is a planning
finding and it is recorded here rather than worked around silently.

**What runs instead.** L in {8, 10, 12}, a **7-point** mutation axis spanning the same
`[0.4 mu_c, 1.6 mu_c]`, seeds 0 to 4, all nine families. **777 cells, about 27 hours.** The
seed count meets section 11.1's stated minimum of five. What is given up is mutation-axis
resolution, from 21 points to 7, and the L = 14 row.

**Why those two and not others.** The seed count is what the confidence intervals in section
11.4 rest on, so it is not the thing to cut. The mutation axis is the one place where
measurement says resolution buys least: WP6 found the bond dimension required by Baseline C
is **flat** across `mu / mu_c` from 0.4 to 1.6, and G-1 found the spectral gap varies smoothly
over the same range, so a 7-point axis resolves everything either of them shows. L = 14 is
deferred rather than dropped, and section 11.2's rule already excludes cells without a
trustworthy reference, so a partial size axis is a case the manifest can already express.

**A correction to Baseline A, with the measurement that forced it.** Section 11.3 says
Wright-Fisher spends its budget on samples. At L = 10 on an NK K = 2 cell:

| | seconds | cosine | total variation |
|---|---|---|---|
| ladder over N = 1e3 to 1e6 | 59.1 | 0.999830 | 9.35e-3 |
| N = 1e6 once, 3000 generations | 23.0 | 0.999830 | 9.35e-3 |
| N = 1e6 once, 12000 generations | 105.8 | 0.999735 | 1.06e-2 |

The ladder costs 2.6 times more for a bit-identical answer, because a generation in
genotype-count space is `O(L 2^L)` and independent of N. And **more generations makes it
worse**: drift is injected once per generation, so a longer chain accumulates noise faster
than time-averaging removes it.

**Baseline A therefore cannot spend a larger budget at all.** Its accuracy is set by a drift
floor. Section 11.3's protocol assumes methods improve with budget, and one of the three does
not, so a WP7 cell where Baseline A looks weak is a cell where it sits at its floor rather
than one where it was starved. Every Baseline A record carries
`budget_is_not_the_binding_constraint` so this cannot be misread from the artefact alone.

---

### Revision 21: WP7: the sweep had no quantum route, and Route A cannot afford the grid
Registered before the quantum pass runs, and after the classical pass had already started.

**The first fault, and it is mine.** The sweep runner's `METHODS` held only
`baseline_a_wright_fisher`, `baseline_b_exact_class` and `baseline_c_tensor_network`. Section
11.5's decision criterion requires that "a quantum route achieves cosine >= 0.90" while "the
compute-matched tensor-network baseline achieves cosine < 0.80". **A sweep holding only the
baselines cannot answer G-7 either way.** It would have completed 777 clean cells and left the
axis the decision turns on empty. Same shape as the `ORDER` defect: a run that succeeds while
doing less than it claims.

Two quantum routes are added. `route_b_qsvt_filter` applies the eigenstate filter to the
operator, which is what the block encoding G-2 verified to 1.7e-12 implements; simulating the
circuit in every cell is not affordable and would measure Qiskit rather than the method.
`route_a_varqite` runs the variational route directly.

**The second fault is not a fault but a result. Route A cannot finish a cell within the
declared budget anywhere on this grid.** Measured at L = 6: 198 s and 235 s of a 300 s
allotment, at cosine 0.999993 and 0.999995. Its cost scales as `n_parameters^2 * 2^L` with
`n_parameters = L(L + 3)`, so L = 8, the smallest size in the grid, is about ten times that,
roughly 2100 s against 300 allotted. Route B reaches the same accuracy in **0.3 s**.

Running Route A over all 777 cells would therefore spend about 65 hours establishing that it
runs out of time. It runs instead on a **declared probe**: L = 8, seed 0, `mu / mu_c` in
{0.4, 1.0, 1.6}, every family, 27 cells. Every other cell records Route A as inapplicable
with this revision as the reason, which is a statement about the budget and not about the
method's accuracy.

**What this means for G-7, stated before the data exists.** The quantum side of the boundary
map is Route B. Route A's exclusion is itself a finding and belongs in the result: the
variational route, which is the NISQ-runnable one, cannot reach any cell of the declared grid
inside the declared budget. That is consistent with G-R.9's barren-plateau measurement and
with the concern docs/notes.md raised about budget protocols disadvantaging imaginary time, and it
should be reported as a limitation of Route A rather than buried as a missing column.

**A hardening that came out of finding this.** `run_cell` catches a method's exception so one
failure cannot lose a cell, which also means a method broken in *every* cell produces a sweep
that looks complete and is full of nulls. Route A did exactly that in its first smoke test, an
`AttributeError` on every call. The manifest now counts errors per method and the runner
prints them, so a column of nulls announces itself.

**note on revision 21, correcting an extrapolation and a gap in the budget accounting.**

The revision above estimated Route A at about 2100 s per cell at L = 8, extrapolated from
`n_parameters^2 * 2^L`. **Measured, it is 510 s**, so the extrapolation was four times too
pessimistic. The probe design is unchanged and the conclusion is unchanged, because 510 s
still exceeds the 300 s allotment section 11.3 declares, but the number quoted above is an
estimate and this one is a measurement.

The measurement also exposed a gap in the budget accounting that affects every method. Each
one checks its remaining budget **between** units of work, so a single unit that overruns is
never caught: Route A converged on its first imaginary-time rung after 510 s and reported
`budget_exhausted = False`, which is true as written and misleading as read. Section 11.3
calls the budget a fairness firewall, and a cell in which one method was allowed 1.7 times
its allotment is not a compute-matched comparison.

Every method record now carries `over_budget`, set from measured seconds against allotted
seconds independently of what the method thinks it did, and the manifest counts them. A cell
where any method is over budget can be excluded or reported as such when G-7 is scored,
rather than being silently treated as matched.

---

### Revision 22: G-4: a replacement for the criterion that could not be executed
Registered before the replacement criterion runs, and after criterion 1 has already been run
and recorded. revision 15 withdrew criterion 2 as unexecutable and said in advance that
G-4 would not be claimed as passed until a replacement was registered. This is that
replacement. The withdrawal is not revisited here and the original record stands.

**What stays withdrawn.** "Throughput within 5x of the reference community implementation"
cannot be evaluated, for the two independent reasons in docs/notes.md: the two implementations sit
in different complexity classes by construction, so the test would pass by three orders of
magnitude without establishing anything, and no reference implementation exists in the pinned
image to compare against. Neither reason is an outcome anyone measured and disliked. Both
were visible from the implementation and the environment before the check ran.

**Criterion 2b, time to accuracy.** The baseline must reach total variation `<= 0.02` against
the analytic single-peak quasispecies at `L = 8`, `mu = 0.10`, within **300 s** of wall clock
in the pinned image, on the `(N, generations)` ladder already declared for criterion 1. The
check reports the cheapest configuration that reaches the target and the wall clock it took.

Three things make this a criterion rather than a formality.

**The threshold is not ours to choose.** 300 s is the WP7 per-cell per-method allotment fixed
in section 11.3 at `L <= 12`, set before any of this was measured and for a different reason.
A baseline that cannot reach the accuracy WP7 needs inside the budget WP7 grants is not a
fair reference for WP7, which is the entire purpose criterion 2 was serving. Picking a number
from the measurement already held would have been circular, so the number comes from
somewhere the measurement cannot reach.

**It can fail.** Nothing about the count-space representation guarantees the target is
reachable at all: the accuracy floor is set by drift at finite `N`, and if the smallest
sufficient `N` had turned out to cost more than the budget, this would fail. That is the
property the withdrawn criterion lacked.

**It is representation independent.** Time to a fixed accuracy is comparable across a
count-space and an individual-based implementation without either one being penalised for its
data structure, so a community reference dropped into the image later can be measured against
the same target with nothing rerun.

**What remains blocked, and is recorded as blocked.** The cross-implementation comparison
itself. Criterion 2b establishes that the baseline is fast enough for the use WP7 puts it to.
It does not establish that it is as fast as the best available forward simulator, and no
claim of that kind is made anywhere from this check. That comparison waits on a reference in
the image, which is a disk-budget decision under the rule in docs/notes.md on a machine with 42 GB free.

**Correction to docs/notes.md.** That decision record quotes total variation `0.0051` at
`N = 10^6` from an exploratory run. The check's own record measures **`0.004664`** on the
registered configuration. The conclusion is unchanged and the recorded value is the one to
cite.

---

### Revision 23: G-6: a per-cell deadline at L = 14, and what a stopped cell is allowed to mean
Registered before any `L = 14` cell has been measured, and after `L = 8`, `10` and `12` have.
The measurement that prompts it is below, and it is a cost measurement rather than a result.

**What the grid actually costs.** Measured in the pinned image, mean wall clock per cell:

| L | cells measured | mean s | worst s |
|---|---|---|---|
| 8 | 79 | 6.8 | 60 |
| 10 | 80 | 22.5 | 180 |
| 12 | 68 | 235.6 | 5760 |

The step from `L = 10` to `L = 12` costs 10.5 times, against 3.3 for the step before it, so
the growth is accelerating rather than geometric at a fixed ratio. Cost tracks the chi a cell
has to climb to: cells settling at chi = 4 average 15.6 s and cells settling at chi = 64
average 2790 s. The single worst cell, house of cards at `mu = mu_c`, took **1.6 hours** and
returned chi = 64, the ceiling at that size.

**Why that is a problem at `L = 14`.** The ceiling there is 128 rather than 64 and the
dimension is 16384 rather than 4096, and 45 cells are registered. Extrapolating the measured
step ratio gives roughly 39 minutes a cell, and the rugged cells that climb to 128 would sit
far above that. Thirty to a hundred hours is the plausible range for one check.

**What is not being done.** The grid is not cut. No family, size, mutation rate or seed is
removed, because a reduction chosen after seeing which cells are expensive would quietly
remove exactly the cells the map is most interesting about.

**Criterion 2b registered: a per-cell wall-clock allotment of 900 s at `L >= 14`.** The ladder
stops before starting a rung it cannot afford and the cell reports the largest chi it did try
and the best cosine it saw, marked `budget_limited`. The check sits between rungs, never
inside an evolution, so a cell can overshoot by at most one rung and no state is ever reported
half-evolved.

Four things about this limit.

**The number is borrowed, not chosen.** 900 s is what section 11.3 allots a method per cell at
`L >= 14`. It was fixed for the WP7 sweep, before any of this was measured and for an
unrelated purpose, so it cannot have been tuned to produce a convenient outcome here. This is
the same reasoning revision 22 used for G-4.

**It applies only at `L >= 14`.** Sizes up to 12 are affordable at 5.1 hours for the whole
grid, and capping a measurement that runs fine would discard real results for nothing. A 300 s
cap at `L = 12` would have thrown away the chi = 64 the worst cell returned.

**It is non-destructive when it does not bind.** If `L = 14` turns out cheaper than the
extrapolation, no cell reaches the limit and the artefact is identical to the one the
unamended grid would have produced. That is why it can be registered before the size it
governs has been measured: it bounds cost without changing what is measured.

**A stopped cell is not a failure, and criterion 1 must not treat it as one.** The whole risk
in a limit like this is that it manufactures a result: in a record, a cell the clock stopped
looks exactly like a cell where the tensor network genuinely cannot hold the state, and those
are opposite findings. Criterion 1 therefore counts only cells that were given the full ladder
and still missed the threshold. Stopped cells are listed separately, with the chi reached, so
they can be finished later without redoing anything else. Criterion 2's map is complete when
every cell either carries a chi or carries a stated reason it does not.

This follows docs/notes.md, which recommended exactly this shape after the WP7 tensor-network
baseline overran its allotment on 37 per cent of `L = 12` cells because it stops on
convergence and never looks at the clock. The recommendation there was a deadline-aware method
that hands back what it holds. This applies it.

**Second change, not a criterion: the check now checkpoints.** It previously built its record
only at the end, so an interruption at nine tenths produced nothing. That turned one LAPACK
failure into fourteen hours lost and made this very decision cost a whole run rather than its
remainder. Cells are appended to a gitignored scratch file as they land and reused on restart,
guarded by a fingerprint of every registered constant so a checkpoint from a different grid is
refused rather than blended into this one.

**note on revision 23, written after the check ran.** The limit works and the wording
above overstates it. "Overshoot by at most one rung" is true and hides that a rung is not
itself bounded: the clock is checked before a rung starts, never inside one, so the real
ceiling is 900 s plus however long the rung that crosses the line takes. Measured: an
`L = 14` spin-glass cell ran **3759 s** against the 900 s limit, four times over, because it
was under the line when it began chi = 16 and that single rung took roughly 2900 s. A tighter
bound needs the deadline pushed down into `evolve`, which would mean interrupting an
imaginary-time evolution midway and reporting a state that is neither converged nor a fair
reading of that chi. That trade was not taken and should be reconsidered before any run at
`L >= 16`.

**What the limit actually cost.** Eight of 285 cells, all at `L = 14`, three of them within
0.002 of the threshold at the chi they reached. The family it blocked entirely is house of
cards, all five mutation rates. Everything else at `L = 14` resolved at chi <= 16, and block
resolved at chi = 2 throughout.

---

### Revision 24: a bounded probe at L = 14 and 16, registered to expect nothing
Registered before the probe runs, after G-7 has been answered as a null. Requested in the
review of 13 August, which asked for one bounded attempt at the corner G-7 named, with a
stopping rule and a budget fixed in advance, and with the expected outcome written down first.

**What the probe asks.** G-7's null is bounded at `L = 12`. Two measurements point at the same
corner beyond it. Section 4.28: the tensor network is exact at `L = 8` and `L = 10` and falls
to cosine 0.8758 at `L = 12`, worst on house of cards near `mu_c`. Section 4.31: G-6 could not
reach `chi = 32` on that same family at `L = 14` inside 900 s. If the classical reference
strains there while a quantum route holds, that is a different result from the one now
recorded.

**The grid.** House of cards at `L = 14` and `L = 16`, `mu / mu_c` in {0.9, 1.0, 1.1}, three
seeds. NK at `K = 4` runs alongside on the same points as a control: without it, a strain seen
only in house of cards cannot be told apart from a strain that arrives at every family once `L`
is large enough. Methods are the compute-matched tensor network and Route B. Route A is
excluded, on the measurement in revision 21 and section 4.32 that it exceeds its allotment on
27 of 27 cells already.

**The stopping rule, fixed here.** The probe stops at whichever comes first:

1. every cell resolved;
2. **fourteen days** of wall clock consumed, counted from first launch;
3. the exact reference itself becoming the bottleneck, which is declared as any cell where
   computing the Perron vector exceeds the time the two methods under test are allotted
   together. A comparison whose referee is slower than its players measures the referee.

Whatever has completed when the rule fires is what gets reported. Partial coverage is stated
per cell, as G-7 states its exclusions, and cells never reached are listed as never reached.

**The decision, using G-7's conditions unchanged.** A positive needs a compute-matched tensor
network below cosine 0.80 while a quantum route reaches 0.90, with separated bootstrap
intervals across at least three of the seeds run. The seed floor is three rather than five
because the grid is three seeds; that weakening is stated here rather than discovered later,
and a positive found on three seeds would be reported as a lead requiring confirmation, not as
a result.

**The expected outcome, stated in advance: nothing.** Section 4.15 measured the thing that
matters here and it points the other way. Rough Mount Fuji and house of cards saturate the
*operator* ceiling while their *states* need `chi = 2` to `4`. Operator cost and state cost are
different quantities, and it is state cost that decides whether a tensor network can hold the
quasispecies. The `L = 14` house-of-cards cells in G-6 were stopped by a clock, not by a wall:
their best cosines run 0.013 to 0.984 below the critical `chi`, which says the search was cut
short rather than that the answer was far away. Nothing measured so far shows the mechanism a
positive would need.

This expectation is registered so that a null here is an outcome and not a disappointment, and
so that a positive, if it comes, arrives against a prediction on the record rather than into a
space left conveniently empty.

**What the probe cannot do.** It cannot extend the boundary map. The grid is one family plus a
control at two sizes, not a map, and its purpose is to close a specific question with evidence
rather than leave a reviewer to ask it. A null closes the door at `L = 16` for house of cards
near `mu_c`. It says nothing about other families, other mutation rates, or larger sizes, and
the report must not imply otherwise.


---

### Revision 25: G-8: what runs on hardware
**Device: `ibm_marrakesh`.** Heron r2, 156 qubits. Selected because it executes: it began running
one second after submission, where the alternative sat queued for hours without starting. It is
not the lowest-error device available, and the artefact records its calibration timestamp, error
rates and layout so a reader can weigh that.

**What is submitted.** The mutation-rate sweep across the error threshold at `L = 2, 3, 4` on the
single-peak landscape, `mu / mu_c` from 0.4 to 1.6: 23 sweep circuits plus 28 readout calibration
circuits, 51 in all, at 4096 shots each.

**Transpiled against the device at optimisation level 3**, seeded so the free `--mode isa`
inspection describes the circuits that are actually submitted:

| L | depth | two-qubit checks |
|---|---|---|
| 2 | 13 | 2 |
| 3 | 45 | 10 |
| 4 | 58 | 18 |

This is roughly five times the depth the run plan assumed at `L = 4` and twice its two-qubit
count, so the expected error is correspondingly larger.

**Readout mitigation is the full `2^n` assignment matrix**, 4 + 8 + 16 = 28 circuits rather than
the 18 a tensored scheme needs, with calibration circuits pinned to the physical qubits the data
circuits use. G-R.8 validated its mitigated cosines with this estimator, so using a cheaper one
here would answer a different question. A layout chosen independently would produce an assignment
matrix for the wrong qubits and mitigate confidently in the wrong direction.

**A shot-noise floor is recorded beside every measurement.** The square-root decode magnifies
sampling error on rare outcomes, so total variation overstates device error unless the floor is
shown next to it: at `L = 4` a noiseless device at the same shot count already scores 0.043.
Cosine is the primary metric for this reason.

**Route B is absent, and that is a result rather than a gap.** 1024 walk-operator queries on 5 to
9 ancillas is a deep coherent circuit and squarely fault-tolerant territory. Section 4.35's
two-currency comparison predicts exactly this.

**The prediction, and the one thing worth keeping from how this was run.** G-R.8's
`superconducting_heron_like` `single_peak` cases predict mitigated cosine 0.99941, 0.99808 and
0.99467 at `mu / mu_c` of 0.40, 0.60 and 0.80 for `L = 2, 3, 4`. **That prediction was written
down before the device ran, and was not adjusted afterwards.** Measured: 0.99930, 0.99595 and
0.99461, all within 2.2e-3. The agreement is worth something only because the order is on the
record, so the timing is stated here and nowhere else in this file needs to argue about it.

**Provenance.** The pinned image has no `qiskit-ibm-runtime`, so WP8 runs outside the pinned image and the
record is written to `results/_local/` as non-evidence under the rule in docs/notes.md. docs/notes.md records why the
alternatives were declined. No claim may cite G-8 as reproduced evidence without that qualifier.

**G-8 is a feasibility check.** No accuracy threshold is set and no advantage is claimed.
