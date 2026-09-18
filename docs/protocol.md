# Settings

Parameters, thresholds and time allocations used by the checks in `experiments/` and by the
comparison sweep. The SHA-256 of this file is written into every result record.

## Numerical conventions

| Convention | Rule |
|---|---|
| Fitness encoding | Spin convention, `a_i Z_i` and `b_ij Z_i Z_j`. |
| Biological normalisation | Quasispecies distributions are L1-normalised and non-negative. |
| Quantum-state normalisation | L2 internally, converted to L1 at readout. |
| Target operator | Dominant eigenvector of the generator `W = diag(f) + mu sum_i (X_i - I)`, which is sign-definite, so L1 and L2 normalisation select the same ray. |
| Qubit ordering | Qiskit little-endian. Bitstring and integer conversions go through `quasarstack/io/conventions.py`. |
| Eigensolvers | Dense at L <= 10 in the sweep, `scipy.sparse.linalg.eigsh` above. No dense operator above L = 12. |
| Random numbers | `numpy.random.default_rng(seed)` with the seeds listed below. float64 throughout. |

## Validation against closed-form solutions (`experiments/wp_r/`)

| Check | Quantity | Threshold | Configurations |
|---|---|---|---|
| G-R.1 | Oracle against exact diagonalisation, max absolute error | < 1e-9 | L = 2 to 10, seven mutation rates, single-peak, uniform and seeded additive landscapes |
| G-R.2 | Compiled Hamiltonian ground state against the analytic quasispecies, cosine | >= 0.999999 | 40 configurations |
| G-R.3 | Trotterised propagator, cosine at dtau = 0.01, and fitted error exponent | >= 0.999; exponent in [1.8, 2.2], R^2 >= 0.99 | dtau from 0.25 to 0.0078 |
| G-R.4 | Error threshold on the qubit operator against the class reduction, max absolute surplus difference | < 1e-3 | L = 4, 6, 8; mu from 0.01 to 3.00 |
| G-R.5 | Rugged NK landscapes against exact diagonalisation, cosine | >= 0.99999 | ten (L, K) cells, seeds 0 to 9 |
| G-R.6 | Variational imaginary-time evolution, cosine; circuit depth at tau = 2.5 and 20 | >= 0.999; identical | L = 3 to 6 |
| G-R.7 | Motta imaginary-time evolution, cosine; energy rise per step | >= 0.95; <= 1e-10 | L = 3 to 6, generator weight 2 |
| G-R.8 | Simulated device noise with readout mitigation, mitigated cosine | >= 0.98 | L = 2 to 4, two device models, 40,000 shots |
| G-R.9 | Gradient variance against system size, fitted decay base | in [0.30, 0.55], R^2 >= 0.95 | L = 2 to 8, 400 samples |
| G-R.10 | Pauli terms, single-peak projector against sparse additive-plus-epistasis form, ratio at L = 12 | >= 50 | L = 4 to 12 |

## Spectral analysis (G-1)

1. Every closed form is reproduced to relative error < 1e-6.
2. The minimum of the spectral gap lies within 5% of the analytic mu_c under both finite-size
   readings, at L = 6, 8 and 10.
3. Every operator-structure statement in `docs/theory.md` resolves to a test or record.

Grid: 41 points over [0.2 mu_c, 2.0 mu_c] per instance, refined to 1,500 points; NK with
K in {0, 1, 2, 4}; dense L up to 12, class-reduced L up to 64; seeds 0 to 9. Gaps below 1e-9
are recomputed at 60 decimal digits.

## Filtering route (G-2)

1. Cosine >= 0.95 against the analytic quasispecies, L = 2 to 6, noiseless.
2. The block encoding satisfies its defining property to 1e-10.
3. The derived polynomial degree agrees with the empirically sufficient degree within a
   factor of 2.

## Landscape families (G-3)

1. Every landscape reproduces exactly from its seed.
2. NK at K = 0 equals the additive family to 1e-12.
3. Strict local optima rise and correlation length falls with K in the seed mean, at L = 10
   and 12.

Families: single peak, additive with pairwise epistasis, NK (K in {0, 1, 2, 3, 4, 6}), spin
glass (+/-1 couplings), Rough Mount Fuji (roughness in {0, 0.1, 0.3, 1, 3}), House of Cards,
block (sizes 1, 2, 4). Seeds 0 to 9.

## Wright-Fisher baseline (G-4)

1. Total variation < 0.02 against the analytic single-peak quasispecies at N = 1e6, L = 8.
2. Total variation <= 0.02 reached within 300 s at L = 8.

Populations 1e3 to 1e6, 4,000 generations, 20% burn-in, time step 0.01, seeds 0 to 9.

## Exact class solver (G-5)

1. Max absolute error <= 1e-6 against the analytic oracle on every in-class landscape, and a
   refusal on every out-of-class one.
2. Applicability is decided by an explicit predicate on the fitness vector.

L = 4 to 10, mu in {0.05, 0.10, 0.20}, seeds 0 to 9.

## Matrix-product baseline (G-6)

Two-site DMRG (quimb) on the exact matrix-product operator of the generator: relative cutoff
1e-13 on the fitness TT-SVD, discarded weight <= 1e-12 per update, random initial state of
bond dimension min(16, chi) with seed 0, Lanczos tolerance 1e-10 with 10 Krylov vectors,
eigenvalue tolerance 1e-12 relative over two consecutive sweeps, at most 100 sweeps.
Landscapes invariant under complementing every locus add `c prod_i X_i` with
`c = max(range(f), 1)`.

1. Cosine >= 0.999 against exact diagonalisation at sufficient chi, L = 8, 10, 12, 14, every
   family.
2. The bond dimension needed for cosine 0.999 is mapped across family, K, mu and L.
3. The bond dimension of the operator is reported per family, with two locus orderings.
4. The eigenvalue and bond dimension are recorded after every sweep.

Chi ladder 1, 2, 4, ..., 128, each rung stopped at relative tolerance 1e-8 or 30 sweeps;
mu / mu_c in {0.4, 0.7, 1.0, 1.3, 1.6}; seeds 0 and 1, seed 0 at L = 14; 900 s per
configuration at L = 14. Cross-check against matrix-product imaginary-time evolution
(dtau = 0.05) at L = 8 and 10.

## Comparison sweep (G-7)

| Setting | Value |
|---|---|
| Families | single peak, additive with pairwise epistasis, NK (K = 1, 2, 4), Rough Mount Fuji (roughness 0.5), spin glass, block (size 2), House of Cards |
| Sizes | L = 8, 10, 12 |
| Mutation rates | mu / mu_c in {0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6} |
| Seeds | 0 to 4 for seeded families |
| Instances | 777 per arm |
| Methods | Wright-Fisher (N = 1e6, 3,000 generations), exact class solver, DMRG (bond dimension up to 2^(L/2)), variational route (L = 8, seed 0, mu / mu_c in {0.4, 1.0, 1.6}), filtering route (degrees up to 1,024) |
| Time allocation | 300 s per instance per method at L <= 12, 900 s at L >= 14, wall clock, one declared worker count |
| Exclusion | an instance is excluded where any applicable method exceeds its allocation |
| Scoring | cosine and total variation against exact diagonalisation; bootstrap 95% intervals over seeds, 10,000 resamples |

A quantum advantage is a group of instances, reproducible across at least 5 seeds, in which
a quantum route reaches cosine >= 0.90, the matrix-product baseline is below 0.80, the exact
class solver does not apply, and the bootstrap intervals of the two methods do not overlap.

## Hardware run (G-8)

`ibm_marrakesh` (IBM Heron r2). Single-peak landscape at L = 2, 3, 4 across the error
threshold, 23 sweep points and 28 readout calibration circuits, 4,096 shots each,
optimisation level 3, transpiler seed 20260813. Full assignment-matrix readout mitigation.
