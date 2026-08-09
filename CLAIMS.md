# CLAIMS.md — the claims ledger

Every claim the manuscript will make maps to a re-runnable artefact. `make claims` runs
`scripts/check_claims.py`, which verifies that each row below resolves to an artefact that
exists and to a script that produces it. **A claim without a resolvable artefact does not go
in the paper.**

Status values: `planned` (registered, not yet run), `pass` (artefact exists and the gate
passed), `fail` (run performed, gate not met, reported as such), `dropped` (claim withdrawn,
with a reason).

---

## Rebuild of Phases 1–3 (WP-R)

| # | Claim as it will appear | Gate | Evidence artefact | Script | Status |
|---|---|---|---|---|---|
| C1 | The analytic Crow–Kimura oracle agrees with brute-force exact diagonalisation to machine precision, with maximum absolute error 2.4×10⁻¹⁵ over 1701 comparisons | G-R.1 | `results/wp_r/g_r_1.json` | `experiments/wp_r_rebuild/g_r_1_oracle_vs_ed.py` | pass |
| C2 | The compiled qubit Hamiltonian's ground state is the analytic quasispecies, at cosine 1.000000 on 40 of 40 registered configurations, with the operator matching the independently assembled generator to 3.6×10⁻¹⁵ | G-R.2 | `results/wp_r/g_r_2.json` | `experiments/wp_r_rebuild/g_r_2_hamiltonian_vs_oracle.py` | pass |
| C3 | The Trotterised imaginary-time propagator converges to the oracle at cosine 1.000000, and its splitting error is second order with a fitted exponent of 1.995 to 2.000 at R² = 1.00000 | G-R.3 | `results/wp_r/g_r_3.json` | `experiments/wp_r_rebuild/g_r_3_trotter_scaling.py` | pass |
| C4 | The error threshold appears on the qubit representation as a localisation-delocalisation transition matching the analytic prediction to 1.9×10⁻¹⁵, with the sharp-peak threshold converging to μ_c·L → 1 and the transition width collapsing from 0.300 to 0.010 over L = 4 to 20 | G-R.4 | `results/wp_r/g_r_4.json` | `experiments/wp_r_rebuild/g_r_4_error_threshold.py` | pass |
| C4b | The spectral gap at the sharp-peak error threshold closes exponentially in system size, with a fitted decay of 0.717 per site over L = 4 to 12 | G-R.4 | `results/wp_r/g_r_4.json` | `experiments/wp_r_rebuild/g_r_4_error_threshold.py` | pass |
| C5 | Rugged epistatic landscapes are reproduced against brute-force exact diagonalisation, at cosine 1.000000 across 100 NK instances spanning L = 6 to 10 and K = 1 to 7, including the maximally rugged case where the Pauli decomposition is dense | G-R.5 | `results/wp_r/g_r_5.json` | `experiments/wp_r_rebuild/g_r_5_rugged.py` | pass |
| C5b | The NK family varies ruggedness monotonically in K: mean local optima rise from 2.8 to 29.1 and fitness autocorrelation falls from 0.659 to −0.021 at L = 8 | G-R.5 | `results/wp_r/g_r_5.json` | `experiments/wp_r_rebuild/g_r_5_rugged.py` | pass |
| C5c | Imaginary-time evolution at a fixed budget fails to converge on the most rugged instances, and the instances it fails on are those with the smallest spectral gaps | G-R.5 | `results/wp_r/g_r_5.json` | `experiments/wp_r_rebuild/g_r_5_rugged.py` | pass |
| C6 | varQITE reproduces the quasispecies at cosine ≥ 0.99997 across 14 configurations, with circuit depth identical at τ = 2.5 and τ = 20 before and after transpilation | G-R.6 | `results/wp_r/g_r_6.json` | `experiments/wp_r_rebuild/g_r_6_varqite.py` | pass |
| C6b | The McLachlan quantities varQITE needs are obtainable from circuit measurements alone, agreeing with the state-vector computation to 7×10⁻¹⁶ by the parameter-shift and fidelity-shift rules | G-R.6 | `results/wp_r/g_r_6.json` | `experiments/wp_r_rebuild/g_r_6_varqite.py` | pass |
| C6c | The ansatz depth varQITE needs grows faster than the system: at L = 6 a depth of reps = 4 fails to reach 0.999 on every seed while reps = 6 succeeds | G-R.6 | `results/wp_r/g_r_6.json` | `experiments/wp_r_rebuild/g_r_6_varqite.py` | pass |
| C7 | Motta-QITE reproduces the quasispecies at cosine ≥ 0.9989 with the energy descending on every step, across 14 configurations | G-R.7 | `results/wp_r/g_r_7.json` | `experiments/wp_r_rebuild/g_r_7_motta.py` | pass |
| C7b | The Motta generator basis must consist of odd-Y Pauli strings, because a real state needs a real antisymmetric generator; Pauli strings of even Y parity contribute exactly zero to the right-hand side, which is the mechanism of the failure the planning documents record for this method | G-R.7 | `results/wp_r/g_r_7.json` | `experiments/wp_r_rebuild/g_r_7_motta.py` | pass |
| C8 | The pipeline is feasible under realistic simulated device noise with error mitigation, L = 2 to 4 | G-R.8 | `results/wp_r/g_r_8.json` | `experiments/wp_r_rebuild/g_r_8_noise.py` | pass |
| C9 | varQITE gradient variance decays exponentially in system size, bounding the method's reach | G-R.9 | `results/wp_r/g_r_9.json` | `experiments/wp_r_rebuild/g_r_9_barren.py` | planned |
| C10 | The sparse additive-plus-epistasis representation requires 152 times fewer Pauli terms than the single-peak projector at L = 12, 27 against 4108 | G-R.10 | `results/wp_r/g_r_10.json` | `experiments/wp_r_rebuild/g_r_10_pauli_count.py` | pass |
| C32 | The circuit holds the quasispecies in its amplitudes, so a computational-basis measurement returns the distribution squared; recovering it needs an explicit square-root decode, without which the measured distribution sits at total-variation distance 0.22 from the quasispecies while scoring 0.987 on cosine | G-R.8 | `results/wp_r/g_r_8.json` | `experiments/wp_r_rebuild/g_r_8_noise.py` | pass |
| C11 | Analytic-first validation catches implementation errors that produce plausible but incorrect output | — | `docs/validation.md` plus the regression tests that lock each convention | `tests/regression/` | planned |

Note on C11. The three bugs described in the planning documents belong to an implementation
that no longer exists. They will be reported as methodological history, clearly attributed
to the earlier implementation, and each one is locked against recurrence by a regression
test in the rebuilt stack. Any bug the rebuild catches on its own is reported separately and
as new.

---

## WP0 — pre-registration and prior art

| # | Claim | Gate | Artefact | Script | Status |
|---|---|---|---|---|---|
| C30 | The mutation–selection generator is non-conservative but reversible, so the nonreversible-Markov-chain speedup of Claudon, Piquemal and Monmarché (2025) does not apply to it; nonreversibility within this problem class requires direction-specific context-dependent mutation | G-0 | `results/wp0/prior_art_iv_4.json` | `experiments/wp0_prior_art/verify_iv_4_claudon.py` | pass |
| C31 | Reversibility is a property of the mutation operator alone: detailed balance constrains off-diagonal entries and selection is diagonal, so no fitness landscape, however rugged, changes it | G-0 | `results/wp0/prior_art_iv_4.json` | `experiments/wp0_prior_art/verify_iv_4_claudon.py` | pass |

C30 and C31 were not in the original plan. They exist because verification of prior-art
entry IV.4 was pulled forward ahead of WP1, on the grounds that WP2 is the novelty core and
rests entirely on that one reference. See `DECISIONS.md` ADR-0010.

## WP1 — structural and spectral analysis

| # | Claim | Gate | Artefact | Script | Status |
|---|---|---|---|---|---|
| C12 | The mutation-selection generator is a non-conservative linear operator whose Perron eigenvector is the quasispecies, with the structural properties derived, not asserted | G-1.3 | `docs/theory.md` | — | planned |
| C13 | The spectral gap of the generator is mapped across ruggedness, mutation rate and system size, and closes at the error threshold | G-1.1, G-1.2 | `results/wp1/g_1.json` | `experiments/wp1_spectral/g_1_gap_map.py` | planned |
| C14 | The condition number of the generator degrades in a characterised way approaching the error threshold and with ruggedness | G-1.3 | `results/wp1/g_1.json` | `experiments/wp1_spectral/g_1_gap_map.py` | planned |
| C15 | Pauli-term count, qubit count and depth scaling are measured per landscape family | G-1.3 | `results/wp1/g_1.json` | `experiments/wp1_spectral/g_1_gap_map.py` | planned |

| C33 | The additive family has an exact spectral gap, 2·minᵢ√(aᵢ²+μ²), verified against dense diagonalisation to 2.9×10⁻¹⁴, and it is independent of system size with λ₂ L-fold degenerate; that family is therefore a ruler and cannot support an advantage claim | G-1.1 | `results/wp1/g_1.json` | `experiments/wp1_spectral/g_1_gap_map.py` | planned |
| C34 | The sharp-peak error threshold sits at μ·L → height with a 1/L correction, and the collapse across peak heights is exact to five digits from L = 8 to 1024; above the threshold the gap saturates at exactly 2μ, agreeing to twelve digits at L = 128 | G-1.1 | `results/wp1/g_1.json` | `experiments/wp1_spectral/g_1_gap_map.py` | planned |
| C35 | The gap minimum at the threshold is an avoided crossing too sharp for double precision: a 1500-point grid overestimates it by a factor of 19 at L = 32, and two LAPACK routines agree to 10⁻¹⁶ while both are wrong, so the gap map carries an arbitrary-precision Sturm-bisection path | G-1.1 | `results/wp1/g_1.json` | `experiments/wp1_spectral/g_1_gap_map.py` | planned |
---

## WP2 — Route B, QSVT

| # | Claim | Gate | Artefact | Script | Status |
|---|---|---|---|---|---|
| C16 | A block encoding of the shifted mutation-selection operator is constructed and satisfies its defining property | G-2.2 | `results/wp2/g_2.json` | `experiments/wp2_qsvt/verify_block_encoding.py` | planned |
| C17 | A QSVT eigenvalue transform amplifies the dominant eigenvector and reproduces the analytic quasispecies at small system size | G-2.1 | `results/wp2/eigen_transform.json` | `experiments/wp2_qsvt/eigen_transform_validation.py` | planned |
| C18 | Route B resource scaling is derived as a function of the measured spectral gap and matches the empirical requirement | G-2.3 | `results/wp2/resources.json` | `experiments/wp2_qsvt/resource_estimate.py` | planned |
| C19 | Route A and Route B are compared head to head on the same landscapes at the same accuracy target | G-2 | `results/wp2/route_comparison.parquet` | `experiments/wp2_qsvt/route_comparison.py` | planned |

| C36 | Route B's polynomial degree is linear in α/Δ and not square root, because the target eigenvalue lies inside the encoded spectrum by construction and Chebyshev acceleration needs it outside; the derived degree matches the empirically sufficient degree within a factor of 2 on every configuration | G-2.3 | `results/wp2/g_2.json` | `experiments/wp2_qsvt/g_2_route_b.py` | planned |
---

## WP3 to WP6 — landscapes and classical baselines

| # | Claim | Gate | Artefact | Script | Status |
|---|---|---|---|---|---|
| C20 | Seven landscape families are implemented, reproduce exactly from seed, and ruggedness increases monotonically with K | G-3 | `results/wp3/g_3.json` | `experiments/wp3_landscapes/g_3_families.py` | planned |
| C21 | The Wright-Fisher baseline converges to the analytic quasispecies and is competitive with the reference community implementation | G-4 | `results/wp4/wf_validation.json` | `experiments/wp4_wright_fisher/validate.py` | planned |
| C22 | The Dixit-Srivastava-Vishnoi baseline matches the analytic oracle where it applies, and its applicability boundary is mapped | G-5 | `results/wp5/dv_applicability.json` | `experiments/wp5_dixit_vishnoi/applicability.py` | planned |
| C23 | Tensor-network imaginary-time evolution converges to exact diagonalisation where both run | G-6.1 | `results/wp6/mps_vs_ed.parquet` | `experiments/wp6_mps/cross_validate.py` | planned |
| C24 | The bond dimension required to hold fixed accuracy is mapped across ruggedness, mutation rate and system size | G-6.2 | `results/wp6/chi_hardness.parquet` | `experiments/wp6_mps/chi_sweep.py` | planned |
| C25 | The MPS comparison is scoped honestly: MPO bond dimensions are reported per family and structural disadvantage on long-range families is stated, not exploited | G-6.3 | `results/wp6/mpo_bond_dims.json` | `experiments/wp6_mps/mpo_analysis.py` | planned |

| C37 | No landscape family is both meaningfully rugged and anchored to a master sequence: Rough Mount Fuji keeps its optimum at genotype 0 in 97% of instances at roughness 0.3, where it has 1.4 local optima, and 25% at roughness 1.0, where it has 121, with retention worsening as L grows; the order parameter is therefore measured from each instance's own fittest genotype | G-3 | `results/wp3/g_3.json` | `experiments/wp3_landscapes/g_3_families.py` | planned |
| C38 | The Sherrington-Kirkpatrick spin glass is the only rugged family whose Pauli count stays polynomial, at L(L−1)/2 + L + 1 terms against 2^L for Rough Mount Fuji at any non-zero roughness, so the biological ruggedness axis and the compilation-cost axis are different axes | G-3 | `results/wp3/g_3.json` | `experiments/wp3_landscapes/g_3_families.py` | planned |
---

## WP7 — the boundary map

| # | Claim | Gate | Artefact | Script | Status |
|---|---|---|---|---|---|
| C26 | The compute-budget protocol was fixed before the sweep and applied to every method, including QUASAR's classical optimisation time | G-7 | `GATES.md` section 11.3 plus per-cell budget fields | `quasarstack/classical/budget.py` | planned |
| C27 | Every grid cell is either scored or explicitly excluded, with the exclusion reason recorded | G-7 | `results/wp7/sweep_manifest.json` | `scripts/sweep_runner.py` | planned |
| C28 | The quantum-classical boundary for mutation-selection dynamics is mapped, with the decision gate answered positively or as a null with an explicit crossover bound | G-7 | `results/wp7/boundary.parquet`, `figures/F7_boundary_map.png` | `experiments/wp7_boundary_map/phase_diagram.py` | planned |

---

## WP8 — live QPU

| # | Claim | Gate | Artefact | Script | Status |
|---|---|---|---|---|---|
| C29 | Validated circuits were executed on a live quantum processor, with job identifiers and both raw and mitigated distributions reported as measured | G-8 | `results/wp8/qpu_runs.json` | `experiments/wp8_live_qpu/submit.py` | planned |

---

## Claims withdrawn after measurement

Recorded rather than deleted. A claim that the evidence did not support is part of the
record, and removing it silently is how a project ends up only reporting what worked.

| Claim, as the planning documents state it | What was measured | Status |
|---|---|---|
| "Antagonistic epistasis lowers the error threshold" | Not supported in the uniform pairwise family. Negative uniform coupling moves the fitness optimum off the master sequence to Hamming class 1, 2 and 2 at L = 4, 6 and 8, with multiplicities 4, 15 and 28, so there is no master sequence to delocalise from and the question is ill-posed rather than merely noisy. Testing it needs a family that keeps the master optimal while varying curvature, which is WP3 work. Artefact `results/wp_r/g_r_4.json`, reasoning in `DECISIONS.md` ADR-0011 | **dropped** |

The companion claim, that synergistic epistasis raises the threshold, **is** supported:
the half-surplus crossover rises from 0.388 to 0.859 to 1.325 as coupling grows at L = 8,
convergently in size, with the master sequence remaining optimal throughout.

## Claims explicitly not made

Registered here so that the absence is deliberate and visible, not an oversight.

- No claim of quantum advantage or speedup for evolutionary dynamics.
- No claim of superiority over tensor-network methods in general, only over the standard,
  well-tuned MPS implementation actually tested, at the system sizes actually tested.
- No claim of clinical, therapeutic or diagnostic utility.
- No claim of a first biologically faithful simulator, a breakthrough, or a game change.
- No claim of simulating whole genomes or chromosomes.
