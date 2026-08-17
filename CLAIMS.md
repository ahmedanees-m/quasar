# CLAIMS.md: the claims ledger

Every claim the manuscript will make maps to a re-runnable artefact. `make claims` runs
`scripts/check_claims.py`, which verifies that each row below resolves to an artefact that
exists and to a script that produces it. **A claim without a resolvable artefact does not go
in the paper.**

Status values: `planned` (specified, not yet run), `pass` (artefact exists and the gate
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
| C9 | varQITE gradient variance decays exponentially in system size, bounding the method's reach | G-R.9 | `results/wp_r/g_r_9.json` | `experiments/wp_r_rebuild/g_r_9_barren.py` | pass |
| C10 | The sparse additive-plus-epistasis representation requires 152 times fewer Pauli terms than the single-peak projector at L = 12, 27 against 4108 | G-R.10 | `results/wp_r/g_r_10.json` | `experiments/wp_r_rebuild/g_r_10_pauli_count.py` | pass |
| C32 | The circuit holds the quasispecies in its amplitudes, so a computational-basis measurement returns the distribution squared; recovering it needs an explicit square-root decode, without which the measured distribution sits at total-variation distance 0.22 from the quasispecies while scoring 0.987 on cosine | G-R.8 | `results/wp_r/g_r_8.json` | `experiments/wp_r_rebuild/g_r_8_noise.py` | pass |
| C11 | Analytic-first validation catches implementation errors that produce plausible but incorrect output | none | `docs/validation.md` plus the regression tests that lock each convention | `tests/regression/` | planned |

Note on C11. The three bugs described in the planning documents belong to an implementation
that no longer exists. They will be reported as methodological history, clearly attributed
to the earlier implementation, and each one is locked against recurrence by a regression
test in the rebuilt stack. Any bug the rebuild catches on its own is reported separately and
as new.

---

## WP0: specification and prior art

| # | Claim | Gate | Artefact | Script | Status |
|---|---|---|---|---|---|
| C30 | The mutation–selection generator is non-conservative but reversible, so the nonreversible-Markov-chain speedup of Claudon, Piquemal and Monmarché (2025) does not apply to it; nonreversibility within this problem class requires direction-specific context-dependent mutation | G-0 | `results/wp0/prior_art_iv_4.json` | `experiments/wp0_prior_art/verify_iv_4_claudon.py` | pass |
| C31 | Reversibility is a property of the mutation operator alone: detailed balance constrains off-diagonal entries and selection is diagonal, so no fitness landscape, however rugged, changes it | G-0 | `results/wp0/prior_art_iv_4.json` | `experiments/wp0_prior_art/verify_iv_4_claudon.py` | pass |

C30 and C31 were not in the original plan. They exist because verification of prior-art
entry IV.4 was pulled forward ahead of WP1, on the grounds that WP2 is the novelty core and
rests entirely on that one reference. See `DECISIONS.md` ADR-0010.

## WP1: structural and spectral analysis

| # | Claim | Gate | Artefact | Script | Status |
|---|---|---|---|---|---|
| C12 | The mutation-selection generator is a non-conservative linear operator whose Perron eigenvector is the quasispecies, with the structural properties derived, not asserted | G-1.3 | `docs/theory.md` | none | pass |
| C13 | The spectral gap of the generator is mapped across ruggedness, mutation rate and system size, and closes at the error threshold | G-1.1, G-1.2 | `results/wp1/g_1.json` | `experiments/wp1_spectral/g_1_gap_map.py` | pass |
| C14 | The condition number of the generator degrades in a characterised way approaching the error threshold and with ruggedness | G-1.3 | `results/wp1/g_1.json` | `experiments/wp1_spectral/g_1_gap_map.py` | pass |
| C15 | Pauli-term count, qubit count and depth scaling are measured per landscape family | G-1.3 | `results/wp1/g_1.json` | `experiments/wp1_spectral/g_1_gap_map.py` | pass |

| C33 | The additive family has an exact spectral gap, 2·minᵢ√(aᵢ²+μ²), verified against dense diagonalisation to 2.9×10⁻¹⁴, and it is independent of system size with λ₂ L-fold degenerate; that family is therefore a ruler and cannot support an advantage claim | G-1.1 | `results/wp1/g_1.json` | `experiments/wp1_spectral/g_1_gap_map.py` | pass |
| C34 | The sharp-peak error threshold sits at μ·L → height with a 1/L correction, and the collapse across peak heights is exact to five digits from L = 8 to 1024; above the threshold the gap saturates at exactly 2μ, agreeing to twelve digits at L = 128 | G-1.1 | `results/wp1/g_1.json` | `experiments/wp1_spectral/g_1_gap_map.py` | pass |
| C35 | The gap minimum at the threshold is an avoided crossing too sharp for double precision: a 1500-point grid overestimates it by a factor of 19 at L = 32, and two LAPACK routines agree to 10⁻¹⁶ while both are wrong, so the gap map carries an arbitrary-precision Sturm-bisection path | G-1.1 | `results/wp1/g_1.json` | `experiments/wp1_spectral/g_1_gap_map.py` | pass |
| C39 | The gap minimum locates the error threshold only asymptotically: at L = 6, 8, 10 it sits 19 to 30% away from the analytic μ_c under both readings, converging inside 1% by L = 48. G-1 criterion 2 therefore **fails as registered**, for reasons that are a property of the model at small L rather than of the implementation | G-1.2 | `results/wp1/g_1.json` | `experiments/wp1_spectral/g_1_gap_map.py` | fail |
---

## WP2: Route B, QSVT

| # | Claim | Gate | Artefact | Script | Status |
|---|---|---|---|---|---|
| C16 | A block encoding of the shifted mutation-selection operator is constructed and satisfies its defining property. Measured as G-2 criterion 2: worst max abs error 1.70e-12 against a 1e-10 tolerance, zero unitarity failures, across 23 configurations verified inside the 12-qubit budget revision 17 registered | G-2.2 | `results/wp2/g_2.json` | `experiments/wp2_qsvt/g_2_route_b.py` | pass |
| C17 | A QSVT eigenvalue transform amplifies the dominant eigenvector and reproduces the analytic quasispecies at small system size. Measured as G-2 criterion 1: worst cosine 0.95024 against a 0.95 threshold, zero configurations failing to reach it, with the Chebyshev series itself accurate to 3.82e-13 | G-2.1 | `results/wp2/g_2.json` | `experiments/wp2_qsvt/g_2_route_b.py` | pass |
| C18 | Route B resource scaling is derived as a function of the measured spectral gap and matches the empirical requirement. **Registered failure, and the failure is the result.** Measured: the predicted QSVT degree overshoots the empirical requirement by up to 3.40 times against an allowed factor of 2.0, on 4 of the configurations tested. The cost model is loose, and loose in the safe direction: it asks for more degree than is needed, so a resource estimate built on it overstates rather than understates. Criteria 1 and 2 of the same gate pass, at worst cosine 0.9502 against 0.95 and block-encoding error 1.7e-12 against 1e-10 | G-2.3 | `results/wp2/g_2.json` | `experiments/wp2_qsvt/g_2_route_b.py` | fail |
| C19 | Route A and Route B are compared head to head on the same landscapes at the same accuracy target. Measured on the 27 cells where both ran: Route B reaches min cosine 0.99973 against Route A's 0.97580, is more accurate on 19 of 27, and is **1129 times faster** at 0.43 s against 480.6 s. Route A exceeds its compute allotment on **27 of 27** cells and Route B on none. The comparison lives in the WP7 quantum sweep rather than the separate script the plan named, because the sweep already runs both routes per cell | G-2 | `results/wp7/sweep_registered_quantum.jsonl` | `scripts/sweep_runner.py` | pass |

| C36 | Route B's polynomial degree is **linear in α/Δ and not square root**, because the target eigenvalue lies inside the encoded spectrum by construction and Chebyshev acceleration needs it outside | G-2.3 | `results/wp2/g_2.json` | `experiments/wp2_qsvt/g_2_route_b.py` | pass |
| C40 | The derived degree does **not** match the empirically sufficient degree within a factor of two: 4 of 65 configurations fall outside, worst ratio 3.40, so **G-2 criterion 3 fails**. Every failure is in the additive family and the formula always overestimates, never underestimates | G-2.3 | `results/wp2/g_2.json` | `experiments/wp2_qsvt/g_2_route_b.py` | fail |
| C41 | The degree bound is loose in the direction that is safe for a resource estimate, and loosest where the spectrum is most spread: it assumes all unwanted weight sits immediately below λ₂, and the additive family, whose λ₂ is L-fold degenerate and whose spectrum spans sums of ±√(aᵢ²+μ²), places most of that weight far below λ₂ where the filter suppresses it harder | G-2.3 | `results/wp2/g_2.json` | `experiments/wp2_qsvt/g_2_route_b.py` | pass |
---

## WP3 to WP6: landscapes and classical baselines

| # | Claim | Gate | Artefact | Script | Status |
|---|---|---|---|---|---|
| C20 | Seven landscape families are implemented, reproduce exactly from seed, and ruggedness increases monotonically with K. Measured: 360 of 360 landscapes reproduce bit-identically, NK at K = 0 equals additive to 4.0e-15 against a 1e-12 tolerance, and strict local optima rise 1.0 to 98.7 while correlation length falls 5.49 to 1.13 across K at L = 12 | G-3 | `results/wp3/g_3.json` | `experiments/wp3_landscapes/g_3_families.py` | pass |
| C21 | The Wright-Fisher baseline converges to the analytic quasispecies, and reaches the accuracy WP7 needs inside the budget WP7 grants. Measured: total variation 0.004664 at N = 1e6 against a 0.02 threshold, and 0.02 reached in 28.04 s against a 300 s allotment. **The competitiveness half of the original claim is withdrawn, not met**: it compared two different complexity classes and no reference implementation exists in the pinned image. See revision 15, revision 22 and ADR-0018 | G-4 | `results/wp4/g_4.json` | `experiments/wp4_wright_fisher/g_4_wright_fisher.py` | pass |
| C22 | The polynomial-time baseline matches the analytic oracle where it applies, refuses where it does not, and its applicability boundary is an explicit predicate. Measured: worst error 1.375e-10 against a 1e-6 tolerance over 442 cases, zero in-class refusals, zero out-of-class solves; 58 of 326 instances are in the class. **The Dixit-Srivastava-Vishnoi attribution is not claimed** and remains flagged in PRIOR_ART entry II.1 | G-5 | `results/wp5/g_5.json` | `experiments/wp5_exact_class/g_5_exact_class.py` | pass |
| C23 | Tensor-network imaginary-time evolution converges to exact diagonalisation where both run. Measured: cosine >= 0.999 on **285 of 285** configurations that were given the full ladder, zero failures to converge. Eight cells at L = 14 were stopped by revision 23's wall-clock limit and are reported as stopped, with the largest chi tried and the best cosine seen, rather than as failures | G-6.1 | `results/wp6/g_6.json` | `experiments/wp6_mps/g_6_tensor_network.py` | pass |
| C24 | The bond dimension required to hold fixed accuracy is mapped across ruggedness, mutation rate and system size. Measured: the requirement **rises with mutation rate and plateaus above the threshold**, max chi of 16, 16, 64, 64, 64 at mu/mu_c of 0.4, 0.7, 1.0, 1.3, 1.6, against a median of 4 throughout. It does not peak at mu_c | G-6.2 | `results/wp6/g_6.json` | `experiments/wp6_mps/g_6_tensor_network.py` | pass |
| C25 | The MPS comparison is scoped explicitly: MPO bond dimensions are reported per family and structural disadvantage on long-range families is stated, not exploited. The record also separates the operator's bond dimension, which sets the cost of one step, from the state's, which sets whether the quasispecies is representable at all | G-6.3 | `results/wp6/g_6_3.json` | `experiments/wp6_mps/mpo_analysis.py` | pass |

| C37 | No landscape family is both multi-peaked and anchored to a master sequence, so the order parameter is measured from each instance's own fittest genotype. Measured across 18 configurations at L = 8: every family with more than 1.2 strict local optima has its optimum away from genotype 0, and every family that keeps genotype 0 has at most 1.2. **Read with the strict count only.** Counted non-strictly the single-peak landscape shows 248 optima while being anchored, because its plateau ties are not peaks; the planning documents' figures of 1.4 optima at roughness 0.3 and 121 at roughness 1.0 are specification targets and the rebuild measures 1.2 and 12.7, so the qualitative claim reproduces and the quantitative one does not | G-3 | `results/wp3/g_3.json` | `experiments/wp3_landscapes/g_3_families.py` | pass |
| C38 | The Sherrington-Kirkpatrick spin glass is the only rugged family whose Pauli count stays polynomial, so the biological ruggedness axis and the compilation-cost axis are different axes. Measured at L = 8: spin glass 37 terms, exactly the predicted L(L-1)/2 + L + 1 = 37, against 264 for Rough Mount Fuji at every non-zero roughness including 0.1, where the landscape is still smooth and still anchored, and 263 for NK at K = 6 | G-3 | `results/wp3/g_3.json` | `experiments/wp3_landscapes/g_3_families.py` | pass |
---

## WP7: the boundary map

| # | Claim | Gate | Artefact | Script | Status |
|---|---|---|---|---|---|
| C26 | The compute-budget protocol was fixed before the sweep and applied to every method, including QUASAR's classical optimisation time. Measured: every method record carries `seconds_used` beside `seconds_allotted`, `over_budget` is recomputed from the measurement rather than trusted, and **198 cells were excluded for overrunning**, 27 of them Route A and 171 the tensor network. The protocol bit the methods it was written to constrain, including the classical reference | G-7 | `results/wp7/sweep_manifest_registered.json`, `results/wp7/g_7.json` | `scripts/sweep_runner.py` | pass |
| C27 | Every grid cell is either scored or explicitly excluded, with the exclusion reason recorded, and the excluded share is reported per size rather than folded into a total. Measured: 777 of 777 classical and 777 of 777 quantum cells recorded, 152 groups scored, 67 excluded, every exclusion carrying the reason `over budget`. ADR-0019 | G-7 | `results/wp7/sweep_manifest_registered.json`, `results/wp7/g_7.json` | `scripts/sweep_runner.py`, `scripts/score_g7.py` | pass |
| C28 | The quantum-classical boundary for mutation-selection dynamics is mapped, with the decision gate answered as a **registered null** carrying an explicit bound and a tally of which condition each group failed. Measured: **all 152 scored groups fail condition 2**, the compute-matched tensor network never falling below 0.80, so no quantum result could have produced a positive. The null is bounded at L = 12, the largest size holding a valid reference | G-7 | `results/wp7/g_7.json` | `scripts/score_g7.py` | pass |

---

## WP8: live QPU

| # | Claim | Gate | Artefact | Script | Status |
|---|---|---|---|---|---|
| C29 | Validated circuits were executed on a live quantum processor, with job identifiers, backend, calibration timestamp, transpiled depth, two-qubit counts and both raw and mitigated distributions reported as measured. **Not reproduced evidence:** the pinned image has no `qiskit-ibm-runtime`, so under ADR-0012 the record is written to `results/_local/` and ADR-0020 requires that qualifier to travel with any claim citing it. Feasibility only, no advantage claimed | G-8 | `experiments/wp8_live_qpu/qpu_sweep.py` | `experiments/wp8_live_qpu/qpu_sweep.py` | planned |

---

## Claims withdrawn after measurement

Recorded rather than deleted. A claim that the evidence did not support is part of the
record, and removing it silently is how a project ends up only reporting what worked.

| Claim, as the planning documents state it | What was measured | Status |
| C30 | The G-7 null does not depend on the budget exclusion rule. The tensor network overruns its allotment on 64.1% of `L = 12` cells, and the rule removes exactly the cells where the classical reference is most strained. Scored both ways: with exclusions the worst tensor-network cosine is 0.999981, without them it is 0.875797, and **zero** cells fall below the 0.80 threshold either way | G-7 | `results/wp7/g_7_budget_sensitivity.json` | `scripts/budget_sensitivity.py` | pass |
| C31 | Baseline B's polynomial-time class is a strict superset of the class-invariant class of arXiv:1203.1287, so no WP7 cell is believed classically hard while being covered by that prior work. Additive landscapes with distinct per-site coefficients lie outside class invariance and inside Baseline B. The `(L+1)`-dimensional reduction is attributable to Swetina and Schuster (1982), which arXiv:1203.1287 itself cites for it, not to that paper | none | `PRIOR_ART.md` II.1 and II.1a | `quasarstack/classical/exact_class.py` | pass |
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
