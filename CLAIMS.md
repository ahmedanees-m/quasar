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
| C1 | The analytic Crow–Kimura oracle agrees with brute-force exact diagonalisation to machine precision | G-R.1 | `results/wp_r/g_r_1.json` | `experiments/wp_r_rebuild/g_r_1_oracle_vs_ed.py` | planned |
| C2 | The compiled qubit Hamiltonian's ground state is the analytic quasispecies across all tested configurations | G-R.2 | `results/wp_r/g_r_2.json` | `experiments/wp_r_rebuild/g_r_2_hamiltonian_vs_oracle.py` | planned |
| C3 | The Trotterised circuit converges to the oracle with the expected second-order step-size scaling | G-R.3 | `results/wp_r/g_r_3.json` | `experiments/wp_r_rebuild/g_r_3_trotter_scaling.py` | planned |
| C4 | The error threshold appears on the circuit as a localisation-delocalisation transition at the analytically predicted mutation rate | G-R.4 | `results/wp_r/g_r_4.json` | `experiments/wp_r_rebuild/g_r_4_error_threshold.py` | planned |
| C5 | Rugged epistatic landscapes are reproduced against brute-force exact diagonalisation | G-R.5 | `results/wp_r/g_r_5.json` | `experiments/wp_r_rebuild/g_r_5_rugged.py` | planned |
| C6 | varQITE reproduces the quasispecies at circuit depth constant in imaginary time | G-R.6 | `results/wp_r/g_r_6.json` | `experiments/wp_r_rebuild/g_r_6_varqite.py` | planned |
| C7 | Motta-QITE reproduces the quasispecies with monotonically descending energy and no barren plateau | G-R.7 | `results/wp_r/g_r_7.json` | `experiments/wp_r_rebuild/g_r_7_motta.py` | planned |
| C8 | The pipeline is feasible under realistic simulated device noise with error mitigation, L = 2 to 4 | G-R.8 | `results/wp_r/g_r_8.json` | `experiments/wp_r_rebuild/g_r_8_noise.py` | planned |
| C9 | varQITE gradient variance decays exponentially in system size, bounding the method's reach | G-R.9 | `results/wp_r/g_r_9.json` | `experiments/wp_r_rebuild/g_r_9_barren.py` | planned |
| C10 | The sparse additive-plus-epistasis representation requires far fewer Pauli terms than the single-peak projector | G-R.10 | `results/wp_r/g_r_10.json` | `experiments/wp_r_rebuild/g_r_10_pauli_count.py` | planned |
| C11 | Analytic-first validation catches implementation errors that produce plausible but incorrect output | — | `docs/validation.md` plus the regression tests that lock each convention | `tests/regression/` | planned |

Note on C11. The three bugs described in the planning documents belong to an implementation
that no longer exists. They will be reported as methodological history, clearly attributed
to the earlier implementation, and each one is locked against recurrence by a regression
test in the rebuilt stack. Any bug the rebuild catches on its own is reported separately and
as new.

---

## WP1 — structural and spectral analysis

| # | Claim | Gate | Artefact | Script | Status |
|---|---|---|---|---|---|
| C12 | The mutation-selection generator is a non-conservative linear operator whose Perron eigenvector is the quasispecies, with the structural properties derived, not asserted | G-1.3 | `docs/theory.md` | — | planned |
| C13 | The spectral gap of the generator is mapped across ruggedness, mutation rate and system size, and closes at the error threshold | G-1.1, G-1.2 | `results/wp1/gap_map.parquet` | `experiments/wp1_spectral/gap_map.py` | planned |
| C14 | The condition number of the generator degrades in a characterised way approaching the error threshold and with ruggedness | G-1.3 | `results/wp1/conditioning.parquet` | `experiments/wp1_spectral/conditioning.py` | planned |
| C15 | Pauli-term count, qubit count and depth scaling are measured per landscape family | G-1.3 | `results/wp1/resources.json` | `experiments/wp1_spectral/resources.py` | planned |

---

## WP2 — Route B, QSVT

| # | Claim | Gate | Artefact | Script | Status |
|---|---|---|---|---|---|
| C16 | A block encoding of the shifted mutation-selection operator is constructed and satisfies its defining property | G-2.2 | `results/wp2/block_encoding.json` | `experiments/wp2_qsvt/verify_block_encoding.py` | planned |
| C17 | A QSVT eigenvalue transform amplifies the dominant eigenvector and reproduces the analytic quasispecies at small system size | G-2.1 | `results/wp2/eigen_transform.json` | `experiments/wp2_qsvt/eigen_transform_validation.py` | planned |
| C18 | Route B resource scaling is derived as a function of the measured spectral gap and matches the empirical requirement | G-2.3 | `results/wp2/resources.json` | `experiments/wp2_qsvt/resource_estimate.py` | planned |
| C19 | Route A and Route B are compared head to head on the same landscapes at the same accuracy target | G-2 | `results/wp2/route_comparison.parquet` | `experiments/wp2_qsvt/route_comparison.py` | planned |

---

## WP3 to WP6 — landscapes and classical baselines

| # | Claim | Gate | Artefact | Script | Status |
|---|---|---|---|---|---|
| C20 | Seven landscape families are implemented, reproduce exactly from seed, and ruggedness increases monotonically with K | G-3 | `results/wp3/landscape_stats.parquet` | `experiments/wp3_landscapes/stats.py` | planned |
| C21 | The Wright-Fisher baseline converges to the analytic quasispecies and is competitive with the reference community implementation | G-4 | `results/wp4/wf_validation.json` | `experiments/wp4_wright_fisher/validate.py` | planned |
| C22 | The Dixit-Srivastava-Vishnoi baseline matches the analytic oracle where it applies, and its applicability boundary is mapped | G-5 | `results/wp5/dv_applicability.json` | `experiments/wp5_dixit_vishnoi/applicability.py` | planned |
| C23 | Tensor-network imaginary-time evolution converges to exact diagonalisation where both run | G-6.1 | `results/wp6/mps_vs_ed.parquet` | `experiments/wp6_mps/cross_validate.py` | planned |
| C24 | The bond dimension required to hold fixed accuracy is mapped across ruggedness, mutation rate and system size | G-6.2 | `results/wp6/chi_hardness.parquet` | `experiments/wp6_mps/chi_sweep.py` | planned |
| C25 | The MPS comparison is scoped honestly: MPO bond dimensions are reported per family and structural disadvantage on long-range families is stated, not exploited | G-6.3 | `results/wp6/mpo_bond_dims.json` | `experiments/wp6_mps/mpo_analysis.py` | planned |

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

## Claims explicitly not made

Registered here so that the absence is deliberate and visible, not an oversight.

- No claim of quantum advantage or speedup for evolutionary dynamics.
- No claim of superiority over tensor-network methods in general, only over the standard,
  well-tuned MPS implementation actually tested, at the system sizes actually tested.
- No claim of clinical, therapeutic or diagnostic utility.
- No claim of a first biologically faithful simulator, a breakthrough, or a game change.
- No claim of simulating whole genomes or chromosomes.
