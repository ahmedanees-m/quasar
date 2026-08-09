# QUASAR

**Quantum algorithms for mutation–selection dynamics: formulation, methods, and an honest
assessment of the quantum–classical boundary.**

QUASAR studies whether quantum algorithms help compute the quasispecies distribution of
molecular evolution, and where the boundary against the best classical methods actually
falls. It rests on a proven correspondence: the Crow–Kimura and Eigen mutation–selection
models are exactly transverse-field Ising spin chains evolving in imaginary time. Mutation
rate is the transverse field, per-site fitness is the longitudinal field, epistasis is the
ZZ coupling, the quasispecies is the Perron eigenvector, and the error catastrophe is a
localisation–delocalisation phase transition.

---

## What this is, and what it is not

**This is.**

- A formal quantum-algorithmic characterisation of the mutation–selection operator: Perron
  structure, spectral gap, conditioning, resource scaling. That characterisation does not
  exist in the literature.
- Two quantum routes compared head to head on the same biological problem. Route A is
  heuristic near-term imaginary-time evolution. Route B is QSVT Perron-vector extraction,
  which connects to provable-complexity results rather than heuristics.
- A biology-specific quantum–classical boundary map, benchmarked against three classical
  baselines, one of which is a computational-biology result that physics benchmarks do not
  include.

**This is not.**

- A claim of quantum advantage. None is made, and the current literature argues against one
  for this problem class at accessible scales.
- A claim of superiority over tensor networks in general. The comparison is scoped to a
  standard, well-tuned MPS implementation at the system sizes actually tested, and the
  manuscript says so.
- A claim of clinical or therapeutic utility. Biological applications are outlook, not
  results.
- A simulator of whole genomes. That would be quantum-washing.

The correspondence this project implements is over twenty-five years old, and classical
methods already solve the standard cases efficiently, including in closed form and in
polynomial time for structured landscapes. Those facts lead the introduction rather than
being buried. The open question is narrower and real: whether quantum methods offer anything
in the regime the classical guarantees do not cover, which is rugged, broken-symmetry,
strong-epistasis landscapes near the error threshold. A null answer is a publishable result
and is pre-registered as such.

---

## Status

Rebuilding Phases 1–3 (work package WP-R). See `GATES.md` section 3 for the registered
thresholds and `DECISIONS.md` ADR-0001 for why the rebuild is happening.

| Work package | Content | Status |
|---|---|---|
| WP0 | Pre-registration and prior-art dossier | `GATES.md` and `CLAIMS.md` done; entry IV.4 verified with a finding that changes WP2; 19 entries still to verify |
| WP-R | Rebuild and re-validate Phases 1–3 | G-R.1 to G-R.5 passed; G-R.6 next |
| WP1 | Spectral and structural analysis | not started |
| WP2 | Route B, QSVT Perron-vector extraction | not started |
| WP3 | Landscape families | not started |
| WP4 | Baseline A, Wright–Fisher | not started |
| WP5 | Baseline B, Dixit–Srivastava–Vishnoi | not started |
| WP6 | Baseline C, tensor-network imaginary time | not started |
| WP7 | Grid sweep and boundary map | not started |
| WP8 | Live QPU execution | not started |
| WP9 | Manuscript and red-team | not started |

### Gates passed in this repository

Only gates with a committed artefact appear here. Numbers reported in the planning
documents belong to an earlier implementation that was lost; they are registered as targets
in `GATES.md`, not carried over as results.

| Gate | What it establishes | Threshold | Measured | Artefact |
|---|---|---|---|---|
| G-R.1 | The analytic oracle agrees with brute-force exact diagonalisation, over 1701 comparisons spanning L = 2 to 10 and seven mutation rates | max abs error < 1e-9 | **2.4e-15** | `results/wp_r/g_r_1.json` |
| G-R.2 | The compiled qubit Hamiltonian's ground state is the analytic quasispecies, on 40 registered configurations | cosine ≥ 0.999999, 40 of 40 | **1.000000**, 40 of 40 | `results/wp_r/g_r_2.json` |
| G-R.3 | The Trotterised imaginary-time propagator converges, and its splitting error is second order | cosine ≥ 0.999; exponent in [1.8, 2.2] at R² ≥ 0.99 | **1.0000000**; **1.995 to 2.000** at R² = **1.00000** | `results/wp_r/g_r_3.json` |
| G-R.4 | The error threshold on the qubit representation matches the analytic prediction, over a 300-point sweep at L = 4, 6, 8 | max abs Δm < 1e-3 | **1.9e-15** | `results/wp_r/g_r_4.json` |
| G-R.5 | Rugged NK landscapes reproduce brute-force exact diagonalisation, 100 instances at L = 6 to 10 and K = 1 to 7 | cosine ≥ 0.99999, every instance | **1.000000**, 0 failing | `results/wp_r/g_r_5.json` |

**G-R.1.** Two independent analytic routes and one structure-blind reference agree: the
closed-form product state for additive fitness, the Hamming-class tridiagonal reduction for
permutation-symmetric fitness, and diagonalisation of the full 2^L generator. On the family
where both analytic routes apply, all three are compared.

**G-R.2.** Beyond the registered cosine, the compiled Pauli operator was compared entry by
entry against the generator assembled independently in `analytic/exact_diag.py`, agreeing to
3.6e-15. That is the stricter check: an endianness error permutes the computational basis
and leaves the spectrum untouched, so a spectral comparison alone would not see it.

**G-R.3.** The splitting exponent is fitted against `exp(-H tau)` computed without
splitting, not against the quasispecies, so the residual from a finite `tau` cannot flatten
it. Convergence is scored separately against the oracle. The propagator is not a
hardware-runnable circuit, because imaginary-time evolution is non-unitary; the
hardware-faithful routes are varQITE and Motta-QITE, gates G-R.6 and G-R.7.

**G-R.4.** Three diagnostics came out of it, none of them pass conditions. The sharp-peak
threshold converges to the infinite-size prediction, with `mu_c × L` reaching 1.000 by
L = 20 against a peak height of 1.0 while the width collapses from 0.300 to 0.010. The
spectral gap at that threshold closes exponentially, a fitted decay of **0.717 per site**
over L = 4 to 12, which is the first measured evidence here for how hard the critical region
is and feeds WP1 gate G-1.2 directly. And one claim from the planning documents did not
survive: see below.

**G-R.5.** The first family with no structure for the compiler to exploit: at K = 7 the
Pauli decomposition saturates at 2^L + L terms and the two routes still agree to machine
precision. Ruggedness is monotone in K, with mean local optima rising 2.8 to 29.1 and
fitness autocorrelation falling 0.659 to −0.021 at L = 8. The Trotterised route ran
alongside as a diagnostic and found something the gate was not asking about.

### A budget problem that would have biased the boundary map

Imaginary-time evolution at a fixed budget failed to reach the gate's accuracy on 3 of 40
rugged instances, and they were **the three with the smallest spectral gaps**: 0.0276,
0.0476 and 0.0835. Mean shortfall by connectivity runs 8.6e-12 at K = 1 up to 1.7e-4 at
K = 7.

That is not a defect in the method. Imaginary time suppresses the leading contaminant as
`exp(-gap × tau)`, so the budget it needs scales as `1 / gap`. But `GATES.md` section 11.3
currently gives every method equal wall-clock, and G-R.4 measured the gap at the error
threshold closing at 0.717 per site. Under that protocol an imaginary-time route would score
badly in the rugged, small-gap, near-threshold cells **because it was under-budgeted, not
because the method is unsuited**, and those cells are exactly the candidate quantum-relevant
regime the boundary map exists to examine.

`DECISIONS.md` ADR-0013 sets out three fairness protocols and recommends reporting both
accuracy-at-fixed-budget and budget-needed-for-accuracy, so the protocol's effect is visible
rather than buried. It needs settling before WP7 runs.

### A claim that did not survive measurement

The planning documents state that antagonistic epistasis lowers the error threshold. In the
uniform pairwise family it does not, and the reason is structural rather than statistical:
negative uniform coupling moves the fitness optimum **off the master sequence** to Hamming
class 1, 2 and 2 at L = 4, 6 and 8, with multiplicities up to 28. There is no master
sequence left to delocalise from, so the question is ill-posed in that family.

The companion claim, that synergistic epistasis raises it, is supported and converges in
size. The failing case is kept in the record rather than dropped. `DECISIONS.md` ADR-0011
turns it into a requirement on WP3: any family used as a ruggedness axis must report where
its fitness optimum sits, because WP7 sweeps ruggedness as its main axis and cells that
silently change which genotype is optimal are not comparable.

### A prior-art finding that changes WP2

Verification of prior-art entry IV.4 was pulled ahead of schedule, because Route B is the
paper's novelty core and rests on that single reference. The result:

> The mutation–selection generator is **non-conservative but reversible**, so the
> beyond-quadratic speedup of Claudon, Piquemal and Monmarché (2025), which is bought by
> *non*reversibility and stated for row-stochastic kernels, does not apply. Reversibility is
> a property of the mutation operator alone, so no amount of landscape ruggedness changes
> it. Within this problem class, nonreversibility requires direction-specific
> context-dependent mutation, which is biologically real but is a model extension.

Route B is not dead; its foundation has to change. Measured across 20 operators in
`results/wp0/prior_art_iv_4.json`, with options and a recommendation in `DECISIONS.md`
ADR-0010.

---

## Quickstart

Everything that produces a result record runs inside the pinned Docker image. Nothing is
installed onto a host.

```bash
git clone https://github.com/ahmedanees-m/quasar.git
cd quasar
make docker
make gates
```

For development on a machine without Docker, or to run the fast test suite only:

```bash
make setup
make test
```

---

## Repository map

```
quasarstack/        the package
  analytic/         the ruler: closed-form Crow-Kimura oracle, brute-force exact diag
  hamiltonian/      biology to qubit Pauli operator
  spectral/         WP1: spectral gap, conditioning, Perron structure
  circuit/          Trotterised imaginary-time circuit (modules M, S, E)
  ite/              Route A: varQITE and Motta-QITE
  qsvt/             Route B: block encoding, phase factors, eigenvalue transform
  classical/        landscapes and the three classical baselines, plus the budget protocol
  backends/         noise models, execution pipeline, live QPU submission
  scoring/          cosine, total variation, bootstrap confidence intervals
  io/               result schema, provenance capture, bitstring conventions
experiments/        one directory per work package; scripts read as protocols
tests/              unit, integration, gates, regression
scripts/            run_all_gates, sweep_runner, make_figures, check_claims
infra/              VM and archive sync over SFTP; no credentials in the repository
docs/               theory, methods, baselines, validation, reproduction
```

---

## The documents that carry scientific weight

| File | Role |
|---|---|
| `GATES.md` | Pre-registration. Append-only. Every threshold, the full grid, seeds, the compute-budget protocol, and the decision rule, all fixed before the runs they judge. |
| `PRIOR_ART.md` | The four-literature dossier. Nothing is cited in the manuscript while still marked to-verify. |
| `CLAIMS.md` | The claims ledger. Every manuscript claim maps to an artefact and a script. `make claims` verifies each one resolves. |
| `DECISIONS.md` | Why things are the way they are. Conventions, storage policy, the rebuild decision. |

---

## Reproducibility

A clean clone plus one command reproduces every gate, or the project is not done. Every
stochastic component takes an explicit seed; every result record carries its seeds, its git
commit, and its image tag. Figures are script-generated only, never hand-edited.

**No external datasets are required.** Every input is analytic, exactly computed, or
seeded-synthetic. There is no licensing question, no provenance risk, and no ethics
approval to obtain.

---

## Compute layout

- **Laptop.** Authoring, git, fast tests, figure scripts over computed JSON, manuscript.
- **VM.** All Docker containers, all inference, all sweeps. Nothing installed on the host.
- **Drive archive.** Complete result set, figures, image tarballs, versioned backups.

Code moves between machines by git. Data moves by SFTP. See `DECISIONS.md` ADR-0006 through
ADR-0008.

---

## Licence and citation

Apache-2.0. See `LICENSE`. Citation metadata is in `CITATION.cff`; a Zenodo DOI is minted at
release.

---

## Author

Anees Ahmed Mahaboob Ali, Gene Therapy Laboratory, VIT Vellore.

In collaboration with Dr Delhibabu Radhakrishnan (School of Computer Science and
Engineering) and Dr Everette Jacob Remington Nelson (School of Bio Sciences and Technology),
VIT Vellore.
