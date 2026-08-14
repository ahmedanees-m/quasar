# QUASAR

Quantum algorithms for mutation-selection dynamics: formulation, methods, and a measured
quantum-classical boundary.

[![ci](https://github.com/ahmedanees-m/quasar/actions/workflows/ci.yml/badge.svg)](https://github.com/ahmedanees-m/quasar/actions/workflows/ci.yml)
[![nightly](https://github.com/ahmedanees-m/quasar/actions/workflows/nightly.yml/badge.svg)](https://github.com/ahmedanees-m/quasar/actions/workflows/nightly.yml)
[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![licence](https://img.shields.io/badge/licence-Apache%202.0-blue.svg)](LICENSE)
[![code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![gates](https://img.shields.io/badge/gates-16%20recorded-brightgreen.svg)](CLAIMS.md)
[![provenance](https://img.shields.io/badge/provenance-21%2F21-brightgreen.svg)](scripts/check_results_provenance.py)

## Overview

QUASAR asks whether quantum algorithms help compute the quasispecies distribution of molecular
evolution, and where the boundary against the best classical methods falls.

The work rests on an exact correspondence: the Crow-Kimura and Eigen mutation-selection models
map onto transverse-field Ising chains evolving in imaginary time. Mutation rate is the
transverse field, per-site fitness the longitudinal field, epistasis the ZZ coupling, the
quasispecies the Perron eigenvector, and the error catastrophe a localisation transition.

Two quantum routes are compared on identical problems:

- **Route A**, variational imaginary-time evolution by McLachlan's principle. Near-term in
  shape: shallow circuits, no ancillas, constant depth in imaginary time, driven by a classical
  optimisation loop.
- **Route B**, QSVT eigenstate filtering through qubitisation walk operators. Fault-tolerant in
  shape: a single deep coherent circuit on 5 to 9 ancillas with no optimisation.

Three classical baselines are held to the same per-cell compute budget: a Wright-Fisher
finite-population sampler, an exact polynomial-time solver for the landscape classes that admit
one, and a matrix-product-state reference.

## Results

**No quantum advantage is claimed, and none was found.** The boundary sweep covers 777 cells
across landscape family, ruggedness, mutation rate and system size. Of 152 scored groups, every
one fails the advantage criterion, and the classical tensor-network reference never falls below
cosine 0.80 against the analytic quasispecies. The null is bounded at `L = 12`, the largest size
at which the sweep held a valid reference.

The mechanism behind the null is the useful part. Where the Pauli expansion of the generator is
sparse, the matrix-product bond dimension is small; where the bond dimension saturates, the
expansion is dense as well. The two costs rise together, so there is no regime in the sweep
where a quantum route is cheap and the classical reference is not.

**The exclusion rule does not manufacture the result.** The compute budget removes 64.1% of
`L = 12` cells, and those are exactly the cells where the classical reference is most strained.
Scoring with every excluded cell restored moves the worst tensor-network cosine from 0.999981 to
0.875797 and still leaves zero cells below the 0.80 threshold. The analysis is an artefact, not
an assurance: see `scripts/budget_sensitivity.py`.

**Hardware.** The mutation-rate sweep across the error threshold ran on an IBM Heron r2 device
at `L = 2, 3, 4`, 51 circuits and 208,896 shots. After readout mitigation the recovered
distributions match the analytic quasispecies to cosine 0.99930, 0.99595 and 0.99461, agreeing
with the simulated noise model to within 2.1e-3 at every point where a prediction existed. The
error threshold is visible in the hardware error profile, with the largest departures below the
critical mutation rate. This is a feasibility and validation result at sizes that are
exactly solvable classically; it is not evidence of advantage.

A secondary finding: full `2^n` readout mitigation improved 7 of 23 hardware points against 16
of 23 in simulation, and degraded results at `L = 4`. The assignment matrix carries its own
sampling noise, and inverting it costs more than the readout error it corrects at these shot
counts.

## Method

Every result is a committed artefact produced by a script that anyone can rerun.

- **Acceptance criteria are written before the runs they judge** and live in `GATES.md`. A gate
  states its statistic, its threshold, and the artefact it must write. A threshold is not
  lowered to accommodate a result; a failing gate is reported as a failure.
- **Every claim maps to an artefact.** `CLAIMS.md` is the ledger, and
  `scripts/check_claims.py` fails if a claim names a file that does not exist.
- **Every committed record proves where it came from.** Records carry the commit, the container
  image tag, the platform and a hash of the specification. Records produced outside the pinned
  image are written to a separate tree and are not treated as evidence.
- **Failures are recorded, not removed.** Three gates are on the ledger as failures, and
  `DECISIONS.md` carries twenty architecture decision records including the ones that document
  mistakes.

## Reproducing

```bash
make setup      # install the pinned environment
make test       # fast unit and regression suite
make gates      # run every gate and write its artefact
make figures    # regenerate every figure from committed artefacts
make claims     # verify each claim resolves to an artefact
```

A full run covers 16 gate scripts producing 21 records and takes over twenty hours, dominated
by the matrix-product baseline. Individual work packages can be run alone:

```bash
python scripts/run_all_gates.py --wp wp_r
python scripts/run_all_gates.py --list
```

Results are compared against their committed versions with:

```bash
python scripts/compare_reproduction.py
```

## Repository layout

| Path | Contents |
|---|---|
| `quasarstack/` | Library: analytic references, Hamiltonian construction, imaginary-time evolution, QSVT, classical baselines, scoring |
| `experiments/` | One gate script per acceptance criterion, each writing a JSON record |
| `scripts/` | Sweep runner, scorers, figure generation, ledger and provenance checks |
| `results/` | Committed result records |
| `figures/` | Figures, regenerated from records only |
| `tests/` | Unit, regression and gate tests |
| `GATES.md` | Gate specification: statistics, thresholds, grids, seeds, budgets |
| `CLAIMS.md` | Claim to artefact ledger |
| `DECISIONS.md` | Architecture decision records |
| `PRIOR_ART.md` | Prior-art dossier with verification status per entry |

## Gate summary

| Work package | Subject | Status |
|---|---|---|
| WP-R | Validation suite: oracle, Hamiltonian, Trotter, threshold, ruggedness, varQITE, QITE, noise, gradients, Pauli count | 10 of 10 recorded |
| WP1 | Spectral gap map and closed-form references | recorded, criterion 2 fails |
| WP2 | QSVT eigenstate filtering | recorded, split verdict |
| WP3 | Landscape families and ruggedness axes | recorded |
| WP4 | Wright-Fisher baseline | recorded |
| WP5 | Polynomial-time landscape class | recorded |
| WP6 | Matrix-product baseline and bond dimension analysis | recorded |
| WP7 | Boundary sweep and advantage verdict | recorded, null |
| WP8 | Hardware execution | recorded |

A failing gate is a result. WP1 criterion 2 fails because the finite-size threshold location
converges to the asymptotic value only above `L = 48`, which the record states rather than
hides.

## Scope

This work does not claim quantum advantage, superiority over tensor networks in general, or
clinical utility. The comparison is scoped to a well-tuned matrix-product implementation at the
sizes actually tested, and the hardware result is feasibility at sizes solvable exactly by
classical means. Route B cannot run on present hardware: 1024 walk-operator queries on 5 to 9
ancillas is a deep coherent circuit and squarely fault-tolerant territory. That absence is
reported as part of the resource comparison rather than omitted.

## Citation

```bibtex
@software{quasar,
  author  = {Mahaboob Ali, Anees Ahmed},
  title   = {QUASAR: Quantum Algorithms for Mutation-Selection Dynamics},
  year    = {2026},
  url     = {https://github.com/ahmedanees-m/quasar}
}
```

## Licence

Apache License 2.0. See [LICENSE](LICENSE).
