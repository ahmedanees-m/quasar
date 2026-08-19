# quasar

Quantum simulation of mutation-selection dynamics. Implements the Crow-Kimura to
transverse-field Ising correspondence as a quantum circuit and benchmarks it against classical
methods.

[![ci](https://github.com/ahmedanees-m/quasar/actions/workflows/ci.yml/badge.svg)](https://github.com/ahmedanees-m/quasar/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![licence](https://img.shields.io/badge/licence-Apache%202.0-blue.svg)](LICENSE)

Code for "Quantum simulation of mutation-selection dynamics reveals a structural obstruction to
quantum advantage", submitted.

## Background

The Crow-Kimura and Eigen mutation-selection models map exactly onto transverse-field Ising
chains evolving in imaginary time. Mutation rate is the transverse field, per-locus fitness the
longitudinal field, epistasis the ZZ coupling, the quasispecies the Perron eigenvector, and the
error catastrophe a localisation transition.

Two quantum routes are compared on identical problems. Route A is variational imaginary-time
evolution by McLachlan's principle: shallow circuits, no ancillas, depth constant in imaginary
time, driven by a classical optimisation loop. Route B is QSVT eigenstate filtering through
qubitisation walk operators: one deep coherent circuit on 5 to 9 ancillas, no optimisation loop.

Three classical baselines run under the same per-cell time allocation: a Wright-Fisher
finite-population sampler, an exact solver for the landscape classes that admit one, and a
matrix-product-state reference.

## Install

    docker build -t quasar:v1 .

Anything that writes a result record runs in that image. For editing and fast tests:

    make setup

## Run

    make test        # fast unit and regression suite
    make gates       # everything, in the image, >20 h
    make claims      # check each number in the paper resolves to a record

Individual work packages:

    python scripts/run_all_gates.py --list
    python scripts/run_all_gates.py --wp wp_r

To compare a fresh run against the committed records:

    python scripts/compare_reproduction.py

A full run is 16 scripts producing 21 records, dominated by the matrix-product baseline at
12.7 h.

## Layout

    quasarstack/analytic     closed-form solutions, exact diagonalisation
    quasarstack/hamiltonian  generator to Pauli operators
    quasarstack/circuit      Trotterised propagator
    quasarstack/ite          variational and Motta imaginary-time evolution
    quasarstack/qsvt         block encoding, phase factors, eigenstate filter
    quasarstack/spectral     gap, conditioning, order parameter
    quasarstack/classical    Wright-Fisher, exact solver, matrix product states
    quasarstack/scoring      cosine, total variation, bootstrap
    quasarstack/backends     noise models, IBM Quantum submission
    quasarstack/io           record schema, provenance, storage

    experiments/             one script per check, each writing a JSON record
    scripts/                 sweep runner, scorers, archival tooling
    results/                 committed records
    docs/                    protocol, results index, notes, theory, validation

`scripts/archive_hardware.py` deposits the raw processor measurements,
`scripts/rescore_hardware.py` rebuilds the hardware result from them offline, and
`scripts/make_manifest.py` writes and verifies a SHA-256 manifest of a deposit.

## Results

Each number in the paper is mapped to the record it came from in
[docs/results-index.md](docs/results-index.md). Thresholds, grids, seeds and time allocations
are in [docs/protocol.md](docs/protocol.md), and [docs/notes.md](docs/notes.md) has the
reasoning behind the awkward decisions.

The boundary sweep covers 777 cells across landscape family, ruggedness, mutation rate and
system size. Of 152 scored groups every one fails the advantage criterion, and the
tensor-network reference never falls below cosine 0.80 against the analytic quasispecies.

The mechanism is the useful part. Where the generator's Pauli expansion is sparse the
matrix-product bond dimension is small, and where the bond dimension saturates the expansion is
dense. The two costs rise together, so nothing in the sweep is cheap for a quantum route and
expensive classically.

Rescoring with every time-excluded cell restored moves the worst tensor-network cosine from
0.999981 to 0.875797 and still leaves nothing below 0.80. See `scripts/budget_sensitivity.py`.

Three checks are recorded as failures rather than removed. The threshold-location check fails
because the finite-size threshold converges to the asymptotic value only above L = 48.

## Hardware

The mutation-rate sweep across the error threshold ran on an IBM Heron r2 device at L = 2, 3
and 4, 51 circuits and 208,896 shots. After readout mitigation the recovered distributions match
the analytic quasispecies to cosine 0.99930, 0.99595 and 0.99461, within 2.1e-3 of the noise
model recorded beforehand.

Full 2^n readout mitigation improved 7 of 23 points on hardware against 16 of 23 in simulation,
and made L = 4 worse. The assignment matrix carries its own sampling noise, and inverting it
costs more than the readout error it corrects at these shot counts.

## Scope

No advantage found at these sizes. The comparison is against a standard matrix-product baseline
at L up to 12, which is a lower bound on classical capability: a stronger tensor-network method
would widen the gap, not narrow it. Hardware runs reach 4 loci, which is exactly solvable
classically. Route B can't run on present hardware at all, since 1024 walk-operator queries on 5
to 9 ancillas is fault-tolerant territory, and that's reported as part of the resource
comparison.

## Requirements

Python 3.12, Qiskit 2.5.1. Full pinned set in `environment.lock.txt`. IBM Quantum credentials
are needed only for hardware runs.

## Development note

This codebase was written with AI-assisted tooling. All results come from the pinned pipeline in
this repository and reproduce from a clean checkout; the authors are responsible for
correctness.

## Citation

```bibtex
@software{quasar,
  author = {Mahaboob Ali, Anees Ahmed and Nelson, Everette Jacob Remington and Delhibabu, Radhakrishnan},
  title  = {quasar: quantum simulation of mutation-selection dynamics},
  year   = {2026},
  url    = {https://github.com/ahmedanees-m/quasar}
}
```

## Licence

Apache-2.0 for code, CC-BY-4.0 for data and records. See [LICENSE](LICENSE).
