# quasar

Code for "Quantum simulation of mutation-selection dynamics reveals a structural obstruction to
quantum advantage" (submitted).

[![ci](https://github.com/ahmedanees-m/quasar/actions/workflows/ci.yml/badge.svg)](https://github.com/ahmedanees-m/quasar/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![licence](https://img.shields.io/badge/licence-Apache%202.0-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22828425.svg)](https://doi.org/10.5281/zenodo.22828425)

The Crow-Kimura and Eigen mutation-selection models map onto transverse-field Ising chains
evolving in imaginary time. Mutation rate is the transverse field, per-locus fitness the
longitudinal field, epistasis the ZZ coupling, and the quasispecies is the Perron eigenvector of
the generator.

This repository implements that mapping as a quantum circuit and compares two quantum routes,
variational imaginary-time evolution and QSVT eigenstate filtering, with three classical
baselines: Wright-Fisher sampling, an exact solver for the landscape classes that admit one, and
DMRG on matrix-product states.

## Install

Everything that writes a record runs inside the Docker image:

    docker build -t quasar:v1 .

For development and the fast tests:

    make setup

## Usage

    make test      # unit and regression tests
    make gates     # every check, inside the image, about nine hours

Single work packages:

    python scripts/run_all_gates.py --list
    python scripts/run_all_gates.py --wp wp_r

The comparison sweep and its scoring:

    python scripts/sweep_runner.py --wp 7 --grid full --methods classical --workers 8
    python scripts/sweep_runner.py --wp 7 --grid full --methods quantum
    python scripts/score_g7.py

`scripts/rescore_hardware.py` reproduces the hardware result from the raw counts.
`scripts/make_manifest.py` writes and verifies a SHA-256 manifest of a data deposit.

## Layout

    quasarstack/analytic     closed-form solutions, exact diagonalisation
    quasarstack/hamiltonian  generator to Pauli operators
    quasarstack/circuit      Trotterised propagator
    quasarstack/ite          variational and Motta imaginary-time evolution
    quasarstack/qsvt         block encoding, phase factors, eigenstate filter
    quasarstack/spectral     gap, conditioning, order parameter
    quasarstack/classical    landscapes, Wright-Fisher, exact solver, DMRG
    quasarstack/scoring      cosine, total variation, bootstrap
    quasarstack/backends     noise models, IBM Quantum submission
    quasarstack/io           record schema, conventions, storage

    experiments/             one script per check, each writing a JSON record
    scripts/                 sweep runner, scoring, hardware rescoring, manifests
    results/                 records
    docs/                    settings and derivations

## Results

Records are in `results/`, one JSON file per check, and `docs/protocol.md` lists the settings.
No quantum advantage appears at the sizes compared, up to 12 loci: the matrix-product baseline
matches exact diagonalisation on every instance, so no group meets the advantage criterion.
The hardware run reaches 4 loci on an IBM Heron r2 device.

## Requirements

Python 3.12, Qiskit 2.5.1 and quimb 1.14.0, with the pinned set in `environment.lock.txt`.
IBM Quantum credentials are needed only for the hardware run.

## Development note

This codebase was written with AI-assisted tooling. All results come from the pipeline in this
repository and reproduce from a clean checkout; the authors are responsible for correctness.

## Citation

Code and data are archived at Zenodo: https://doi.org/10.5281/zenodo.22828425

```bibtex
@software{quasar,
  author  = {Mahaboob Ali, Anees Ahmed and Nelson, Everette Jacob Remington and Delhibabu, Radhakrishnan},
  title   = {quasar: quantum simulation of mutation-selection dynamics},
  year    = {2026},
  doi     = {10.5281/zenodo.22828425},
  url     = {https://github.com/ahmedanees-m/quasar}
}
```

## Licence

Apache-2.0 for code, CC-BY-4.0 for data and records. See [LICENSE](LICENSE).
