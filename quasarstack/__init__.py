"""quasar: quantum algorithms for mutation-selection dynamics.

Physics and algorithms live in this package; scripts under `experiments/` and `scripts/` call it.

Conventions, listed in `docs/protocol.md`:

- fitness in the spin convention, `a_i Z_i` and `b_ij Z_i Z_j`
- quasispecies distributions L1-normalised and non-negative
- quantum states L2-normalised internally, converted at readout
- Qiskit little-endian ordering, with bitstring conversions in `quasarstack.io.conventions`
- sparse eigensolvers above L = 12
"""

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
