"""QUASAR: quantum algorithms for mutation-selection dynamics.

The package is organised so that physics and algorithms live here and experiment scripts
only orchestrate. An experiment script should read as a protocol, not contain physics.

Binding conventions, enforced by tests rather than by habit, are listed in `GATES.md`
section 1 and justified in `DECISIONS.md`:

- fitness in the spin convention, `a_i Z_i` and `b_ij Z_i Z_j`, never the projector form
- quasispecies distributions L1-normalised and non-negative
- quantum states L2-normalised internally, converted at the decode boundary only
- Qiskit little-endian ordering, with every bitstring conversion routed through
  `quasarstack.io.conventions`
- sparse eigensolvers above L = 12, dense diagonalisation guarded against
"""

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
