# data/

**Deliberately empty. No external datasets are required.**

Every input to this project is analytic, exactly computed, or seeded-synthetic:

- fitness landscapes are generated from a recorded seed by `quasarstack/classical/landscapes.py`
- the reference quasispecies is the closed-form Crow-Kimura solution or sparse exact diagonalisation
- noise models are parameterised device models, not measured calibration dumps

This is a genuine strength and the manuscript states it explicitly: no licensing question,
no provenance risk, no ethics approval, and complete reproducibility from a clean clone.

The single exception is live QPU output in WP8, which is measurement data written to
`results/wp8/` with its job identifiers, not input data.
