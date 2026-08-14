# Changelog

Keep a Changelog format. Semantic versioning. Release tags follow the execution plan:
`v1.0-implementation` after WP-R, `v2.0-boundary-map` after WP7, `v3.0-submission`.

## [Unreleased]

### Added

- Repository skeleton per `QUASAR_engineering_standards.md`.
- `GATES.md`, the kept as a running record specification, with thresholds for WP-R through WP8, the
  full WP7 grid, seed lists, the compute-budget protocol, and the G-7 decision rule.
- `PRIOR_ART.md`, the four-literature dossier, with a per-entry verification flag. Nothing
  may be cited in the manuscript while still marked to-verify.
- `CLAIMS.md`, the claims ledger, and `scripts/check_claims.py` to verify every entry
  resolves to an artefact and a script.
- `DECISIONS.md` with ADR-0001 through ADR-0008, covering the rebuild decision, the three
  numerical conventions that correspond to known silent-failure modes, the Docker-only
  execution policy, the git-for-code and SFTP-for-data split, and the VM storage ceiling.
- `Dockerfile` and `requirements.in` for the pinned execution image, with BLAS thread
  counts pinned so the compute-budget protocol means something.
- `quasarstack/io/conventions.py`, the single source of index, ordering, and normalisation
  conventions, with unit and property-based tests.
- `infra/sync.py` for checksum-verified SFTP transfer between the VM and the Drive archive,
  and `infra/disk_guard.py` to hold QUASAR inside its declared storage ceiling on the
  shared VM.
- CI on every push: lint, format check, types, fast tests, claims-ledger check, and a
  full-history secrets scan. Nightly runs slow tests, gates, and regression.

- The analytic ruler: `quasarstack/analytic/crow_kimura.py` with two exactly solvable
  families, neither of which ever forms the 2^L generator, and
  `quasarstack/analytic/exact_diag.py`, the deliberately structure-blind brute-force
  reference it is checked against.
- `quasarstack/classical/landscapes.py` with the WP-R subset of families, in the spin
  convention only.
- `quasarstack/io/store.py`, which writes result records carrying the commit, the image
  tag, the interpreter, whether the tree was dirty, and the SHA-256 of `GATES.md`, so that
  "the threshold was registered before the run" is checkable rather than asserted.
- `quasarstack/hamiltonian/builder.py`, compiling landscapes into Pauli operators by two
  independent routes, and `quasarstack/scoring/metrics.py` with cosine and total variation
  always reported together.
- `GATES.md` Amendments 1 and 2: the G-R.1 case set and the G-R.2 configuration set, each
  appended before the gate it governs was executed.

### Gates passed

- **G-R.1**, oracle against exact diagonalisation: max absolute error 2.4e-15 against a
  1e-9 threshold, over 1701 comparisons.
- **G-R.2**, compiled Hamiltonian against the oracle: cosine 1.000000 on 40 of 40
  configurations against a 0.999999 threshold, with operator-level agreement to 3.6e-15.
- **G-R.3**, Trotterised imaginary-time evolution: convergence at cosine 1.0000000 against a
  0.999 threshold, fitted splitting exponent 1.995 to 2.000 against bounds of [1.8, 2.2],
  R² = 1.00000 on every configuration.
- **G-R.4**, the error threshold: max absolute surplus difference 1.9e-15 against a 1e-3
  threshold, over 15 cases and a 300-point mutation-rate sweep. The sharp-peak threshold
  converges to `mu_c × L` = 1.000 with the width collapsing from 0.300 to 0.010, and the
  spectral gap at that threshold closes with a fitted decay of 0.717 per site.

- **G-R.5**, rugged NK landscapes: cosine 1.000000 across 100 instances against a 0.99999
  threshold, zero failing, including the maximally rugged K = 7 case where the Pauli
  decomposition saturates at 2^L + L terms. Ruggedness is monotone in K. The Trotterised
  diagnostic exposed the budget fairness problem recorded in ADR-0013.

- **G-R.6**, varQITE: cosine 0.9999741 against a 0.999 threshold across 14 configurations,
  with circuit depth identical at τ = 2.5 and τ = 20 on every one. The McLachlan quantities
  agree with their parameter-shift and fidelity-shift equivalents to 7e-16, which is what
  licenses calling the method hardware-faithful. Adds `quasarstack/ite/varqite.py` and
  `quasarstack/ite/qite_motta.py`.

### Claim withdrawn

- "Antagonistic epistasis lowers the error threshold", from the planning documents, is not
  supported and is dropped. Negative uniform coupling relocates the fitness optimum off the
  master sequence, so the question is ill-posed in that family. ADR-0011 makes reporting the
  optimum's location a requirement on every ruggedness axis, which matters because WP7
  sweeps ruggedness as its main axis.

### Enforcement

- `scripts/check_results_provenance.py`, run in CI, rejects any committed result record that
  cannot show it came from the pinned image on a clean tree. Added after a laptop-produced
  record reached a commit. `write_gate_record` now also redirects records from outside the
  image into a gitignored directory so the situation cannot arise. ADR-0012.

### Prior art

- Entry IV.4, Claudon, Piquemal and Monmarché (2025), verified ahead of schedule and found
  **not to apply**. The generator is non-conservative but reversible, while their speedup is
  bought by nonreversibility and their theorems are stated for row-stochastic kernels. Route
  B needs a different foundation; options in ADR-0010. Also adds
  `quasarstack/spectral/perron.py` and its reversibility diagnostics.
- `docs/validation.md`, mapping each of the three historical failure modes to the
  convention that now locks it out.

### Notes

Values reported in the planning documents belong to an earlier implementation that could not
be located, and are registered in `GATES.md` section 3 as targets for the rebuild rather
than carried over as results. See `DECISIONS.md` ADR-0001.
