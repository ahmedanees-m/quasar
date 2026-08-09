# Changelog

Keep a Changelog format. Semantic versioning. Release tags follow the execution plan:
`v1.0-implementation` after WP-R, `v2.0-boundary-map` after WP7, `v3.0-submission`.

## [Unreleased]

### Added

- Repository skeleton per `QUASAR_engineering_standards.md`.
- `GATES.md`, the append-only pre-registration, with thresholds for WP-R through WP8, the
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
- `GATES.md` Amendment 1: the G-R.1 case set, appended before the gate was executed.
- `docs/validation.md`, mapping each of the three historical failure modes to the
  convention that now locks it out.

### Notes

Values reported in the planning documents belong to an earlier implementation that could not
be located, and are registered in `GATES.md` section 3 as targets for the rebuild rather
than carried over as results. See `DECISIONS.md` ADR-0001.
