# Contributing

## The five principles this repository is built on

1. Every claim in the paper maps to a committed, re-runnable artefact. If a script cannot
   regenerate it, it does not go in the paper.
2. Specify, then run. Thresholds and protocols live in `GATES.md` and are committed, so a
   result is judged against a written criterion rather than one chosen to fit it.
3. The ruler comes first. Validation infrastructure is built before the thing it validates.
4. Honest reporting is a first-class output. Nulls, failures, and scope limits are committed
   artefacts, not omissions.
5. Reproducibility is binary. A clean clone plus one command reproduces every gate, or the
   project is not done.

## Workflow

- `main` is protected and always green. No direct pushes.
- One branch per work package: `wp/<n>-<slug>`, for example `wp/2-qsvt-block-encoding`.
  Small changes use `fix/<slug>` or `docs/<slug>`.
- Pull request into `main` with CI green and the template completed. Squash-merge, so
  `main` reads cleanly and the branch keeps the detail.

## Do not use `git stash` on the Drive-mounted clone

`git stash pop` has failed silently there twice: it reports the files as restored, leaves
the stash entry in place, and writes nothing. Both times the work was recovered with
`git show 'stash@{0}:<file>'`, but the failure looks exactly like success until you check
the content.

Commit before pulling instead, then `git pull --rebase`. Avoid `--autostash` as well, since
it uses the same machinery. The compute VM clone is on a normal filesystem and is unaffected.

## Commit messages

Conventional Commits with project-specific types:

```
feat(qsvt): add LCU block encoding for sparse epistatic Hamiltonians
fix(ite): correct Motta generator linear system
gate(wp7): record G-7 decision-gate outcome
exp(wp1): spectral gap sweep across NK K=0..6
docs(prior-art): add quantum stochastic process literature
test(regression): freeze golden outputs for varQITE L=2-4
chore(deps): pin quimb
```

A fix that the analytic ruler caught references the failure it prevents, because that
traceability becomes a subsection of the paper.

## Code standards

- Python 3.12. `black` at line length 100, `ruff` for linting, `mypy` required on
  `quasarstack/` and advisory on `experiments/`.
- NumPy-style docstrings on every public function.
- Physics and algorithms live in `quasarstack/`. Experiment scripts orchestrate and should
  read as protocols, not contain physics.
- No hidden global state. Every function that consumes randomness takes an explicit `rng`
  or `seed`. `numpy.random.default_rng` only.
- Every algorithmic function documents its conventions: spin or projector, L1 or L2, and
  endianness. Two of this project's historical bugs were convention mismatches, so
  docstrings here are a control, not decoration.
- Side effects, meaning anything that writes a file, are confined to `quasarstack/io/`.

## Tests

Four layers, and a change is not done until the relevant ones exist:

- `tests/unit/` fast, isolated, deterministic. Runs on every push.
- `tests/integration/` cross-module pipelines.
- `tests/gates/` the specified acceptance criteria, one test per gate in `GATES.md`.
  A failing gate test is a red build.
- `tests/regression/` golden outputs. Any change that shifts a golden file must be
  explained in the pull request and logged in `DECISIONS.md`.

Markers: `fast`, `slow`, `gate`, `hardware`. CI on push runs `fast`. Nightly runs
`slow` and `gate`. `hardware` never runs in CI.

## When a gate fails

Open a `validation_failure` issue. The template asks what failed, measured against
threshold, the hypothesis, and whether the threshold was lowered. The answer to the last
question is always no. A failing method is fixed or the failure is reported.

## Secrets

Never in the repository, in any form, at any point in history. Tokens come from the
environment or a gitignored `.env`. Pre-commit runs a secrets scan and CI scans the full
history.
