---
name: Experiment
about: Register an experiment before running it
title: "exp(wpN): "
labels: ["experiment"]
---

Specification is a workflow habit here, not a one-off document. Fill this in **before**
the run, not after.

**Work package.**

**Objective.** What question does this run answer?

**Method.** Which module, which parameters, which seeds.

**Specified threshold.** The number that decides pass or fail, and why that number.

**Expected artefact.** Path under `results/` and the `docs/results-index.md` row it backs.

**Compute budget.** Allotted wall-clock per cell, worker count, image tag.

**Acceptable negative outcome.** What result would count as an honest null rather than a
failure? Registering this in advance is what stops a fallback being presented later as a
planned success.
