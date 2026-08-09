---
name: Validation failure
about: A pre-registered gate did not pass
title: "gate failure: G-"
labels: ["validation-failure", "honesty-check"]
---

A gate failure is a scientific event, not an annoyance.

**Gate.** Which gate, and its statement from `GATES.md`.

**Measured against threshold.** Measured value, registered threshold, and the artefact.

**Reproduction.** Exact command, image tag, git sha, seeds.

**Hypothesis.** What is most likely wrong: the method, the reference, or the threshold's
underlying reasoning?

**Was the threshold lowered?** This must be answered "no". If the threshold's justification
was itself wrong, that is an amendment appended to `GATES.md` with a dated rationale, and it
is a separate discussion from this failure.

**Fix.** What changed, and which regression test now locks it.
