---
name: Validation failure
about: A specified check did not pass
title: "check failure: G-"
labels: ["validation-failure", "honesty-check"]
---

A check failure is a scientific event, not an annoyance.

**Check.** Which check, and its statement from `docs/protocol.md`.

**Measured against threshold.** Measured value, registered threshold, and the artefact.

**Reproduction.** Exact command, image tag, git sha, seeds.

**Hypothesis.** What is most likely wrong: the method, the reference, or the threshold's
underlying reasoning?

**Was the threshold lowered?** This must be answered "no". If the threshold's justification
was itself wrong, that is an amendment appended to `docs/protocol.md` with a dated rationale, and it
is a separate discussion from this failure.

**Fix.** What changed, and which regression test now locks it.
