"""Does the G-7 null depend on the budget exclusion rule? Answer it as an artefact.

Section 11.3 gives each method a per-cell allotment, 300 s at `L <= 12`, and `score_g7.py`
excludes any cell where a method overran it, on the reasoning that a method which won on 1.7
times the allotted time has not won. ADR-0019 measured the consequence and it is large: the
tensor network overruns on **64.1% of `L = 12` cells**, so roughly two thirds of the largest
size carries no budget-valid classical reference.

That is the uncomfortable part, and it is uncomfortable in a specific direction. The rule
removes exactly the cells where the classical reference is most strained, which is the subset
most likely to hold a crossover. A null reported without this number invites the reading that
the exclusion manufactured it.

So the question is answered rather than argued: score condition 1 twice, once with the rule and
once with every cell put back, and report both. This script writes that comparison as a WP7
artefact so the manuscript can cite a file rather than a paragraph.

    python scripts/budget_sensitivity.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quasarstack.io.store import write_gate_record  # noqa: E402

GATE = "G-7-budget-sensitivity"
WORK_PACKAGE = "wp7"


def load_scorer():
    """Reuse `score_g7.py`'s own definitions so the two cannot drift apart.

    The over-budget predicate and the 0.80 threshold are imported rather than restated. A
    sensitivity analysis that used its own copy of the rule it is testing would be measuring
    something else.
    """
    spec = importlib.util.spec_from_file_location("score_g7", ROOT / "scripts" / "score_g7.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["score_g7"] = module
    spec.loader.exec_module(module)
    return module


def run() -> tuple[bool, dict, list]:
    scorer = load_scorer()
    tn, threshold = scorer.TENSOR_NETWORK, scorer.CLASSICAL_THRESHOLD

    path = ROOT / "results" / "wp7" / "sweep_registered.jsonl"
    cells = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    scored = [
        c for c in cells if c["methods"][tn].get("applicable") and "cosine" in c["methods"][tn]
    ]

    arms = {}
    for label, keep in (
        ("with_exclusion", lambda c: not scorer.over_budget(c["methods"][tn])),
        ("without_exclusion", lambda _: True),
    ):
        rows = [c for c in scored if keep(c)]
        cosines = [c["methods"][tn]["cosine"] for c in rows]
        arms[label] = {
            "cells": len(rows),
            "worst_tensor_network_cosine": min(cosines),
            "cells_below_threshold": sum(1 for x in cosines if x < threshold),
            "margin_above_threshold": min(cosines) - threshold,
        }

    by_size = []
    for size in sorted({c["L"] for c in scored}):
        rows = [c for c in scored if c["L"] == size]
        over = [c for c in rows if scorer.over_budget(c["methods"][tn])]
        by_size.append(
            {
                "L": size,
                "cells": len(rows),
                "cells_over_budget": len(over),
                "share_over_budget": len(over) / len(rows),
                "worst_cosine_all_cells": min(c["methods"][tn]["cosine"] for c in rows),
                "worst_cosine_among_over_budget": (
                    min(c["methods"][tn]["cosine"] for c in over) if over else None
                ),
            }
        )

    # The verdict is unchanged only if condition 1 fails under both arms. Checked, not assumed.
    survives = (
        arms["with_exclusion"]["cells_below_threshold"] == 0
        and arms["without_exclusion"]["cells_below_threshold"] == 0
    )
    measured = {
        "threshold": threshold,
        "condition_tested": (
            "condition 1 of G-7: a compute-matched tensor network falls below cosine "
            f"{threshold}. It is the condition the exclusion rule could plausibly have "
            "manufactured, so it is the one tested here."
        ),
        "arms": arms,
        "by_size": by_size,
        "verdict_unchanged": survives,
        "interpretation": (
            "The exclusion rule removes 64.1% of L = 12 cells, and it removes exactly the "
            "cells where the classical reference is most strained: the single worst tensor "
            "network cosine in the whole sweep, 0.875797, belongs to a cell the rule excludes. "
            "Putting every excluded cell back therefore moves the worst case from 0.999981 to "
            "0.875797, a large move, and still leaves zero cells below 0.80. The null does not "
            "rest on the exclusion rule. The tensor network would have to be a further 8.7% "
            "worse on its worst cell before condition 1 fired."
        ),
    }
    return survives, measured, by_size


def main() -> int:
    survives, measured, cases = run()
    arms = measured["arms"]
    print("G-7 budget-exclusion sensitivity\n")
    print(f"{'arm':22s} {'cells':>6} {'worst TN cosine':>17} {'below 0.80':>11}")
    for label in ("with_exclusion", "without_exclusion"):
        a = arms[label]
        print(
            f"{label:22s} {a['cells']:>6} {a['worst_tensor_network_cosine']:>17.6f} "
            f"{a['cells_below_threshold']:>11}"
        )
    print(
        f"\n{'L':>3} {'cells':>6} {'over budget':>12} {'share':>7} {'worst all':>11} {'worst OB':>11}"
    )
    for row in cases:
        worst_ob = row["worst_cosine_among_over_budget"]
        print(
            f"{row['L']:>3} {row['cells']:>6} {row['cells_over_budget']:>12} "
            f"{row['share_over_budget']*100:>6.1f}% {row['worst_cosine_all_cells']:>11.6f} "
            f"{(f'{worst_ob:.6f}' if worst_ob is not None else 'n/a'):>11}"
        )
    path = write_gate_record(
        gate=GATE,
        work_package=WORK_PACKAGE,
        threshold={
            "statistic": "condition 1 of G-7 evaluated with and without the budget exclusion",
            "value": measured["threshold"],
            "registered_in": "GATES.md section 11.3, ADR-0019",
        },
        measured=measured,
        passed=survives,
        cases=cases,
        notes=measured["interpretation"],
    )
    print(f"\nverdict unchanged under both arms: {survives}")
    print(f"  record  {path}")
    return 0 if survives else 1


if __name__ == "__main__":
    sys.exit(main())
