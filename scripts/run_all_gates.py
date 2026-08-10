"""One command, full reproduction: run every pre-registered gate and write its artefact.

Reproducibility is binary. A clean clone plus this script reproduces every gate, or the
project is not done.

Gate scripts live under `experiments/<wp>/` and are named `g_*.py`. Each one is expected to
write a JSON result record under `results/<wp>/`. This driver discovers them, runs them in
declared order, and reports pass or fail per gate without swallowing failures.

    python scripts/run_all_gates.py                 every gate
    python scripts/run_all_gates.py --wp wp_r       one work package
    python scripts/run_all_gates.py --list          show what would run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quasarstack.io.store import evidence_directory  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "experiments"
RESULTS = ROOT / "results"

# Declared execution order. A work package runs only after the ones it depends on.
# WP-R gates the rest, because nothing downstream is meaningful against an unvalidated
# stack. See GATES.md section 3.
ORDER = [
    "wp_r_rebuild",
    "wp1_spectral",
    "wp2_qsvt",
    "wp3_landscapes",
    "wp4_wright_fisher",
    "wp5_exact_class",
    "wp6_mps",
    "wp7_boundary_map",
    "wp8_live_qpu",
]


def discover(only: str | None) -> list[Path]:
    """Every gate script, in dependency order, with nothing silently left out.

    `ORDER` used to be an allowlist and any work package missing from it was skipped in
    silence while the runner still printed a pass count and exited zero. Four gates were
    written, committed, launched and reported as a clean run without ever executing: G-3,
    G-4, G-5 and G-6 sat in directories nobody had added to the list. That is the same shape
    as ADR-0014, a mechanism that fails by doing nothing and says it succeeded.

    `ORDER` now sequences the packages it knows about, and any package holding a `g_*.py`
    that is *not* listed raises rather than being skipped. Adding a work package should
    force a decision about where it belongs in the order, not quietly opt it out of the
    suite.
    """
    listed = {name for name in ORDER}
    found = {
        directory.name
        for directory in EXPERIMENTS.iterdir()
        if directory.is_dir() and any(directory.glob("g_*.py"))
    }
    unlisted = sorted(found - listed)
    if unlisted:
        raise SystemExit(
            f"gate scripts live in work packages missing from ORDER: {unlisted}. "
            f"Add them to scripts/run_all_gates.py in the position their dependencies "
            f"require. Skipping them silently is how G-3 to G-6 were reported as a clean "
            f"run without executing."
        )

    scripts: list[Path] = []
    for package in ORDER:
        if only and only not in package:
            continue
        directory = EXPERIMENTS / package
        if not directory.is_dir():
            continue
        scripts.extend(sorted(directory.glob("g_*.py")))
    return scripts


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wp", default=None, help="run only this work package")
    parser.add_argument("--list", action="store_true", help="list gates without running them")
    args = parser.parse_args()

    scripts = discover(args.wp)

    if args.list:
        if not scripts:
            print("no gate scripts found")
        for script in scripts:
            print(script.relative_to(ROOT))
        return 0

    if not scripts:
        print("No gate scripts found yet. WP-R is the first work package to add them; see")
        print("GATES.md section 3 for the registered thresholds and CLAIMS.md for the")
        print("artefact each gate must write.")
        return 0

    RESULTS.mkdir(exist_ok=True)
    sha = git_sha()
    summary: list[dict[str, object]] = []
    failed = 0

    for script in scripts:
        rel = script.relative_to(ROOT).as_posix()
        print(f"\n=== {rel} ===", flush=True)
        started = time.monotonic()
        proc = subprocess.run([sys.executable, str(script)], cwd=ROOT)
        elapsed = time.monotonic() - started
        ok = proc.returncode == 0
        failed += 0 if ok else 1
        summary.append({"gate": rel, "passed": ok, "seconds": round(elapsed, 2)})
        print(f"--- {'PASS' if ok else 'FAIL'} in {elapsed:.1f}s", flush=True)

    manifest = {"git_sha": sha, "gates": summary, "failed": failed}
    (evidence_directory() / "gate_run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\n{len(summary) - failed}/{len(summary)} gates passed at {sha[:8]}")
    if failed:
        print("A failing gate is a scientific event. Open a validation_failure issue.")
        print("The threshold is not lowered.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
