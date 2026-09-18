"""Run every gate script and write its result record.

Gate scripts live under `experiments/<package>/` and are named `g_*.py`. Each writes a JSON
record under `results/<package>/`. Scripts run in the order below and each is reported as
pass or fail.

    python scripts/run_all_gates.py                 every gate
    python scripts/run_all_gates.py --wp wp_r       one package
    python scripts/run_all_gates.py --list          show what would run

Set ``QUASAR_RESUME=1`` to skip gates already completed at the current commit.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quasarstack.io.store import output_directory  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "experiments"
RESULTS = ROOT / "results"

# Execution order. G-1 reads the wp0 and wp_r records, so both run first.
ORDER = [
    "wp0_reversibility",
    "wp_r",
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
    """Every gate script, in execution order.

    A package holding a `g_*.py` that is not listed in `ORDER` raises, so no gate is skipped.
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
            f"gate scripts in packages missing from ORDER: {unlisted}. "
            f"Add them to scripts/run_all_gates.py."
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
    parser.add_argument("--wp", default=None, help="run only this package")
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
        print("no gate scripts found")
        return 0

    RESULTS.mkdir(exist_ok=True)
    sha = git_sha()
    summary: list[dict[str, object]] = []
    failed = 0

    # Completed gates are recorded per commit so an interrupted run can resume. The ledger sits
    # inside the repository because the container mounts only the repository. Gitignored.
    ledger_path = ROOT / ".quasar_repro_completed.json"
    resuming = os.environ.get("QUASAR_RESUME") == "1"
    done: dict[str, object] = {}
    if resuming and ledger_path.is_file():
        with contextlib.suppress(Exception):
            entry = json.loads(ledger_path.read_text(encoding="utf-8"))
            if entry.get("git_sha") == sha:
                done = entry.get("gates") or {}
            else:
                print(
                    f"ledger is for {str(entry.get('git_sha'))[:8]}, not {sha[:8]}: starting fresh"
                )
    if done:
        print(f"resuming: {len(done)} gates already run at {sha[:8]}, they will be skipped")

    for script in scripts:
        rel = script.relative_to(ROOT).as_posix()
        if rel in done:
            print(f"\n=== {rel} ===\n--- SKIP, already run at this commit", flush=True)
            summary.append(dict(done[rel], gate=rel, skipped_as_already_run=True))
            failed += 0 if done[rel].get("passed") else 1
            continue
        print(f"\n=== {rel} ===", flush=True)
        started = time.monotonic()
        proc = subprocess.run([sys.executable, str(script)], cwd=ROOT)
        elapsed = time.monotonic() - started
        ok = proc.returncode == 0
        failed += 0 if ok else 1
        entry = {"passed": ok, "seconds": round(elapsed, 2)}
        summary.append(dict(entry, gate=rel))
        done[rel] = entry
        with contextlib.suppress(Exception):
            ledger_path.write_text(
                json.dumps({"git_sha": sha, "gates": done}, indent=2), encoding="utf-8"
            )
        print(f"--- {'PASS' if ok else 'FAIL'} in {elapsed:.1f}s", flush=True)

    manifest = {"git_sha": sha, "gates": summary, "failed": failed}
    # The manifest covers the whole run, so it goes to the root of the results tree.
    (output_directory("", announce=False) / "gate_run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\n{len(summary) - failed}/{len(summary)} gates passed at {sha[:8]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
