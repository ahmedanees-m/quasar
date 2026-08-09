"""Storage ceiling guard for the compute VM.

The VM root filesystem is shared with other projects and is about 91% full. QUASAR is held
to a hard ceiling so that no other project is ever affected. See DECISIONS.md ADR-0008.

Run this before any sweep. It exits non-zero if the ceiling would be breached, which stops
the sweep rather than filling the disk.

    python infra/disk_guard.py
    python infra/disk_guard.py --require-free-gb 12
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Hard ceiling for everything QUASAR occupies on the VM, in gigabytes. Registered in
# DECISIONS.md ADR-0008. Raising it is a decision, not a convenience.
QUASAR_CEILING_GB = 40.0

# Minimum free space that must remain on the filesystem after QUASAR's footprint, so that
# other projects on the shared VM are never squeezed.
DEFAULT_REQUIRED_FREE_GB = 8.0

GB = 1024**3


def tree_size_gb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:
            continue
    return total / GB


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="QUASAR working tree to measure")
    parser.add_argument("--require-free-gb", type=float, default=DEFAULT_REQUIRED_FREE_GB)
    parser.add_argument(
        "--ceiling-gb", type=float, default=QUASAR_CEILING_GB, help="QUASAR footprint ceiling"
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    usage = shutil.disk_usage(root)
    free_gb = usage.free / GB
    total_gb = usage.total / GB
    footprint_gb = tree_size_gb(root)

    print(f"filesystem   {total_gb:8.1f} GB total, {free_gb:8.1f} GB free")
    print(f"quasar tree  {footprint_gb:8.1f} GB at {root}")
    print(f"ceiling      {args.ceiling_gb:8.1f} GB, required free {args.require_free_gb:.1f} GB")

    problems = []
    if footprint_gb > args.ceiling_gb:
        problems.append(
            f"QUASAR footprint {footprint_gb:.1f} GB exceeds its {args.ceiling_gb:.1f} GB ceiling. "
            f"Archive results to the Drive with `make sync-up` and prune the local copy."
        )
    if free_gb < args.require_free_gb:
        problems.append(
            f"only {free_gb:.1f} GB free, below the {args.require_free_gb:.1f} GB reserved for "
            f"other projects on this shared machine."
        )

    if problems:
        print()
        for p in problems:
            print(f"BLOCKED: {p}")
        return 1

    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
