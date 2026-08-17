"""Write a SHA-256 manifest of a directory tree, and verify one.

An archive that cannot be checked is an archive nobody can trust after the first copy. This
writes one line per file, sorted by path, so two deposits can be compared with a text diff and
a single altered byte is visible.

    python scripts/make_manifest.py write <directory> [--out MANIFEST.sha256]
    python scripts/make_manifest.py verify <directory> [--manifest MANIFEST.sha256]
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

SKIP_DIRECTORIES = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
DEFAULT_NAME = "MANIFEST.sha256"


def digest(path: pathlib.Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def walk(root: pathlib.Path, manifest_name: str) -> list[pathlib.Path]:
    found = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if SKIP_DIRECTORIES.intersection(path.relative_to(root).parts):
            continue
        if path.name == manifest_name:
            continue
        found.append(path)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["write", "verify"])
    parser.add_argument("directory")
    parser.add_argument("--out", default=DEFAULT_NAME)
    parser.add_argument("--manifest", default=DEFAULT_NAME)
    arguments = parser.parse_args()

    root = pathlib.Path(arguments.directory).resolve()
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")

    if arguments.action == "write":
        target = root / arguments.out
        files = walk(root, arguments.out)
        lines = [f"{digest(p)}  {p.relative_to(root).as_posix()}" for p in files]
        total = sum(p.stat().st_size for p in files)
        target.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        print(f"{len(lines)} files, {total / 1e6:.1f} MB -> {target}")
        return 0

    manifest = root / arguments.manifest
    if not manifest.is_file():
        raise SystemExit(f"no manifest at {manifest}")
    recorded = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        checksum, _, relative = line.partition("  ")
        recorded[relative] = checksum

    present = {p.relative_to(root).as_posix(): p for p in walk(root, arguments.manifest)}
    missing = sorted(set(recorded) - set(present))
    added = sorted(set(present) - set(recorded))
    changed = [
        name
        for name in sorted(set(recorded) & set(present))
        if digest(present[name]) != recorded[name]
    ]

    for name in missing:
        print(f"MISSING  {name}")
    for name in added:
        print(f"ADDED    {name}")
    for name in changed:
        print(f"CHANGED  {name}")
    ok = len(recorded) - len(missing) - len(changed)
    print(f"\n{ok}/{len(recorded)} files verified")
    if missing or added or changed:
        print("the archive does not match its manifest")
        return 1
    print("the archive matches its manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
