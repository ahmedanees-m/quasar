"""SFTP transfer between the compute VM and the Drive archive.

Code moves between machines by git. This module moves everything else: result records,
figures, image tarballs. rclone is deliberately not used. See DECISIONS.md ADR-0007.

Credentials come from the environment, never from this file. Populate a gitignored `.env`
from `.env.example`, or export the variables directly:

    QUASAR_VM_HOST     hostname or IP of the compute VM
    QUASAR_VM_USER     login user
    QUASAR_VM_PASS     password (or set QUASAR_VM_KEY to a private key path instead)
    QUASAR_VM_KEY      path to a private key, preferred over a password
    QUASAR_VM_ROOT     absolute path of the repository working tree on the VM
    QUASAR_ARCHIVE     absolute path of the Drive archive directory on the laptop

Usage:

    python infra/sync.py up   results          push results to the archive
    python infra/sync.py down results          pull results from the archive
    python infra/sync.py verify results        checksum both sides, report drift
"""

from __future__ import annotations

import hashlib
import os
import posixpath
import stat
import sys
from pathlib import Path

try:
    import paramiko
except ImportError:  # pragma: no cover - the laptop always has it, the image does not need it
    paramiko = None  # type: ignore[assignment]

SKIP = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "scratch"}
CHUNK = 1 << 20


def _load_dotenv(path: Path) -> None:
    """Read KEY=VALUE lines from a .env file into the environment without overwriting."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"{name} is not set. Copy .env.example to .env and fill it in, or export the "
            f"variable. Credentials are never stored in this repository."
        )
    return value


def _load_key(path: str) -> paramiko.PKey:
    """Load a private key without being told its type.

    paramiko has no auto-detecting loader, and asking for the wrong class raises rather
    than falling through, so each supported type is tried in turn. Ed25519 is first because
    it is what this project generates.
    """
    # Looked up by name rather than by attribute, because paramiko drops key classes across
    # major versions: DSSKey is gone in 4.x, and hard-coding it makes this fail on load.
    candidates = [
        getattr(paramiko, name, None) for name in ("Ed25519Key", "ECDSAKey", "RSAKey", "DSSKey")
    ]
    errors = []
    for key_class in [c for c in candidates if c is not None]:
        try:
            return key_class.from_private_key_file(path)
        except paramiko.SSHException as exc:
            errors.append(f"{key_class.__name__}: {exc}")
    raise SystemExit(f"could not load private key {path}\n  " + "\n  ".join(errors))


def connect() -> tuple[paramiko.SFTPClient, paramiko.Transport]:
    """Open an SFTP session to the compute VM using key auth if available, else password."""
    if paramiko is None:
        raise SystemExit("paramiko is not installed. pip install paramiko")
    host = _require("QUASAR_VM_HOST")
    user = _require("QUASAR_VM_USER")
    key_path = os.environ.get("QUASAR_VM_KEY")
    transport = paramiko.Transport((host, int(os.environ.get("QUASAR_VM_PORT", "22"))))
    if key_path:
        transport.connect(username=user, pkey=_load_key(key_path))
    else:
        transport.connect(username=user, password=_require("QUASAR_VM_PASS"))
    sftp = paramiko.SFTPClient.from_transport(transport)
    if sftp is None:
        raise SystemExit("could not open an SFTP channel")
    return sftp, transport


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _remote_sha256(sftp: paramiko.SFTPClient, remote: str) -> str:
    h = hashlib.sha256()
    with sftp.open(remote, "rb") as fh:
        fh.prefetch()
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _rmkdir(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    parts = [p for p in remote_dir.split("/") if p]
    cur = "/" if remote_dir.startswith("/") else ""
    for part in parts:
        cur = posixpath.join(cur, part) if cur else part
        try:
            sftp.stat(cur)
        except OSError:
            sftp.mkdir(cur)


def push(sftp: paramiko.SFTPClient, local_root: Path, remote_root: str) -> int:
    """Copy a local tree to the VM, verifying every file by checksum after transfer."""
    count = 0
    for path in sorted(local_root.rglob("*")):
        if any(part in SKIP for part in path.parts):
            continue
        rel = path.relative_to(local_root).as_posix()
        remote = posixpath.join(remote_root, rel)
        if path.is_dir():
            _rmkdir(sftp, remote)
            continue
        _rmkdir(sftp, posixpath.dirname(remote))
        sftp.put(str(path), remote)
        if _sha256(path) != _remote_sha256(sftp, remote):
            raise SystemExit(f"checksum mismatch after transfer: {rel}")
        count += 1
    return count


def pull(sftp: paramiko.SFTPClient, remote_root: str, local_root: Path) -> int:
    """Copy a VM tree to the archive, verifying every file by checksum after transfer."""
    count = 0
    local_root.mkdir(parents=True, exist_ok=True)
    try:
        entries = sftp.listdir_attr(remote_root)
    except OSError:
        return 0
    for entry in entries:
        if entry.filename in SKIP:
            continue
        remote = posixpath.join(remote_root, entry.filename)
        local = local_root / entry.filename
        if entry.st_mode is not None and stat.S_ISDIR(entry.st_mode):
            count += pull(sftp, remote, local)
        else:
            # getfo rather than get. paramiko's get() stats the local file afterwards and
            # compares sizes, and the Google Drive mount reports size lazily, so a
            # correctly transferred file raises "size mismatch" on a stale stat. Writing
            # through a handle skips that check, and the SHA-256 comparison below is a
            # stronger one anyway: it catches corruption, not just truncation.
            with local.open("wb") as handle:
                sftp.getfo(remote, handle)
            if _sha256(local) != _remote_sha256(sftp, remote):
                raise SystemExit(f"checksum mismatch after transfer: {remote}")
            count += 1
    return count


def main() -> None:
    _load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    action, subtree = sys.argv[1], sys.argv[2]

    vm_root = _require("QUASAR_VM_ROOT")
    archive = Path(_require("QUASAR_ARCHIVE"))

    sftp, transport = connect()
    try:
        if action == "up":
            n = push(sftp, archive / subtree, posixpath.join(vm_root, subtree))
            print(f"pushed {n} files to {vm_root}/{subtree}")
        elif action == "down":
            n = pull(sftp, posixpath.join(vm_root, subtree), archive / subtree)
            print(f"pulled {n} files to {archive / subtree}")
        elif action == "verify":
            local_files = {
                p.relative_to(archive / subtree).as_posix(): _sha256(p)
                for p in (archive / subtree).rglob("*")
                if p.is_file() and not any(part in SKIP for part in p.parts)
            }
            drift = []
            for rel, digest in sorted(local_files.items()):
                remote = posixpath.join(vm_root, subtree, rel)
                try:
                    if _remote_sha256(sftp, remote) != digest:
                        drift.append((rel, "differs"))
                except OSError:
                    drift.append((rel, "missing on VM"))
            if drift:
                for rel, why in drift:
                    print(f"DRIFT {why}: {rel}")
                raise SystemExit(1)
            print(f"verified {len(local_files)} files, no drift")
        else:
            raise SystemExit(f"unknown action {action!r}")
    finally:
        sftp.close()
        transport.close()


if __name__ == "__main__":
    main()
