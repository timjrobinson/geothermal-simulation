"""Content-addressed raw store (doc 04 §3, §8.1).

Verbatim source files are stored immutably at ``raw/<sha256>/<original_name>``,
keyed by the whole-file SHA-256. Re-uploading identical bytes de-dups to the same
``<sha256>`` directory (doc 04 §8.1: "re-upload of identical bytes de-dups"). The
returned hash is the provenance root (`provenance.raw_file_id` → here, doc 02 §9).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

__all__ = ["RawRef", "sha256_bytes", "RawStore"]


def sha256_bytes(data: bytes) -> str:
    """Whole-content SHA-256 hex digest — the content address (doc 04 §8.1)."""
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class RawRef:
    """A stored raw file: its content address and on-disk path (doc 04 §8.1)."""

    sha256: str
    path: Path


class RawStore:
    """Immutable content-addressed store rooted at a ``raw/`` directory (doc 04 §3)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put_bytes(self, name: str, data: bytes) -> RawRef:
        """Store ``data`` under its SHA-256; identical bytes de-dup (doc 04 §8.1).

        The file lands at ``raw/<sha256>/<name>``. Writing identical content again
        is a no-op (the bytes are immutable) and returns the same ``sha256`` path.
        """
        digest = sha256_bytes(data)
        target_dir = self.root / digest
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / name
        if not target.exists():
            # immutable: write once, never mutate in place (doc 04 §8.1)
            target.write_bytes(data)
        return RawRef(digest, target)

    def put_file(self, src: str | Path, name: str | None = None) -> RawRef:
        """Store a file from disk by content address (doc 04 §8.1)."""
        src = Path(src)
        return self.put_bytes(name or src.name, src.read_bytes())

    def path_for(self, sha256: str, name: str) -> Path:
        """The canonical on-disk path for ``(sha256, name)`` — ``raw/<sha256>/<name>``."""
        return self.root / sha256 / name

    def get_bytes(self, sha256: str, name: str) -> bytes:
        """Read back stored content by its address (doc 04 §8.1)."""
        return self.path_for(sha256, name).read_bytes()
