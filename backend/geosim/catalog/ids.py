"""Kind-prefixed ULID identifiers (doc 02 §1, doc 04 §2.4).

Every catalog primary key is a ``TEXT`` ULID — time-ordered, sortable, and
prefixed by a short token naming its kind so an id is self-describing in logs,
URLs, and provenance edges (doc 02 §1 reserves the prefix set). The prefix is the
``<token>_`` part; the remainder is a canonical 26-char Crockford-base32 ULID.

Prefixes (doc 02 §1 / doc 04 §2.4 *Reconciliation*):
``prj_ ds_ obs_ pm_ feat_ fem_ prov_ well_ run_ ver_`` plus the layer/view/job/
raw-file/fused-layer/spatial-frame ids this catalog needs.
"""

from __future__ import annotations

from enum import StrEnum

import ulid

__all__ = ["IdKind", "new_id", "prefix_of", "is_kind"]


class IdKind(StrEnum):
    """The kind tokens that prefix a ULID primary key (doc 02 §1)."""

    PROJECT = "prj"
    DATASET = "ds"
    OBSERVATION = "obs"
    PROPERTY_MODEL = "pm"
    FEATURE = "feat"
    FUSED_MODEL = "fem"
    FUSED_LAYER = "flay"
    PROVENANCE = "prov"
    WELL = "well"
    RUN = "run"
    VERSION = "ver"
    RAW_FILE = "raw"
    LAYER = "lyr"
    VIEW = "view"
    JOB = "job"


def new_id(kind: IdKind | str) -> str:
    """Mint a fresh kind-prefixed ULID, e.g. ``ds_01J9Z3...`` (doc 02 §1)."""
    token = kind.value if isinstance(kind, IdKind) else str(kind)
    return f"{token}_{ulid.ULID()}"


def prefix_of(identifier: str) -> str:
    """Return the kind token of an id (the part before the first ``_``)."""
    token, _, _ = identifier.partition("_")
    return token


def is_kind(identifier: str, kind: IdKind | str) -> bool:
    """True if ``identifier`` carries the ``kind`` prefix."""
    token = kind.value if isinstance(kind, IdKind) else str(kind)
    return prefix_of(identifier) == token
