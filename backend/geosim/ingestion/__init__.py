"""Ingestion pipeline & adapter registry (doc 03 ENTIRELY + doc 08 registry).

Every byte that enters the model passes through an **adapter** (doc 03 §1): a pure
format reader declaring what it can handle (``sniff``) and parsing raw bytes into a
:class:`ParseResult` of native-frame, native-unit ``Raw*`` primitives. The shared
pipeline then runs identically for every method (doc 03 §3, §7):

    upload → store-raw → detect → parse → normalize → write → register

- :mod:`.base` — the :class:`IngestionAdapter` Protocol + the pre-normalization
  ``Raw*`` twins (:class:`SourceRef`, :class:`RawSource`, :class:`RawObservation`,
  :class:`RawPropertyModel`, :class:`RawFeature`, :class:`ParseResult`) +
  :class:`IngestWarning`/:class:`IngestReport` (doc 03 §1, §6).
- :mod:`.registry` — the adapter registry **integrated with** ``geosim.plugins`` (doc 08):
  the :func:`adapter` decorator, entry-point discovery, and ``sniff()``-based
  :func:`detect` (doc 03 §1, §7 step 3).
- :mod:`.normalize` — the post-parse pipeline: CRS+vertical → Engineering, units →
  canonical, 1D/2D→3D placement, delegated to ``geosim.spatial`` (doc 03 §3).
- :mod:`.writer` — write via ``geosim.storage`` + register catalog rows (doc 03 §7 6–7).
- :mod:`.pipeline` — :func:`ingest_file`, the high-level orchestrator (inline, RQ-ready)
  with sha256+adapter+version+params idempotency (doc 03 §8).
- :mod:`.adapters` — first-party adapters; the package auto-imports its siblings so a new
  adapter is one self-registering file (doc 03 §9).

The M1 ``write + register`` slice (:func:`seed_m1_project`, :mod:`.seed`) is retained.
"""

from __future__ import annotations

from . import adapters as _adapters  # noqa: F401 — triggers first-party adapter auto-registration
from .base import (
    GEOMETRY_KINDS,
    SUPPORT_KINDS,
    IngestionAdapter,
    IngestReport,
    IngestStatus,
    IngestWarning,
    ParseResult,
    Provenance,
    RawFeature,
    RawObservation,
    RawPropertyModel,
    RawSource,
    Severity,
    SourceRef,
)
from .normalize import (
    NormalizationError,
    NormalizedBundle,
    NormFeature,
    NormObservation,
    NormParams,
    NormPropertyModel,
    normalize,
)
from .pipeline import frame_for_bundle, idempotency_key, ingest_file
from .registry import (
    DetectionError,
    adapter,
    adapter_named,
    adapters,
    adapters_for,
    detect,
    discover_entry_points,
    register_adapter,
)
from .seed import seed_m1_project
from .writer import WriteContext, write_and_register

__all__ = [
    # adapter contract + Raw* twins (doc 03 §1)
    "IngestionAdapter", "SourceRef", "RawSource",
    "RawObservation", "RawPropertyModel", "RawFeature", "ParseResult", "Provenance",
    "GEOMETRY_KINDS", "SUPPORT_KINDS",
    # report / warnings (doc 03 §6)
    "IngestReport", "IngestStatus", "IngestWarning", "Severity",
    # registry (doc 03 §1, doc 08)
    "adapter", "register_adapter", "adapters", "adapters_for", "adapter_named",
    "detect", "discover_entry_points", "DetectionError",
    # normalize (doc 03 §3)
    "normalize", "NormalizedBundle", "NormObservation", "NormPropertyModel",
    "NormFeature", "NormParams", "NormalizationError",
    # write + register (doc 03 §7)
    "WriteContext", "write_and_register",
    # pipeline (doc 03 §7, §8)
    "ingest_file", "idempotency_key", "frame_for_bundle",
    # M1 slice (kept working)
    "seed_m1_project",
]
