"""The ingestion adapter contract + pre-normalization ``Raw*`` twins (doc 03 §1, §6).

An adapter does exactly two things (doc 03 §1): **declare what it can handle**
(``sniff``) and **parse raw bytes into a** :class:`ParseResult` (``parse``). It never
touches storage, the catalog, or coordinate transforms — the pipeline (doc 03 §7,
:mod:`geosim.ingestion.pipeline`) and normalizer (doc 03 §3, :mod:`.normalize`) do that.

The :class:`RawObservation` / :class:`RawPropertyModel` / :class:`RawFeature`
dataclasses are the **pre-normalization** twins of the doc-02 primitives
(``ObservationSet`` / ``PropertyModel`` / ``GeologicalFeature``). They carry
**native-frame** coordinates (in :class:`SourceRef`'s CRS/datum) and **native units**
(declared per-property in :attr:`ParseResult.units`). The normalizer (doc 03 §3) turns
them into the canonical doc-02 primitives in the Engineering Frame with canonical units.

Geometry is classified by the **frozen vocabulary** (doc 02 §3–§4): observations by
``geometry_kind`` ∈ ``points | soundings | profile2d | traces | raster2d | wellcurve |
tensor``; property models by ``support`` ∈ ``volume | grid2d | section | mesh``.

The :class:`IngestionAdapter` here is the doc-03 *authoring* protocol (richer
``sniff(sample, filename)`` / ``parse(source)`` signatures). It is structurally
compatible with — and registered through — the unified ``geosim.plugins`` registry
(doc 08); :mod:`.registry` adapts between the two surfaces.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "GEOMETRY_KINDS",
    "SUPPORT_KINDS",
    "Severity",
    "IngestStatus",
    "SourceRef",
    "RawSource",
    "RawObservation",
    "RawPropertyModel",
    "RawFeature",
    "Provenance",
    "IngestWarning",
    "ParseResult",
    "IngestReport",
    "IngestionAdapter",
]

# Frozen geometry / support vocabularies (doc 02 §3–§4, doc 03 §10).
GEOMETRY_KINDS: frozenset[str] = frozenset(
    {"points", "soundings", "profile2d", "traces", "raster2d", "wellcurve", "tensor"}
)
SUPPORT_KINDS: frozenset[str] = frozenset({"volume", "grid2d", "section", "mesh"})


class Severity(str, Enum):
    """Warning severity (doc 03 §6). ``high`` is reserved for silent-wrong-unit class
    issues (missing units → canonical-unit assumption is the worst failure mode)."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class IngestStatus(str, Enum):
    """Terminal ingest status (doc 03 §6)."""

    OK = "ok"
    OK_WITH_WARNINGS = "ok_with_warnings"
    FAILED = "failed"


@dataclass
class SourceRef:
    """Where a quantity's coordinates/units live BEFORE normalization (doc 03 §1).

    Consumed by doc 01's ``frame.to_engineering()`` + ``units.to_canonical()``. The
    adapter declares this; it does **not** reproject or unit-convert itself (doc 03 §2).
    """

    crs: str | None            # EPSG code, WKT2, or None if unknown/local
    vertical_datum: str | None = None  # EPSG, "ellipsoidal", "local", or None
    horizontal_unit: str = "m"  # e.g. "m", "deg", "ft"
    # "elevation_up" | "depth_below_surface" | "depth_below_datum" | "MD" (doc 03 §3a)
    z_convention: str = "elevation_up"


@dataclass
class RawSource:
    """The raw bytes + identity handed to an adapter (doc 03 §1, §7 step 2).

    Content-addressed by :attr:`sha256`; the adapter reads :attr:`data` (or
    :attr:`path` for streaming large files) and uses :attr:`filename` for extension
    sniffing. The pipeline fills this from the raw store (doc 04 §3).
    """

    filename: str
    data: bytes | None = None
    path: str | None = None
    sha256: str | None = None
    media_type: str | None = None
    # Upload-time user hints (doc 03 §7 step 1): override detection / supply CRS.
    method_hint: str | None = None
    crs_hint: str | None = None

    def sample(self, n: int = 65536) -> bytes:
        """A cheap header sample for ``sniff()`` (doc 03 §7 step 3)."""
        if self.data is not None:
            return self.data[:n]
        if self.path is not None:
            with open(self.path, "rb") as fh:
                return fh.read(n)
        return b""


@dataclass
class RawObservation:
    """Pre-normalization twin of doc-02 ``ObservationSet`` (doc 03 §1, §10).

    Coordinates are native-frame ``(N, 3)`` in :class:`SourceRef`'s CRS/units; values
    are native-unit columns keyed by ``property_type`` (doc 01 §5 key). Paired sigma
    columns ride in :attr:`sigma` (``role:"sigma"``, doc 02 §3). ``acquired_at`` (ISO-8601
    UTC, doc 02 §1/§8) is set for 4D methods.
    """

    geometry_kind: str
    coords: Any                                   # (N, 3) array-like, native frame
    values: dict[str, Any] = field(default_factory=dict)  # property_type -> (N,) values
    sigma: dict[str, Any] = field(default_factory=dict)   # property_type -> (N,) 1σ
    primary_property: str | None = None
    acquired_at: Sequence[str] | None = None      # ISO-8601 UTC per record (4D, doc 02 §8)
    meta: dict[str, Any] = field(default_factory=dict)  # methodData/acquisition blob


@dataclass
class RawPropertyModel:
    """Pre-normalization twin of doc-02 ``PropertyModel`` — an already-inverted/gridded
    field (doc 03 §1, §2). ``support`` ∈ :data:`SUPPORT_KINDS`.

    ``values`` is the native-unit field array (``[z, y, x]`` Z-up for ``volume``);
    ``origin``/``spacing`` are native-frame ``(z, y, x)``. ``sigma`` is the co-registered
    per-cell 1σ (doc 02 §6) when present.
    """

    property: str
    values: Any                                   # ndarray, native units
    origin: tuple[float, float, float]            # (z, y, x) native frame
    spacing: tuple[float, float, float]           # (z, y, x)
    support: str = "volume"
    sigma: Any | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RawFeature:
    """Pre-normalization twin of doc-02 ``GeologicalFeature`` (doc 03 §1, §5).

    Geometry is native-frame vertices/rings; the normalizer reprojects + drapes to the
    surface model where Z is absent (geology maps are 2.5D, doc 03 §5).
    """

    feature_type: str
    geometry: Any                                 # native-frame geometry (GeoJSON-like)
    props: dict[str, Any] = field(default_factory=dict)
    store_format: str = "geojson"


@dataclass
class Provenance:
    """Adapter-supplied provenance seed (doc 02 §7, populated by the pipeline, doc 03 §8).

    The adapter records *what it knows* (source unit/CRS, parse params); the pipeline
    adds the raw-file hash, adapter name+version, and normalization params before writing
    the catalog ``provenance`` row.
    """

    process: str = "ingest"                       # doc 02 §7 Step op, e.g. "ingest:ert-stg"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestWarning:
    """A structured ingest warning (doc 03 §6)."""

    code: str
    severity: Severity = Severity.LOW
    message: str = ""
    locus: str | None = None                      # e.g. "row 41", "property:resistivity"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "locus": self.locus,
        }


@dataclass
class ParseResult:
    """Everything an adapter emits from one file (doc 03 §1, §10 decision #1).

    All coords/units below stay in the file's NATIVE crs/datum/units, declared via
    :attr:`source` (default frame) and :attr:`units` (property -> source unit). The
    pipeline normalizes in place. :attr:`records_total` / :attr:`records_dropped` feed the
    >10% partial-file rule (doc 03 §6).
    """

    observations: list[RawObservation] = field(default_factory=list)
    property_models: list[RawPropertyModel] = field(default_factory=list)
    features: list[RawFeature] = field(default_factory=list)
    source: SourceRef | None = None
    units: dict[str, str] = field(default_factory=dict)
    provenance: Provenance | None = None
    warnings: list[IngestWarning] = field(default_factory=list)
    records_total: int = 0
    records_dropped: int = 0

    def is_empty(self) -> bool:
        return not (self.observations or self.property_models or self.features)


@dataclass
class IngestReport:
    """The structured report produced by every ingest (doc 03 §6, stored + shown in UI).

    Holds primitive counts, the warning list, the drop ratio, and the terminal
    :attr:`status`. The >10%-records-dropped rule escalates to ``failed`` (doc 03 §6/§10
    decision #7).
    """

    status: IngestStatus = IngestStatus.OK
    n_observations: int = 0
    n_property_models: int = 0
    n_features: int = 0
    records_total: int = 0
    records_dropped: int = 0
    warnings: list[IngestWarning] = field(default_factory=list)
    dataset_id: str | None = None
    project_id: str | None = None
    raw_file_id: str | None = None
    idempotency_key: str | None = None
    reused: bool = False                          # idempotent re-ingest hit (doc 03 §8)
    message: str | None = None

    # >10% of records dropped escalates the whole file to failed (doc 03 §6/§10 #7).
    DROP_FAIL_THRESHOLD: float = 0.10

    @property
    def drop_ratio(self) -> float:
        if self.records_total <= 0:
            return 0.0
        return self.records_dropped / self.records_total

    def add_warning(self, w: IngestWarning) -> None:
        self.warnings.append(w)

    def finalize(self, *, drop_threshold: float | None = None) -> IngestStatus:
        """Compute the terminal status from drops + warnings (doc 03 §6).

        - Already ``failed`` (header/identity error) stays failed.
        - ``> drop_threshold`` of records dropped → ``failed`` (doc 03 §10 #7).
        - Any warning → ``ok_with_warnings``; otherwise ``ok``.
        """
        thr = self.DROP_FAIL_THRESHOLD if drop_threshold is None else drop_threshold
        if self.status is IngestStatus.FAILED:
            return self.status
        if self.records_total > 0 and self.drop_ratio > thr:
            self.status = IngestStatus.FAILED
            if not self.message:
                self.message = (
                    f"{self.records_dropped}/{self.records_total} records dropped "
                    f"({self.drop_ratio:.0%} > {thr:.0%}) — escalated to failed (doc 03 §6)"
                )
            return self.status
        self.status = (
            IngestStatus.OK_WITH_WARNINGS if self.warnings else IngestStatus.OK
        )
        return self.status

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "counts": {
                "observations": self.n_observations,
                "property_models": self.n_property_models,
                "features": self.n_features,
            },
            "records_total": self.records_total,
            "records_dropped": self.records_dropped,
            "drop_ratio": self.drop_ratio,
            "warnings": [w.to_dict() for w in self.warnings],
            "dataset_id": self.dataset_id,
            "project_id": self.project_id,
            "raw_file_id": self.raw_file_id,
            "idempotency_key": self.idempotency_key,
            "reused": self.reused,
            "message": self.message,
        }


@runtime_checkable
class IngestionAdapter(Protocol):
    """The doc-03 §1 ingestion-adapter authoring contract.

    Declares what it can handle and parses raw bytes → :class:`ParseResult`. It does
    **not** reproject, unit-convert, write, or register (doc 03 §1–§2). ``method`` /
    ``submethod`` MUST be a canonical pair from doc 02 §2 (enforced at registration,
    doc 08 §8). The richer ``sniff(sample, filename)`` / ``parse(source)`` signatures are
    bridged to the plugins-registry surface by :mod:`.registry`.
    """

    method: str                      # canonical MethodKey (doc 02 §2)
    submethod: str | None            # canonical submethod (doc 02 §2) or None
    name: str                        # unique adapter id, e.g. "ert-stg-v1"
    version: str                     # adapter version (idempotency key, doc 03 §8)
    extensions: Sequence[str]        # [".stg", ".dat"]
    media_types: Sequence[str]       # optional MIME hints

    def sniff(self, sample: bytes, filename: str) -> float:
        """Confidence in [0,1] that this adapter handles the file (cheap header check)."""
        ...

    def parse(self, source: RawSource) -> ParseResult:
        """Full parse → native-frame :class:`ParseResult` (doc 03 §1, §7 step 4)."""
        ...


def as_xyz(coords: Any) -> np.ndarray:
    """Coerce coords to a float ``(N, 3)`` array, padding a missing Z with 0 (doc 03 §3)."""
    arr = np.asarray(coords, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] == 2:
        arr = np.column_stack([arr, np.zeros(len(arr))])
    if arr.shape[1] != 3:
        raise ValueError(f"coords must be (N,2) or (N,3); got shape {arr.shape}")
    return arr
