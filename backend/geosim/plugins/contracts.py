"""The six extension-point contracts (doc 08 §4) + execution mode (doc 08 §2.1).

These are the *stable surface* a plugin imports (doc 08 §9): the six Protocols/specs
defined here, plus ``PropertyType`` (REUSED from ``geosim.spatial.property_types`` — doc
01 §5 / doc 08 §4b owns it; we do NOT redefine it). Each interface is fixed only to the
depth registration needs; the behavioural contract is owned by the sibling doc flagged
in its docstring.

Extension points (doc 08 §1 table):
  (a) ``IngestionAdapter``  — doc 03      (d) ``ForwardModel``     — doc 05
  (b) ``PropertyType``      — doc 01 §5   (e) ``RendererSpec``     — doc 06
  (c) ``Transform``         — doc 07      (f) ``InversionEngine``  — doc 10 (later)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

# (b) Property type — REUSE the doc 01 §5 declarative registry; never redefine (doc 08 §4b).
from geosim.spatial.property_types import PropertyType

__all__ = [
    "ExecutionMode",
    "PropertyType",
    "IngestionAdapter",
    "Transform",
    "ForwardModel",
    "InversionEngine",
    "TransferFunction",
    "RendererSpec",
]


class ExecutionMode(str, Enum):
    """Process-isolation axis, per contribution (doc 08 §2.1).

    Orthogonal to the (global) trust axis — NOT a security feature. Defaults to
    ``IN_PROCESS``; heavy/conflicting engines (SimPEG, PyGIMLi — doc 10) opt out.
    """

    IN_PROCESS = "in_process"          # DEFAULT — lightweight trusted contributions
    WORKER_PROCESS = "worker_process"  # heavy CPU jobs (RQ/Redis tier, doc 04)
    CONTAINER = "container"            # conflicting/heavy native deps in a dedicated image
    REMOTE_WORKER = "remote_worker"    # remote/GPU worker; only DTOs cross the boundary

    @classmethod
    def coerce(cls, value: str | ExecutionMode | None) -> ExecutionMode:
        """Coerce a declared value to an ``ExecutionMode``; ``None`` ⇒ default in_process."""
        if value is None:
            return cls.IN_PROCESS
        if isinstance(value, cls):
            return value
        return cls(value)  # raises ValueError on a non-canonical mode (doc 08 §8)


# --------------------------------------------------------------------------- (a) adapter


@runtime_checkable
class IngestionAdapter(Protocol):
    """Ingestion adapter — **[doc 03 binds here]** (doc 08 §4a).

    ``method``/``submethod`` MUST be a canonical pair from doc 02 §2 — never invented
    variants. Doc 03 owns parsing rules and the per-method format table; this fixes only
    the signature and that ``parse`` returns the OVERVIEW §3 normalized primitive.
    """

    method: str                 # canonical MethodKey (doc 02 §2): "gravity", "mt", ...
    submethod: str | None       # canonical submethod (doc 02 §2) or None
    formats: list[str]          # native format keys it claims (OVERVIEW §3)

    def sniff(self, raw: Any) -> float:
        """0..1 confidence it can parse ``raw`` (a RawFile)."""
        ...

    def parse(self, raw: Any, ctx: Any) -> Any:
        """Parse ``raw`` → NormalizedBundle (doc 03)."""
        ...


# ------------------------------------------------------------------------- (c) transform


@runtime_checkable
class Transform(Protocol):
    """Rock-physics transform — **[doc 07 binds here]** (doc 08 §4c).

    ``inputs``/``outputs`` are property-type keys; a transform may register a *new*
    output property type as part of its bundle. Doc 07 owns the maths + uncertainty
    propagation + the fused-grid resampling it runs on.
    """

    key: str
    inputs: list[str]           # property-type keys it consumes
    outputs: list[str]          # property-type keys it produces (often new ones)

    def apply(self, fields: dict[str, Any], params: dict) -> dict[str, Any]:
        """Apply the transform to ``fields`` (property-key → Field)."""
        ...


# --------------------------------------------------------------------- (d) forward model


@runtime_checkable
class ForwardModel(Protocol):
    """Forward model — **[doc 05 binds here]** (doc 08 §4d).

    Contract: given the synthetic earth, emit a native-format file the *same method's*
    adapter can ingest — closing the OVERVIEW §8 round-trip. Doc 05 owns the physics.
    """

    method: str                 # canonical MethodKey (doc 02 §2)
    submethod: str | None       # canonical submethod (doc 02 §2)

    def simulate(self, earth: Any, geom: Any, noise: Any) -> Any:
        """Simulate → RawFile (a native-format file, OVERVIEW §8)."""
        ...


# ------------------------------------------------------------------- (f) inversion engine


@runtime_checkable
class InversionEngine(Protocol):
    """Inversion engine — **[doc 10 binds here, later]** (doc 08 §4f, Phase 6).

    Listed now so the registry shape doesn't change later: inversion is *just another
    contribution type*, run as a background job. Heavy/conflicting engines declare
    ``executionMode: container | remote_worker`` (doc 08 §2.1).
    """

    key: str                    # "simpeg.dc", "pygimli.ert", "simpeg.joint"
    methods: list[str]          # canonical MethodKeys it can invert (doc 02 §2)

    def invert(self, observations: list, mesh: Any, config: dict, job: Any) -> Any:
        """Run the inversion → PropertyModel, as a job."""
        ...


# ------------------------------------------------------------- (e) renderer / transfer fn


@dataclass(frozen=True)
class TransferFunction:
    """Opacity/colour ramp + isovalue defaults for a renderer (doc 06, declarative).

    Serialized into ``/capabilities`` so the frontend (which owns the implementation)
    can configure the matching client renderer (doc 08 §7.2).
    """

    colormap: str = "viridis"
    scaling: str = "linear"                      # "linear" | "log"
    opacity_curve: tuple[tuple[float, float], ...] = ()  # alpha control points [(v, a), ...]
    isovalues: tuple[float, ...] = ()
    value_range: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "colormap": self.colormap,
            "scaling": self.scaling,
            "opacity_curve": [list(p) for p in self.opacity_curve],
            "isovalues": list(self.isovalues),
            "value_range": list(self.value_range) if self.value_range is not None else None,
        }


@dataclass(frozen=True)
class RendererSpec:
    """Renderer / transfer function — **[doc 06 binds here]** (doc 08 §4e, declarative).

    Backend-registered as a *serializable spec* so the frontend discovers it via
    ``/capabilities``; the *implementation* is frontend code resolved by ``key`` from a
    fixed client catalog (doc 08 §7.2). Unknown keys degrade gracefully on the client.
    """

    key: str                                     # "volume.raymarch", "wellpath.tube", ...
    applies_to: list[str] = field(default_factory=list)  # property keys / primitive kinds
    default_transfer_function: TransferFunction = field(default_factory=TransferFunction)
    ui_panel: str | None = None                  # optional custom React panel id

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "applies_to": list(self.applies_to),
            "default_transfer_function": self.default_transfer_function.to_dict(),
            "ui_panel": self.ui_panel,
        }
