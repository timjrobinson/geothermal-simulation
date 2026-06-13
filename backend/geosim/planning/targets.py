"""Drill targets + model enrichment (doc 09 §3).

A :class:`DrillTarget` is the subsurface volume we want the well to reach (doc 09 §3.1):
a **point** bullseye + tolerance, or a **zone** (centroid + reference to the producing
feature/isosurface). At creation the target is **enriched** by a single points-mode fused
sample at the bullseye (doc 09 §3.2): temperature, favorability, lithology and their
uncertainties (doc 07) are stamped on, tied to ``model_version``. If the fused model is
re-derived, the cached snapshot's ``model_version`` no longer matches → the target is
**stale** (doc 09 §3.3 "re-sample" badge), rather than silently drifting.

Enrichment REUSES the fused-grid trilinear path sampler (the same one
:func:`geosim.fusion.sample_path` uses) at a single point, plus the shared σ→confidence
mapping from :mod:`.predict`, so a target's numbers agree with the predicted log's.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from geosim.catalog import FusedModel
from geosim.spatial import to_display

from ._sampling import sample_layers_with_sigma
from .predict import sigma_to_confidence

__all__ = [
    "DrillTarget",
    "TargetTolerance",
    "SampledValue",
    "TargetEnrichment",
    "enrich_target",
]


@dataclass
class TargetTolerance:
    """Acceptable miss around the bullseye (doc 09 §3.3)."""

    radius_m: float = 50.0
    tvd_window_m: float = 25.0


@dataclass
class SampledValue:
    """A sampled scalar + its uncertainty (doc 09 §3.3 ``sampled.*``)."""

    value: float | None
    sigma: float | None = None
    confidence: float | None = None

    def to_payload(self) -> dict:
        return {"value": self.value, "sigma": self.sigma, "confidence": self.confidence}


@dataclass
class TargetEnrichment:
    """The model snapshot stamped on a target at creation (doc 09 §3.3 ``sampled``)."""

    temperature_c: SampledValue | None = None
    favorability: SampledValue | None = None
    lithology: str | None = None
    depth_tvd_m: float | None = None
    model_version: str | None = None
    properties: dict[str, SampledValue] = field(default_factory=dict)

    def to_payload(self) -> dict:
        return {
            "temperatureC": self.temperature_c.to_payload() if self.temperature_c else None,
            "favorability": self.favorability.to_payload() if self.favorability else None,
            "lithology": self.lithology,
            "depthTVD_m": self.depth_tvd_m,
            "modelVersion": self.model_version,
            "properties": {k: v.to_payload() for k, v in self.properties.items()},
        }


@dataclass
class DrillTarget:
    """A drilling target feature (doc 09 §3.3).

    ``location`` is Engineering ``(x, y, z)`` metres (z canonical, +up). ``kb_elev_m`` is the
    reference elevation used to derive the cached ``depthTVD_m`` view. The ``enrichment``
    snapshot is filled by :func:`enrich_target` and carries ``model_version`` for stale
    detection (doc 09 §3.3).
    """

    id: str
    name: str
    project_id: str
    kind: str  # "point" | "zone"
    location: tuple[float, float, float]
    tolerance: TargetTolerance = field(default_factory=TargetTolerance)
    desired_temperature_c: float | None = None
    min_temperature_c: float | None = None
    geological_unit: str | None = None
    rationale: str | None = None
    zone_ref: dict | None = None
    kb_elev_m: float = 0.0
    enrichment: TargetEnrichment | None = None
    provenance: dict = field(default_factory=dict)

    def is_stale(self, current_model_version: str) -> bool:
        """True if the cached enrichment was sampled against a different fused model (§3.3)."""
        if self.enrichment is None:
            return True
        return self.enrichment.model_version != current_model_version

    def to_payload(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "projectId": self.project_id,
            "kind": self.kind,
            "location": {"x": self.location[0], "y": self.location[1], "z": self.location[2]},
            "tolerance": asdict(self.tolerance),
            "desiredTemperatureC": self.desired_temperature_c,
            "minTemperatureC": self.min_temperature_c,
            "geologicalUnit": self.geological_unit,
            "rationale": self.rationale,
            "zoneRef": self.zone_ref,
            "sampled": self.enrichment.to_payload() if self.enrichment else None,
            "provenance": self.provenance,
        }


def enrich_target(
    session: Session,
    fem: FusedModel,
    target: DrillTarget,
    *,
    storage_root: str | Path | None = None,
    properties: list[str] | None = None,
) -> DrillTarget:
    """Stamp the points-mode fused enrichment onto ``target`` (doc 09 §3.2).

    One trilinear sample (value + σ) at the bullseye across the resampled fused layers.
    Temperature is converted to °C for the display-facing ``temperatureC`` field (canonical
    K internally, doc 09 §5 units note); favorability/other props are carried natively.
    Lithology is read from a ``lithology_class`` layer when present (nearest class label).
    The snapshot is tied to ``fem.id`` as ``model_version`` for stale detection (§3.3).
    """
    x, y, z = (float(c) for c in target.location)
    point_zyx = np.array([[z, y, x]], dtype=float)
    sampled = sample_layers_with_sigma(
        session, fem, point_zyx, properties=properties, storage_root=storage_root
    )

    enrichment = TargetEnrichment(model_version=fem.id)
    enrichment.depth_tvd_m = target.kb_elev_m - z

    for prop, (vals, sig) in sampled.items():
        v = float(vals[0]) if np.isfinite(vals[0]) else None
        s = float(sig[0]) if (sig is not None and np.isfinite(sig[0])) else None
        conf = sigma_to_confidence(prop, v, s)
        sv = SampledValue(value=v, sigma=s, confidence=conf)
        if prop == "temperature":
            tc = None if v is None else float(to_display(v, "temperature"))
            sc = None if s is None else float(s)  # σ is a difference: K and °C magnitudes match
            enrichment.temperature_c = SampledValue(value=tc, sigma=sc, confidence=conf)
        elif prop == "favorability":
            enrichment.favorability = sv
        elif prop == "lithology_class":
            enrichment.lithology = None if v is None else str(int(round(v)))
        else:
            enrichment.properties[prop] = sv

    target.enrichment = enrichment
    return target
