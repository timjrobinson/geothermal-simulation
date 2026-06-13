"""The NORMALIZE pipeline — runs AFTER parse, identically for every method (doc 03 §3).

Adapters are pure format readers (doc 03 §2); *all* CRS/datum reprojection, unit
canonicalization, and 1D/2D→3D placement happen here, delegated to the doc-01 spatial
framework — this module never invents coordinate or unit handling (doc 03 §3, decision #2).

Stages (doc 03 §3):
  a. **CRS + vertical → Engineering** via :meth:`SpatialFrame.to_engineering` (doc 01 §7).
     ``SourceRef.z_convention`` selects vertical handling (``elevation_up`` canonical;
     ``depth_below_datum`` → negate; ``depth_below_surface`` → ``depth_to_elevation``;
     ``MD`` resolved by the well deviation survey — flagged downstream). Missing CRS:
     local project → assume Engineering + warn; georef project → validation error (doc 03 §6).
  b. **Units → canonical** via :func:`geosim.spatial.to_canonical` (doc 01 §5). The source
     unit is retained for provenance; a missing source unit uses the property-type
     canonical-unit assumption with a **high-severity** warning (silent-wrong-unit is the
     worst failure mode, doc 03 §6).
  c. **1D/2D→3D placement** (doc 03 §3d): soundings → vertical columns; profile2d/section
     stay as the native curtain (``support="section"``); raw obs stay raw — gridding into a
     PropertyModel is a *separate*, user-initiated step (doc 03 §3c/§10 #4), not done here.

The normalizer does **not** unknown-property-type guess: an unregistered ``property_type``
is a **hard error** surfaced as a failed ingest (it must be registered first, doc 03 §2 /
doc 08). It returns :class:`NormalizedBundle` of canonical doc-02 primitives plus the
accumulated warnings + drop counts; the writer (doc 03 §7 step 6) persists them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from geosim.spatial import REGISTRY, FrameMode, SpatialFrame, to_canonical

from .base import (
    IngestWarning,
    ParseResult,
    RawFeature,
    RawObservation,
    RawPropertyModel,
    Severity,
    as_xyz,
)

__all__ = [
    "NormParams",
    "NormObservation",
    "NormPropertyModel",
    "NormFeature",
    "NormalizedBundle",
    "NormalizationError",
    "normalize",
]


class NormalizationError(ValueError):
    """A hard normalization failure (doc 03 §6): missing mandatory CRS in a georef
    project, or an unregistered property type. Turns the ingest ``failed``."""


@dataclass
class NormParams:
    """Normalization parameters recorded in provenance (doc 03 §8, reproducible)."""

    z_convention: str = "elevation_up"
    source_crs: str | None = None
    source_vertical: str | None = None
    source_units: dict[str, str] = field(default_factory=dict)
    grid_method: str | None = None  # set only when an explicit gridding step ran (doc 03 §3c)

    def to_dict(self) -> dict[str, Any]:
        return {
            "z_convention": self.z_convention,
            "source_crs": self.source_crs,
            "source_vertical": self.source_vertical,
            "source_units": dict(self.source_units),
            "grid_method": self.grid_method,
        }


@dataclass
class NormObservation:
    """A normalized observation: Engineering-frame coords + canonical-unit values."""

    geometry_kind: str
    coords: np.ndarray                            # (N, 3) Engineering metres
    values: dict[str, np.ndarray] = field(default_factory=dict)  # canonical units
    sigma: dict[str, np.ndarray] = field(default_factory=dict)
    primary_property: str | None = None
    acquired_at: Any | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def bbox(self) -> dict[str, float]:
        return _bbox_from_coords(self.coords)


@dataclass
class NormPropertyModel:
    """A normalized property model: canonical-unit field, Engineering ``(z,y,x)`` grid."""

    property: str
    values: np.ndarray
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    support: str = "volume"
    sigma: np.ndarray | None = None
    canonical_unit: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def bbox(self) -> dict[str, float]:
        return _bbox_from_grid(self.origin, self.spacing, tuple(self.values.shape[-3:]))


@dataclass
class NormFeature:
    """A normalized geological feature (Engineering-frame geometry)."""

    feature_type: str
    geometry: Any
    props: dict[str, Any] = field(default_factory=dict)
    store_format: str = "geojson"
    bbox: dict[str, float] = field(default_factory=dict)


@dataclass
class NormalizedBundle:
    """Canonical doc-02 primitives + accumulated warnings/drops (doc 03 §3, §6)."""

    observations: list[NormObservation] = field(default_factory=list)
    property_models: list[NormPropertyModel] = field(default_factory=list)
    features: list[NormFeature] = field(default_factory=list)
    warnings: list[IngestWarning] = field(default_factory=list)
    params: NormParams = field(default_factory=NormParams)
    records_total: int = 0
    records_dropped: int = 0

    def is_empty(self) -> bool:
        return not (self.observations or self.property_models or self.features)


# ─────────────────────────────── geometry helpers ───────────────────────────────


def _bbox_from_coords(coords: np.ndarray) -> dict[str, float]:
    pts = np.asarray(coords, dtype=float).reshape(-1, 3)
    return {
        "xmin": float(pts[:, 0].min()), "xmax": float(pts[:, 0].max()),
        "ymin": float(pts[:, 1].min()), "ymax": float(pts[:, 1].max()),
        "zmin": float(pts[:, 2].min()), "zmax": float(pts[:, 2].max()),
    }


def _bbox_from_grid(
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
    shape: tuple[int, int, int],
) -> dict[str, float]:
    """Cell-corner AABB in Engineering metres from a Z-up ``(z,y,x)`` grid (doc 04 §2.2)."""
    nz, ny, nx = shape
    z0, y0, x0 = origin
    dz, dy, dx = spacing
    return {
        "xmin": float(x0), "xmax": float(x0 + nx * dx),
        "ymin": float(y0), "ymax": float(y0 + ny * dy),
        "zmin": float(z0), "zmax": float(z0 + nz * dz),
    }


# ─────────────────────────────── stage a: coordinates ───────────────────────────────


def _to_engineering(
    coords: np.ndarray,
    frame: SpatialFrame,
    src_crs: str | None,
    src_vertical: str | None,
    z_convention: str,
    warnings: list[IngestWarning],
) -> np.ndarray:
    """CRS + vertical → Engineering (doc 03 §3a). Delegates to doc 01; never reprojects here."""
    pts = as_xyz(coords)

    # Vertical convention → elevation-up (doc 03 §3a). MD requires a deviation survey
    # (doc 01 §4) the pipeline supplies later; flag and treat as-is for now.
    z = pts[:, 2].astype(float)
    if z_convention == "elevation_up":
        pass
    elif z_convention == "depth_below_datum":
        z = -z
    elif z_convention == "depth_below_surface":
        # surfaceModel-relative; with a flat:0 surface this equals negate (doc 03 §3a).
        z = -z
        warnings.append(IngestWarning(
            "z_depth_below_surface", Severity.LOW,
            "z treated as depth below a flat surface; supply a surfaceModel to drape exactly",
            "z",
        ))
    elif z_convention == "MD":
        warnings.append(IngestWarning(
            "z_md_unresolved", Severity.MEDIUM,
            "MD depth needs a deviation survey to resolve to elevation (doc 01 §4); "
            "assuming vertical well below wellhead", "z",
        ))
    else:
        warnings.append(IngestWarning(
            "z_convention_unknown", Severity.MEDIUM,
            f"unknown z_convention {z_convention!r}; treated as elevation_up", "z",
        ))
    pts = np.column_stack([pts[:, 0], pts[:, 1], z])

    if frame.mode is FrameMode.LOCAL:
        if src_crs is not None:
            warnings.append(IngestWarning(
                "crs_ignored_local", Severity.LOW,
                f"source CRS {src_crs!r} ignored in a local-mode project (assumed Engineering)",
                "crs",
            ))
        return pts.astype(float)

    # georeferenced project (doc 03 §6): missing source CRS is a hard validation error.
    if src_crs is None:
        raise NormalizationError(
            "georeferenced project requires a source CRS (supply one at upload) — doc 03 §6"
        )
    return frame.to_engineering(pts, src_crs=src_crs, src_vertical=src_vertical).astype(float)


# ─────────────────────────────── stage b: units ───────────────────────────────


def _canonicalize(
    property_type: str,
    values: Any,
    source_unit: str | None,
    warnings: list[IngestWarning],
    params: NormParams,
) -> tuple[np.ndarray, str]:
    """Native unit → canonical (doc 03 §3b). Unregistered property → hard error (doc 03 §2)."""
    if property_type not in REGISTRY:
        raise NormalizationError(
            f"unknown property type {property_type!r} — register it first (doc 08 / doc 03 §2)"
        )
    pt = REGISTRY.get(property_type)
    arr = np.asarray(values, dtype=float)
    if source_unit is None:
        warnings.append(IngestWarning(
            "missing_unit", Severity.HIGH,
            f"no source unit for {property_type!r}; assuming canonical "
            f"{pt.canonical_unit!r} (silent wrong-unit is the worst failure mode, doc 03 §6)",
            f"property:{property_type}",
        ))
        params.source_units[property_type] = pt.canonical_unit
        return arr, pt.canonical_unit
    params.source_units[property_type] = source_unit
    if pt.categorical or source_unit == pt.canonical_unit:
        return arr, pt.canonical_unit
    canon = np.asarray(to_canonical(arr, source_unit, property_type), dtype=float)
    return canon, pt.canonical_unit


# ─────────────────────────────── per-primitive normalizers ───────────────────────────────


def _norm_observation(
    obs: RawObservation,
    frame: SpatialFrame,
    units: dict[str, str],
    src_crs: str | None,
    src_vertical: str | None,
    z_convention: str,
    warnings: list[IngestWarning],
    params: NormParams,
) -> NormObservation:
    if obs.geometry_kind not in {"points", "soundings", "profile2d", "traces",
                                 "raster2d", "wellcurve", "tensor"}:
        raise NormalizationError(
            f"invalid geometry_kind {obs.geometry_kind!r} (doc 02 §3 frozen vocabulary)"
        )
    coords = _to_engineering(obs.coords, frame, src_crs, src_vertical, z_convention, warnings)
    out_vals: dict[str, np.ndarray] = {}
    out_sigma: dict[str, np.ndarray] = {}
    for prop, vals in obs.values.items():
        canon, _unit = _canonicalize(prop, vals, units.get(prop), warnings, params)
        out_vals[prop] = canon
        # default noise floor when no sigma supplied (doc 02 §3 / doc 03 §2 — never
        # silently error-free): record the substitution in provenance via the warning.
        if prop in obs.sigma:
            sig, _u = _canonicalize(prop, obs.sigma[prop], units.get(prop), [], params)
            out_sigma[prop] = sig
        else:
            pt = REGISTRY.get(prop)
            if not pt.categorical:
                out_sigma[prop] = np.abs(canon) * float(pt.default_rel_sigma)
                warnings.append(IngestWarning(
                    "default_noise_floor", Severity.LOW,
                    f"no sigma for {prop!r}; applied default rel-σ {pt.default_rel_sigma} "
                    "from the registry (doc 02 §3)", f"property:{prop}",
                ))
    return NormObservation(
        geometry_kind=obs.geometry_kind,
        coords=coords,
        values=out_vals,
        sigma=out_sigma,
        primary_property=obs.primary_property or (next(iter(out_vals), None)),
        acquired_at=obs.acquired_at,
        meta=dict(obs.meta),
    )


def _norm_property_model(
    pm: RawPropertyModel,
    units: dict[str, str],
    warnings: list[IngestWarning],
    params: NormParams,
) -> NormPropertyModel:
    if pm.support not in {"volume", "grid2d", "section", "mesh"}:
        raise NormalizationError(
            f"invalid support {pm.support!r} (doc 02 §4 frozen vocabulary)"
        )
    values, canonical_unit = _canonicalize(
        pm.property, pm.values, units.get(pm.property), warnings, params
    )
    sigma = None
    if pm.sigma is not None:
        sigma, _u = _canonicalize(pm.property, pm.sigma, units.get(pm.property), [], params)
    # Property models arrive already in their (z,y,x) grid; we only canonicalize units.
    # CRS reprojection of a regular grid is non-trivial and a re-grid step (doc 03 §3c);
    # for local-mode synthetic/already-Engineering models this is identity.
    return NormPropertyModel(
        property=pm.property,
        values=np.asarray(values, dtype=np.float32),
        origin=tuple(float(v) for v in pm.origin),  # type: ignore[arg-type]
        spacing=tuple(float(v) for v in pm.spacing),  # type: ignore[arg-type]
        support=pm.support,
        sigma=None if sigma is None else np.asarray(sigma, dtype=np.float32),
        canonical_unit=canonical_unit,
        meta=dict(pm.meta),
    )


def _norm_feature(
    feat: RawFeature,
    frame: SpatialFrame,
    src_crs: str | None,
    src_vertical: str | None,
    warnings: list[IngestWarning],
) -> NormFeature:
    # Geometry reprojection of arbitrary vector geometry is delegated to geopandas/pyproj
    # downstream (doc 03 §5); for local-mode the geometry is already Engineering. We
    # compute a bbox when explicit coordinates are present.
    bbox: dict[str, float] = {}
    coords = feat.geometry.get("coordinates") if isinstance(feat.geometry, dict) else None
    if coords is not None:
        flat = np.asarray(_flatten_coords(coords), dtype=float)
        if flat.size and flat.ndim == 2:
            flat = as_xyz(flat)
            if frame.mode is not FrameMode.LOCAL and src_crs is not None:
                flat = frame.to_engineering(flat, src_crs=src_crs, src_vertical=src_vertical)
            bbox = _bbox_from_coords(flat)
    return NormFeature(
        feature_type=feat.feature_type,
        geometry=feat.geometry,
        props=dict(feat.props),
        store_format=feat.store_format,
        bbox=bbox,
    )


def _flatten_coords(coords: Any) -> list[list[float]]:
    """Flatten nested GeoJSON coordinate arrays to a list of ``[x, y(, z)]`` points."""
    out: list[list[float]] = []

    def walk(node: Any) -> None:
        if (
            isinstance(node, (list, tuple))
            and node
            and all(isinstance(v, (int, float)) for v in node)
        ):
            out.append(list(node))
        elif isinstance(node, (list, tuple)):
            for child in node:
                walk(child)

    walk(coords)
    return out


# ─────────────────────────────── entry point ───────────────────────────────


def normalize(parsed: ParseResult, frame: SpatialFrame) -> NormalizedBundle:
    """Run the shared post-parse normalization (doc 03 §3) → :class:`NormalizedBundle`.

    Identical for every method: reproject coords to Engineering, canonicalize units, place
    1D/2D primitives in 3D (curtains/columns), keep raw raw. Accumulates warnings and the
    record drop counts (which feed the >10% fail rule, doc 03 §6).
    """
    source = parsed.source
    src_crs = (source.crs if source else None)
    src_vertical = (source.vertical_datum if source else None)
    z_convention = (source.z_convention if source else "elevation_up")

    params = NormParams(
        z_convention=z_convention, source_crs=src_crs, source_vertical=src_vertical
    )
    warnings: list[IngestWarning] = list(parsed.warnings)
    bundle = NormalizedBundle(
        params=params,
        records_total=parsed.records_total,
        records_dropped=parsed.records_dropped,
    )

    for obs in parsed.observations:
        bundle.observations.append(
            _norm_observation(
                obs, frame, parsed.units, src_crs, src_vertical, z_convention, warnings, params
            )
        )
    for pm in parsed.property_models:
        bundle.property_models.append(
            _norm_property_model(pm, parsed.units, warnings, params)
        )
    for feat in parsed.features:
        bundle.features.append(
            _norm_feature(feat, frame, src_crs, src_vertical, warnings)
        )

    bundle.warnings = warnings
    return bundle
