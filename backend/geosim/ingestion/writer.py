"""WRITE + REGISTER stages (doc 03 §7 steps 6–7) — persist + index the normalized bundle.

The writer chooses storage (doc 03 §2: "adapter does not choose storage; the writer
does, keyed by support geometry") and inserts the catalog rows atomically (doc 03 §7
step 7 — the dataset becomes visible only once registration commits):

- **PropertyModel** (``support`` ∈ ``volume``) → a doc-02 Zarr v3 group via
  :func:`geosim.storage.write_property_model` (pyramid + sibling ``_sigma``, doc 02 §10),
  then a ``property_models`` catalog row carrying shape/unit/levels/bbox (doc 04 §2.4).
- **Observation** → an ``observations`` row; small value tables go inline as
  ``values_json`` (doc 04 §2.4 ``values_json`` for small obs), bbox in Engineering metres.
- **Feature** → a ``features`` row with inline GeoJSON geometry (doc 02 §5).

Provenance is written **first** (doc 02 §7 — no dataset without provenance), populated
with the raw-file hash + adapter name+version + source CRS/unit + normalization params
(doc 03 §8). All rows insert in one transaction; a flush between dependency groups keeps
SQLite FK checks happy (matching the M1 seed path).
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy.orm import Session

from geosim.api.frame_io import frame_row_kwargs
from geosim.catalog import (
    Dataset,
    Feature,
    IdKind,
    Observation,
    Project,
    PropertyModel,
    Provenance,
    ProvenanceInput,
    RawFile,
    SpatialFrameRow,
    new_id,
)
from geosim.spatial import SpatialFrame
from geosim.storage import (
    SIGMA_SUFFIX,
    GridSpec,
    ProjectLayout,
    open_property_model,
    write_property_model,
)

from .base import IngestReport
from .normalize import NormalizedBundle, NormFeature, NormObservation, NormPropertyModel

__all__ = ["WriteContext", "write_and_register"]


class WriteContext:
    """Identity + provenance metadata threaded into the write/register step (doc 03 §7)."""

    def __init__(
        self,
        *,
        project_id: str,
        dataset_id: str,
        provenance_id: str,
        layout: ProjectLayout,
        method: str,
        submethod: str | None,
        process: str,
        process_version: str | None,
        params: dict[str, Any],
        source_crs: str | None,
        source_unit: str | None,
        raw_file_id: str | None,
        created_by: str,
        name: str,
        tags: list[str] | None = None,
    ) -> None:
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.provenance_id = provenance_id
        self.layout = layout
        self.method = method
        self.submethod = submethod
        self.process = process
        self.process_version = process_version
        self.params = params
        self.source_crs = source_crs
        self.source_unit = source_unit
        self.raw_file_id = raw_file_id
        self.created_by = created_by
        self.name = name
        self.tags = tags or []


def _union_bbox(boxes: list[dict[str, float]]) -> dict[str, float]:
    """Union of Engineering-metre AABBs (doc 04 §2.2); falls back to a zero box."""
    boxes = [b for b in boxes if b]
    if not boxes:
        return {"xmin": 0.0, "xmax": 0.0, "ymin": 0.0, "ymax": 0.0, "zmin": 0.0, "zmax": 0.0}
    return {
        "xmin": min(b["xmin"] for b in boxes), "xmax": max(b["xmax"] for b in boxes),
        "ymin": min(b["ymin"] for b in boxes), "ymax": max(b["ymax"] for b in boxes),
        "zmin": min(b["zmin"] for b in boxes), "zmax": max(b["zmax"] for b in boxes),
    }


def _dataset_kind(bundle: NormalizedBundle) -> str:
    """The dominant dataset kind (doc 02 §2): property models > observations > features."""
    if bundle.property_models:
        return "propertyModel"
    if bundle.observations:
        return "obs"
    return "feature"


def _write_property_model(
    session: Session, ctx: WriteContext, pm: NormPropertyModel, project_id: str
) -> tuple[PropertyModel, dict[str, float]]:
    pm_id = new_id(IdKind.PROPERTY_MODEL)
    zarr_path = ctx.layout.zarr_path(f"{ctx.dataset_id}_{pm_id}")
    grid = GridSpec(origin=pm.origin, spacing=pm.spacing, cell_ref="center")
    write_property_model(
        zarr_path, pm.property, pm.values, grid=grid, sigma=pm.sigma, overwrite=True
    )
    levels = open_property_model(zarr_path).level_count(pm.property)
    bbox = pm.bbox
    shape_list = [int(s) for s in pm.values.shape]
    row = PropertyModel(
        id=pm_id,
        dataset_id=ctx.dataset_id,
        project_id=project_id,
        property=pm.property,
        canonical_unit=pm.canonical_unit,
        support=pm.support,
        store_uri=str(zarr_path),
        store_format="zarr",
        shape_json=json.dumps(shape_list),
        spacing_json=json.dumps(list(pm.spacing)),
        origin_json=json.dumps(list(pm.origin)),
        bbox_json=json.dumps(bbox),
        has_time=0,
        pyramid_levels=int(levels),
        stats_json=json.dumps(_stats(pm.values)),
        uncertainty_uri=(f"{pm.property}{SIGMA_SUFFIX}" if pm.sigma is not None else None),
    )
    return row, bbox


def _observation_row(
    ctx: WriteContext, obs: NormObservation, project_id: str
) -> tuple[Observation, dict[str, float]]:
    bbox = obs.bbox
    # Small obs go inline (doc 04 §2.4 values_json); coords + value/sigma columns as JSON.
    values_json = json.dumps({
        "coords": obs.coords.tolist(),
        "values": {k: v.tolist() for k, v in obs.values.items()},
        "sigma": {k: v.tolist() for k, v in obs.sigma.items()},
    })
    row = Observation(
        id=new_id(IdKind.OBSERVATION),
        dataset_id=ctx.dataset_id,
        project_id=project_id,
        geometry_kind=obs.geometry_kind,
        primary_property=obs.primary_property,
        geometry_wkb=None,
        values_uri=None,
        values_json=values_json,
        bbox_json=json.dumps(bbox),
        acquired_at=None,
        meta_json=json.dumps(obs.meta) if obs.meta else None,
    )
    return row, bbox


def _feature_row(
    ctx: WriteContext, feat: NormFeature, project_id: str
) -> tuple[Feature, dict[str, float]]:
    bbox = feat.bbox or {
        "xmin": 0.0, "xmax": 0.0, "ymin": 0.0, "ymax": 0.0, "zmin": 0.0, "zmax": 0.0
    }
    row = Feature(
        id=new_id(IdKind.FEATURE),
        dataset_id=ctx.dataset_id,
        project_id=project_id,
        feature_type=feat.feature_type,
        store_uri=None,
        store_format=feat.store_format,
        geometry_wkb=None,
        bbox_json=json.dumps(bbox),
        has_time=0,
        props_json=json.dumps({"geometry": feat.geometry, "props": feat.props}),
    )
    return row, bbox


def write_and_register(
    session: Session,
    ctx: WriteContext,
    bundle: NormalizedBundle,
    frame: SpatialFrame,
    report: IngestReport,
    *,
    create_project: bool = True,
) -> None:
    """Write bulk stores + insert catalog rows atomically (doc 03 §7 steps 6–7).

    Inserts provenance first (doc 02 §7), then the dataset, then its primitives, then the
    provenance-input edges; one ``commit`` makes the dataset visible. Fills ``report`` with
    the registered ids + primitive counts.
    """
    project_id = ctx.project_id
    boxes: list[dict[str, float]] = []

    # Bulk writes first (outside the row inserts); Zarr is content-addressed on disk.
    pm_rows: list[PropertyModel] = []
    for pm in bundle.property_models:
        row, bbox = _write_property_model(session, ctx, pm, project_id)
        pm_rows.append(row)
        boxes.append(bbox)

    obs_rows: list[Observation] = []
    for obs in bundle.observations:
        row, bbox = _observation_row(ctx, obs, project_id)
        obs_rows.append(row)
        boxes.append(bbox)

    feat_rows: list[Feature] = []
    for feat in bundle.features:
        row, bbox = _feature_row(ctx, feat, project_id)
        feat_rows.append(row)
        boxes.append(bbox)

    extent = _union_bbox(boxes)
    extent_json = json.dumps(extent)

    # Provenance target points at the primary primitive (first PM, else the dataset).
    target_id = pm_rows[0].id if pm_rows else ctx.dataset_id
    target_kind = _dataset_kind(bundle)

    provenance = Provenance(
        id=ctx.provenance_id,
        project_id=project_id,
        target_kind=target_kind,
        target_id=target_id,
        process=ctx.process,
        process_version=ctx.process_version,
        params_json=json.dumps(ctx.params),
        source_crs=ctx.source_crs,
        source_unit=ctx.source_unit,
        raw_file_id=ctx.raw_file_id,
    )

    dataset = Dataset(
        id=ctx.dataset_id,
        project_id=project_id,
        name=ctx.name,
        method=ctx.method,
        submethod=ctx.submethod,
        kind=target_kind,
        status="ready",
        extent_json=extent_json,
        time_extent_json=None,
        spatial_frame_id=project_id,
        origin_crs=ctx.source_crs,
        provenance_id=ctx.provenance_id,
        version_root_id=ctx.dataset_id,
        version_seq=1,
        version_parent_id=None,
        tags_json=json.dumps(ctx.tags),
        meta_json=json.dumps({"normalization": bundle.params.to_dict()}),
        created_by=ctx.created_by,
    )

    # Insert in dependency order (doc 03 §7 step 7 atomic).
    if create_project:
        project = Project(
            id=project_id, name=ctx.name, storage_root=str(ctx.layout.storage_root)
        )
        project.spatial_frame = SpatialFrameRow(
            project_id=project_id, **frame_row_kwargs(frame)
        )
        session.add(project)
        session.flush()

    session.add(provenance)
    session.flush()
    session.add(dataset)
    session.flush()
    for row in pm_rows:
        session.add(row)
    for row in obs_rows:
        session.add(row)
    for row in feat_rows:
        session.add(row)
    if ctx.raw_file_id:
        session.add(ProvenanceInput(
            provenance_id=ctx.provenance_id, input_kind="rawFile", input_id=ctx.raw_file_id
        ))
    session.commit()

    report.dataset_id = ctx.dataset_id
    report.project_id = project_id
    report.n_property_models = len(pm_rows)
    report.n_observations = len(obs_rows)
    report.n_features = len(feat_rows)


def _register_raw_file(
    session: Session,
    *,
    project_id: str,
    filename: str,
    rel_path: str,
    sha256: str,
    nbytes: int,
    media_type: str | None,
) -> str:
    """Insert (or reuse) a ``raw_files`` row keyed by content hash (doc 04 §3, §8.1)."""
    existing = (
        session.query(RawFile)
        .filter(RawFile.project_id == project_id, RawFile.sha256 == sha256)
        .first()
    )
    if existing is not None:
        return existing.id
    raw_id = new_id(IdKind.RAW_FILE)
    session.add(RawFile(
        id=raw_id, project_id=project_id, filename=filename, rel_path=rel_path,
        sha256=sha256, bytes=nbytes, media_type=media_type,
    ))
    session.flush()
    return raw_id


def _stats(values: Any) -> dict[str, float]:
    import numpy as np

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "count": 0}
    return {
        "min": float(finite.min()), "max": float(finite.max()),
        "mean": float(finite.mean()), "count": int(finite.size),
    }
