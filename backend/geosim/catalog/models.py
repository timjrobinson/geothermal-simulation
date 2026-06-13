"""SQLAlchemy 2.0 ORM models for the catalog DB (doc 04 §2.4).

These are the **physical** catalog tables — the index/source-of-truth for
metadata (doc 04 §1, §2). Bulk samples never live here: rows hold pointers
(``store_uri``/``values_uri``), bounding boxes in **Engineering metres** (doc 01
§1, doc 04 §2.2), shapes, units, and stats only. The logical field semantics are
owned by doc 02 (§2–§11); doc 04 §2.3 pins each logical field to the physical
home reproduced here, so the two docs cannot drift.

Conventions (doc 04 §2.4):
- Every PK is a ``TEXT`` kind-prefixed ULID (see ``geosim.catalog.ids``).
- ``created_at`` / ``updated_at`` are **epoch-milliseconds** integers.
- Open/extensible metadata rides in ``*_json`` columns (the R&D-plugin
  requirement, OVERVIEW §4) so new survey methods add fields without migrations.
- ``provenance_id`` is NOT NULL on ``datasets`` — there is no dataset without
  provenance (doc 02 §7); the provenance row is created first in the same
  ingest/fusion transaction.
"""

from __future__ import annotations

import time

from sqlalchemy import (
    BLOB,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

__all__ = [
    "Base",
    "now_ms",
    "Project",
    "SpatialFrameRow",
    "Dataset",
    "PropertyModel",
    "FusedModel",
    "FusedLayer",
    "Observation",
    "Feature",
    "Provenance",
    "ProvenanceInput",
    "RawFile",
    "Layer",
    "View",
    "Job",
]


def now_ms() -> int:
    """Current wall-clock time as epoch-milliseconds (doc 04 §2.4 timestamps)."""
    return int(time.time() * 1000)


class Base(DeclarativeBase):
    """Declarative base for every catalog table."""


# ───────────────────────── projects ─────────────────────────
class Project(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    storage_root: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=now_ms)
    updated_at: Mapped[int] = mapped_column(
        Integer, nullable=False, default=now_ms, onupdate=now_ms
    )

    # FK children cascade on project delete (doc 04 §2.4 ON DELETE CASCADE).
    spatial_frame: Mapped[SpatialFrameRow] = relationship(
        back_populates="project", cascade="all, delete-orphan", uselist=False,
        passive_deletes=True,
    )
    datasets: Mapped[list[Dataset]] = relationship(
        back_populates="project", cascade="all, delete-orphan", passive_deletes=True
    )


# ── spatial_frame: 1:1 with project; the doc-01 SpatialFrame, serialized ──
class SpatialFrameRow(Base):
    __tablename__ = "spatial_frame"

    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    mode: Mapped[str] = mapped_column(Text, nullable=False)  # 'georeferenced'|'local'
    horizontal_crs: Mapped[str | None] = mapped_column(Text)
    vertical_datum: Mapped[str | None] = mapped_column(Text)
    anchor_json: Mapped[str | None] = mapped_column(Text)
    rotation_deg: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    axis_convention: Mapped[str] = mapped_column(Text, nullable=False, default="ENU")
    length_unit: Mapped[str] = mapped_column(Text, nullable=False, default="m")
    roi_json: Mapped[str] = mapped_column(Text, nullable=False)
    depth_range_json: Mapped[str] = mapped_column(Text, nullable=False)
    surface_model: Mapped[str | None] = mapped_column(Text)
    frame_json: Mapped[str] = mapped_column(Text, nullable=False)

    project: Mapped[Project] = relationship(back_populates="spatial_frame")


# ───────────────────────── datasets ─────────────────────────
class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    method: Mapped[str] = mapped_column(Text, nullable=False)  # canonical MethodKey (doc 02 §2)
    submethod: Mapped[str | None] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(Text, nullable=False)  # obs|propertyModel|feature|fusedModel
    status: Mapped[str] = mapped_column(Text, nullable=False)  # ingesting|ready|error
    extent_json: Mapped[str] = mapped_column(Text, nullable=False)  # Engineering m (index src §2.5)
    time_extent_json: Mapped[str | None] = mapped_column(Text)  # null ⇒ static (doc 02)
    spatial_frame_id: Mapped[str] = mapped_column(
        String, ForeignKey("spatial_frame.project_id"), nullable=False
    )
    origin_crs: Mapped[str | None] = mapped_column(Text)
    # NOT NULL — every dataset has exactly one provenance (doc 02 §7). Provenance is
    # created first within the same ingest/fusion transaction (doc 04 §2.4 note).
    provenance_id: Mapped[str] = mapped_column(
        String, ForeignKey("provenance.id"), nullable=False
    )
    version_root_id: Mapped[str] = mapped_column(String, nullable=False)  # VersionInfo.rootId (§9)
    version_seq: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    version_parent_id: Mapped[str | None] = mapped_column(String)  # null for v1
    tags_json: Mapped[str | None] = mapped_column(Text)
    meta_json: Mapped[str | None] = mapped_column(Text)  # methodData/acquisition blob (doc 02)
    created_by: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=now_ms)
    updated_at: Mapped[int] = mapped_column(
        Integer, nullable=False, default=now_ms, onupdate=now_ms
    )

    project: Mapped[Project] = relationship(back_populates="datasets")
    property_models: Mapped[list[PropertyModel]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan", passive_deletes=True
    )
    fused_models: Mapped[list[FusedModel]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan", passive_deletes=True
    )
    observations: Mapped[list[Observation]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan", passive_deletes=True
    )
    features: Mapped[list[Feature]] = relationship(
        back_populates="dataset", cascade="all, delete-orphan", passive_deletes=True
    )


# ─────────────────── property_models (3D/4D continuous fields) ───────────────────
class PropertyModel(Base):
    __tablename__ = "property_models"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id: Mapped[str] = mapped_column(
        String, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    property: Mapped[str] = mapped_column(Text, nullable=False)  # PropertyTypeKey (doc 01 §5)
    canonical_unit: Mapped[str] = mapped_column(Text, nullable=False)  # doc 01 units registry
    support: Mapped[str] = mapped_column(Text, nullable=False)  # volume|grid2d|mesh (doc 02 §4)
    store_uri: Mapped[str] = mapped_column(Text, nullable=False)
    store_format: Mapped[str] = mapped_column(Text, nullable=False, default="zarr")
    shape_json: Mapped[str] = mapped_column(Text, nullable=False)  # [nz,ny,nx] or [nt,nz,ny,nx]
    spacing_json: Mapped[str | None] = mapped_column(Text)
    origin_json: Mapped[str | None] = mapped_column(Text)
    bbox_json: Mapped[str] = mapped_column(Text, nullable=False)  # Engineering m (index source)
    has_time: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    pyramid_levels: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    stats_json: Mapped[str | None] = mapped_column(Text)
    uncertainty_uri: Mapped[str | None] = mapped_column(Text)  # '<property>_sigma' (doc 02 §6)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=now_ms)
    updated_at: Mapped[int] = mapped_column(
        Integer, nullable=False, default=now_ms, onupdate=now_ms
    )

    dataset: Mapped[Dataset] = relationship(back_populates="property_models")


# ─────────────────── fused_models (the FusedEarthModel CONTAINER grid) ───────────────────
class FusedModel(Base):
    __tablename__ = "fused_models"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # 'fem_...'
    dataset_id: Mapped[str] = mapped_column(
        String, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    grid_type: Mapped[str] = mapped_column(Text, nullable=False, default="regular_voxel")
    store_uri: Mapped[str] = mapped_column(Text, nullable=False)
    store_format: Mapped[str] = mapped_column(Text, nullable=False, default="zarr")
    shape_json: Mapped[str] = mapped_column(Text, nullable=False)  # VolumeSupport (doc 02 §4)
    spacing_json: Mapped[str] = mapped_column(Text, nullable=False)
    origin_json: Mapped[str] = mapped_column(Text, nullable=False)
    bbox_json: Mapped[str] = mapped_column(Text, nullable=False)  # Engineering m (index source)
    has_time: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    time_extent_json: Mapped[str | None] = mapped_column(Text)  # TimeAxis if 4D (doc 02 §11)
    pyramid_levels: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=now_ms)
    updated_at: Mapped[int] = mapped_column(
        Integer, nullable=False, default=now_ms, onupdate=now_ms
    )

    dataset: Mapped[Dataset] = relationship(back_populates="fused_models")
    layers: Mapped[list[FusedLayer]] = relationship(
        back_populates="fused_model", cascade="all, delete-orphan", passive_deletes=True
    )


# ── fused_layers: each native PropertyModel resampled INTO the fused grid (doc 02 §11) ──
class FusedLayer(Base):
    __tablename__ = "fused_layers"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # layerId
    fused_model_id: Mapped[str] = mapped_column(
        String, ForeignKey("fused_models.id", ondelete="CASCADE"), nullable=False
    )
    source_property_model_id: Mapped[str] = mapped_column(
        String, ForeignKey("property_models.id"), nullable=False
    )
    source_version: Mapped[str] = mapped_column(Text, nullable=False)  # pinned doc-02 version
    property: Mapped[str] = mapped_column(Text, nullable=False)  # PropertyTypeKey (doc 01 §5)
    resample_op_json: Mapped[str] = mapped_column(Text, nullable=False)  # {method, params} (doc 07)
    sigma_array: Mapped[str | None] = mapped_column(Text)  # resampled 1σ in fused Zarr (doc 02 §11)
    valid_mask: Mapped[str | None] = mapped_column(Text)  # coverage mask path
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=now_ms)

    fused_model: Mapped[FusedModel] = relationship(back_populates="layers")


# ─────────────────── observations (raw measured survey data) ───────────────────
class Observation(Base):
    __tablename__ = "observations"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id: Mapped[str] = mapped_column(
        String, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    geometry_kind: Mapped[str] = mapped_column(Text, nullable=False)  # doc 02 §3 geometryKind
    primary_property: Mapped[str | None] = mapped_column(Text)  # null for raw traces/tensors
    geometry_wkb: Mapped[bytes | None] = mapped_column(BLOB)  # Engineering-frame geometry
    values_uri: Mapped[str | None] = mapped_column(Text)  # array file if bulk
    values_json: Mapped[str | None] = mapped_column(Text)  # inline values for small obs
    bbox_json: Mapped[str] = mapped_column(Text, nullable=False)
    acquired_at: Mapped[int | None] = mapped_column(Integer)  # time of measurement (4D)
    meta_json: Mapped[str | None] = mapped_column(Text)  # methodData/acquisition blob (doc 02)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=now_ms)

    dataset: Mapped[Dataset] = relationship(back_populates="observations")


# ─────────────────── features (vector geological interpretation) ───────────────────
class Feature(Base):
    __tablename__ = "features"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("datasets.id", ondelete="CASCADE")
    )
    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    feature_type: Mapped[str] = mapped_column(Text, nullable=False)  # doc 02 §5 featureKind
    store_uri: Mapped[str | None] = mapped_column(Text)
    store_format: Mapped[str] = mapped_column(Text, nullable=False)  # gltf|vtk|geojson|laz|3dtiles
    geometry_wkb: Mapped[bytes | None] = mapped_column(BLOB)  # simplified geom for picking/index
    bbox_json: Mapped[str] = mapped_column(Text, nullable=False)
    has_time: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    props_json: Mapped[str | None] = mapped_column(Text)  # per-feature attributes (doc 02 §5)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=now_ms)
    updated_at: Mapped[int] = mapped_column(
        Integer, nullable=False, default=now_ms, onupdate=now_ms
    )

    dataset: Mapped[Dataset] = relationship(back_populates="features")


# ─────────────────── provenance (lineage DAG) ───────────────────
class Provenance(Base):
    __tablename__ = "provenance"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    target_kind: Mapped[str] = mapped_column(Text, nullable=False)  # doc 02 §2 target kind
    target_id: Mapped[str] = mapped_column(Text, nullable=False)
    process: Mapped[str] = mapped_column(Text, nullable=False)  # 'ingest:ert-stg'|'fuse:resample'
    process_version: Mapped[str | None] = mapped_column(Text)
    params_json: Mapped[str | None] = mapped_column(Text)
    source_crs: Mapped[str | None] = mapped_column(Text)  # original CRS before doc-01 reprojection
    source_unit: Mapped[str | None] = mapped_column(Text)  # original unit before canonicalization
    raw_file_id: Mapped[str | None] = mapped_column(String, ForeignKey("raw_files.id"))
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=now_ms)

    inputs: Mapped[list[ProvenanceInput]] = relationship(
        back_populates="provenance", cascade="all, delete-orphan", passive_deletes=True
    )


class ProvenanceInput(Base):
    __tablename__ = "provenance_inputs"

    provenance_id: Mapped[str] = mapped_column(
        String, ForeignKey("provenance.id", ondelete="CASCADE"), primary_key=True
    )
    input_kind: Mapped[str] = mapped_column(Text, primary_key=True)
    input_id: Mapped[str] = mapped_column(Text, primary_key=True)

    provenance: Mapped[Provenance] = relationship(back_populates="inputs")


# ─────────────────── raw_files (raw store index) ───────────────────
class RawFile(Base):
    __tablename__ = "raw_files"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    filename: Mapped[str] = mapped_column(Text, nullable=False)  # original name
    rel_path: Mapped[str] = mapped_column(Text, nullable=False)  # raw/<sha256>/<filename> (§3)
    sha256: Mapped[str] = mapped_column(Text, nullable=False)  # content address / dedupe
    bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    media_type: Mapped[str | None] = mapped_column(Text)  # detected format
    uploaded_at: Mapped[int] = mapped_column(Integer, nullable=False, default=now_ms)


# ─────────────────── layers / views (viewer presentation state) ───────────────────
class Layer(Base):
    __tablename__ = "layers"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    source_kind: Mapped[str] = mapped_column(Text, nullable=False)  # doc 02 §2 kinds
    source_id: Mapped[str] = mapped_column(Text, nullable=False)
    render_type: Mapped[str] = mapped_column(Text, nullable=False)  # volume|slice|iso|mesh|points
    visible: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    z_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    style_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=now_ms)
    updated_at: Mapped[int] = mapped_column(
        Integer, nullable=False, default=now_ms, onupdate=now_ms
    )


class View(Base):
    __tablename__ = "views"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    camera_json: Mapped[str | None] = mapped_column(Text)  # pose, target, fov
    layer_set_json: Mapped[str | None] = mapped_column(Text)  # [{layer_id, overrides}]
    time_json: Mapped[str | None] = mapped_column(Text)  # current t for 4D
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=now_ms)


# ─────────────────── jobs (async work) ───────────────────
class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[str] = mapped_column(
        String, ForeignKey("projects.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(Text, nullable=False)  # ingest|fuse|iso|pyramid|export|gc|...
    status: Mapped[str] = mapped_column(Text, nullable=False)  # queued|running|succeeded|failed|cancelled  # noqa: E501
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)  # 0..1
    message: Mapped[str | None] = mapped_column(Text)
    params_json: Mapped[str] = mapped_column(Text, nullable=False)
    result_json: Mapped[str | None] = mapped_column(Text)
    error_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False, default=now_ms)
    started_at: Mapped[int | None] = mapped_column(Integer)
    finished_at: Mapped[int | None] = mapped_column(Integer)
