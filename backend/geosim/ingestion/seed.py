"""M1 'write + register' seed path (doc 03 §7 steps 6–7, doc 02 §7).

``seed_m1_project`` synthesizes one resistivity volume (doc 05, via
:func:`geosim.synthgen.build_resistivity_volume`) and runs the terminal ingest steps:

1. **write** — the doc-02 PropertyModel Zarr v3 group with a multiscale pyramid +
   sibling ``_sigma`` (doc 02 §10.2/§10.3) via :func:`geosim.storage.write_property_model`,
   under the project's ``arrays/`` bulk store (doc 04 §3).
2. **register** — insert catalog rows in one transaction (doc 03 §7 step 7, atomic):
   ``project`` + local-mode ``spatial_frame`` (doc 01 §2), the mandatory ``provenance``
   row first (``process="synthesize"``, doc 02 §7 — no dataset without provenance),
   then the ``dataset`` (``kind=propertyModel``) and the ``property_model`` row carrying
   the shape/unit/levels/bbox index pointers (doc 04 §2.4).

Bounding boxes are stored in **Engineering metres** (doc 01 §1, doc 04 §2.2/§2.5),
derived from the volume's ``origin``/``spacing``/``shape`` (``[z, y, x]`` Z-up).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

import geosim
from geosim.api.frame_io import frame_row_kwargs
from geosim.catalog import (
    Dataset,
    IdKind,
    Project,
    PropertyModel,
    Provenance,
    ProvenanceInput,
    SpatialFrameRow,
    new_id,
)
from geosim.spatial import REGISTRY, Aabb, DepthRange, SpatialFrame
from geosim.storage import (
    SIGMA_SUFFIX,
    GridSpec,
    PropertyModelReader,
    ensure_project_layout,
    open_property_model,
    write_property_model,
)
from geosim.synthgen import build_resistivity_volume

__all__ = ["seed_m1_project"]

_AGENT = "geosim.ingestion.seed_m1_project"
_TOOL = "geosim.synthgen.build_resistivity_volume"


def _engineering_bbox(
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
    shape: tuple[int, int, int],
) -> dict[str, float]:
    """Cell-corner AABB in Engineering metres from a Z-up ``(z,y,x)`` grid (doc 04 §2.2).

    ``origin``/``spacing`` are ``(z, y, x)``; the Engineering axes are X=East, Y=North,
    Z=Up (doc 01 §1). The box spans ``[origin, origin + n*spacing]`` per axis.
    """
    nz, ny, nx = shape
    z0, y0, x0 = origin
    dz, dy, dx = spacing
    return {
        "xmin": float(x0), "xmax": float(x0 + nx * dx),
        "ymin": float(y0), "ymax": float(y0 + ny * dy),
        "zmin": float(z0), "zmax": float(z0 + nz * dz),
    }


def seed_m1_project(
    session: Session,
    storage_root: str | Path,
    name: str = "m1-resistivity",
    *,
    shape: tuple[int, int, int] = (32, 32, 32),
    spacing: tuple[float, float, float] = (25.0, 25.0, 25.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    seed: int = 42,
) -> dict[str, str]:
    """Synthesize + write + register one M1 resistivity PropertyModel (doc 03 §7).

    Creates a ``Project`` with a local-mode ``SpatialFrame`` (doc 01 §2), materializes
    the bulk-store directory tree (doc 04 §3), writes the doc-02 PropertyModel Zarr
    group (pyramid + ``_sigma``), and inserts the provenance/dataset/property_model
    catalog rows. Deterministic for a fixed ``(shape, spacing, origin, seed)``.

    Returns ``{"project_id", "dataset_id", "property_model_id"}``.
    """
    storage_root = Path(storage_root)
    vol = build_resistivity_volume(shape=shape, spacing=spacing, origin=origin, seed=seed)
    pt = REGISTRY.get(vol.property)

    # ── ids (kind-prefixed ULIDs, doc 02 §1) ──────────────────────────────────────
    project_id = new_id(IdKind.PROJECT)
    dataset_id = new_id(IdKind.DATASET)
    property_model_id = new_id(IdKind.PROPERTY_MODEL)
    provenance_id = new_id(IdKind.PROVENANCE)

    # ── project + local-mode SpatialFrame (doc 01 §2) ─────────────────────────────
    bbox = _engineering_bbox(vol.origin, vol.spacing, tuple(vol.values.shape))
    frame = SpatialFrame(
        roi=Aabb(bbox["xmin"], bbox["xmax"], bbox["ymin"], bbox["ymax"]),
        depth_range=DepthRange(bbox["zmin"], bbox["zmax"]),
    )

    layout = ensure_project_layout(storage_root, project_id)

    project = Project(id=project_id, name=name, storage_root=str(storage_root))
    project.spatial_frame = SpatialFrameRow(project_id=project_id, **frame_row_kwargs(frame))

    # ── write: doc-02 PropertyModel Zarr group with pyramid + _sigma (doc 02 §10) ─
    zarr_path = layout.zarr_path(dataset_id)
    grid = GridSpec(origin=vol.origin, spacing=vol.spacing, cell_ref="center")
    write_property_model(
        zarr_path,
        vol.property,
        vol.values,
        grid=grid,
        sigma=vol.sigma,
        overwrite=True,
    )
    reader: PropertyModelReader = open_property_model(zarr_path)
    levels = reader.level_count(vol.property)

    store_uri = str(zarr_path)
    bbox_json = json.dumps(bbox)
    shape_list = [int(s) for s in vol.values.shape]  # [nz, ny, nx]

    stats = _value_stats(vol.values)

    # ── register: provenance FIRST (doc 02 §7), then dataset + property_model ─────
    provenance = Provenance(
        id=provenance_id,
        project_id=project_id,
        target_kind="propertyModel",
        target_id=property_model_id,
        process="synthesize",  # doc 02 §7 Step op
        process_version=geosim.__version__,
        params_json=json.dumps(
            {
                "agent": _AGENT,
                "tool": _TOOL,
                "code": {"module": _TOOL, "gitSha": None},
                "params": {
                    "shape": shape_list,
                    "spacing": list(vol.spacing),
                    "origin": list(vol.origin),
                    "seed": seed,
                    "scene": "unit-cube-v1",
                },
            }
        ),
        source_crs=None,  # synthetic / local-mode (doc 02 §2 originCrs null)
        source_unit=pt.canonical_unit,
    )

    dataset = Dataset(
        id=dataset_id,
        project_id=project_id,
        name=name,
        method="ert",  # canonical MethodKey (doc 02 §2); resistivity is an ERT-family field
        submethod=None,
        kind="propertyModel",
        status="ready",
        extent_json=bbox_json,
        time_extent_json=None,  # static (doc 02)
        spatial_frame_id=project_id,
        origin_crs=None,
        provenance_id=provenance_id,
        version_root_id=dataset_id,
        version_seq=1,
        version_parent_id=None,
        tags_json=json.dumps(["synthetic", "m1"]),
        meta_json=json.dumps({"source": "synthgen", "scene": "unit-cube-v1", "seed": seed}),
        created_by=_AGENT,
    )

    property_model = PropertyModel(
        id=property_model_id,
        dataset_id=dataset_id,
        project_id=project_id,
        property=vol.property,
        canonical_unit=pt.canonical_unit,
        support="volume",  # doc 02 §4 VolumeSupport
        store_uri=store_uri,
        store_format="zarr",
        shape_json=json.dumps(shape_list),
        spacing_json=json.dumps(list(vol.spacing)),
        origin_json=json.dumps(list(vol.origin)),
        bbox_json=bbox_json,
        has_time=0,
        pyramid_levels=int(levels),
        stats_json=json.dumps(stats),
        uncertainty_uri=f"{vol.property}{SIGMA_SUFFIX}",  # sibling subgroup (doc 02 §6)
    )

    # Insert in dependency order (doc 03 §7 step 7 atomic): project + frame, then the
    # mandatory provenance row (doc 02 §7), then the dataset that FKs both, then the
    # property_model. Flush between groups so SQLite FK checks see the parents.
    session.add(project)
    session.flush()
    session.add(provenance)
    session.flush()
    session.add(dataset)
    session.flush()
    session.add(property_model)
    session.add(
        ProvenanceInput(
            provenance_id=provenance_id,
            input_kind="synthetic",
            input_id=f"seed:{seed}",
        )
    )
    session.commit()

    # Cache the frame next to the bulk stores (DB is canonical; doc 04 §3).
    layout.frame_json.write_text(json.dumps(_frame_dict(frame), indent=2))

    return {
        "project_id": project_id,
        "dataset_id": dataset_id,
        "property_model_id": property_model_id,
    }


def _value_stats(values: Any) -> dict[str, float]:
    import numpy as np

    finite = values[np.isfinite(values)]
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "count": int(finite.size),
    }


def _frame_dict(frame: SpatialFrame) -> dict[str, Any]:
    from geosim.api.frame_io import frame_to_dict

    return frame_to_dict(frame)
