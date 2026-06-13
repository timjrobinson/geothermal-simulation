"""The Fused Earth Model grid — a regular-voxel CONTAINER (doc 07 §1, doc 02 §11).

The fused grid is a **comparison/compositing** grid, not a super-resolution grid
(doc 07 §1.1). It is a CONTAINER with no single property of its own (doc 07 §2.1):
native :class:`~geosim.catalog.PropertyModel`\\s are RESAMPLED IN as referenced,
read-only :class:`~geosim.catalog.FusedLayer`\\s — the originals are never modified.

Auto-resolution heuristic (doc 07 §1.1):

- default ``dx=dy=dz`` = **median of native cell sizes** across the loaded property
  models, clamped to ``[roi_extent/512, roi_extent/64]``;
- hard cap ``nx·ny·nz <= 256³`` (≈ 16.7 M cells) for the level-0 brick — if the auto
  spacing would exceed it, spacing is coarsened until the cap holds;
- spacing is overridable per project.

Axis order is ``[z, y, x]`` (Z-up), matching doc 02 §10.2 / :mod:`geosim.storage`:
``shape=[nz,ny,nx]`` with ``origin``/``spacing`` ordered ``(z, y, x)``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr
from sqlalchemy.orm import Session

from geosim.catalog import (
    Dataset,
    FusedModel,
    IdKind,
    Project,
    PropertyModel,
    Provenance,
    new_id,
)
from geosim.storage import GridSpec, ProjectLayout

__all__ = [
    "FusedGrid",
    "DEFAULT_CELL_CAP",
    "auto_resolution",
    "build_fused_model",
    "fused_grid_from_row",
    "open_fused_group",
]

# doc 07 §1.1 — hard cap on the default level-0 brick (256³ ≈ 16.7 M cells).
DEFAULT_CELL_CAP = 256**3
_CLAMP_LO_DIV = 512  # roi_extent / 512 (finest allowed default)
_CLAMP_HI_DIV = 64  # roi_extent / 64 (coarsest allowed default)


@dataclass(frozen=True)
class FusedGrid:
    """Regular-voxel fused-grid geometry in the Engineering Frame (doc 07 §1, doc 02 §11).

    ``origin``/``spacing`` are ``(z, y, x)`` Engineering metres (Z-up); ``shape`` is
    ``(nz, ny, nx)``. The grid is a pure geometric container — it holds no property of
    its own; resampled layers are written into its Zarr group (:mod:`.resample`).
    """

    origin: tuple[float, float, float]  # (z0, y0, x0), Engineering m
    spacing: tuple[float, float, float]  # (dz, dy, dx), Engineering m
    shape: tuple[int, int, int]  # (nz, ny, nx)

    @property
    def n_cells(self) -> int:
        nz, ny, nx = self.shape
        return int(nz) * int(ny) * int(nx)

    def to_grid_spec(self) -> GridSpec:
        """A :class:`geosim.storage.GridSpec` for this grid (cell-centred)."""
        return GridSpec(origin=self.origin, spacing=self.spacing, cell_ref="center")

    def axis_coords(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cell-centre coordinate vectors ``(z, y, x)`` along each axis (Engineering m)."""
        oz, oy, ox = self.origin
        dz, dy, dx = self.spacing
        nz, ny, nx = self.shape
        return (
            oz + dz * np.arange(nz, dtype=float),
            oy + dy * np.arange(ny, dtype=float),
            ox + dx * np.arange(nx, dtype=float),
        )

    def bbox(self) -> dict[str, float]:
        """Engineering-metre AABB spanning the cell centres (doc 04 §2.2 index source)."""
        z, y, x = self.axis_coords()
        return {
            "xmin": float(x.min()), "xmax": float(x.max()),
            "ymin": float(y.min()), "ymax": float(y.max()),
            "zmin": float(z.min()), "zmax": float(z.max()),
        }


def _native_cell_size(pm: PropertyModel) -> float | None:
    """Mean of a native model's ``(dz,dy,dx)`` spacing — its characteristic cell size."""
    if not pm.spacing_json:
        return None
    sp = [abs(float(v)) for v in json.loads(pm.spacing_json) if float(v) != 0.0]
    return float(np.mean(sp)) if sp else None


def auto_resolution(
    native_spacings: list[float],
    roi_extent: tuple[float, float, float],
    *,
    cell_cap: int = DEFAULT_CELL_CAP,
) -> tuple[float, float, float]:
    """Auto fused spacing ``(dz, dy, dx)`` per doc 07 §1.1.

    ``native_spacings`` are the characteristic cell sizes of the native models;
    ``roi_extent`` is the ``(z, y, x)`` span of the fused volume in Engineering metres.
    Returns an **isotropic** spacing (anisotropy is off by default, doc 07 §1.1):

    1. base = median native spacing (fallback: a 1/256 of the largest extent);
    2. clamp to ``[max_extent/512, max_extent/64]``;
    3. coarsen (double) until ``nx·ny·nz <= cell_cap``.
    """
    ez, ey, ex = (float(abs(e)) for e in roi_extent)
    max_extent = max(ez, ey, ex, 1.0)

    spacings = [s for s in native_spacings if s and s > 0.0]
    base = float(np.median(spacings)) if spacings else max_extent / 256.0

    lo = max_extent / _CLAMP_LO_DIV
    hi = max_extent / _CLAMP_HI_DIV
    spacing = float(np.clip(base, lo, hi))

    # Coarsen to honour the hard cell cap (doc 07 §1.1).
    def n_cells(sp: float) -> int:
        return int(np.ceil(ez / sp) + 1) * int(np.ceil(ey / sp) + 1) * int(np.ceil(ex / sp) + 1)

    while spacing > 0.0 and n_cells(spacing) > cell_cap:
        spacing *= 2.0
    return (spacing, spacing, spacing)


def _grid_from_extent(
    origin: tuple[float, float, float],
    extent: tuple[float, float, float],
    spacing: tuple[float, float, float],
) -> FusedGrid:
    """Build a :class:`FusedGrid` covering ``extent`` from ``origin`` at ``spacing``."""
    shape = tuple(
        max(1, int(np.floor(e / s + 1e-9)) + 1) for e, s in zip(extent, spacing, strict=True)
    )
    return FusedGrid(origin=origin, spacing=spacing, shape=shape)  # type: ignore[arg-type]


def build_fused_model(
    session: Session,
    layout: ProjectLayout,
    project_id: str,
    *,
    source_property_model_ids: list[str] | None = None,
    bbox: dict[str, float] | None = None,
    spacing: tuple[float, float, float] | None = None,
    cell_cap: int = DEFAULT_CELL_CAP,
    name: str = "fused",
    created_by: str = "system:fusion",
) -> tuple[FusedModel, FusedGrid]:
    """Create a regular-voxel FusedEarthModel container (doc 07 §1, doc 02 §11).

    Writes the ``provenance`` + ``datasets`` (kind ``fusedModel``) + ``fused_models``
    catalog rows and materialises an (empty) Zarr group that resampled layers are later
    written into (:mod:`.resample`). The grid holds **no property** of its own — it is a
    container (doc 07 §2.1); a project may hold several (doc 02 §11 A2).

    Resolution is auto-chosen from the loaded native models' spacings (doc 07 §1.1) unless
    ``spacing`` overrides it. The footprint defaults to the union of the named sources'
    bboxes (falling back to the whole project frame if no sources are given).
    """
    project = session.get(Project, project_id)
    if project is None:
        raise ValueError(f"unknown project {project_id!r}")

    pms = _resolve_sources(session, project_id, source_property_model_ids)
    box = bbox or _sources_bbox(session, project_id, pms)
    origin = (box["zmin"], box["ymin"], box["xmin"])
    extent = (box["zmax"] - box["zmin"], box["ymax"] - box["ymin"], box["xmax"] - box["xmin"])

    if spacing is None:
        native = [s for s in (_native_cell_size(pm) for pm in pms) if s is not None]
        spacing = auto_resolution(native, extent, cell_cap=cell_cap)

    grid = _grid_from_extent(origin, extent, spacing)

    fem_ds_id = new_id(IdKind.DATASET)
    fem_id = new_id(IdKind.FUSED_MODEL)
    prov_id = new_id(IdKind.PROVENANCE)

    # The fused Zarr group is a container; layers (value + sigma + mask) are added later.
    store_path = layout.zarr_path(fem_id)
    root = zarr.create_group(store=str(store_path), overwrite=True)
    root.attrs["geosim"] = {
        "kind": "fusedModel",
        "layoutDoc": "07 §1 / 02 §11",
        "axisOrder": ["z", "y", "x"],
        "gridType": "regular_voxel",
        "origin": list(grid.origin),
        "spacing": list(grid.spacing),
        "shape": list(grid.shape),
        "layers": [],
    }

    grid_bbox = grid.bbox()
    bbox_json = json.dumps(grid_bbox)

    session.add(Provenance(
        id=prov_id, project_id=project_id, target_kind="fusedModel", target_id=fem_id,
        process="fuse:grid", process_version="1.0.0",
        params_json=json.dumps({
            "spacing": list(grid.spacing), "shape": list(grid.shape),
            "sources": [pm.id for pm in pms], "cellCap": cell_cap,
        }),
    ))
    session.flush()
    session.add(Dataset(
        id=fem_ds_id, project_id=project_id, name=name, method="fusion", kind="fusedModel",
        status="ready", extent_json=bbox_json, spatial_frame_id=project_id,
        provenance_id=prov_id, version_root_id=fem_ds_id, version_seq=1,
        created_by=created_by,
    ))
    session.flush()
    fem = FusedModel(
        id=fem_id, dataset_id=fem_ds_id, project_id=project_id, grid_type="regular_voxel",
        store_uri=str(store_path), store_format="zarr",
        shape_json=json.dumps(list(grid.shape)),
        spacing_json=json.dumps(list(grid.spacing)),
        origin_json=json.dumps(list(grid.origin)),
        bbox_json=bbox_json, has_time=0, pyramid_levels=1,
    )
    session.add(fem)
    session.commit()
    session.refresh(fem)
    return fem, grid


def _resolve_sources(
    session: Session, project_id: str, ids: list[str] | None
) -> list[PropertyModel]:
    """The native property models to size the grid from (doc 07 §1.1)."""
    if ids:
        out = []
        for pmid in ids:
            pm = session.get(PropertyModel, pmid)
            if pm is None or pm.project_id != project_id:
                raise ValueError(f"property model {pmid!r} not found in project {project_id!r}")
            out.append(pm)
        return out
    return list(
        session.query(PropertyModel)
        .filter(PropertyModel.project_id == project_id, PropertyModel.support == "volume")
        .all()
    )


def _sources_bbox(
    session: Session, project_id: str, pms: list[PropertyModel]
) -> dict[str, float]:
    """Union the source bboxes; fall back to the project frame ROI × depth range."""
    boxes = [json.loads(pm.bbox_json) for pm in pms]
    if boxes:
        return {
            "xmin": min(b["xmin"] for b in boxes), "xmax": max(b["xmax"] for b in boxes),
            "ymin": min(b["ymin"] for b in boxes), "ymax": max(b["ymax"] for b in boxes),
            "zmin": min(b["zmin"] for b in boxes), "zmax": max(b["zmax"] for b in boxes),
        }
    project = session.get(Project, project_id)
    frame = project.spatial_frame
    roi = json.loads(frame.roi_json)
    dr = json.loads(frame.depth_range_json)
    return {
        "xmin": float(roi["xmin"]), "xmax": float(roi["xmax"]),
        "ymin": float(roi["ymin"]), "ymax": float(roi["ymax"]),
        "zmin": float(dr["zmin"]), "zmax": float(dr["zmax"]),
    }


def fused_grid_from_row(fem: FusedModel) -> FusedGrid:
    """Reconstruct a :class:`FusedGrid` from a persisted ``fused_models`` row."""
    return FusedGrid(
        origin=tuple(json.loads(fem.origin_json)),  # type: ignore[arg-type]
        spacing=tuple(json.loads(fem.spacing_json)),  # type: ignore[arg-type]
        shape=tuple(json.loads(fem.shape_json)),  # type: ignore[arg-type]
    )


def open_fused_group(fem: FusedModel, storage_root: str | Path | None = None) -> zarr.Group:
    """Open a fused model's Zarr group read-write (for layer writes / reads)."""
    path = Path(fem.store_uri)
    if not path.is_absolute() and storage_root is not None:
        path = Path(storage_root) / fem.store_uri
    return zarr.open_group(str(path), mode="a")
