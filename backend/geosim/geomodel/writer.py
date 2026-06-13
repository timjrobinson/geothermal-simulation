"""Catalog the implicit model: a lithology PropertyModel + unitSolid features (doc 02 §5/§10.2).

:func:`persist_geomodel` takes a computed :class:`~geosim.geomodel.builder.GeoModelResult`
and writes it into a project's catalog + bulk stores exactly like any other interpretation
product (doc 02 §7 — nothing exists without provenance):

1. A **lithology PropertyModel** — the categorical ``lithology_class`` key (doc 02 §10.2),
   written via :func:`geosim.storage.write_property_model` as a ``[z, y, x]`` Zarr v3 group
   of **hard integer labels** (float-stored), then annotated with the doc-02 **categories**
   attribute table that decodes the labels. The smooth class-probability axis is preserved
   alongside as a ``classProbability`` group attr (doc 02 §10.2 allows either form).
2. One **unitSolid GeologicalFeature per stratigraphic unit** (doc 02 §5) — its blocky
   solid packed to ``.glb`` via :func:`geosim.storage.write_glb`, dropped under the
   project ``meshes/`` store, and catalogued with per-feature props + provenance.

Every write threads a single :class:`~geosim.catalog.Provenance` row (process
``model:gempy-implicit``) so the lineage DAG records the build (doc 04 §2.4).
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
    Feature,
    IdKind,
    Project,
    PropertyModel,
    Provenance,
    new_id,
    now_ms,
)
from geosim.storage import GridSpec, ProjectLayout, write_glb, write_property_model

from .builder import GeoModelResult

__all__ = ["PersistedGeoModel", "persist_geomodel"]

_PROCESS = "model:gempy-implicit"  # doc 02 §7 provenance process key
_LITHOLOGY_KEY = "lithology_class"  # categorical PropertyType (doc 02 §10.2)


@dataclass
class PersistedGeoModel:
    """Ids the build created (doc 04 §2.4) — the API echoes these back."""

    dataset_id: str
    property_model_id: str
    provenance_id: str
    unit_solid_feature_ids: list[str]
    lithology_store_uri: str


def _bbox_from_frame(spec_extent: list[float]) -> dict[str, float]:
    xmin, xmax, ymin, ymax, zmin, zmax = spec_extent
    return {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax, "zmin": zmin, "zmax": zmax}


def _mesh_bbox(verts: np.ndarray) -> dict[str, float]:
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    return {
        "xmin": float(lo[0]), "xmax": float(hi[0]),
        "ymin": float(lo[1]), "ymax": float(hi[1]),
        "zmin": float(lo[2]), "zmax": float(hi[2]),
    }


def persist_geomodel(
    session: Session,
    layout: ProjectLayout,
    project_id: str,
    result: GeoModelResult,
    *,
    extent: list[float],
    created_by: str = "geomodel",
    input_feature_ids: list[str] | None = None,
) -> PersistedGeoModel:
    """Write + catalog ``result`` as a lithology PropertyModel + unitSolid features.

    All rows are created in one transaction off ``session`` (provenance first, doc 04 §2.4).
    ``extent`` is the GemPy ``[xmin..zmax]`` the grid was built over (the dataset/property
    bbox in Engineering m). Returns the created ids.
    """
    if session.get(Project, project_id) is None:
        raise ValueError(f"project {project_id!r} not found")

    bbox = _bbox_from_frame(extent)
    prov_id = new_id(IdKind.PROVENANCE)
    ds_id = new_id(IdKind.DATASET)

    prov = Provenance(
        id=prov_id, project_id=project_id, target_kind="propertyModel", target_id=ds_id,
        process=_PROCESS, process_version="gempy-2025.2",
        params_json=json.dumps({"resolution": list(result.shape_zyx)}),
    )
    session.add(prov)
    if input_feature_ids:
        from geosim.catalog import ProvenanceInput

        for fid in input_feature_ids:
            session.add(ProvenanceInput(provenance_id=prov_id, input_kind="feature", input_id=fid))
    session.flush()

    session.add(Dataset(
        id=ds_id, project_id=project_id, name="implicit-geomodel", method="geomodel",
        submethod="gempy", kind="propertyModel", status="ready",
        extent_json=json.dumps(bbox), spatial_frame_id=project_id, provenance_id=prov_id,
        version_root_id=ds_id, version_seq=1, created_by=created_by,
        meta_json=json.dumps({"categories": result.categories}),
    ))
    session.flush()

    pm_id = _write_lithology_model(session, layout, project_id, ds_id, result, bbox)
    feature_ids = _write_unit_solids(session, layout, project_id, ds_id, result, created_by)

    session.commit()
    return PersistedGeoModel(
        dataset_id=ds_id,
        property_model_id=pm_id,
        provenance_id=prov_id,
        unit_solid_feature_ids=feature_ids,
        lithology_store_uri=str(layout.zarr_path(ds_id)),
    )


def _write_lithology_model(
    session: Session,
    layout: ProjectLayout,
    project_id: str,
    ds_id: str,
    result: GeoModelResult,
    bbox: dict[str, float],
) -> str:
    """Write the ``lithology_class`` Zarr group (hard labels) + categories attr (doc 02 §10.2)."""
    store_path = layout.zarr_path(ds_id)
    grid = GridSpec(origin=result.origin_zyx, spacing=result.spacing_zyx, cell_ref="center")
    labels = np.round(result.lith_zyx).astype(np.float32)
    write_property_model(store_path, _LITHOLOGY_KEY, labels, grid=grid, overwrite=True)

    # Annotate the level arrays with the doc-02 categories table + a class-probability summary
    # (the smooth membership is a derived axis; we record its presence/shape, doc 02 §10.2).
    group = zarr.open_group(str(store_path), mode="a")
    n_levels = len(group[_LITHOLOGY_KEY].attrs["multiscales"][0]["datasets"])
    for lv in range(n_levels):
        arr = group[f"{_LITHOLOGY_KEY}/{lv}"]
        attrs = dict(arr.attrs)
        attrs["categories"] = result.categories
        arr.attrs.update(attrs)
    group[_LITHOLOGY_KEY].attrs["classProbability"] = {
        "axis": "class", "nClasses": int(result.class_prob.shape[0]),
        "categories": [c for c in result.categories if not c["isFault"]],
    }

    nz, ny, nx = result.shape_zyx
    pm_id = new_id(IdKind.PROPERTY_MODEL)
    session.add(PropertyModel(
        id=pm_id, dataset_id=ds_id, project_id=project_id, property=_LITHOLOGY_KEY,
        canonical_unit="dimensionless", support="volume",
        store_uri=str(store_path), store_format="zarr",
        shape_json=json.dumps([nz, ny, nx]),
        spacing_json=json.dumps(list(result.spacing_zyx)),
        origin_json=json.dumps(list(result.origin_zyx)),
        bbox_json=json.dumps(bbox),
        pyramid_levels=n_levels,
        stats_json=json.dumps({"categories": result.categories}),
    ))
    session.flush()
    return pm_id


def _write_unit_solids(
    session: Session,
    layout: ProjectLayout,
    project_id: str,
    ds_id: str,
    result: GeoModelResult,
    created_by: str,
) -> list[str]:
    """Pack each unit's blocky solid to ``.glb`` and catalog a ``unitSolid`` feature (doc 02 §5)."""
    feature_ids: list[str] = []
    color_by_name = {c["name"]: c for c in result.categories}
    for name, (verts, tris) in result.unit_meshes.items():
        feat_id = new_id(IdKind.FEATURE)
        glb = write_glb(
            verts, tris,
            extras={"featureId": feat_id, "unit": name, "featureKind": "unitSolid"},
        )
        glb_path = Path(layout.meshes) / f"{feat_id}.glb"
        glb_path.parent.mkdir(parents=True, exist_ok=True)
        glb_path.write_bytes(glb)

        cat = color_by_name.get(name, {})
        session.add(Feature(
            id=feat_id, dataset_id=ds_id, project_id=project_id,
            feature_type="unitSolid", store_uri=str(glb_path), store_format="gltf",
            bbox_json=json.dumps(_mesh_bbox(verts)),
            props_json=json.dumps({
                "unit": name, "lithoId": cat.get("id"),
                "triangleCount": int(tris.shape[0]), "vertexCount": int(verts.shape[0]),
                "source": _PROCESS,
            }),
            created_at=now_ms(),
        ))
        feature_ids.append(feat_id)
    session.flush()
    return feature_ids
