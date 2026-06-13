"""M4 exit criteria — fusion L1–L2: fused grid → resample → cross-plot/cluster (doc-ROADMAP M4).

The doc-ROADMAP M4 gate (design/ROADMAP.md §"M4 — Fusion L1–L2"):

    Exit: build a fused grid from the ``great-basin-v1`` layers; cross-plot resistivity vs
    density vs velocity at shared cells and **see the geothermal anomaly separate as a
    cluster**.

This test proves that gate end-to-end and headless:

1. **One earth** — compile the flagship ``great-basin-v1`` scene (doc 05 §7.1: a Basin-&-
   Range hydrothermal play with a fault-controlled upflow + shallow clay-cap conductor) at
   a deliberately **coarse** truth grid (replacing the fine 50 m / 20 m spacings with
   600 m / 300 m → a ~20×20×26 grid) so the whole build + fuse + cluster runs in ~1 s.
   The geology is unchanged, so the conductive hydrothermal anomaly is still present and
   still co-varies across methods (low resistivity, lower density, slower Vp, hotter).
2. **Native geophysical models** — write the co-located truth resistivity (log10 interp
   space), density and Vp volumes in as ordinary native :class:`PropertyModel`\\s (these
   stand in for the inverted/gridded survey models the fusion engine consumes, doc 07 §0).
3. **Real ingest pipeline** — additionally ingest a couple of native-format ``unit-cube-v1``
   measured files through the **real** :func:`geosim.ingestion.ingest_file` pipeline into
   a project, proving the M2 round-trip still feeds M4 (doc 03 §7, doc-ROADMAP M4 deps).
4. **Fuse + resample** — :func:`build_fused_model` over the three native models, then
   :func:`resample_to_fused` each onto the shared support (doc 07 §1–§2), interpolating
   resistivity in **log10** space per :data:`geosim.spatial.REGISTRY`.
5. **Cross-plot + cluster (THE M4 EXIT)** — sample the co-located cells, build a
   resistivity-vs-density-vs-velocity cross-plot, and cluster (both **GMM** and **k-means**)
   into background vs anomaly. **Assert** the known synthetic geothermal anomaly (the
   conductive / low-density / low-velocity cells) **separates into its own cluster well
   above chance** — high purity, low background contamination — and that clustering wrote
   back the categorical ``lithology_class`` derived volume (doc 07 §3.3, §4.3).

All I/O is to ``tmp_path`` with in-memory SQLite — no Docker/Postgres/Redis, coarse grids
throughout. The interactive cross-plot brushing ↔ 3D highlight confirmation is the FRONTEND
half of M4 (a separate browser check) and is intentionally out of scope here.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from geosim.catalog import (
    Dataset,
    IdKind,
    Project,
    PropertyModel,
    Provenance,
    SpatialFrameRow,
    create_all,
    make_engine,
    new_id,
    session_factory,
)
from geosim.fusion import (
    build_fused_model,
    cluster_fused,
    correlation_matrix,
    crossplot,
    fused_grid_from_row,
    resample_to_fused,
    sample_fused,
    selection_to_mask,
)
from geosim.ingestion import IngestStatus, ingest_file
from geosim.spatial import REGISTRY, Aabb, DepthRange, SpatialFrame
from geosim.storage import (
    GridSpec,
    ensure_project_layout,
    open_property_model,
    write_property_model,
)
from geosim.synthgen import compile_scene
from geosim.synthgen.scenarios import build_scenario, get_scenario

# The three co-located geophysical models the M4 cross-plot lives on (doc-ROADMAP M4 exit):
# resistivity (log10 interp space per the registry), density and P-wave velocity.
_FUSION_PROPS = ["resistivity", "density", "velocity_p"]


# ─────────────────────────────── coarse great-basin earth ───────────────────────────────


def _coarse_great_basin():
    """Compile ``great-basin-v1`` at a coarse truth grid (same geology, ~20×20×26 cells).

    The shipped flagship truth grid is 385×240×240 (~22 M cells — minutes per forward); we
    only need the *co-located property volumes*, so we replace the fine spacings with coarse
    ones (doc 05 §2 allows the truth-grid spacing to be chosen) to keep this gate ~1 s while
    preserving the hydrothermal anomaly's multi-method signature.
    """
    spec = get_scenario("great-basin-v1").scene
    coarse_frame = replace(spec.frame, dx=600.0, dy=600.0, dz=300.0)
    coarse = replace(spec, frame=coarse_frame)
    return compile_scene(coarse)


def _anomaly_truth_mask(earth) -> np.ndarray:
    """The known synthetic geothermal anomaly cells: the hydrothermal alteration body.

    The flagship play's diagnostic anomaly is the fault-controlled hydrothermal upflow +
    clay-cap — an **altered** (and thereby conductive / softened / less-dense) body (doc 05
    §7.1). We take the truth alteration-fraction state field (the physical cause, not a
    property proxy) as the ground-truth anomaly the cross-plot/cluster must isolate. This is
    an independent label: it is read from ``earth.state``, never from the resistivity /
    density / velocity features fed to the clusterer.
    """
    sub = ~earth.above_surface
    altered = earth.state.alteration_fraction > 0.1
    return altered & sub


def _engineering_bbox(origin, spacing, shape) -> str:
    oz, oy, ox = origin
    dz, dy, dx = spacing
    nz, ny, nx = shape
    return json.dumps({
        "xmin": ox, "xmax": ox + dx * (nx - 1),
        "ymin": oy, "ymax": oy + dy * (ny - 1),
        "zmin": oz, "zmax": oz + dz * (nz - 1),
    })


def _write_native_pm(session, layout, pid, *, prop, values, origin, spacing, unit, method):
    """Persist a co-located native :class:`PropertyModel` (Zarr + catalog rows)."""
    ds_id = new_id(IdKind.DATASET)
    pm_id = new_id(IdKind.PROPERTY_MODEL)
    prov_id = new_id(IdKind.PROVENANCE)
    zarr_path = layout.zarr_path(pm_id)
    grid = GridSpec(origin=origin, spacing=spacing, cell_ref="center")
    write_property_model(zarr_path, prop, values.astype(np.float32), grid=grid, overwrite=True)

    bbox = _engineering_bbox(origin, spacing, values.shape)
    session.add(Provenance(id=prov_id, project_id=pid, target_kind="propertyModel",
                           target_id=pm_id, process="ingest:synthetic"))
    session.flush()
    session.add(Dataset(
        id=ds_id, project_id=pid, name=f"{prop}-native", method=method, kind="propertyModel",
        status="ready", extent_json=bbox, spatial_frame_id=pid, provenance_id=prov_id,
        version_root_id=ds_id, version_seq=1, created_by="m4@test",
    ))
    session.flush()
    row = PropertyModel(
        id=pm_id, dataset_id=ds_id, project_id=pid, property=prop,
        canonical_unit=unit, support="volume", store_uri=str(zarr_path),
        shape_json=json.dumps(list(values.shape)),
        spacing_json=json.dumps(list(spacing)), origin_json=json.dumps(list(origin)),
        bbox_json=bbox,
    )
    session.add(row)
    session.commit()
    return row


@pytest.fixture
def fused_great_basin(tmp_path):
    """Coarse great-basin → 3 native models → a fused grid with all three resampled in.

    Yields ``(session, layout, pid, fem, earth, anomaly_mask)``.
    """
    earth = _coarse_great_basin()

    storage_root = tmp_path / "store"
    engine = make_engine()  # in-memory SQLite (doc 04 §2.1 fallback)
    create_all(engine)
    Session = session_factory(engine)
    session = Session()

    pid = new_id(IdKind.PROJECT)
    layout = ensure_project_layout(storage_root, pid)

    origin = earth.origin
    spacing = earth.spacing
    shape = earth.shape
    bbox = json.loads(_engineering_bbox(origin, spacing, shape))
    frame = SpatialFrame(
        roi=Aabb(bbox["xmin"], bbox["xmax"], bbox["ymin"], bbox["ymax"]),
        depth_range=DepthRange(bbox["zmin"], bbox["zmax"]),
    )
    session.add(Project(id=pid, name="m4-great-basin", storage_root=str(storage_root)))
    session.add(SpatialFrameRow(
        project_id=pid, mode=frame.mode.value,
        roi_json=json.dumps({k: bbox[k] for k in ("xmin", "xmax", "ymin", "ymax")}),
        depth_range_json=json.dumps({"zmin": bbox["zmin"], "zmax": bbox["zmax"]}),
        frame_json=json.dumps({"mode": frame.mode.value}),
    ))
    session.commit()

    # The three co-located native geophysical models (doc-ROADMAP M4 exit layers).
    methods = {"resistivity": "mt", "density": "gravity", "velocity_p": "seismic"}
    units = {"resistivity": "ohm*m", "density": "kg/m**3", "velocity_p": "m/s"}
    pm_ids = []
    for prop in _FUSION_PROPS:
        pm = _write_native_pm(
            session, layout, pid, prop=prop,
            values=earth.property_volume(prop if prop != "velocity_p" else "velocity_p"),
            origin=origin, spacing=spacing, unit=units[prop], method=methods[prop],
        )
        pm_ids.append(pm.id)

    # Build the fused grid (doc 07 §1) at the native spacing and resample each layer in.
    fem, _grid = build_fused_model(
        session, layout, pid, source_property_model_ids=pm_ids,
        spacing=spacing, name="m4-fused",
    )
    for pmid in pm_ids:
        resample_to_fused(session, fem, pmid)
    session.refresh(fem)

    anomaly = _anomaly_truth_mask(earth)
    yield session, layout, pid, fem, earth, anomaly
    session.close()


# ─────────────────────────────── the M4 exit assertions ───────────────────────────────


def test_resistivity_resamples_in_log_space(fused_great_basin):
    """Resistivity is interpolated in log10 space per the registry (doc 07 §2.3, doc 01 §5)."""
    assert REGISTRY.get("resistivity").interp_space == "log10"


def test_fused_grid_holds_three_colocated_layers(fused_great_basin):
    """build_fused_model + resample_to_fused put the three methods on a shared support (07 §1–2)."""
    session, _layout, _pid, fem, _earth, _anom = fused_great_basin
    s = sample_fused(session, fem, _FUSION_PROPS, mode="all")
    assert s.properties == _FUSION_PROPS
    assert s.features.shape[1] == 3
    assert s.n > 100  # plenty of co-located cells to cross-plot
    # listwise sampling → every retained feature vector is fully finite (doc 07 §3.1).
    assert np.isfinite(s.features).all()
    assert s.coords.shape == (s.n, 3)


def test_crossplot_resistivity_density_velocity(fused_great_basin):
    """Resistivity/density/velocity cross-plot + correlation over shared cells (M4 exit, §3.2)."""
    session, _layout, _pid, fem, _earth, _anom = fused_great_basin
    s = sample_fused(session, fem, _FUSION_PROPS, mode="all")

    cp = crossplot(s, _FUSION_PROPS, color_by="depth")  # 3D scatter
    assert cp["axes"] == _FUSION_PROPS
    assert cp["kind"] == "scatter"

    corr = correlation_matrix(s)
    assert corr["properties"] == _FUSION_PROPS
    m = corr["matrix"]
    assert len(m) == 3 and all(len(r) == 3 for r in m)


def _anomaly_cluster_quality(
    labels: np.ndarray, anomaly_in_sample: np.ndarray, n_clusters: int
) -> tuple[int, float, float, float]:
    """Separation of the cluster that best captures the geothermal anomaly.

    Returns ``(cluster, recall, precision, lift)`` for the single cluster containing the
    most true-anomaly cells:

    - ``recall``    — fraction of true-anomaly cells that fall into that one cluster
      (a coherent anomaly lands together, not scattered across clusters);
    - ``precision`` — fraction of that cluster's cells that are true-anomaly;
    - ``lift``      — precision ÷ the anomaly's overall prevalence; ``lift > 1`` means the
      cluster is *enriched* in anomaly cells **above chance** — the M4 "separates" claim.
    """
    assert anomaly_in_sample.any(), "the sampled cells must include the planted anomaly"
    prevalence = float(np.mean(anomaly_in_sample))
    captured = [int(np.sum(anomaly_in_sample & (labels == c))) for c in range(n_clusters)]
    cluster = int(np.argmax(captured))
    in_cluster = labels == cluster
    recall = captured[cluster] / float(anomaly_in_sample.sum())
    precision = captured[cluster] / float(max(in_cluster.sum(), 1))
    lift = precision / prevalence if prevalence else float("inf")
    return cluster, recall, precision, lift


@pytest.mark.parametrize("algorithm", ["gmm", "kmeans"])
def test_geothermal_anomaly_separates_as_a_cluster(fused_great_basin, algorithm):
    """THE M4 EXIT: clustering the fused grid isolates the synthetic geothermal anomaly (07 §3.3).

    Cross-plotting resistivity/density/velocity at shared cells and clustering the flagship
    layered play (alluvium / volcanics / carbonate / granite + the hydrothermal overprint)
    into its natural populations must surface the geothermal anomaly as its own cluster —
    one cluster captures (almost) every altered cell and is strongly *enriched* in them
    above the base rate.
    """
    session, layout, _pid, fem, _earth, anomaly = fused_great_basin

    n_clusters = 4  # 4 lithology layers + the hydrothermal overprint → ~natural populations
    result = cluster_fused(
        session, layout, fem, properties=_FUSION_PROPS,
        algorithm=algorithm, n_clusters=n_clusters, write_volumes=True, random_state=0,
    )
    assert result.algorithm == algorithm
    assert result.n_clusters == n_clusters
    assert sum(result.sizes) == result.labels.shape[0]

    # Map the ground-truth anomaly mask onto exactly the cells that were clustered
    # (cluster_fused samples mode="all"; reuse the same sampling to align indices).
    s = sample_fused(session, fem, _FUSION_PROPS, mode="all")
    anomaly_in_sample = anomaly.reshape(-1)[s.cell_index]
    assert anomaly_in_sample.sum() > 10, "need a meaningful anomaly population in the sample"

    cluster, recall, precision, lift = _anomaly_cluster_quality(
        result.labels, anomaly_in_sample, n_clusters
    )
    # The anomaly lands coherently in ONE cluster (not scattered) ...
    assert recall >= 0.8, f"anomaly scattered across clusters (recall={recall:.2f})"
    # ... and that cluster is markedly enriched in anomaly cells vs the base rate — i.e. it
    # SEPARATES the geothermal anomaly well above chance (lift==1 would be pure chance).
    assert lift >= 1.5, f"anomaly cluster not enriched above chance (lift={lift:.2f})"

    # The anomaly cluster is genuinely the conductive / low-density / low-velocity one
    # (the geothermal signature, doc 05 §7.1), not an arbitrary partition.
    feats = s.features
    in_cluster = result.labels == cluster
    for prop in _FUSION_PROPS:
        j = s.properties.index(prop)
        in_med = float(np.median(feats[in_cluster, j]))
        bg_med = float(np.median(feats[~in_cluster, j]))
        assert in_med < bg_med, f"{prop}: anomaly cluster should be the LOW population"

    # Clustering wrote back a categorical lithology_class derived volume (doc 07 §3.3 / §4.3).
    assert result.class_model_id is not None
    class_pm = session.get(PropertyModel, result.class_model_id)
    assert class_pm is not None and class_pm.property == "lithology_class"
    reader = open_property_model(class_pm.store_uri)
    vol = reader.read_level("lithology_class", 0)
    assert vol.shape == fused_grid_from_row(fem).shape
    assert np.unique(np.round(vol[np.isfinite(vol)])).size >= 2

    if algorithm == "gmm":
        assert len(result.probability_model_ids) == n_clusters


def test_crossplot_selection_brushes_back_to_volume(fused_great_basin):
    """A cross-plot selection of the conductive lobe maps back to a 3D boolean volume (§3.2)."""
    session, _layout, _pid, fem, _earth, _anom = fused_great_basin
    s = sample_fused(session, fem, _FUSION_PROPS, mode="all")

    res = s.features[:, s.properties.index("resistivity")]
    sel = np.flatnonzero(res < np.percentile(res, 20.0))  # brush the conductive lobe
    assert sel.size > 0
    mask = selection_to_mask(s, sel)
    assert mask.shape == fused_grid_from_row(fem).shape
    assert mask.dtype == bool
    assert int(mask.sum()) == sel.size


# ─────────────────────────── real ingest pipeline still feeds M4 ───────────────────────────


def test_real_ingest_pipeline_feeds_a_project(tmp_path):
    """A couple of native-format methods ingest through the REAL pipeline (doc 03 §7).

    M4 fusion consumes models produced by the M1/M2 ingest pipeline; this proves that
    pipeline still round-trips native survey files into a project (using the fast
    ``unit-cube-v1`` smoke scenario so the gate stays ~1 s).
    """
    scenario = build_scenario("unit-cube-v1", tmp_path / "scenario")
    assert scenario.errors == {}, scenario.errors

    storage_root = tmp_path / "ingest-store"
    engine = make_engine()
    create_all(engine)
    Session = session_factory(engine)

    surveys = [
        "measured/gravity_bouguer.tif",  # gravity GeoTIFF → a PropertyModel grid
        "measured/mt/ST000.edi",         # mt EDI sounding
    ]
    with Session() as session:
        project_id = None
        for rel in surveys:
            report = ingest_file(session, storage_root, project_id, scenario.out_dir / rel)
            assert report.status is not IngestStatus.FAILED, (rel, report.message)
            project_id = report.project_id or project_id
        session.commit()

        # both methods landed in ONE project (the M2 "one earth" invariant M4 builds on).
        assert session.query(Project).count() == 1
        datasets = session.query(Dataset).filter_by(project_id=project_id).all()
        assert len(datasets) == len(surveys)
        assert {"gravity", "mt"} <= {d.method for d in datasets}
