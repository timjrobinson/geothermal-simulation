"""Feature serving + 4-D backend tests (doc 04 §9.2, doc 02 §5/§8, doc 06 §5.3/§5.4/§9.4).

Synthesizes a SMALL scenario with the :mod:`geosim.synthgen` T0 forwards — a deviated
well (LAS + deviation CSV), a microseismic QuakeML+CSV catalog, and an InSAR LOS GeoTIFF
time-series — ingests each through :func:`geosim.ingestion.ingest_file` into one project,
then drives the :mod:`geosim.api.features` router through a FastAPI ``TestClient``:

- ``GET /projects/{pid}/features`` lists the ingested features and the ``has_time`` filter
  isolates the (time-bearing) microseismic cloud (doc 02 §8).
- ``GET /features/{id}/geometry`` serves a **loadable binary glTF** for a horizon surface
  (server-side GeoJSON→mesh conversion, doc 06 §5.3) and a GeoJSON ``Feature`` for the well
  path / horizon line.
- ``GET /features/{id}/points`` returns the microseismic 4-D cloud, time-window filtered.
- ``GET /wells/{id}/trajectory`` re-integrates the min-curvature polyline + MD/TVD and joins
  the LAS log curves vs MD (doc 06 §5.3 tube colouring).
- ``GET /projects/{pid}/time-extent`` unions the microseismic + InSAR epochs (doc 06 §9.4).

In-memory SQLite + a tmp ``storage_root`` (doc 04 §2.1 fallback) — no Docker/Postgres.
"""

from __future__ import annotations

import json
import struct
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from geosim.api import Settings, create_app
from geosim.catalog import Dataset as DatasetRow
from geosim.catalog import Feature as FeatureRow
from geosim.catalog import IdKind, new_id
from geosim.synthgen import (
    AnomalySpec,
    FaultSpec,
    FrameSpec,
    GeothermSpec,
    LayerSpec,
    SceneSpec,
    SurfaceSpec,
    compile_scene,
)
from geosim.synthgen.forward import Acquisition, get_forward

# InSAR adapter's TimeAxis convention (geosim.ingestion.adapters.insar): 2026-01-01 UTC +
# 12-day repeat pass per epoch. We mirror it to populate the dataset TimeAxis the 4-D
# writer does not yet persist (so the union endpoint sees the InSAR epochs, doc 02 §8).
_INSAR_EPOCH0 = datetime(2026, 1, 1, tzinfo=UTC)
_INSAR_STEP_S = 12 * 24 * 3600
_INSAR_N = 3


def _scene() -> SceneSpec:
    """A SMALL Basin-&-Range scene with a fault-controlled plume (mirrors the forward test)."""
    return SceneSpec(
        id="features-serving-v1",
        seed=5,
        frame=FrameSpec(
            xmin=-600, xmax=600, ymin=-600, ymax=600,
            zmin=-1000, zmax=300, dx=200, dy=200, dz=200,
        ),
        surface=SurfaceSpec(kind="tilted-block", base_elev=150.0, tilt_x=0.15),
        layers=(
            LayerSpec("alluvium", "surface", (50.0, 80.0)),
            LayerSpec("volcanics", "conformable", (150.0, 250.0)),
            LayerSpec("basement_granite", "conformable", "fill"),
        ),
        faults=(
            FaultSpec("range-front", trace=((-600, -100), (600, 200)),
                      kind="normal", dip=60, dip_azimuth=90, throw=200, is_conduit=True),
        ),
        geotherm=GeothermSpec(surface_temp=15.0, gradient=45.0),
        anomalies=(
            AnomalySpec(
                "upflow", footprint_center=(0.0, 0.0), footprint_radius_xy=300.0,
                top_elev=150.0, bottom_elev=-900.0, controlled_by="range-front",
                temp_peak=220.0, alteration_frac=0.9, porosity_boost=0.04,
                salinity_tds=8000.0, fracture_density=0.5,
                clay_cap_top_elev=100.0, clay_cap_thickness=150.0,
            ),
        ),
        rock_physics="default-v1",
    )


def _insar_epochs() -> list[str]:
    return [
        (_INSAR_EPOCH0 + timedelta(seconds=i * _INSAR_STEP_S)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for i in range(_INSAR_N)
    ]


@pytest.fixture(scope="module")
def synthesized(tmp_path_factory):
    """Forward-simulate + ingest the well / microseismic / InSAR scenario into one project."""
    from geosim.ingestion import ingest_file

    earth = compile_scene(_scene())
    out_dir = tmp_path_factory.mktemp("synth")
    storage = tmp_path_factory.mktemp("storage")
    acq = Acquisition(
        ms_n_events=24,
        insar_n_epochs=_INSAR_N,
        insar_pixel=200.0,
        params={"out_dir": str(out_dir)},
    )
    rng = np.random.default_rng(7)

    arts: dict[str, list[Path]] = {}
    for method, sub in [("welllog", None), ("microseismic", None), ("insar", None)]:
        arts[method] = [a.path for a in get_forward(method, sub).simulate(earth, acq, rng)]

    def _pick(method: str, *suffixes: str) -> Path:
        return next(
            (p for p in arts[method] if p.name.endswith(suffixes)), arts[method][0]
        )

    settings = Settings(storage_root=storage)
    app = create_app(settings)
    Session = app.state.session_factory
    session = Session()

    # welllog creates the project (feature + wellcurve obs); the rest join it.
    r_well = ingest_file(session, storage, None, _pick("welllog", ".las"))
    pid = r_well.project_id
    assert r_well.status.value.startswith("ok")
    r_ms = ingest_file(session, storage, pid, _pick("microseismic", ".quakeml", ".qml"))
    assert r_ms.status.value.startswith("ok")
    r_insar = ingest_file(session, storage, pid, _pick("insar", ".tif", ".tiff"))
    assert r_insar.status.value.startswith("ok")

    # The 4-D PropertyModel writer does not yet persist the InSAR TimeAxis (doc 02 §8 gap);
    # populate the dataset's time_extent_json as a complete writer would, so the global
    # time-extent union spans the InSAR epochs (doc 06 §9.4).
    insar_ds = (
        session.query(DatasetRow)
        .filter(DatasetRow.project_id == pid, DatasetRow.method == "insar")
        .one()
    )
    insar_ds.time_extent_json = json.dumps(
        {"epochs": _insar_epochs(), "unit": "ISO-8601-UTC"}
    )

    # Add a horizon SURFACE feature (a draped Engineering grid) so the glTF surface
    # conversion path is exercised (doc 06 §5.3); the synthgen seismic horizon ingests as a
    # 2-D LineString, which serves as GeoJSON.
    ny, nx = 4, 5
    xs = np.linspace(-400.0, 400.0, nx)
    ys = np.linspace(-400.0, 400.0, ny)
    nodes = [
        [float(x), float(y), float(-300.0 + 0.1 * x + 0.05 * y)]
        for y in ys for x in xs
    ]
    horizon_id = new_id(IdKind.FEATURE)
    pts = np.asarray(nodes, dtype=float)
    bbox = {
        "xmin": float(pts[:, 0].min()), "xmax": float(pts[:, 0].max()),
        "ymin": float(pts[:, 1].min()), "ymax": float(pts[:, 1].max()),
        "zmin": float(pts[:, 2].min()), "zmax": float(pts[:, 2].max()),
    }
    session.add(FeatureRow(
        id=horizon_id,
        dataset_id=None,
        project_id=pid,
        feature_type="horizon",
        store_uri=None,
        store_format="geojson",
        bbox_json=json.dumps(bbox),
        has_time=0,
        props_json=json.dumps({
            "geometry": {"type": "MultiPoint", "coordinates": nodes},
            "props": {"name": "top-volcanics", "grid": {"ny": ny, "nx": nx}},
        }),
    ))
    session.commit()

    client = TestClient(app)
    try:
        yield {
            "client": client,
            "pid": pid,
            "horizon_id": horizon_id,
            "insar_epochs": _insar_epochs(),
        }
    finally:
        session.close()


# ─────────────────────────────── feature list ───────────────────────────────


def test_features_list_includes_ingested_kinds(synthesized):
    client, pid = synthesized["client"], synthesized["pid"]
    resp = client.get(f"/projects/{pid}/features")
    assert resp.status_code == 200
    feats = resp.json()
    kinds = {f["featureKind"] for f in feats}
    assert {"wellPath", "pointCloud", "horizon"} <= kinds

    by_kind = {f["featureKind"]: f for f in feats}
    assert by_kind["wellPath"]["geometryEndpoint"] == "geojson"
    assert by_kind["pointCloud"]["geometryEndpoint"] == "points"
    assert by_kind["horizon"]["geometryEndpoint"] == "gltf"  # draped grid → surface mesh


def test_features_has_time_filter_isolates_microseismic(synthesized):
    client, pid = synthesized["client"], synthesized["pid"]
    timed = client.get(f"/projects/{pid}/features", params={"has_time": True}).json()
    assert timed and all(f["hasTime"] for f in timed)
    assert {f["featureKind"] for f in timed} == {"pointCloud"}

    static = client.get(f"/projects/{pid}/features", params={"has_time": False}).json()
    assert {"wellPath", "horizon"} <= {f["featureKind"] for f in static}
    assert all(not f["hasTime"] for f in static)


def test_features_kind_filter(synthesized):
    client, pid = synthesized["client"], synthesized["pid"]
    only = client.get(
        f"/projects/{pid}/features", params={"featureKind": "wellPath"}
    ).json()
    assert only and all(f["featureKind"] == "wellPath" for f in only)


def test_feature_list_unknown_project_404(synthesized):
    resp = synthesized["client"].get("/projects/prj_does_not_exist/features")
    assert resp.status_code == 404


# ─────────────────────────────── geometry serving ───────────────────────────────


def _well_feature(client, pid):
    feats = client.get(f"/projects/{pid}/features").json()
    return next(f for f in feats if f["featureKind"] == "wellPath")


def test_horizon_surface_yields_loadable_glb(synthesized):
    client, fid = synthesized["client"], synthesized["horizon_id"]
    resp = client.get(f"/features/{fid}/geometry")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("model/gltf-binary")
    glb = resp.content

    # glTF 2.0 §4.4.3 binary container: magic 'glTF', version 2, declared length matches.
    magic, version, total = struct.unpack("<III", glb[:12])
    assert magic == 0x46546C67
    assert version == 2
    assert total == len(glb)

    json_len, json_type = struct.unpack("<II", glb[12:20])
    assert json_type == 0x4E4F534A  # "JSON"
    doc = json.loads(glb[20:20 + json_len])
    prim = doc["meshes"][0]["primitives"][0]
    assert prim["mode"] == 4  # TRIANGLES
    assert "POSITION" in prim["attributes"]
    # POSITION accessor carries the REQUIRED min/max bounds (glTF 2.0 §5.1.1).
    pos = doc["accessors"][prim["attributes"]["POSITION"]]
    assert pos["type"] == "VEC3" and len(pos["min"]) == 3 and len(pos["max"]) == 3
    # a (4,5) grid → (3*4) cells * 2 triangles = 24 triangles → 72 indices
    idx_acc = doc["accessors"][prim["indices"]]
    assert idx_acc["count"] == (4 - 1) * (5 - 1) * 2 * 3
    assert doc["asset"]["extras"]["featureId"] == fid


def test_wellpath_geometry_is_geojson(synthesized):
    client, pid = synthesized["client"], synthesized["pid"]
    well = _well_feature(client, pid)
    resp = client.get(f"/features/{well['id']}/geometry")
    assert resp.status_code == 200
    assert "json" in resp.headers["content-type"]
    gj = resp.json()
    assert gj["type"] == "Feature"
    assert gj["geometry"]["type"] == "LineString"
    assert gj["properties"]["featureKind"] == "wellPath"


def test_feature_detail(synthesized):
    client, pid = synthesized["client"], synthesized["pid"]
    well = _well_feature(client, pid)
    detail = client.get(f"/features/{well['id']}").json()
    assert detail["featureKind"] == "wellPath"
    assert detail["geometryType"] == "LineString"
    assert detail["projectId"] == pid
    assert "wellId" in detail["props"]


# ─────────────────────────────── microseismic points (4-D) ───────────────────────────────


def _ms_feature(client, pid):
    feats = client.get(f"/projects/{pid}/features").json()
    return next(f for f in feats if f["featureKind"] == "pointCloud")


def test_microseismic_points_all(synthesized):
    client, pid = synthesized["client"], synthesized["pid"]
    ms = _ms_feature(client, pid)
    pts = client.get(f"/features/{ms['id']}/points").json()
    assert pts["count"] == len(pts["x"]) == len(pts["y"]) == len(pts["z"]) > 0
    assert len(pts["t"]) == pts["count"]
    assert len(pts["magnitude"]) == pts["count"]
    # times are ISO-8601 UTC strings
    assert all(t.endswith("Z") for t in pts["t"])


def test_microseismic_points_time_window_filters(synthesized):
    client, pid = synthesized["client"], synthesized["pid"]
    ms = _ms_feature(client, pid)
    full = client.get(f"/features/{ms['id']}/points").json()
    times = sorted(full["t"])
    assert len(times) >= 2
    # a window covering only the earliest half should drop later events.
    mid = times[len(times) // 2]
    windowed = client.get(
        f"/features/{ms['id']}/points", params={"t0": times[0], "t1": mid}
    ).json()
    assert 0 < windowed["count"] <= full["count"]
    assert all(times[0] <= t <= mid for t in windowed["t"])

    # an empty future window returns nothing.
    empty = client.get(
        f"/features/{ms['id']}/points", params={"t0": "2099-01-01T00:00:00Z"}
    ).json()
    assert empty["count"] == 0


def test_microseismic_points_bbox_filters(synthesized):
    client, pid = synthesized["client"], synthesized["pid"]
    ms = _ms_feature(client, pid)
    full = client.get(f"/features/{ms['id']}/points").json()
    xs, ys, zs = full["x"], full["y"], full["z"]
    # half-bbox in x around the data → strict subset.
    xmid = (min(xs) + max(xs)) / 2.0
    spec = f"{min(xs)},{xmid},{min(ys)},{max(ys)},{min(zs)},{max(zs)}"
    sub = client.get(f"/features/{ms['id']}/points", params={"bbox": spec}).json()
    assert sub["count"] <= full["count"]
    assert all(x <= xmid + 1e-6 for x in sub["x"])


# ─────────────────────────────── well trajectory ───────────────────────────────


def test_well_trajectory_polyline_md_tvd_and_logs(synthesized):
    client, pid = synthesized["client"], synthesized["pid"]
    well = _well_feature(client, pid)
    traj = client.get(f"/wells/{well['id']}/trajectory").json()

    n = len(traj["polyline"])
    assert n >= 2
    assert len(traj["md"]) == n
    assert len(traj["tvd"]) == n
    # MD is monotonically non-decreasing from the datum (min-curvature integration).
    md = traj["md"]
    assert md[0] <= md[-1]
    assert all(b >= a - 1e-6 for a, b in zip(md, md[1:], strict=False))
    # polyline is Engineering XYZ (3-tuples).
    assert all(len(p) == 3 for p in traj["polyline"])

    # joined LAS curves vs MD for tube colouring (doc 06 §5.3).
    logs = traj["logs"]
    assert logs["curves"], "expected joined well-log curves"
    assert len(logs["md"]) > 0
    # each curve has one sample per MD station.
    for samples in logs["curves"].values():
        assert len(samples) == len(logs["md"])


def test_trajectory_non_well_feature_422(synthesized):
    client, fid = synthesized["client"], synthesized["horizon_id"]
    resp = client.get(f"/wells/{fid}/trajectory")
    assert resp.status_code == 422


# ─────────────────────────────── 4-D time extent union ───────────────────────────────


def test_time_extent_unions_microseismic_and_insar(synthesized):
    client, pid = synthesized["client"], synthesized["pid"]
    te = client.get(f"/projects/{pid}/time-extent").json()
    assert te["count"] == len(te["epochs"]) > 0
    # sorted unique ISO epochs.
    assert te["epochs"] == sorted(set(te["epochs"]))
    assert te["t0"] == te["epochs"][0]
    assert te["t1"] == te["epochs"][-1]

    # the union must span both the InSAR series epochs and the microseismic event times.
    insar_epochs = synthesized["insar_epochs"]
    assert set(insar_epochs) <= set(te["epochs"])

    ms = _ms_feature(client, pid)
    ms_times = client.get(f"/features/{ms['id']}/points").json()["t"]
    assert set(ms_times) <= set(te["epochs"])

    # both an InSAR dataset and a microseismic feature are credited as sources.
    src_kinds = {(s["kind"], s.get("method") or s.get("featureKind")) for s in te["sources"]}
    assert ("dataset", "insar") in src_kinds
    assert ("feature", "pointCloud") in src_kinds


def test_time_extent_unknown_project_404(synthesized):
    resp = synthesized["client"].get("/projects/prj_missing/time-extent")
    assert resp.status_code == 404
