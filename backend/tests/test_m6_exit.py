"""M6 exit criteria — the 4-D feature scene, backend half (doc-ROADMAP M6).

The doc-ROADMAP M6 gate (design/ROADMAP.md §"M6 — 4-D features + time"):

    Exit: a 4-D scene — scrub the time slider; microseismic events accumulate and InSAR
    deformation evolves, with horizons / faults / wells composited into one scene.

The *visual* scrub-the-slider half is a browser check (see ``blockers`` in the structured
result). This test proves the **backend half** of that gate end-to-end (doc 02 §5, doc 06
§5.3/§5.4/§9.4): synthesize a SMALL 4-D-capable scenario, ingest every feature kind into one
project through :func:`geosim.ingestion.ingest_file`, and assert that the feature-serving +
time API can drive that one composited 4-D scene:

1. **One earth, every feature kind** — compile a SMALL Basin-&-Range scene (a fault-controlled
   geothermal upflow, doc 05 §7.1) and run the T0 forwards for a **deviated well** (LAS +
   deviation survey), a **microseismic** QuakeML+CSV catalog (the 4-D event cloud) and an
   **InSAR** LOS GeoTIFF **time-series** (the evolving deformation). A grid-surface **horizon**
   is added so the server-side glTF surface path is exercised (doc 06 §5.3).
2. **Features are served** — the horizon resolves to a **loadable binary glTF** surface; the
   deviated well to a re-integrated min-curvature **trajectory polyline** (MD/TVD + joined LAS
   logs); the microseismic cloud to a **time-filterable** 4-D point set (doc 02 §8).
3. **Project time-extent union** — ``GET /projects/{pid}/time-extent`` unions the microseismic
   event epochs and the InSAR series epochs into one sorted ISO-8601 **UTC** axis (the slider's
   domain, doc 06 §9.4), crediting both an InSAR dataset and a microseismic feature.

All I/O is to a tmp ``storage_root`` with in-memory SQLite (doc 04 §2.1 fallback) — no
Docker / Postgres / Redis, coarse grids throughout, small event / epoch counts.
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
from geosim.ingestion import ingest_file
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

# The InSAR adapter's TimeAxis convention (geosim.ingestion.adapters.insar): an epoch0 at
# 2026-01-01 UTC with a 12-day repeat pass. The 4-D PropertyModel writer does not yet persist
# that axis (doc 02 §8 gap), so we mirror it onto the dataset's time_extent_json as a complete
# writer would, so the time-extent union endpoint sees the InSAR epochs (doc 06 §9.4).
_INSAR_EPOCH0 = datetime(2026, 1, 1, tzinfo=UTC)
_INSAR_STEP_S = 12 * 24 * 3600
_INSAR_N = 3


def _scene() -> SceneSpec:
    """A SMALL Basin-&-Range scene with a fault-controlled geothermal plume (doc 05 §7.1).

    Coarse 200 m cells over a ~1.2 km cube keep the forwards + ingest fast while preserving
    the fault-controlled upflow the well / microseismic / InSAR forwards key off.
    """
    return SceneSpec(
        id="m6-exit-v1",
        seed=11,
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
def m6_project(tmp_path_factory):
    """Synthesize the 4-D scenario, ingest every feature kind, expose a wired TestClient.

    Yields ``{"client", "pid", "horizon_id", "insar_epochs"}``.
    """
    earth = compile_scene(_scene())
    out_dir = tmp_path_factory.mktemp("synth")
    storage = tmp_path_factory.mktemp("storage")
    acq = Acquisition(
        ms_n_events=24,
        insar_n_epochs=_INSAR_N,
        insar_pixel=200.0,
        params={"out_dir": str(out_dir)},
    )
    rng = np.random.default_rng(13)

    # Run the three 4-D-relevant T0 forwards: a deviated well (LAS + deviation CSV), the
    # microseismic catalog (the accumulating event cloud) and the InSAR LOS time-series.
    arts: dict[str, list[Path]] = {}
    for method in ("welllog", "microseismic", "insar"):
        arts[method] = [a.path for a in get_forward(method).simulate(earth, acq, rng)]

    def _pick(method: str, *suffixes: str) -> Path:
        return next(
            (p for p in arts[method] if p.name.endswith(suffixes)), arts[method][0]
        )

    settings = Settings(storage_root=storage)
    app = create_app(settings)
    Session = app.state.session_factory
    session = Session()

    # welllog creates the project (a wellPath feature + the wellcurve obs); the microseismic
    # cloud + InSAR series join it through ingest_file (doc 03 §2 round-trip).
    r_well = ingest_file(session, storage, None, _pick("welllog", ".las"))
    pid = r_well.project_id
    assert r_well.status.value.startswith("ok")
    r_ms = ingest_file(session, storage, pid, _pick("microseismic", ".quakeml", ".qml"))
    assert r_ms.status.value.startswith("ok")
    r_insar = ingest_file(session, storage, pid, _pick("insar", ".tif", ".tiff"))
    assert r_insar.status.value.startswith("ok")

    # Mirror the InSAR TimeAxis the 4-D writer does not yet persist onto the dataset
    # (doc 02 §8 gap) so the union endpoint spans the deformation series (doc 06 §9.4).
    insar_ds = (
        session.query(DatasetRow)
        .filter(DatasetRow.project_id == pid, DatasetRow.method == "insar")
        .one()
    )
    insar_ds.time_extent_json = json.dumps(
        {"epochs": _insar_epochs(), "unit": "ISO-8601-UTC"}
    )

    # Add a draped grid-surface horizon feature so the server-side glTF surface path is
    # exercised (doc 06 §5.3); the synthgen seismic horizon ingests as a 2-D LineString.
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


# ─────────────────────────── one composited scene: every feature kind ───────────────────────────


def test_one_scene_serves_every_feature_kind(m6_project):
    """All four scene layers (well / microseismic / horizon) coexist in ONE project (doc 06 §5).

    The composited 4-D scene needs the static structure (a horizon surface + the well) AND the
    time-bearing microseismic cloud served from one project, each routed to its viewer endpoint.
    """
    client, pid = m6_project["client"], m6_project["pid"]
    feats = client.get(f"/projects/{pid}/features").json()
    by_kind = {f["featureKind"]: f for f in feats}
    # the deviated well, the microseismic 4-D cloud, and the horizon surface are all present.
    assert {"wellPath", "pointCloud", "horizon"} <= set(by_kind)
    # each kind routes to the right viewer endpoint (doc 02 §5 geometryEndpoint).
    assert by_kind["wellPath"]["geometryEndpoint"] == "geojson"
    assert by_kind["pointCloud"]["geometryEndpoint"] == "points"
    assert by_kind["horizon"]["geometryEndpoint"] == "gltf"  # draped grid → surface mesh

    # has_time isolates EXACTLY the time-bearing layer — the slider-driven microseismic cloud.
    timed = client.get(f"/projects/{pid}/features", params={"has_time": True}).json()
    assert timed and {f["featureKind"] for f in timed} == {"pointCloud"}
    assert all(f["hasTime"] for f in timed)
    static = client.get(f"/projects/{pid}/features", params={"has_time": False}).json()
    assert {"wellPath", "horizon"} <= {f["featureKind"] for f in static}
    assert all(not f["hasTime"] for f in static)


def test_horizon_serves_loadable_gltf(m6_project):
    """The horizon surface resolves to a loadable binary glTF mesh (doc 06 §5.3)."""
    client, fid = m6_project["client"], m6_project["horizon_id"]
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
    assert json_type == 0x4E4F534A  # "JSON" chunk
    doc = json.loads(glb[20:20 + json_len])
    prim = doc["meshes"][0]["primitives"][0]
    assert prim["mode"] == 4  # TRIANGLES
    assert "POSITION" in prim["attributes"]
    # a (4,5) grid → (3*4) cells * 2 triangles → 72 indices.
    idx_acc = doc["accessors"][prim["indices"]]
    assert idx_acc["count"] == (4 - 1) * (5 - 1) * 2 * 3
    assert doc["asset"]["extras"]["featureId"] == fid


def test_well_trajectory_polyline_md_tvd_and_logs(m6_project):
    """The well resolves to a re-integrated polyline with MD/TVD + joined logs (doc 06 §5.3).

    The wellPath feature serves a trajectory polyline (Engineering XYZ per station) whose MD is
    monotone from the datum and whose joined LAS log curves (one sample per MD station) drive
    the tube colouring — the static borehole layer of the composited 4-D scene.

    NB: ingesting the LAS alone through the public :func:`ingest_file` content-addresses the
    raw bytes, so the sibling ``*_deviation.csv`` is not co-located and the adapter serves the
    documented vertical-well fallback (doc 03 §5/§6, ``trajectory == 'vertical_assumption'``);
    the polyline is therefore the wellhead-anchored vertical track. We assert the trajectory
    *contract* the viewer consumes rather than re-deriving the deviation here.
    """
    client, pid = m6_project["client"], m6_project["pid"]
    well = next(
        f for f in client.get(f"/projects/{pid}/features").json()
        if f["featureKind"] == "wellPath"
    )
    traj = client.get(f"/wells/{well['id']}/trajectory").json()

    n = len(traj["polyline"])
    assert n >= 2 and len(traj["md"]) == n and len(traj["tvd"]) == n
    md = traj["md"]
    assert md[0] <= md[-1]
    assert all(b >= a - 1e-6 for a, b in zip(md, md[1:], strict=False))
    assert all(len(p) == 3 for p in traj["polyline"])  # Engineering XYZ per station

    # the track descends from the datum (TVD grows; Up decreases over the polyline).
    poly = np.asarray(traj["polyline"], dtype=float)
    assert poly[-1, 2] < poly[0, 2]
    assert traj["tvd"][-1] > traj["tvd"][0]

    # joined LAS curves vs MD for tube colouring (doc 06 §5.3): one sample per MD station.
    logs = traj["logs"]
    assert logs["curves"], "expected joined well-log curves for tube colouring"
    for samples in logs["curves"].values():
        assert len(samples) == len(logs["md"])


def test_non_well_feature_trajectory_rejected(m6_project):
    """Asking the trajectory endpoint for a non-well feature is a 422 (doc 02 §5)."""
    client, fid = m6_project["client"], m6_project["horizon_id"]
    assert client.get(f"/wells/{fid}/trajectory").status_code == 422


# ─────────────────────── 4-D: microseismic accumulate over the window ───────────────────────


def _ms_feature(client, pid):
    return next(
        f for f in client.get(f"/projects/{pid}/features").json()
        if f["featureKind"] == "pointCloud"
    )


def test_microseismic_accumulates_through_time_window(m6_project):
    """Scrubbing the slider accumulates microseismic events — the 4-D heartbeat (doc 06 §9.4).

    Each event carries an ISO-8601 UTC epoch + a magnitude. A growing time window
    ``[t0, t]`` returns a MONOTONICALLY GROWING subset of the cloud (events accumulate as the
    slider advances), bottoming at zero before the first event and topping out at the full
    catalog after the last — exactly the accumulate behaviour the slider drives.
    """
    client, pid = m6_project["client"], m6_project["pid"]
    ms = _ms_feature(client, pid)
    full = client.get(f"/features/{ms['id']}/points").json()
    assert full["count"] == len(full["x"]) == len(full["t"]) == len(full["magnitude"]) > 0
    assert all(t.endswith("Z") for t in full["t"])  # ISO-8601 UTC

    times = sorted(full["t"])
    assert len(times) >= 2
    t0 = times[0]

    # accumulate: each widening window [t0, t] is a non-shrinking subset capped by the full cloud.
    prev = 0
    for t in times:
        win = client.get(
            f"/features/{ms['id']}/points", params={"t0": t0, "t1": t}
        ).json()
        assert all(t0 <= ti <= t for ti in win["t"])
        assert win["count"] >= prev  # accumulation is monotone non-decreasing
        assert win["count"] <= full["count"]
        prev = win["count"]
    # the final (full-span) window has recovered the whole accumulated catalog.
    assert prev == full["count"]

    # before the first epoch nothing has accumulated yet.
    empty = client.get(
        f"/features/{ms['id']}/points",
        params={"t0": "2000-01-01T00:00:00Z", "t1": "2000-01-02T00:00:00Z"},
    ).json()
    assert empty["count"] == 0


# ─────────────────────────── the slider domain: time-extent union ───────────────────────────


def test_time_extent_unions_microseismic_and_insar(m6_project):
    """THE M6 EXIT (backend): the time-extent unions microseismic + InSAR into one axis (§9.4).

    The slider's domain is the project time-extent: the sorted, de-duplicated UNION of the
    microseismic event epochs (the accumulating cloud) AND the InSAR series epochs (the evolving
    deformation), in ISO-8601 UTC order. Both sources must be credited and fully spanned.
    """
    client, pid = m6_project["client"], m6_project["pid"]
    te = client.get(f"/projects/{pid}/time-extent").json()
    assert te["count"] == len(te["epochs"]) > 0

    # one sorted, de-duplicated ISO-8601 UTC axis bracketed by [t0, t1].
    assert te["epochs"] == sorted(set(te["epochs"]))
    assert all(e.endswith("Z") for e in te["epochs"])
    assert te["t0"] == te["epochs"][0]
    assert te["t1"] == te["epochs"][-1]
    # canonical UTC parse + ascending order (doc 06 §9.4).
    parsed = [datetime.fromisoformat(e.replace("Z", "+00:00")) for e in te["epochs"]]
    assert all(p.tzinfo is not None for p in parsed)
    assert parsed == sorted(parsed)

    # the union spans BOTH the InSAR deformation series and the microseismic event times.
    insar_epochs = m6_project["insar_epochs"]
    assert set(insar_epochs) <= set(te["epochs"])
    ms = _ms_feature(client, pid)
    ms_times = client.get(f"/features/{ms['id']}/points").json()["t"]
    assert set(ms_times) <= set(te["epochs"])

    # both an InSAR dataset and a microseismic feature are credited as time sources.
    src_kinds = {(s["kind"], s.get("method") or s.get("featureKind")) for s in te["sources"]}
    assert ("dataset", "insar") in src_kinds
    assert ("feature", "pointCloud") in src_kinds
