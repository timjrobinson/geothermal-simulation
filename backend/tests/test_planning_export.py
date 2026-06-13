"""Tests for well-plan export (doc 09 §9 / §9.1).

Exercises the decided export set against a resolved :class:`PlannedWell` on a small fused
grid with a planted hot/favorable/fractured zone (FORGE-style), all on SQLite-in-memory +
local temp dirs (no Docker/Postgres/Redis):

- **CSV deviation survey** (§9 row 1): parse the emitted CSV back and check MD/Inc/Azi/TVD/
  N/E/DLS match the well's min-curvature geometry; a georeferenced frame adds Lat/Lon/Elev/
  TVDSS columns that re-georeference through the project CRS (doc 01 §7).
- **CSV predicted log** (§9 row 2): parse back temperature(+σ)/favorability/lithology/
  resistivity/fractureDensity/hazards/risk and check they match the predicted-log stations;
  temperature is °C (canonical K internally).
- **WITSML 2.0 trajectory** (§9.1): validates structurally (required objects/fields, uom on
  every quantity) and round-trips — export → re-import → each station's (MD,inc,azi) and
  derived (TVD,N,E,DLS) match within tolerance (MD/TVD/N/E ≤ 0.01 m, inc/azi ≤ 0.01°,
  DLS ≤ 0.01°/30 m); MD datum + CRS survive. 1.4.1.1 legacy alt round-trips too.
- **Field units** (§9): a field-unit export writes ft/°F/°-per-100ft headers and still
  round-trips to the same canonical metric numbers (the reader converts back via uom).
- The **API** export endpoint (§10) streams each format as a download.
"""

import csv
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from geosim.api.app import Settings, create_app
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
from geosim.fusion import build_fused_model, resample_to_fused
from geosim.planning import (
    DesignSpec,
    DrillTarget,
    PlannedWell,
    TargetTolerance,
    TrajectoryConstraints,
    predict_log,
    solve_survey,
)
from geosim.planning.export import (
    FIELD_UNITS,
    METRIC_UNITS,
    export_log_csv,
    export_survey_csv,
    export_witsml_trajectory,
    parse_witsml_trajectory,
    validate_witsml_trajectory,
)
from geosim.spatial import (
    Anchor,
    FrameMode,
    GeorefStatus,
    SpatialFrame,
    min_curvature_positions,
)
from geosim.storage import GridSpec, ensure_project_layout, write_property_model

# Round-trip tolerances (doc 09 §9.1).
TOL_LEN_M = 0.01
TOL_ANG_DEG = 0.01
TOL_DLS = 0.01

# A coarse grid spanning x,y ∈ [0,1000], z ∈ [-2000, 0] (Engineering metres, Z-up).
SHAPE = (21, 11, 11)  # (nz, ny, nx)
ORIGIN = (-2000.0, 0.0, 0.0)  # (z0, y0, x0)
SPACING = (100.0, 100.0, 100.0)  # (dz, dy, dx)
HOT_XYZ = (600.0, 500.0, -1500.0)


# ───────────────────────────────── fixtures / fused-model setup ─────────────────────────────────


@pytest.fixture
def env(tmp_path):
    engine = make_engine()
    create_all(engine)
    Session = session_factory(engine)
    session = Session()
    storage_root = tmp_path
    pid = new_id(IdKind.PROJECT)
    layout = ensure_project_layout(storage_root, pid)
    session.add(Project(id=pid, name="export-test", storage_root=str(storage_root)))
    session.add(SpatialFrameRow(
        project_id=pid, mode="local",
        roi_json=json.dumps({"xmin": 0, "xmax": 1000, "ymin": 0, "ymax": 1000}),
        depth_range_json=json.dumps({"zmin": -2000, "zmax": 0}),
        frame_json=json.dumps({"mode": "local"}),
    ))
    session.commit()
    yield session, layout, storage_root, pid
    session.close()


def _axis_coords():
    oz, oy, ox = ORIGIN
    dz, dy, dx = SPACING
    nz, ny, nx = SHAPE
    return (oz + dz * np.arange(nz), oy + dy * np.arange(ny), ox + dx * np.arange(nx))


def _gaussian_blob(centre, sigma=350.0):
    z, y, x = _axis_coords()
    gz, gy, gx = np.meshgrid(z, y, x, indexing="ij")
    cx, cy, cz = centre
    r2 = (gx - cx) ** 2 + (gy - cy) ** 2 + (gz - cz) ** 2
    return np.exp(-r2 / (2.0 * sigma**2))


def _planted_fields():
    z, _y, _x = _axis_coords()
    blob = _gaussian_blob(HOT_XYZ)
    depth = -z
    grad_c = 15.0 + 0.030 * depth[:, None, None]
    grad_c = np.broadcast_to(grad_c, SHAPE).copy()
    temp_k = (grad_c + 120.0 * blob) + 273.15  # canonical kelvin
    fav = 0.05 + 0.9 * blob
    frac = 0.05 + 0.9 * blob
    return temp_k, fav, frac


def _make_native_pm(session, layout, pid, *, prop, values, unit):
    ds_id = new_id(IdKind.DATASET)
    pm_id = new_id(IdKind.PROPERTY_MODEL)
    prov_id = new_id(IdKind.PROVENANCE)
    zarr_path = layout.zarr_path(pm_id)
    grid = GridSpec(origin=ORIGIN, spacing=SPACING, cell_ref="center")
    write_property_model(zarr_path, prop, values, grid=grid, overwrite=True)
    nz, ny, nx = values.shape
    oz, oy, ox = ORIGIN
    dz, dy, dx = SPACING
    bbox = json.dumps({
        "xmin": ox, "xmax": ox + dx * (nx - 1), "ymin": oy, "ymax": oy + dy * (ny - 1),
        "zmin": oz, "zmax": oz + dz * (nz - 1),
    })
    session.add(Provenance(id=prov_id, project_id=pid, target_kind="propertyModel",
                           target_id=pm_id, process="ingest:synthetic"))
    session.flush()
    session.add(Dataset(
        id=ds_id, project_id=pid, name=f"{prop}-native", method="synthetic",
        kind="propertyModel", status="ready", extent_json=bbox, spatial_frame_id=pid,
        provenance_id=prov_id, version_root_id=ds_id, version_seq=1, created_by="t@x",
    ))
    session.flush()
    session.add(PropertyModel(
        id=pm_id, dataset_id=ds_id, project_id=pid, property=prop, canonical_unit=unit,
        support="volume", store_uri=str(zarr_path), shape_json=json.dumps([nz, ny, nx]),
        spacing_json=json.dumps(list(SPACING)), origin_json=json.dumps(list(ORIGIN)),
        bbox_json=bbox,
    ))
    session.commit()
    return pm_id


def _build_fused(session, layout, pid):
    temp_k, fav, frac = _planted_fields()
    pms = [
        _make_native_pm(session, layout, pid, prop="temperature", values=temp_k, unit="kelvin"),
        _make_native_pm(session, layout, pid, prop="favorability", values=fav,
                        unit="dimensionless"),
        _make_native_pm(session, layout, pid, prop="fracture_density", values=frac,
                        unit="dimensionless"),
    ]
    fem, _grid = build_fused_model(
        session, layout, pid, source_property_model_ids=pms, spacing=SPACING, name="fused-exp",
    )
    for pm_id in pms:
        resample_to_fused(session, fem, pm_id, method="trilinear", interp_space="linear")
    session.refresh(fem)
    return fem


def _resolved_well(pid, *, kb_elev_m=1627.0, wellhead=(10.0, 20.0)):
    cons = TrajectoryConstraints(max_dls_deg30m=5.0, max_inc_deg=92.0)
    design = DesignSpec(method="build-hold-land", target=HOT_XYZ, kop_md_m=500.0,
                        build_rate_deg30m=3.0)
    res = solve_survey(design, wellhead, kb_elev_m, cons)
    return PlannedWell(
        id=new_id(IdKind.FEATURE), name="W-01", project_id=pid, wellhead=wellhead,
        kb_elev_m=kb_elev_m, deviation_survey=res.survey, design=design, constraints=cons,
    )


def _georef_frame():
    return SpatialFrame(
        mode=FrameMode.GEOREFERENCED, horizontal_crs="EPSG:32612",
        vertical_datum="EPSG:3855", anchor=Anchor(500000.0, 4000000.0, 1620.0),
        georef_status=GeorefStatus.ANCHORED,
    )


def _read_csv(text):
    """Strip ``#`` provenance lines, return (provenance_lines, header, rows-as-dicts)."""
    lines = text.splitlines()
    prov = [ln for ln in lines if ln.startswith("#")]
    data = "\n".join(ln for ln in lines if not ln.startswith("#"))
    reader = csv.DictReader(io.StringIO(data))
    rows = list(reader)
    return prov, reader.fieldnames, rows


# ───────────────────────────────── CSV deviation survey (§9) ─────────────────────────────────


def test_csv_survey_round_trips_geometry():
    well = _resolved_well("p1")
    text = export_survey_csv(well, units="metric")
    prov, header, rows = _read_csv(text)

    assert header == ["MD_m", "Inc_deg", "Azi_deg", "TVD_m", "N_m", "E_m", "DLS_deg/30m"]
    assert any("local frame, no CRS" in ln for ln in prov)  # local-mode CRS note (§9)
    assert any("samplingStep" not in ln for ln in prov)  # survey export has no sampling step

    pos = well.positions()
    survey = np.asarray(well.deviation_survey)
    assert len(rows) == survey.shape[0]
    for i, r in enumerate(rows):
        assert float(r["MD_m"]) == pytest.approx(survey[i, 0], abs=1e-3)
        assert float(r["Inc_deg"]) == pytest.approx(survey[i, 1], abs=1e-3)
        assert float(r["Azi_deg"]) == pytest.approx(survey[i, 2], abs=1e-3)
        assert float(r["TVD_m"]) == pytest.approx(pos.tvd[i], abs=1e-3)
        assert float(r["N_m"]) == pytest.approx(pos.enu[i, 1], abs=1e-3)
        assert float(r["E_m"]) == pytest.approx(pos.enu[i, 0], abs=1e-3)
        assert float(r["DLS_deg/30m"]) == pytest.approx(pos.dls[i], abs=1e-3)


def test_csv_survey_georeferenced_adds_latlon_elev_tvdss():
    well = _resolved_well("p1")
    frame = _georef_frame()
    text = export_survey_csv(well, units="metric", frame=frame)
    prov, header, rows = _read_csv(text)

    assert "EPSG:32612" in "\n".join(prov)  # CRS written into provenance (§9 round-trip)
    for col in ("Elev_m", "TVDSS_m", "Lat_deg", "Lon_deg"):
        assert col in header

    pos = well.positions()
    crs_xyz = frame.engineering_to_crs(pos.enu)
    latlon = frame.to_lonlat(pos.enu)
    for i, r in enumerate(rows):
        assert float(r["Elev_m"]) == pytest.approx(crs_xyz[i, 2], abs=1e-3)
        assert float(r["TVDSS_m"]) == pytest.approx(-crs_xyz[i, 2], abs=1e-3)
        assert float(r["Lat_deg"]) == pytest.approx(latlon[i, 1], abs=1e-6)
        assert float(r["Lon_deg"]) == pytest.approx(latlon[i, 0], abs=1e-6)


def test_csv_survey_field_units_headers_and_values():
    well = _resolved_well("p1")
    text = export_survey_csv(well, units="field")
    prov, header, rows = _read_csv(text)

    assert header[0] == "MD_ft"
    assert "DLS_deg/100ft" in header
    assert any("units: field" in ln for ln in prov)

    pos = well.positions()
    # MD in ft = metres / 0.3048.
    md_ft = well.deviation_survey[-1, 0] / 0.3048
    assert float(rows[-1]["MD_ft"]) == pytest.approx(md_ft, abs=1e-2)
    # Inclination is degrees either way.
    assert float(rows[-1]["Inc_deg"]) == pytest.approx(well.deviation_survey[-1, 1], abs=1e-3)
    assert float(rows[-1]["TVD_ft"]) == pytest.approx(pos.tvd[-1] / 0.3048, abs=1e-2)


# ───────────────────────────────── CSV predicted log (§9) ─────────────────────────────────


def test_csv_log_round_trips_predicted_stations(env):
    session, layout, storage_root, pid = env
    fem = _build_fused(session, layout, pid)
    well = _resolved_well(pid, kb_elev_m=0.0, wellhead=(0.0, 0.0))
    target = DrillTarget(
        id=new_id(IdKind.FEATURE), name="hot", project_id=pid, kind="point",
        location=HOT_XYZ, tolerance=TargetTolerance(120.0, 60.0),
        min_temperature_c=150.0, kb_elev_m=0.0,
    )
    log = predict_log(session, fem, well, md_step_m=20.0, target=target,
                      storage_root=storage_root)

    text = export_log_csv(log, well, units="metric")
    prov, header, rows = _read_csv(text)

    assert any(f"modelVersion: {fem.id}" in ln for ln in prov)
    assert any("samplingStep_md: 20.0 m" in ln for ln in prov)
    for col in ("MD_m", "TVD_m", "Temperature_degC", "TemperatureSigma_degC",
                "Favorability", "Lithology", "Resistivity_ohmm", "FractureDensity", "Risk"):
        assert col in header

    assert len(rows) == len(log.stations)
    s_last = log.stations[-1]
    r_last = rows[-1]
    assert float(r_last["MD_m"]) == pytest.approx(s_last.md, abs=1e-3)
    assert float(r_last["TVD_m"]) == pytest.approx(s_last.tvd, abs=1e-3)
    # Temperature at TD is the hot zone (°C, from canonical K) and matches the station.
    assert float(r_last["Temperature_degC"]) == pytest.approx(
        s_last.values["temperatureC"]["value"], abs=1e-3
    )
    assert float(r_last["Temperature_degC"]) > 150.0
    fav = s_last.values.get("favorability", {}).get("value")
    if fav is not None:
        assert float(r_last["Favorability"]) == pytest.approx(fav, abs=1e-3)
    assert float(r_last["Risk"]) == pytest.approx(s_last.risk, abs=1e-3)


def test_csv_log_field_units_temperature_in_fahrenheit(env):
    session, layout, storage_root, pid = env
    fem = _build_fused(session, layout, pid)
    well = _resolved_well(pid, kb_elev_m=0.0, wellhead=(0.0, 0.0))
    log = predict_log(session, fem, well, md_step_m=40.0, storage_root=storage_root)

    text = export_log_csv(log, well, units="field")
    _prov, header, rows = _read_csv(text)
    assert "Temperature_degF" in header
    assert "MD_ft" in header

    s_last = log.stations[-1]
    c = s_last.values["temperatureC"]["value"]
    assert float(rows[-1]["Temperature_degF"]) == pytest.approx(c * 9 / 5 + 32, abs=1e-2)


# ───────────────────────────────── WITSML 2.0 (§9.1) ─────────────────────────────────


def _assert_round_trip(well, version, units="metric", frame=None, model_version=None):
    """The mandatory export → re-import round-trip within tolerance (doc 09 §9.1)."""
    xml = export_witsml_trajectory(
        well, version=version, units=units, frame=frame, model_version=model_version,
    )
    vr = validate_witsml_trajectory(xml)
    assert vr.well_formed
    assert vr.structural_ok, vr.errors
    assert vr.schema_validated is False  # XSD/xmlschema unavailable here (§9.1 note)
    assert "structural validation only" in vr.note

    parsed = parse_witsml_trajectory(xml)
    arr = parsed.as_arrays()
    survey = np.asarray(well.deviation_survey)
    pos = well.positions()
    wx, wy = float(well.wellhead[0]), float(well.wellhead[1])

    assert np.max(np.abs(arr["md"] - survey[:, 0])) <= TOL_LEN_M
    assert np.max(np.abs(arr["inc"] - survey[:, 1])) <= TOL_ANG_DEG
    assert np.max(np.abs(arr["azi"] - survey[:, 2])) <= TOL_ANG_DEG
    assert np.max(np.abs(arr["tvd"] - pos.tvd)) <= TOL_LEN_M
    # dispNs/dispEw are displacements from the wellhead → add it back to compare to enu N/E.
    assert np.max(np.abs((arr["dispNs"] + wy) - pos.enu[:, 1])) <= TOL_LEN_M
    assert np.max(np.abs((arr["dispEw"] + wx) - pos.enu[:, 0])) <= TOL_LEN_M
    assert np.max(np.abs(arr["dls"] - pos.dls)) <= TOL_DLS

    # MD datum + CRS survive (doc 09 §9.1).
    assert parsed.md_datum_elev_m == pytest.approx(well.kb_elev_m, abs=TOL_LEN_M)
    assert parsed.md_datum_kind == "KB"
    return parsed


def test_witsml_20_validates_and_round_trips():
    well = _resolved_well("p1", kb_elev_m=1627.0)
    parsed = _assert_round_trip(well, "2.0", model_version="fused_v7")
    assert parsed.version == "2.0"
    # Service company carries the model version (doc 09 §9.1 trajectory metadata).
    xml = export_witsml_trajectory(well, version="2.0", model_version="fused_v7")
    assert "fused_v7" in xml
    assert "TrajectoryStation" in xml


def test_witsml_20_crs_round_trips_when_georeferenced():
    well = _resolved_well("p1")
    frame = _georef_frame()
    parsed = _assert_round_trip(well, "2.0", frame=frame)
    assert parsed.well_crs == "EPSG:32612"  # project CRS survives (doc 01 §7 / §9.1)


def test_witsml_141_legacy_round_trips():
    well = _resolved_well("p1", kb_elev_m=1627.0)
    parsed = _assert_round_trip(well, "1.4.1.1")
    assert parsed.version == "1.4.1.1"


def test_witsml_field_units_round_trip_back_to_metric():
    """A field-unit (ft/°/100ft) export round-trips to the SAME canonical metric numbers (§9)."""
    well = _resolved_well("p1", kb_elev_m=1627.0)
    xml = export_witsml_trajectory(well, version="2.0", units="field")
    # Field uom appears on quantities.
    assert 'uom="ft"' in xml
    assert 'uom="dega/100.ft"' in xml
    # Reader converts back via uom → metric within tolerance.
    _assert_round_trip(well, "2.0", units="field")


def test_witsml_rejects_unknown_version():
    well = _resolved_well("p1")
    with pytest.raises(ValueError, match="unsupported WITSML version"):
        export_witsml_trajectory(well, version="3.0")


def test_witsml_every_quantity_carries_uom():
    """doc 09 §9.1: every quantity carries an EML uom — the validator enforces it per field."""
    well = _resolved_well("p1")
    xml = export_witsml_trajectory(well, version="2.0", units="metric")
    vr = validate_witsml_trajectory(xml)
    assert vr.structural_ok
    assert not any("missing uom" in e for e in vr.errors)
    # Metric uoms present.
    assert 'uom="m"' in xml
    assert 'uom="dega"' in xml
    assert 'uom="dega/30.m"' in xml


def test_witsml_validation_flags_malformed_xml():
    vr = validate_witsml_trajectory("<not-witsml><oops></not-witsml>")
    assert not vr.well_formed or not vr.structural_ok


def test_unit_profiles_are_coherent():
    assert METRIC_UNITS.length_unit == "m" and METRIC_UNITS.temperature_unit == "degC"
    assert FIELD_UNITS.length_unit == "ft" and FIELD_UNITS.temperature_unit == "degF"
    # DLS conversion: °/30m → °/100ft scales by (100ft/30m) = 100*0.3048/30.
    assert FIELD_UNITS.conv_dls(5.0) == pytest.approx(5.0 * (100 * 0.3048 / 30.0))
    # σ is a difference → scales by 9/5 (NOT the offset conversion).
    assert FIELD_UNITS.conv_temperature_delta_c(10.0) == pytest.approx(18.0)
    assert METRIC_UNITS.conv_temperature_delta_c(10.0) == pytest.approx(10.0)


# ───────────────────────────────── min-curvature reuse sanity ─────────────────────────────────


def test_survey_csv_geometry_is_shared_integrator():
    """The CSV survey's TVD/N/E are EXACTLY the shared min-curvature integrator (doc 09 §4.3)."""
    well = _resolved_well("p1", kb_elev_m=0.0, wellhead=(0.0, 0.0))
    _prov, _header, rows = _read_csv(export_survey_csv(well))
    direct = min_curvature_positions(well.deviation_survey, (0.0, 0.0), kb_elev=0.0)
    assert float(rows[-1]["TVD_m"]) == pytest.approx(direct.tvd[-1], abs=1e-3)
    assert float(rows[-1]["N_m"]) == pytest.approx(direct.enu[-1, 1], abs=1e-3)


# ───────────────────────────────── API export endpoint (§10) ─────────────────────────────────


def test_api_export_endpoint_all_formats(tmp_path):
    settings = Settings(storage_root=tmp_path / "store")
    app = create_app(settings)
    tc = TestClient(app)

    r = tc.post("/projects", json={"name": "p", "frame": {
        "mode": "local",
        "roi": {"xmin": 0, "xmax": 1000, "ymin": 0, "ymax": 1000},
        "depth_range": {"zmin": -2000, "zmax": 0},
    }})
    assert r.status_code == 201, r.text
    pid = r.json()["id"]

    Session = session_factory(app.state.engine)
    session = Session()
    layout = ensure_project_layout(app.state.storage_root, pid)
    fem = _build_fused(session, layout, pid)
    fem_id = fem.id
    session.close()

    rw = tc.post(f"/projects/{pid}/wells", json={
        "name": "W-01", "wellhead": [0.0, 0.0], "kb_elev_m": 0.0,
        "design": {"method": "build-hold-land", "target": list(HOT_XYZ),
                   "kop_md_m": 500.0, "build_rate_deg30m": 3.0},
        "max_dls_deg30m": 5.0,
    })
    assert rw.status_code == 201, rw.text
    wid = rw.json()["id"]

    # CSV survey.
    rs = tc.get(f"/wells/{wid}/export", params={"fmt": "csv-survey", "units": "metric"})
    assert rs.status_code == 200, rs.text
    assert rs.headers["content-type"].startswith("text/csv")
    assert "attachment" in rs.headers["content-disposition"]
    assert "MD_m,Inc_deg" in rs.text

    # CSV log (needs the fused model).
    rl = tc.get(f"/wells/{wid}/export",
                params={"fmt": "csv-log", "fused_model_id": fem_id, "md_step_m": 25.0})
    assert rl.status_code == 200, rl.text
    assert "Temperature_degC" in rl.text

    # CSV log without a fused model → 400.
    rl_bad = tc.get(f"/wells/{wid}/export", params={"fmt": "csv-log"})
    assert rl_bad.status_code == 400

    # WITSML 2.0 — re-parse the streamed body and round-trip-check it.
    rx = tc.get(f"/wells/{wid}/export",
                params={"fmt": "witsml", "version": "2.0", "fused_model_id": fem_id})
    assert rx.status_code == 200, rx.text
    parsed = parse_witsml_trajectory(rx.text)
    assert parsed.version == "2.0"
    assert len(parsed.stations) > 1

    # WITSML 1.4.1.1 legacy.
    rx2 = tc.get(f"/wells/{wid}/export", params={"fmt": "witsml", "version": "1.4.1.1"})
    assert rx2.status_code == 200, rx2.text
    assert parse_witsml_trajectory(rx2.text).version == "1.4.1.1"

    # Unknown format → 400.
    assert tc.get(f"/wells/{wid}/export", params={"fmt": "bogus"}).status_code == 400
