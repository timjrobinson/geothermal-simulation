"""Tests for the shippable scenarios + ``build_scenario`` driver (doc 05 §5, §7).

Builds ``unit-cube-v1`` — the CI smoke scenario (doc 05 §7 row 1) — at a SMALL/coarse
truth grid into ``tmp_path`` and asserts the doc 05 §5 self-contained folder: ``measured/``
native files (gravity + mt + seismic + welllog re-readable with their parsing libraries),
the ``truth/`` scoring oracle (the conductive cube is present), and ``manifest.json``
(seed + per-file synthetic provenance). All I/O is to ``tmp_path`` — headless, no
Docker/Postgres/Redis; the coarse grid keeps the full build to ~1 s.
"""

from __future__ import annotations

import json

import lasio
import numpy as np
import pandas as pd
import rasterio
import segyio

from geosim.storage import open_property_model
from geosim.synthgen.scenarios import (
    SCENARIOS,
    build_scenario,
    get_scenario,
    list_scenarios,
)

# --------------------------------------------------------------------------- registry


def test_scenarios_registered():
    """Both shipped scenarios are registered (doc 05 §7 rows 1 + flagship)."""
    ids = list_scenarios()
    assert "unit-cube-v1" in ids
    assert "great-basin-v1" in ids
    assert set(ids) == set(SCENARIOS)


def test_unit_cube_is_small_and_coarse():
    """The smoke scenario stays small so CI builds run in seconds (doc 05 §7 row 1)."""
    scene = get_scenario("unit-cube-v1").scene
    nz, ny, nx = scene.frame.shape
    assert nz * ny * nx < 50_000  # coarse truth grid
    assert scene.id == "unit-cube-v1"


def test_great_basin_flagship_structure():
    """great-basin-v1 has the flagship structure: 4 layers, conduit fault, plume (§7.1)."""
    scene = get_scenario("great-basin-v1").scene
    assert [layer.unit for layer in scene.layers] == [
        "alluvium",
        "volcanics",
        "carbonate",
        "basement_granite",
    ]
    (fault,) = scene.faults
    assert fault.is_conduit and fault.kind == "normal"
    assert abs(fault.dip - 60.0) < 1e-6 and abs(fault.throw - 700.0) < 1e-6
    (anomaly,) = scene.anomalies
    assert anomaly.kind == "hydrothermal-plume"
    assert anomaly.controlled_by == "range-front"
    assert abs(anomaly.temp_peak - 220.0) < 1e-6


# --------------------------------------------------------------------------- build


def test_build_unit_cube_full_folder(tmp_path):
    """Build unit-cube-v1 → the doc 05 §5 self-contained folder, all parts present."""
    result = build_scenario("unit-cube-v1", tmp_path)
    out = result.out_dir

    # no forward failed (doc 05 §6 T0 for every method)
    assert result.errors == {}, result.errors
    assert result.artifacts

    # scaffolding (doc 05 §5)
    for name in ("scene.jsonc", "acquisition.jsonc", "frame.json", "manifest.json"):
        assert (out / name).exists(), name
    assert (out / "measured").is_dir()
    assert (out / "truth").is_dir()

    # frame.json is the scenario SpatialFrame (doc 01 §2): local + synthetic surface model
    frame = json.loads((out / "frame.json").read_text())
    assert frame["mode"] == "local"
    assert frame["surfaceModel"] == "synthetic:unit-cube-v1"

    # manifest: seed + synthetic provenance per measured file (doc 05 §5)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["seed"] == result.truth.spec.seed
    assert manifest["synthetic"] is True
    assert manifest["measured"]
    for rec in manifest["measured"]:
        assert rec["synthetic"] is True
        assert rec["provenance"]["source"] == "synthgen"
        assert rec["provenance"]["sceneId"] == "unit-cube-v1"
        assert rec["sha256"]
    # truth is recorded but flagged never-ingested (decision #6)
    assert manifest["truth"]
    assert all(rec["ingested"] is False for rec in manifest["truth"])


def test_truth_bundle_and_conductor(tmp_path):
    """truth/ holds the scoring oracle and the conductive cube is present (doc 05 §5)."""
    result = build_scenario("unit-cube-v1", tmp_path)
    truth_dir = result.out_dir / "truth"

    # ground-truth zarr property models + features (doc 05 §5)
    assert (truth_dir / "features.geojson").exists()
    for key in ("resistivity", "density", "velocity_p", "temperature"):
        assert (truth_dir / f"{key}.zarr").exists()

    # the conductive cube: a strong resistivity low vs. the background (doc 05 §7 scoring)
    reader = open_property_model(truth_dir / "resistivity.zarr")
    res = reader.read_level("resistivity")
    assert res.shape == result.truth.shape
    assert np.isfinite(res).all()
    assert float(np.nanmin(res)) < 0.2 * float(np.nanmedian(res))


def test_measured_files_readable(tmp_path):
    """At least gravity + mt + seismic + welllog measured files re-read (doc 05 §5)."""
    result = build_scenario("unit-cube-v1", tmp_path)
    measured = result.out_dir / "measured"

    # gravity: CSV stations re-read with pandas, plausible Bouguer values
    grav = pd.read_csv(measured / "gravity_stations.csv")
    assert {"x", "y", "bouguer_mgal"} <= set(grav.columns)
    assert len(grav) > 0
    assert np.isfinite(grav["bouguer_mgal"]).all()
    # gravity GeoTIFF re-reads with rasterio
    with rasterio.open(measured / "gravity_bouguer.tif") as ds:
        assert ds.count == 1
        assert ds.read(1).size > 0

    # mt: at least one EDI station file written
    edis = sorted((measured / "mt").glob("*.edi"))
    assert edis
    assert edis[0].read_text(encoding="utf-8").lstrip().startswith(">")

    # seismic: SEG-Y re-reads with segyio
    with segyio.open(measured / "seismic_lineAA.segy", "r", ignore_geometry=True) as f:
        assert f.tracecount > 0
        assert len(f.trace[0]) > 0

    # welllog: LAS re-reads with lasio and carries the expected curves
    las = lasio.read(str(measured / "wells" / "UC-1.las"))
    curves = set(las.curves.keys())
    assert {"RES", "VP", "TEMP"} <= curves
    assert las["RES"].size > 0


def test_build_is_deterministic(tmp_path):
    """Same (scene, seed) → byte-identical measured checksums (doc 05 §1 invariant).

    QuakeML is excluded: ObsPy stamps a fresh creation timestamp + random resource IDs
    into the XML wrapper on write, so the *file* differs run-to-run even though the seeded
    event content does not — the byte-identical microseismic catalog CSV proves the seeded
    content is reproducible.
    """
    a = build_scenario("unit-cube-v1", tmp_path / "a")
    b = build_scenario("unit-cube-v1", tmp_path / "b")
    ma = json.loads((a.out_dir / "manifest.json").read_text())
    mb = json.loads((b.out_dir / "manifest.json").read_text())
    sums_a = {
        r["path"]: r["sha256"] for r in ma["measured"] if not r["path"].endswith(".quakeml")
    }
    sums_b = {
        r["path"]: r["sha256"] for r in mb["measured"] if not r["path"].endswith(".quakeml")
    }
    assert sums_a == sums_b
    # the QuakeML wrapper varies, but the seeded catalog content is byte-identical
    catalog = "measured/microseismic_catalog.csv"
    assert sums_a[catalog] == sums_b[catalog]
