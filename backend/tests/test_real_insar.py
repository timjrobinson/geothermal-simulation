"""Real Utah FORGE InSAR ingest — the headerless UTM range-change point-CSV (doc 03 §2).

Ingests the real ``avg_range_mperyr_utm.csv`` (mean LOS range-change RATE, m/yr, ~3.2 M
points in UTM 12 N metres, EPSG:32612) through the full pipeline against a georeferenced
FORGE project, and asserts the adapter:

- detects + parses it to ``ok`` / ``ok_with_warnings`` (NOT failed),
- yields exactly one ``deformation`` ``grid2d`` PropertyModel,
- bins onto a tractable regular grid (<= ~1024 per axis) with a sane FORGE extent,
- converts m/yr -> mm/yr (canonical ``deformation`` unit ``mm``), NaN outside coverage,
- pairs the sibling ``sig_*`` file as the per-cell 1σ.

The synthetic GeoTIFF time-series path + its tests live in ``test_adapters_seismic.py`` and
are untouched (this file only exercises the added real-CSV branch).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geosim.ingestion.adapters.insar import InsarGeotiffAdapter
from geosim.ingestion.base import IngestStatus, RawSource

REAL = (
    Path(__file__).resolve().parents[2]
    / "data" / "utah-forge" / "measured" / "insar" / "avg_range_mperyr_utm.csv"
)

# FORGE UTM 12 N window the real points fall in (sanity bounds, metres).
_E_LO, _E_HI = 320_000.0, 350_000.0
_N_LO, _N_HI = 4_250_000.0, 4_280_000.0


pytestmark = pytest.mark.skipif(
    not REAL.exists(), reason="real Utah FORGE InSAR CSV not downloaded"
)


def _forge_project(tmp_path):
    from geosim.api.frame_io import frame_row_kwargs
    from geosim.catalog.db import create_all, make_engine, session_factory
    from geosim.catalog.ids import IdKind, new_id
    from geosim.catalog.models import Project, SpatialFrameRow
    from geosim.spatial import Aabb, DepthRange, SpatialFrame
    from geosim.storage import ensure_project_layout

    root = tmp_path / "store"
    root.mkdir(parents=True, exist_ok=True)
    engine = make_engine(f"sqlite:///{root / 'catalog.db'}")
    create_all(engine)
    Session = session_factory(engine)

    frame = SpatialFrame.for_real_site(
        lon=-112.89, lat=38.50, surface_elev=1655,
        roi=Aabb(-15000, 15000, -15000, 15000),
        depth_range=DepthRange(-10000, 2000),
    )
    project_id = new_id(IdKind.PROJECT)
    with Session() as s:
        proj = Project(id=project_id, name="FORGE-real", storage_root=str(root))
        proj.spatial_frame = SpatialFrameRow(project_id=project_id, **frame_row_kwargs(frame))
        s.add(proj)
        s.commit()
    ensure_project_layout(root, project_id)
    return Session, root, project_id


def test_real_insar_csv_detected_as_insar():
    """The headerless 3-col UTM point-CSV sniffs to the insar adapter (content + name)."""
    adapter = InsarGeotiffAdapter()
    with open(REAL, "rb") as fh:
        sample = fh.read(4096)
    score = adapter.sniff(sample, REAL.name)
    assert score >= 0.85  # filename carries the range_mperyr hint


def test_real_insar_csv_parses_to_deformation_grid():
    """Direct adapter parse → one deformation grid2d, mm/yr, paired sigma, NaN gaps."""
    adapter = InsarGeotiffAdapter()
    source = RawSource(filename=REAL.name, path=str(REAL), crs_hint="EPSG:4326")
    res = adapter.parse(source)

    assert not res.is_empty()
    assert len(res.property_models) == 1
    pm = res.property_models[0]
    assert pm.property == "deformation"
    assert pm.support == "grid2d"
    assert res.units["deformation"] == "mm"

    # coords stay native UTM; the adapter declares the source CRS so the pipeline reprojects.
    assert res.source is not None
    assert res.source.crs == "EPSG:32612"
    assert res.source.horizontal_unit == "m"

    # (z=1, ny, nx) grid, tractable cell count, NaN where no points fall.
    vals = np.asarray(pm.values)
    assert vals.ndim == 3 and vals.shape[0] == 1
    _, ny, nx = vals.shape
    assert 1 < ny <= 1024 and 1 < nx <= 1024
    assert np.isnan(vals).any()          # gaps outside coverage
    assert np.isfinite(vals).any()       # real signal present

    # mm/yr: m/yr rates are ~1e-2 → ~tens of mm/yr after the 1000x conversion.
    finite = vals[np.isfinite(vals)]
    assert np.nanmax(np.abs(finite)) < 1000.0
    assert np.nanmax(np.abs(finite)) > 1.0

    # grid origin lands inside the FORGE UTM window.
    _, y0, x0 = pm.origin
    assert _E_LO < x0 < _E_HI and _N_LO < y0 < _N_HI

    # sibling sig_*.csv paired as per-cell 1σ on the SAME grid.
    assert pm.sigma is not None
    assert np.asarray(pm.sigma).shape == vals.shape


def test_real_insar_csv_pipeline_ingest_ok(tmp_path):
    """Full pipeline ingest of the real avg CSV → ok / ok_with_warnings (not failed)."""
    from geosim.ingestion import ingest_file

    Session, root, project_id = _forge_project(tmp_path)
    with Session() as s:
        rep = ingest_file(
            s, root, project_id, REAL,
            method_hint="insar", crs_hint="EPSG:4326",
        )
        s.commit()

    assert rep.status in (IngestStatus.OK, IngestStatus.OK_WITH_WARNINGS)
    assert rep.n_property_models == 1
    assert rep.n_observations == 0
    # binning means almost no records are "dropped" (only non-finite rows).
    assert rep.drop_ratio < 0.10
