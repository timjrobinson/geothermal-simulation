"""Real-data ingest test for the Zonge USF transient-EM (TEM) adapter (doc 03 §2 em row).

Ingests a real Utah FORGE WalkTEM ``.usf`` sounding through the full pipeline and asserts
the file ingests ``ok``/``ok_with_warnings`` with a sounding placed at its UTM 12N location.
The synthetic ``.xyz`` path stays covered by ``test_adapters_electrical.py`` (untouched).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from geosim.ingestion.adapters.em import EmXyzAdapter
from geosim.ingestion.base import IngestStatus, RawSource

REPO = Path(__file__).resolve().parents[2]
USF_DIR = REPO / "data" / "utah-forge" / "measured" / "em" / "FORGE_TEM_USF"
# A file whose //EPSG header declares 32612 (the common case).
USF_FILE = USF_DIR / "20170509_192354_203_Station2.usf"


def _any_usf() -> Path:
    if USF_FILE.exists():
        return USF_FILE
    files = sorted(USF_DIR.glob("*.usf"))
    if not files:
        pytest.skip("real FORGE USF files not present")
    return files[0]


def test_usf_adapter_parses_sounding_at_utm() -> None:
    """The USF branch yields a ``soundings`` obs at the file's UTM /LOCATION + raw decay."""
    path = _any_usf()
    src = RawSource(filename=path.name, data=path.read_bytes())
    adapter = EmXyzAdapter()

    assert adapter.sniff(src.data[:4096], src.filename) >= 0.6  # claims `.usf`/`//USF`

    pr = adapter.parse(src)
    assert pr.observations, "expected at least one sounding observation"
    obs = pr.observations[0]
    assert obs.geometry_kind == "soundings"

    # CRS comes from //EPSG (UTM 12N) so the normalizer can reproject.
    assert pr.source is not None
    assert pr.source.crs == "EPSG:32612"
    assert pr.source.horizontal_unit == "m"

    # Location is a plausible FORGE UTM 12N easting/northing.
    x, y, _z = obs.coords[0]
    assert 300_000 < x < 400_000
    assert 4_200_000 < y < 4_300_000

    # Raw transient registered: monotone-ish increasing time gates + finite voltages.
    decay = obs.meta["transient"]
    times = np.asarray(decay["time_s"], dtype=float)
    volts = np.asarray(decay["voltage"], dtype=float)
    assert times.size > 0 and times.size == volts.size
    assert np.all(times > 0)
    assert np.all(np.isfinite(volts))


def test_usf_ingests_ok_through_pipeline(tmp_path) -> None:
    """End-to-end: a real USF ingests ok / ok_with_warnings into a georeferenced project."""
    from geosim.api.frame_io import frame_row_kwargs
    from geosim.catalog.db import create_all, make_engine, session_factory
    from geosim.catalog.ids import IdKind, new_id
    from geosim.catalog.models import Project, SpatialFrameRow
    from geosim.ingestion import ingest_file
    from geosim.spatial import Aabb, DepthRange, SpatialFrame
    from geosim.storage import ensure_project_layout

    path = _any_usf()

    frame = SpatialFrame.for_real_site(
        lon=-112.89, lat=38.50, surface_elev=1655,
        roi=Aabb(-15000, 15000, -15000, 15000),
        depth_range=DepthRange(-10000, 2000),
    )
    root = tmp_path / "store"
    root.mkdir()
    engine = make_engine(f"sqlite:///{root / 'catalog.db'}")
    create_all(engine)
    Session = session_factory(engine)
    project_id = new_id(IdKind.PROJECT)
    with Session() as s:
        proj = Project(id=project_id, name="USF test", storage_root=str(root))
        proj.spatial_frame = SpatialFrameRow(
            project_id=project_id, **frame_row_kwargs(frame)
        )
        s.add(proj)
        s.commit()
    ensure_project_layout(root, project_id)

    with Session() as s:
        rep = ingest_file(s, root, project_id, path, method_hint="em")
        s.commit()

    assert rep.status in (IngestStatus.OK, IngestStatus.OK_WITH_WARNINGS), rep.message
    assert rep.n_observations >= 1
