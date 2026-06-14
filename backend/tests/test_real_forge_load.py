"""End-to-end verification that the real Utah FORGE field files ingest.

Builds a georeferenced project (same frame as ``data/load_utah_forge.py``) and runs the
normal ingestion pipeline over one representative real native file per method. Each must
ingest with status != ``failed`` (``ok`` or ``ok_with_warnings``). This guards the
real-format adapter branches without touching the synthetic ``test_adapters_*`` suites.

Skips gracefully if the real dataset is not present on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FORGE = REPO / "data" / "utah-forge"
MEASURED = FORGE / "measured"

# one representative real file per method + the source-CRS hint the loader uses
CASES = {
    "gravity": (MEASURED / "gravity" / "Utah_FORGE_Gravity_Data.txt", "EPSG:4326"),
    "mt": (MEASURED / "mt" / "edi" / "RHS12001.edi", "EPSG:4326"),
    "em": (MEASURED / "em" / "FORGE_TEM_USF" / "20170509_192354_203_Station2.usf", None),
    "welllog": (MEASURED / "welllog" / "58-32" / "58-32_DSI_Sonic.las", None),
    "insar": (MEASURED / "insar" / "avg_range_mperyr_utm.csv", "EPSG:4326"),
}

pytestmark = pytest.mark.skipif(
    not MEASURED.exists(), reason="real Utah FORGE dataset not present"
)


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    import json

    from geosim.api.frame_io import frame_row_kwargs
    from geosim.catalog.db import create_all, make_engine, session_factory
    from geosim.catalog.ids import IdKind, new_id
    from geosim.catalog.models import Project, SpatialFrameRow
    from geosim.spatial import Aabb, DepthRange, SpatialFrame
    from geosim.storage import ensure_project_layout

    f = json.loads((FORGE / "frame.json").read_text())
    frame = SpatialFrame.for_real_site(
        lon=f["anchor_lonlat"][0],
        lat=f["anchor_lonlat"][1],
        surface_elev=f["surface_elev_m"],
        roi=Aabb(**f["roi"]),
        depth_range=DepthRange(**f["depth_range"]),
    )

    root = tmp_path_factory.mktemp("forge-verify")
    engine = make_engine(f"sqlite:///{root / 'catalog.db'}")
    create_all(engine)
    Session = session_factory(engine)
    project_id = new_id(IdKind.PROJECT)
    with Session() as s:
        proj = Project(id=project_id, name="FORGE verify", storage_root=str(root))
        proj.spatial_frame = SpatialFrameRow(
            project_id=project_id, **frame_row_kwargs(frame)
        )
        s.add(proj)
        s.commit()
    ensure_project_layout(root, project_id)
    return root, project_id, Session


@pytest.mark.parametrize("method", sorted(CASES))
def test_real_forge_file_ingests(method, project):
    from geosim.ingestion import ingest_file

    root, project_id, Session = project
    path, crs_hint = CASES[method]
    if not path.exists():
        pytest.skip(f"{method} sample missing: {path.name}")

    with Session() as s:
        rep = ingest_file(
            s, root, project_id, path, method_hint=method, crs_hint=crs_hint
        )
        s.commit()

    status = rep.status.value if hasattr(rep.status, "value") else str(rep.status)
    n = rep.n_observations + rep.n_property_models + rep.n_features
    assert status != "failed", f"{method} {path.name} failed: {rep.warnings}"
    assert status in {"ok", "ok_with_warnings"}, f"unexpected status {status}"
    assert n >= 1, f"{method} produced no primitives"
