"""Load the real Utah FORGE dataset into a simulator project.

Creates a georeferenced project from ``data/utah-forge/frame.json`` (Engineering Frame
anchored at the FORGE site, UTM 12N) and ingests every native file under
``data/utah-forge/measured/`` through the normal ingestion pipeline — reprojecting each
into the project frame. Best-effort: it reports per-file success/failure (real field
files carry vendor quirks the synthetic-trained adapters may not yet handle).

    backend/.venv/bin/python data/load_utah_forge.py --storage-root .devdata-forge
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORGE = HERE / "utah-forge"

# native files the ingestion adapters read directly (skip .zip/.pdf/.docx/.sh/.xml/.json)
NATIVE_EXT = {".edi", ".las", ".usf", ".csv", ".txt", ".segy", ".sgy", ".xyz", ".stg"}
# per-method source-CRS hint (geographic methods carry lon/lat; others self-declare)
CRS_HINT = {"gravity": "EPSG:4326", "mt": "EPSG:4326", "insar": "EPSG:4326"}


def build_frame():
    from geosim.spatial import Aabb, DepthRange, SpatialFrame

    f = json.loads((FORGE / "frame.json").read_text())
    return SpatialFrame.for_real_site(
        lon=f["anchor_lonlat"][0], lat=f["anchor_lonlat"][1], surface_elev=f["surface_elev_m"],
        roi=Aabb(**f["roi"]), depth_range=DepthRange(**f["depth_range"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--storage-root", default=".devdata-forge")
    ap.add_argument("--db", default=None, help="SQLAlchemy URL (default: sqlite under storage-root)")
    ap.add_argument("--limit-mb", type=float, default=200.0, help="skip native files larger than this")
    args = ap.parse_args()

    from geosim.api.frame_io import frame_row_kwargs
    from geosim.catalog.db import create_all, make_engine, session_factory
    from geosim.catalog.ids import IdKind, new_id
    from geosim.catalog.models import Project, SpatialFrameRow
    from geosim.ingestion import ingest_file
    from geosim.storage import ensure_project_layout

    root = Path(args.storage_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    db_url = args.db or f"sqlite:///{root / 'catalog.db'}"
    engine = make_engine(db_url)
    create_all(engine)
    Session = session_factory(engine)

    frame = build_frame()
    project_id = new_id(IdKind.PROJECT)
    with Session() as s:
        proj = Project(id=project_id, name="Utah FORGE (real data)", storage_root=str(root))
        proj.spatial_frame = SpatialFrameRow(project_id=project_id, **frame_row_kwargs(frame))
        s.add(proj)
        s.commit()
    ensure_project_layout(root, project_id)
    print(f"project {project_id}  ({frame.horizontal_crs}, anchor "
          f"{frame.anchor.easting:.0f}E/{frame.anchor.northing:.0f}N)\n")

    files = []
    for p in sorted((FORGE / "measured").rglob("*")):
        if p.is_file() and p.suffix.lower() in NATIVE_EXT:
            low = p.name.lower()
            if "metadata" in low or "readme" in low:  # descriptive, not data
                continue
            files.append(p)

    ok = warn = fail = skip = 0
    for p in files:
        method = p.relative_to(FORGE / "measured").parts[0]
        if method in {"16A", "58-32"}:
            method = "welllog"
        size_mb = p.stat().st_size / 1e6
        if size_mb > args.limit_mb:
            print(f"  · skip (>{args.limit_mb:.0f} MB)  {p.name}")
            skip += 1
            continue
        try:
            with Session() as s:
                rep = ingest_file(
                    s, root, project_id, p,
                    method_hint=method, crs_hint=CRS_HINT.get(method),
                )
                s.commit()
            status = rep.status.value if hasattr(rep.status, "value") else str(rep.status)
            n = rep.n_observations + rep.n_property_models + rep.n_features
            mark = {"failed": "✗"}.get(status, "✓")
            print(f"  {mark} [{method:>11}] {p.name}  → {status} ({n} primitives, "
                  f"{rep.records_total - rep.records_dropped}/{rep.records_total} records)")
            if status == "failed":
                fail += 1
            elif status == "ok_with_warnings":
                warn += 1
            else:
                ok += 1
        except Exception as e:  # noqa: BLE001 — best-effort over real-world files
            print(f"  ✗ [{method:>11}] {p.name}  → error: {str(e)[:120]}")
            fail += 1

    print(f"\nIngested {ok} ok, {warn} with warnings, {fail} failed, {skip} skipped "
          f"(>{args.limit_mb:.0f} MB) of {len(files)} native files.")
    print(f"Storage: {root}\nDB: {db_url}\n"
          f"Point the API at it (Settings(database_url, storage_root)) and open the viewer.")


if __name__ == "__main__":
    main()
