"""Persistent, seeded development server — `make demo`.

The default FastAPI app (``Settings()``) is intentionally ephemeral: in-memory SQLite +
a temp ``storage_root`` (doc 04 §2.1/§3). That's fine for tests and for the frontend's
client-side mock, but to actually *see* data in the viewer the backend needs a persistent
catalog + storage with at least one project seeded.

This launcher builds an on-disk SQLite catalog + storage root under ``--data-dir``, seeds
one synthetic resistivity ``PropertyModel`` project if the catalog is empty, and serves the
app with uvicorn. The frontend (``make run-frontend``) can then load it via the discovery
panel (or ``?id=<propertyModelId>``).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import uvicorn
from sqlalchemy import select

from geosim.api.app import Settings, create_app
from geosim.catalog.models import Project
from geosim.ingestion.seed import seed_m1_project


def main() -> None:
    ap = argparse.ArgumentParser(description="Persistent seeded GeoSim dev server")
    ap.add_argument("--data-dir", default=".devdata", help="persistent catalog + storage dir")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reseed", action="store_true", help="wipe the data dir and reseed")
    args = ap.parse_args()

    data = Path(args.data_dir).resolve()
    if args.reseed and data.exists():
        shutil.rmtree(data)
    storage = data / "storage"
    storage.mkdir(parents=True, exist_ok=True)
    db_path = data / "catalog.db"

    settings = Settings(database_url=f"sqlite:///{db_path}", storage_root=str(storage))
    app = create_app(settings)

    Session = app.state.session_factory
    with Session() as session:
        if session.execute(select(Project)).first() is None:
            ids = seed_m1_project(session, app.state.storage_root, name="dev-resistivity")
            session.commit()
            pm = ids["property_model_id"]
            print(f"[dev] seeded project={ids['project_id']} property_model={pm}")
            print(f"[dev] open the viewer at:  http://localhost:5173/?id={pm}")
        else:
            print("[dev] catalog already seeded (use --reseed to rebuild)")

    print(f"[dev] data dir: {data}")
    print(f"[dev] serving GeoSim API on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
