"""M1 exit criteria — end-to-end on the backend (doc 03 §7, doc 05, doc 04 §9.2/§9.3, doc 06 §1.3).

This is the M1 milestone gate: it exercises the *whole* backend slice through the public
surfaces only — no internal Zarr poking. It:

1. seeds a project via :func:`geosim.ingestion.seed_m1_project` (synthgen → write → register,
   doc 03 §7 steps 6–7) into the app's catalog + ``storage_root``, then
2. drives the FastAPI ``TestClient`` to assert:
   - ``GET /property-models/{id}`` meta is correct (property/unit/shape/levels/origin/spacing/
     colormap/scaling/stats/frame, doc 04 §9.2),
   - ``GET /property-models/{id}/volume?level=0`` returns a decodable little-endian float32
     buffer of the right ``(z, y, x)`` shape (the M1 single-resident path, doc 06 §1.3),
   - ``POST /property-models/{id}/slice`` returns a plane of the right shape (doc 04 §9.3).

Runs entirely on the no-service tier — SQLite in-memory + a tmp ``storage_root`` (doc 04
§2.1/§3); no Docker/Postgres/Redis.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from geosim.api import Settings, create_app
from geosim.ingestion import seed_m1_project

SHAPE = (16, 16, 16)  # (nz, ny, nx)
SPACING = (25.0, 25.0, 25.0)  # (dz, dy, dx)
ORIGIN = (0.0, 0.0, 0.0)  # (z0, y0, x0)
SEED = 42
PROPERTY = "resistivity"


@pytest.fixture
def seeded(tmp_path):
    """An app over a tmp storage root, seeded by the real M1 ingest path."""
    storage_root = tmp_path / "store"
    app = create_app(Settings(storage_root=storage_root))

    Session = app.state.session_factory
    with Session() as session:
        ids = seed_m1_project(
            session,
            storage_root,
            shape=SHAPE,
            spacing=SPACING,
            origin=ORIGIN,
            seed=SEED,
        )
    return app, ids


def test_m1_exit_meta_volume_and_slice(seeded):
    app, ids = seeded
    pm_id = ids["property_model_id"]
    nz, ny, nx = SHAPE
    client = TestClient(app)

    # ── meta (doc 04 §9.2) ───────────────────────────────────────────────────────────
    r = client.get(f"/property-models/{pm_id}")
    assert r.status_code == 200, r.text
    meta = r.json()
    assert meta["id"] == pm_id
    assert meta["property"] == PROPERTY
    assert meta["canonicalUnit"]  # from the property-type registry (doc 01 §5)
    assert meta["shape"] == [nz, ny, nx]
    assert meta["origin"] == list(ORIGIN)
    assert meta["spacing"] == list(SPACING)
    assert meta["levels"] >= 1
    assert meta["colormap"]  # seeded from the property-type registry (doc 01 §5)
    assert meta["scaling"]
    assert meta["hasSigma"] is True  # synthgen co-registers a 1σ array (doc 02 §6)
    # Synthetic field is the layered halfspace + conductive blob (doc 05 §2.2).
    assert meta["stats"]["min"] is not None and meta["stats"]["max"] is not None
    assert meta["stats"]["min"] <= 20.0  # conductive blob core (5–20 Ω·m)
    assert meta["stats"]["max"] <= 600.0  # plausible background band
    assert meta["frame"] is not None  # project SpatialFrame summary (doc 01 §2)

    # ── volume: decodable f32 buffer of the right shape (doc 06 §1.3) ─────────────────
    rv = client.get(f"/property-models/{pm_id}/volume?level=0")
    assert rv.status_code == 200, rv.text
    assert rv.headers["content-type"] == "application/octet-stream"
    assert rv.headers["x-volume-byte-order"] == "little"
    assert rv.headers["x-volume-dtype"] == "float32"
    assert json.loads(rv.headers["x-volume-shape"]) == [nz, ny, nx]
    assert json.loads(rv.headers["x-volume-origin"]) == list(ORIGIN)
    assert json.loads(rv.headers["x-volume-spacing"]) == list(SPACING)

    buf = np.frombuffer(rv.content, dtype="<f4")
    assert buf.size == nz * ny * nx  # exactly one f32 per cell
    vol = buf.reshape(nz, ny, nx)  # round-trips to the (z, y, x) volume
    assert np.all(np.isfinite(vol))  # synthetic volume has no no-data cells (doc 05)
    # The decoded buffer is the same field synthgen authored (deterministic, doc 05 §1).
    from geosim.synthgen import build_resistivity_volume

    expected = build_resistivity_volume(
        shape=SHAPE, spacing=SPACING, origin=ORIGIN, seed=SEED
    ).values
    np.testing.assert_array_equal(vol, expected)

    # ── slice: a constant-z plane is the right (ny, nx) shape (doc 04 §9.3) ───────────
    rs = client.post(
        f"/property-models/{pm_id}/slice",
        json={"plane": "z", "position": nz // 2, "level": 0, "encoding": "f32"},
    )
    assert rs.status_code == 200, rs.text
    header = json.loads(rs.headers["x-slice-header"])
    assert header["width"] == nx and header["height"] == ny
    plane = np.frombuffer(rs.content, dtype="<f4").reshape(header["height"], header["width"])
    assert plane.shape == (ny, nx)
    np.testing.assert_array_equal(plane, expected[nz // 2, :, :])
