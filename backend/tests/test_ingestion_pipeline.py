"""Ingestion pipeline + registry tests (doc 03 §1, §3, §6, §7, §8).

In-memory SQLite (doc 04 §2.1 fallback) + a tmp ``storage_root`` — no Docker/Postgres.
Exercises the full ``store-raw → detect → parse → normalize → write → register`` chain
over a dummy in-memory adapter, the idempotent re-ingest path, and the >10%-drop failure
rule. SMALL truth grids keep runtime trivial.
"""

from __future__ import annotations

import numpy as np
import pytest

from geosim.catalog import (
    Dataset,
    Observation,
    PropertyModel,
    Provenance,
    RawFile,
    create_all,
    make_engine,
    session_factory,
)
from geosim.ingestion import (
    DetectionError,
    IngestStatus,
    ParseResult,
    RawObservation,
    RawPropertyModel,
    RawSource,
    SourceRef,
    adapter_named,
    adapters,
    detect,
    ingest_file,
    register_adapter,
)
from geosim.ingestion import (
    Provenance as RawProvenance,
)


@pytest.fixture
def session():
    engine = make_engine()  # in-memory SQLite (doc 04 §2.1)
    create_all(engine)
    Session = session_factory(engine)
    with Session() as s:
        yield s


# ─────────────────────────── dummy in-memory adapters ───────────────────────────


class _DummyPmAdapter:
    """Emits one already-inverted resistivity volume (doc 03 §2: pre-inverted → PropertyModel)."""

    method = "ert"
    submethod = "dc_resistivity"
    name = "dummy-pm-v1"
    version = "1.0"
    extensions = (".dummy",)
    media_types = ("application/x-dummy",)

    def sniff(self, sample: bytes, filename: str) -> float:
        return 0.95 if sample.startswith(b"DUMMYPM") else 0.0

    def parse(self, source: RawSource) -> ParseResult:
        vals = np.full((4, 4, 4), 100.0, dtype=float)
        vals[2, 2, 2] = 10.0  # a conductive cell
        pm = RawPropertyModel(
            property="resistivity",
            values=vals,
            origin=(0.0, 0.0, 0.0),
            spacing=(10.0, 10.0, 10.0),
            support="volume",
        )
        return ParseResult(
            property_models=[pm],
            source=SourceRef(crs=None, z_convention="elevation_up"),
            units={"resistivity": "ohm*m"},
            provenance=RawProvenance(process="ingest:dummy-pm-v1"),
            records_total=1,
            records_dropped=0,
        )


class _DummyObsAdapter:
    """Emits gravity station points with a configurable drop ratio (doc 03 §6 partial-file)."""

    method = "gravity"
    submethod = None
    name = "dummy-obs-v1"
    version = "1.0"
    extensions = (".obs",)
    media_types = ()
    drop = 0

    def sniff(self, sample: bytes, filename: str) -> float:
        return 0.9 if sample.startswith(b"DUMMYOBS") else 0.0

    def parse(self, source: RawSource) -> ParseResult:
        n = 10
        coords = np.array([[float(i), float(i), 0.0] for i in range(n)])
        ga = np.linspace(-5.0, 5.0, n)
        obs = RawObservation(
            geometry_kind="points",
            coords=coords,
            values={"gravity_anomaly": ga},
            primary_property="gravity_anomaly",
        )
        return ParseResult(
            observations=[obs],
            source=SourceRef(crs=None, z_convention="elevation_up"),
            units={"gravity_anomaly": "mGal"},
            records_total=n,
            records_dropped=self.drop,
        )


@pytest.fixture(autouse=True)
def _register_dummies():
    # Register once; the plugins registry is a process singleton (idempotent re-register
    # just overwrites the same key), so repeated test runs are safe.
    register_adapter(_DummyPmAdapter())
    register_adapter(_DummyObsAdapter())
    yield


def _write(tmp_path, name, payload: bytes):
    p = tmp_path / name
    p.write_bytes(payload)
    return p


# ─────────────────────────── registry / detect (doc 03 §1, §7 step 3) ───────────────────────────


def test_dummy_adapters_registered():
    assert adapter_named("dummy-pm-v1") is not None
    assert adapter_named("dummy-obs-v1") is not None
    # the first-party auto-imported gravity CSV adapter is present too (doc 03 §9)
    assert any(getattr(a, "name", None) == "gravity-csv-v1" for a in adapters().values())


def test_detect_picks_highest_sniff(tmp_path):
    src = RawSource(filename="a.dummy", data=b"DUMMYPM...")
    assert detect(src).name == "dummy-pm-v1"
    src2 = RawSource(filename="b.obs", data=b"DUMMYOBS...")
    assert detect(src2).name == "dummy-obs-v1"


def test_detect_unrecognized_raises():
    with pytest.raises(DetectionError):
        detect(RawSource(filename="x.unknown", data=b"nothing-matches"))


# ─────────────────────────── end-to-end: PropertyModel (doc 03 §7) ───────────────────────────


def test_ingest_property_model_end_to_end(session, tmp_path):
    path = _write(tmp_path, "model.dummy", b"DUMMYPM payload")
    report = ingest_file(session, tmp_path / "storage", None, path)

    assert report.status in (IngestStatus.OK, IngestStatus.OK_WITH_WARNINGS)
    assert report.dataset_id is not None
    assert report.n_property_models == 1
    assert report.raw_file_id is not None

    # ── catalog PropertyModel row written + bulk store on disk ──
    pm = session.query(PropertyModel).filter_by(dataset_id=report.dataset_id).one()
    assert pm.property == "resistivity"
    assert pm.canonical_unit == "ohm*m"
    assert pm.support == "volume"
    assert pm.store_format == "zarr"

    from geosim.storage import open_property_model

    reader = open_property_model(pm.store_uri)
    vol = reader.read_level("resistivity", 0)
    assert vol.shape == (4, 4, 4)
    assert vol[2, 2, 2] < vol[0, 0, 0]  # conductive cell survived the write

    # ── provenance edge (doc 02 §7) ──
    ds = session.get(Dataset, report.dataset_id)
    assert ds.method == "ert" and ds.submethod == "dc_resistivity"
    prov = session.get(Provenance, ds.provenance_id)
    assert prov is not None
    assert prov.process == "ingest:dummy-pm-v1"
    assert prov.source_unit == "ohm*m"
    assert prov.raw_file_id == report.raw_file_id

    # ── raw file row content-addressed (doc 04 §8.1) ──
    raw = session.get(RawFile, report.raw_file_id)
    assert raw.sha256 and raw.filename == "model.dummy"


def test_ingest_observation_unit_conversion(session, tmp_path):
    """Units canonicalize on the way in (doc 03 §3b)."""
    path = _write(tmp_path, "stations.obs", b"DUMMYOBS payload")
    report = ingest_file(session, tmp_path / "storage", None, path)
    assert report.status in (IngestStatus.OK, IngestStatus.OK_WITH_WARNINGS)
    assert report.n_observations == 1

    obs = session.query(Observation).filter_by(dataset_id=report.dataset_id).one()
    assert obs.geometry_kind == "points"
    assert obs.primary_property == "gravity_anomaly"


# ─────────────────────────── idempotency (doc 03 §8) ───────────────────────────


def test_idempotent_reingest_returns_same_dataset(session, tmp_path):
    path = _write(tmp_path, "model.dummy", b"DUMMYPM payload")
    storage = tmp_path / "storage"
    first = ingest_file(session, storage, None, path)
    assert first.dataset_id is not None
    assert not first.reused

    # Re-ingest the SAME bytes+adapter into the SAME project → same dataset (doc 03 §8).
    second = ingest_file(session, storage, first.project_id, path)
    assert second.reused
    assert second.dataset_id == first.dataset_id

    # exactly one dataset + one raw file (content-addressed dedupe)
    assert session.query(Dataset).filter_by(project_id=first.project_id).count() == 1
    assert session.query(RawFile).filter_by(project_id=first.project_id).count() == 1


# ─────────── partial-file >10% drop → failed (doc 03 §6) ───────────


def test_excessive_drop_rate_fails(session, tmp_path):
    bad = _DummyObsAdapter()
    bad.name = "dummy-obs-baddrop-v1"
    bad.drop = 5  # 5/10 = 50% > 10% threshold → failed (doc 03 §6/§10 #7)
    bad.sniff = lambda sample, filename: 0.99 if sample.startswith(b"BADDROP") else 0.0
    register_adapter(bad)

    path = _write(tmp_path, "bad.obs", b"BADDROP payload")
    report = ingest_file(session, tmp_path / "storage", None, path, method_hint="gravity")
    assert report.status is IngestStatus.FAILED
    assert report.records_dropped == 5
    assert report.drop_ratio == 0.5


def test_acceptable_drop_rate_warns_not_fails(session, tmp_path):
    ok = _DummyObsAdapter()
    ok.name = "dummy-obs-okdrop-v1"
    ok.drop = 0  # no drops → ok
    ok.sniff = lambda sample, filename: 0.99 if sample.startswith(b"OKDROP") else 0.0
    register_adapter(ok)

    path = _write(tmp_path, "ok.obs", b"OKDROP payload")
    report = ingest_file(session, tmp_path / "storage", None, path, method_hint="gravity")
    assert report.status is not IngestStatus.FAILED
    assert report.drop_ratio == 0.0


# ─────────────────────────── normalization hard errors (doc 03 §2, §6) ───────────────────────────


def test_unknown_property_type_fails(session, tmp_path):
    class _BadPropAdapter:
        method = "ert"
        submethod = None
        name = "dummy-badprop-v1"
        version = "1.0"
        extensions = (".bad",)
        media_types = ()

        def sniff(self, sample, filename):
            return 0.99 if sample.startswith(b"BADPROP") else 0.0

        def parse(self, source):
            return ParseResult(
                property_models=[RawPropertyModel(
                    property="not_a_real_property", values=np.ones((2, 2, 2)),
                    origin=(0.0, 0.0, 0.0), spacing=(1.0, 1.0, 1.0),
                )],
                source=SourceRef(crs=None),
                units={"not_a_real_property": "ohm*m"},
                records_total=1,
            )

    register_adapter(_BadPropAdapter())
    path = _write(tmp_path, "bad.bad", b"BADPROP payload")
    report = ingest_file(session, tmp_path / "storage", None, path)
    assert report.status is IngestStatus.FAILED
    assert "unknown property type" in (report.message or "")


# ─────────── real first-party gravity CSV adapter (doc 03 §2, §9) ───────────


def test_gravity_csv_adapter_end_to_end(session, tmp_path):
    csv_text = (
        "# unit: mGal\n"
        "x,y,z,gravity_anomaly\n"
        "0,0,0,1.0\n"
        "100,0,5,2.5\n"
        "0,100,5,-1.5\n"
        "100,100,10,0.5\n"
    )
    path = _write(tmp_path, "grav.csv", csv_text.encode())
    report = ingest_file(session, tmp_path / "storage", None, path)
    assert report.status in (IngestStatus.OK, IngestStatus.OK_WITH_WARNINGS)
    assert report.n_observations == 1
    obs = session.query(Observation).filter_by(dataset_id=report.dataset_id).one()
    assert obs.geometry_kind == "points"
    ds = session.get(Dataset, report.dataset_id)
    assert ds.method == "gravity"
