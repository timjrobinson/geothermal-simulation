"""The high-level ingest orchestrator — ``store-raw → detect → parse → normalize →
write → register`` (doc 03 §7), inline + RQ-ready.

:func:`ingest_file` runs the whole doc-03 pipeline for one uploaded file against one
project, returning an :class:`~geosim.ingestion.base.IngestReport` (doc 03 §6). It is
written to run inline (FastAPI ``BackgroundTasks`` for small files) but is structured as
discrete stages so a worker (RQ/Celery, doc 03 §7) can drive the same chain.

Pipeline stages (doc 03 §7):
  2. **store-raw** — verbatim into the content-addressed raw store; sha256 is the
     idempotency root (doc 03 §8, doc 04 §8.1).
  3. **detect** — ``registry.detect`` runs ``sniff()`` (highest score wins; user
     ``method_hint`` overrides), failing into a ``failed`` report (not a crash).
  4. **parse** — ``adapter.parse(source)`` → ``ParseResult`` (native frame/units).
  5. **normalize** — :func:`geosim.ingestion.normalize.normalize` (CRS+units+placement).
  6–7. **write + register** — :func:`geosim.ingestion.writer.write_and_register` (atomic).

**Idempotency** (doc 03 §8): the key is ``sha256(raw) + adapter.name + adapter.version +
normalization params``. The same key already in the catalog → skip parse and return the
existing dataset (``reused=True``); the raw bytes dedupe by content hash regardless.

**Partial-file policy** (doc 03 §6): the adapter's ``records_total``/``records_dropped``
feed the >10% rule via :meth:`IngestReport.finalize`; a normalization hard error
(unregistered property, missing CRS in a georef project) yields ``failed``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

import geosim
from geosim.catalog import (
    Dataset,
    IdKind,
    Project,
    Provenance,
    new_id,
)
from geosim.spatial import Aabb, DepthRange, FrameMode, SpatialFrame
from geosim.storage import RawStore, ensure_project_layout

from .base import (
    IngestReport,
    IngestStatus,
    IngestWarning,
    RawSource,
    Severity,
)
from .normalize import NormalizationError, normalize
from .registry import DetectionError, detect
from .writer import WriteContext, _register_raw_file, write_and_register

__all__ = ["ingest_file", "idempotency_key", "frame_for_bundle"]

_DEFAULT_AGENT = "geosim.ingestion.ingest_file"


def idempotency_key(
    sha256: str, adapter_name: str, adapter_version: str, params: dict[str, Any]
) -> str:
    """Content + adapter + version + normalization params → idempotency key (doc 03 §8).

    Stable across runs (sorted JSON), so a re-ingest of the same bytes with the same
    adapter/params resolves to the same dataset.
    """
    payload = json.dumps(
        {"sha256": sha256, "adapter": adapter_name, "version": adapter_version,
         "params": params},
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def frame_for_bundle(bundle: Any) -> SpatialFrame:
    """Build a local-mode :class:`SpatialFrame` sized to a normalized bundle (doc 01 §2).

    A new project ingested from a local/non-georeferenced file lands in Engineering space
    (doc 03 §6 worst case: "lands in local/Engineering with a loud warning"); the ROI/depth
    range are taken from the union bbox so the frame encloses the data.
    """
    boxes: list[dict[str, float]] = []
    for obs in bundle.observations:
        boxes.append(obs.bbox)
    for pm in bundle.property_models:
        boxes.append(pm.bbox)
    for feat in bundle.features:
        if feat.bbox:
            boxes.append(feat.bbox)
    if not boxes:
        return SpatialFrame(mode=FrameMode.LOCAL)
    xmin = min(b["xmin"] for b in boxes)
    xmax = max(b["xmax"] for b in boxes)
    ymin = min(b["ymin"] for b in boxes)
    ymax = max(b["ymax"] for b in boxes)
    zmin = min(b["zmin"] for b in boxes)
    zmax = max(b["zmax"] for b in boxes)
    return SpatialFrame(
        mode=FrameMode.LOCAL,
        roi=Aabb(xmin, xmax, ymin, ymax),
        depth_range=DepthRange(zmin, zmax),
    )


def _existing_dataset_for_key(
    session: Session, project_id: str, key: str
) -> Dataset | None:
    """Find a dataset previously ingested under ``key`` in this project (doc 03 §8).

    The idempotency key is stored in the provenance ``params_json`` under
    ``"idempotency_key"``; we match it through the dataset's provenance.
    """
    rows = (
        session.query(Dataset, Provenance)
        .join(Provenance, Dataset.provenance_id == Provenance.id)
        .filter(Dataset.project_id == project_id)
        .all()
    )
    for ds, prov in rows:
        if not prov.params_json:
            continue
        try:
            params = json.loads(prov.params_json)
        except json.JSONDecodeError:
            continue
        if params.get("idempotency_key") == key:
            return ds
    return None


def ingest_file(
    session: Session,
    storage_root: str | Path,
    project_id: str | None,
    path: str | Path,
    *,
    method_hint: str | None = None,
    crs_hint: str | None = None,
    name: str | None = None,
    created_by: str = _DEFAULT_AGENT,
    drop_threshold: float | None = None,
) -> IngestReport:
    """Ingest one file end-to-end (doc 03 §7) → :class:`IngestReport`.

    If ``project_id`` is ``None`` a new project is created with a local-mode frame sized to
    the data (doc 01 §2). ``method_hint`` overrides format detection; ``crs_hint`` supplies
    a source CRS for a georeferenced project (doc 03 §7 step 1). Re-ingesting identical
    bytes+adapter+params returns the existing dataset (``reused=True``, doc 03 §8). Any
    detection / normalization failure produces a ``failed`` report rather than raising.
    """
    path = Path(path)
    storage_root = Path(storage_root)
    report = IngestReport()

    # ── identify / create project up front so the raw store + layout exist ──
    creating_project = project_id is None
    if project_id is None:
        project_id = new_id(IdKind.PROJECT)
    layout = ensure_project_layout(storage_root, project_id)

    # ── stage 2: store-raw (content-addressed, sha256 = idempotency root) ──
    data = path.read_bytes()
    raw_store = RawStore(layout.raw)
    raw_ref = raw_store.put_bytes(path.name, data)
    report.raw_file_id = None

    source = RawSource(
        filename=path.name, data=data, path=str(raw_ref.path), sha256=raw_ref.sha256,
        method_hint=method_hint, crs_hint=crs_hint,
    )

    # ── stage 3: detect ──
    try:
        adapter_obj = detect(source)
    except DetectionError as e:
        report.status = IngestStatus.FAILED
        report.message = str(e)
        report.add_warning(IngestWarning("detect_failed", Severity.HIGH, str(e), source.filename))
        return report

    adapter_name = getattr(adapter_obj, "name", getattr(adapter_obj, "method", "adapter"))
    adapter_version = getattr(adapter_obj, "version", "v1")

    # ── stage 4: parse (native frame/units) ──
    try:
        parsed = _parse(adapter_obj, source)
    except Exception as e:  # an adapter crash is a hard failure for THIS file (doc 03 §6)
        report.status = IngestStatus.FAILED
        report.message = f"parse failed: {e}"
        report.add_warning(IngestWarning("parse_failed", Severity.HIGH, str(e), source.filename))
        return report

    if crs_hint and parsed.source is not None and parsed.source.crs is None:
        parsed.source.crs = crs_hint  # user-supplied CRS (doc 03 §7 step 1)

    report.records_total = parsed.records_total
    report.records_dropped = parsed.records_dropped

    # ── frame: reuse the project's, else size one to the (un-normalized) data ──
    frame = _project_frame(session, project_id) if not creating_project else None

    # ── stage 5: normalize ──
    try:
        bundle = normalize(parsed, frame or SpatialFrame(mode=FrameMode.LOCAL))
        if creating_project:
            frame = frame_for_bundle(bundle)
            # re-normalize is unnecessary (local-mode identity); reuse the bundle.
    except NormalizationError as e:
        report.status = IngestStatus.FAILED
        report.message = str(e)
        report.add_warning(IngestWarning("normalize_failed", Severity.HIGH, str(e)))
        return report

    for w in bundle.warnings:
        report.add_warning(w)

    if bundle.is_empty():
        report.status = IngestStatus.FAILED
        report.message = "adapter produced no primitives"
        return report

    # ── idempotency check (doc 03 §8): same content+adapter+params → reuse ──
    norm_params = bundle.params.to_dict()
    key = idempotency_key(raw_ref.sha256, adapter_name, adapter_version, norm_params)
    report.idempotency_key = key
    if not creating_project:
        existing = _existing_dataset_for_key(session, project_id, key)
        if existing is not None:
            report.reused = True
            report.dataset_id = existing.id
            report.project_id = project_id
            report.status = IngestStatus.OK
            report.message = "idempotent re-ingest: returned existing dataset (doc 03 §8)"
            return report

    # ── register the raw file row (dedup by content hash) ──
    rel_path = str(raw_ref.path.relative_to(layout.root))
    report.raw_file_id = None  # filled after we ensure the project row exists
    dataset_id = new_id(IdKind.DATASET)
    provenance_id = new_id(IdKind.PROVENANCE)

    process = f"ingest:{adapter_name}"
    params = {
        "agent": created_by,
        "adapter": adapter_name,
        "adapter_version": adapter_version,
        "idempotency_key": key,
        "normalization": norm_params,
        "raw_sha256": raw_ref.sha256,
    }

    pt_unit = _first_source_unit(norm_params)

    # If we're creating the project, write_and_register makes the Project row; the raw_file
    # row must FK to it, so for a new project we create the project first, then the raw row.
    if creating_project:
        _ensure_project_row(session, project_id, name or path.stem, storage_root, frame)
    report.raw_file_id = _register_raw_file(
        session,
        project_id=project_id, filename=path.name, rel_path=rel_path,
        sha256=raw_ref.sha256, nbytes=len(data), media_type=source.media_type,
    )

    ctx = WriteContext(
        project_id=project_id,
        dataset_id=dataset_id,
        provenance_id=provenance_id,
        layout=layout,
        method=adapter_obj.method,
        submethod=getattr(adapter_obj, "submethod", None),
        process=process,
        process_version=geosim.__version__,
        params=params,
        source_crs=norm_params.get("source_crs"),
        source_unit=pt_unit,
        raw_file_id=report.raw_file_id,
        created_by=created_by,
        name=name or path.stem,
        tags=["ingested", adapter_name],
    )

    try:
        write_and_register(
            session, ctx, bundle, frame, report, create_project=False
        )
    except Exception as e:
        session.rollback()
        report.status = IngestStatus.FAILED
        report.message = f"write/register failed: {e}"
        report.add_warning(IngestWarning("register_failed", Severity.HIGH, str(e)))
        return report

    report.finalize(drop_threshold=drop_threshold)
    if report.status is IngestStatus.FAILED:
        # >10% drop escalation AFTER commit is reported but the data is already written;
        # surface it loudly (doc 03 §6) — the dataset row remains for audit.
        report.message = report.message or "escalated to failed by >10% record drop (doc 03 §6)"
    return report


# ─────────────────────────────── stage helpers ───────────────────────────────


def _parse(adapter_obj: Any, source: RawSource):
    """Call the adapter's ``parse`` (doc-03 ``parse(source)`` or plugins ``parse(raw, ctx)``)."""
    try:
        return adapter_obj.parse(source)
    except TypeError:
        return adapter_obj.parse(source, None)


def _project_frame(session: Session, project_id: str) -> SpatialFrame:
    from geosim.api.frame_io import frame_from_row
    from geosim.catalog import SpatialFrameRow

    row = session.get(SpatialFrameRow, project_id)
    if row is None:
        return SpatialFrame(mode=FrameMode.LOCAL)
    return frame_from_row(row)


def _ensure_project_row(
    session: Session,
    project_id: str,
    name: str,
    storage_root: Path,
    frame: SpatialFrame,
) -> None:
    """Create the ``Project`` + ``spatial_frame`` rows if absent (doc 04 §2.4)."""
    from geosim.api.frame_io import frame_row_kwargs
    from geosim.catalog import SpatialFrameRow

    if session.get(Project, project_id) is not None:
        return
    project = Project(id=project_id, name=name, storage_root=str(storage_root))
    project.spatial_frame = SpatialFrameRow(project_id=project_id, **frame_row_kwargs(frame))
    session.add(project)
    session.flush()


def _first_source_unit(norm_params: dict[str, Any]) -> str | None:
    units = norm_params.get("source_units") or {}
    for v in units.values():
        return v
    return None
