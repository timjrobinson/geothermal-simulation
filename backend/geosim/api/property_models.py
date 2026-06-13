"""PropertyModel serving endpoints — the M1 volume/slice/zarr surface (doc 04 §9.2/§9.3).

This router is the HTTP read surface over a PropertyModel Zarr v3 group written by
:mod:`geosim.storage` (doc 02 §10.2). It serves four shapes (doc 04 §9.2, doc 06 §12):

- ``GET /property-models/{id}`` — meta JSON: property/unit/scaling/colormap/displayRange
  (property-type registry, doc 01 §5), level-0 ``shape``/``origin``/``spacing``, pyramid
  ``levels``, and NaN-aware ``stats`` (``min``/``max``/``p1``/``p99``) plus the project
  frame summary.
- ``GET /property-models/{id}/zarr/{path}`` — **Zarr-over-HTTP passthrough** of the
  on-disk store objects (``zarr.json`` group/array meta + ``c/<bz>/<by>/<bx>`` chunks)
  with HTTP ``Range``, ``ETag``/``If-None-Match`` and ``Cache-Control: immutable`` for
  chunks (doc 04 §9.2).
- ``GET /property-models/{id}/volume`` — **the M1 single-resident path** (doc 06 §1.3):
  server-DECODES a Zarr level into a contiguous little-endian float32 ``(z, y, x)`` buffer
  (NaN no-data) as ``application/octet-stream`` with ``X-Volume-*`` headers, sidestepping
  browser-Blosc. A sibling ``/volume/meta`` returns the same shape/origin/spacing as JSON.
- ``POST /property-models/{id}/slice`` — axis-aligned plane → raw f32 body (doc 04 §9.3
  default ``encoding:"f32"``) + an ``X-Slice-Header`` JSON (width/height/dx/dy/plane_basis).

The router shares the catalog session + ``storage_root`` injected onto ``app.state`` by
:func:`geosim.api.create_app`; it reuses :func:`geosim.storage.open_property_model` and
never reimplements the Zarr layout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.orm import Session

from geosim.catalog import Project as ProjectRow
from geosim.catalog import PropertyModel as PropertyModelRow
from geosim.storage import PropertyModelReader, open_property_model

from .frame_io import frame_from_row, frame_to_dict
from .schemas import (
    PropertyModelMeta,
    PropertyModelStats,
    SliceHeader,
    SliceRequest,
    VolumeMeta,
)

__all__ = ["build_property_model_router"]

_CHUNK_BYTES = 1024 * 1024  # streaming read granularity for the zarr passthrough


def _nan_stats(arr: np.ndarray) -> PropertyModelStats:
    """NaN-aware ``min``/``max``/``p1``/``p99`` over a level (doc 04 §9.2 ``stats``).

    Returns all-``None`` for a fully-masked (all-NaN) or empty array.
    """
    flat = np.asarray(arr, dtype=np.float64).ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return PropertyModelStats()
    p1, p99 = np.percentile(finite, [1.0, 99.0])
    return PropertyModelStats(
        min=float(finite.min()),
        max=float(finite.max()),
        p1=float(p1),
        p99=float(p99),
    )


def build_property_model_router(
    session_dep: Any,
) -> APIRouter:
    """Build the ``/property-models`` router wired to the app's catalog + storage DI.

    ``session_dep`` is the same ``Depends(get_session)`` marker :func:`create_app` builds
    over its ``session_factory``; the storage root is read off ``request.app.state`` so
    this router shares the app's injected services (doc 04 §9).
    """
    router = APIRouter(prefix="/property-models", tags=["property-models"])

    def _row_or_404(session: Session, pm_id: str) -> PropertyModelRow:
        row = session.get(PropertyModelRow, pm_id)
        if row is None:
            raise HTTPException(status_code=404, detail="property model not found")
        return row

    def _store_path(request: Request, row: PropertyModelRow) -> Path:
        """Resolve the on-disk Zarr group for a PropertyModel row.

        Honours an absolute ``store_uri`` (what the storage writer records); otherwise
        resolves it against the app's ``storage_root`` (doc 04 §3).
        """
        uri = row.store_uri
        path = Path(uri)
        if not path.is_absolute():
            path = Path(request.app.state.storage_root) / uri
        return path

    def _open(request: Request, row: PropertyModelRow) -> PropertyModelReader:
        path = _store_path(request, row)
        if not path.exists():
            raise HTTPException(status_code=404, detail="property model store missing")
        return open_property_model(path)

    def _resolve_property(reader: PropertyModelReader, requested: str | None) -> str:
        props = reader.properties
        if requested is not None:
            if requested not in props:
                raise HTTPException(
                    status_code=404, detail=f"property {requested!r} not in model"
                )
            return requested
        if not props:
            raise HTTPException(status_code=404, detail="property model has no properties")
        return props[0]

    def _frame_summary(session: Session, row: PropertyModelRow) -> dict[str, Any] | None:
        project = session.get(ProjectRow, row.project_id)
        if project is None or project.spatial_frame is None:
            return None
        return frame_to_dict(frame_from_row(project.spatial_frame))

    # ──────────────────────────────── meta (doc 04 §9.2) ────────────────────────────────
    @router.get("/{pm_id}", response_model=PropertyModelMeta)
    def get_meta(
        pm_id: str, request: Request, session: Session = session_dep
    ) -> PropertyModelMeta:
        row = _row_or_404(session, pm_id)
        reader = _open(request, row)
        prop = _resolve_property(reader, None)
        attrs = reader.attrs(prop, 0)
        level0 = reader.read_level(prop, 0)
        return PropertyModelMeta(
            id=pm_id,
            property=prop,
            canonicalUnit=attrs.get("canonicalUnit", row.canonical_unit),
            scaling=attrs.get("scaling", "linear"),
            colormap=attrs.get("colormap"),
            displayRange=attrs.get("displayRange"),
            shape=list(level0.shape),
            origin=list(attrs.get("origin", [0.0, 0.0, 0.0])),
            spacing=list(attrs.get("spacing", [1.0, 1.0, 1.0])),
            levels=reader.level_count(prop),
            stats=_nan_stats(level0),
            frame=_frame_summary(session, row),
            hasSigma=reader.has_sigma(prop),
        )

    # ──────────────────────── Zarr-over-HTTP passthrough (doc 04 §9.2) ────────────────────
    @router.get("/{pm_id}/zarr/{path:path}")
    def get_zarr_object(
        pm_id: str,
        path: str,
        request: Request,
        session: Session = session_dep,
        range_header: str | None = Header(default=None, alias="range"),
        if_none_match: str | None = Header(default=None, alias="if-none-match"),
    ) -> Response:
        """Serve a raw on-disk store object (``zarr.json`` or a chunk) (doc 04 §9.2).

        Chunks (under a ``/c/`` path component) are immutable & content-addressed, so they
        get a strong ``ETag`` + ``Cache-Control: immutable``; ``zarr.json`` meta is mutable
        meta, so it is not marked immutable. Supports a single ``Range: bytes=a-b``.
        """
        row = _row_or_404(session, pm_id)
        base = _store_path(request, row).resolve()
        # Resolve + contain the requested path inside the store (no traversal).
        target = (base / path).resolve()
        if base not in target.parents and target != base:
            raise HTTPException(status_code=403, detail="path escapes store")
        if not target.is_file():
            raise HTTPException(status_code=404, detail="zarr object not found")

        stat = target.stat()
        size = stat.st_size
        # Strong ETag over the immutable identity (path + size + mtime).
        etag = f'"{abs(hash((str(target), size, stat.st_mtime_ns)))}"'

        if if_none_match is not None and if_none_match.strip() == etag:
            return Response(status_code=304)

        headers = {"ETag": etag, "Accept-Ranges": "bytes"}
        media = (
            "application/json"
            if target.name == "zarr.json"
            else "application/octet-stream"
        )
        if media == "application/octet-stream":
            # Chunk bytes are immutable & content-addressed (doc 04 §9.2/§10).
            headers["Cache-Control"] = "public, max-age=31536000, immutable"

        if range_header:
            start, end = _parse_range(range_header, size)
            if start is None:
                return Response(
                    status_code=416,
                    headers={"Content-Range": f"bytes */{size}"},
                )
            length = end - start + 1
            with open(target, "rb") as fh:
                fh.seek(start)
                body = fh.read(length)
            headers["Content-Range"] = f"bytes {start}-{end}/{size}"
            return Response(
                content=body,
                status_code=206,
                media_type=media,
                headers=headers,
            )

        data = target.read_bytes()
        return Response(content=data, media_type=media, headers=headers)

    # ──────────────────── M1 single-resident volume (doc 06 §1.3) ────────────────────────
    @router.get("/{pm_id}/volume")
    def get_volume(
        pm_id: str,
        request: Request,
        level: int = 0,
        property: str | None = None,
        session: Session = session_dep,
    ) -> StreamingResponse:
        """Server-DECODE a level into a contiguous LE float32 ``(z,y,x)`` buffer (doc 06 §1.3).

        This is the M1 single-resident path: it sidesteps browser-Blosc by decoding the
        Zarr level server-side into ``application/octet-stream`` (C-contiguous, z-major,
        NaN no-data). Shape/origin/spacing ride along as ``X-Volume-*`` headers, and the
        full sidecar is at ``/volume/meta``.
        """
        row = _row_or_404(session, pm_id)
        reader = _open(request, row)
        prop = _resolve_property(reader, property)
        _check_level(reader, prop, level)
        arr = np.ascontiguousarray(reader.read_level(prop, level), dtype="<f4")
        attrs = reader.attrs(prop, level)
        buf = arr.tobytes(order="C")
        headers = {
            "X-Volume-Shape": json.dumps(list(arr.shape)),
            "X-Volume-Origin": json.dumps(list(attrs.get("origin", [0.0, 0.0, 0.0]))),
            "X-Volume-Spacing": json.dumps(list(attrs.get("spacing", [1.0, 1.0, 1.0]))),
            "X-Volume-Level": str(level),
            "X-Volume-Property": prop,
            "X-Volume-Dtype": "float32",
            "X-Volume-Byte-Order": "little",
            "Content-Length": str(len(buf)),
        }

        def _iter() -> Any:
            for i in range(0, len(buf), _CHUNK_BYTES):
                yield buf[i : i + _CHUNK_BYTES]

        return StreamingResponse(
            _iter(), media_type="application/octet-stream", headers=headers
        )

    @router.get("/{pm_id}/volume/meta", response_model=VolumeMeta)
    def get_volume_meta(
        pm_id: str,
        request: Request,
        level: int = 0,
        property: str | None = None,
        session: Session = session_dep,
    ) -> VolumeMeta:
        """The JSON sidecar for ``/volume`` (shape/origin/spacing at ``level``, doc 06 §1.3)."""
        row = _row_or_404(session, pm_id)
        reader = _open(request, row)
        prop = _resolve_property(reader, property)
        _check_level(reader, prop, level)
        arr = reader.read_level(prop, level)
        attrs = reader.attrs(prop, level)
        return VolumeMeta(
            id=pm_id,
            property=prop,
            level=level,
            shape=list(arr.shape),
            origin=list(attrs.get("origin", [0.0, 0.0, 0.0])),
            spacing=list(attrs.get("spacing", [1.0, 1.0, 1.0])),
        )

    # ──────────────────────────────── slice (doc 04 §9.3) ────────────────────────────────
    @router.post("/{pm_id}/slice")
    def post_slice(
        pm_id: str,
        body: SliceRequest,
        request: Request,
        session: Session = session_dep,
    ) -> Response:
        """Axis-aligned plane → raw f32 body + ``X-Slice-Header`` JSON (doc 04 §9.3).

        ``plane`` ∈ ``x|y|z``; ``position`` indexes that axis (M1 index addressing). The
        raw float32 default keeps slice colours locked to the volume's transfer function
        (doc 04 §9.3 / doc 06 §12). The returned plane is row-major ``(height, width)``.
        """
        if body.encoding != "f32":
            raise HTTPException(
                status_code=400,
                detail=f"M1 slice supports encoding 'f32' only (got {body.encoding!r})",
            )
        if body.plane not in ("x", "y", "z"):
            raise HTTPException(
                status_code=400, detail="plane must be one of x|y|z (M1)"
            )
        row = _row_or_404(session, pm_id)
        reader = _open(request, row)
        prop = _resolve_property(reader, body.property)
        _check_level(reader, prop, body.level)
        vol = reader.read_level(prop, body.level)  # (z, y, x)
        nz, ny, nx = vol.shape
        attrs = reader.attrs(prop, body.level)
        oz, oy, ox = (float(v) for v in attrs.get("origin", [0.0, 0.0, 0.0]))
        dz, dy, dx = (float(v) for v in attrs.get("spacing", [1.0, 1.0, 1.0]))

        plane, pos = body.plane, body.position
        # Axis-aligned plane extraction + the in-scene basis (origin,u,v) in Engineering m.
        if plane == "z":  # constant-z → (y, x) image; u=+x, v=+y
            _bounds(pos, nz)
            img = vol[pos, :, :]
            zc = oz + pos * dz
            header = SliceHeader(
                width=nx, height=ny, dx=dx, dy=dy,
                plane_basis={
                    "origin": [ox, oy, zc],
                    "u": [dx, 0.0, 0.0],
                    "v": [0.0, dy, 0.0],
                },
            )
        elif plane == "y":  # constant-y → (z, x) image; u=+x, v=+z
            _bounds(pos, ny)
            img = vol[:, pos, :]
            yc = oy + pos * dy
            header = SliceHeader(
                width=nx, height=nz, dx=dx, dy=dz,
                plane_basis={
                    "origin": [ox, yc, oz],
                    "u": [dx, 0.0, 0.0],
                    "v": [0.0, 0.0, dz],
                },
            )
        else:  # plane == "x": constant-x → (z, y) image; u=+y, v=+z
            _bounds(pos, nx)
            img = vol[:, :, pos]
            xc = ox + pos * dx
            header = SliceHeader(
                width=ny, height=nz, dx=dy, dy=dz,
                plane_basis={
                    "origin": [xc, oy, oz],
                    "u": [0.0, dy, 0.0],
                    "v": [0.0, 0.0, dz],
                },
            )

        body_bytes = np.ascontiguousarray(img, dtype="<f4").tobytes(order="C")
        headers = {
            "X-Slice-Header": header.model_dump_json(),
            "Content-Length": str(len(body_bytes)),
        }
        return Response(
            content=body_bytes,
            media_type="application/octet-stream",
            headers=headers,
        )

    return router


def _bounds(pos: int, n: int) -> None:
    if pos < 0 or pos >= n:
        raise HTTPException(
            status_code=400, detail=f"position {pos} out of range [0,{n})"
        )


def _check_level(reader: PropertyModelReader, prop: str, level: int) -> None:
    levels = reader.level_count(prop)
    if level < 0 or level >= levels:
        raise HTTPException(
            status_code=404, detail=f"level {level} out of range [0,{levels})"
        )


def _parse_range(header: str, size: int) -> tuple[int | None, int | None]:
    """Parse a single ``Range: bytes=a-b`` header → ``(start, end)`` inclusive (doc 04 §9.2).

    Supports ``bytes=a-b``, ``bytes=a-`` (to EOF) and ``bytes=-n`` (suffix). Returns
    ``(None, None)`` for an unsatisfiable / malformed range (caller emits 416).
    """
    unit, _, spec = header.partition("=")
    if unit.strip() != "bytes" or "," in spec:
        return (None, None)
    start_s, _, end_s = spec.strip().partition("-")
    try:
        if start_s == "":  # suffix range: last N bytes
            n = int(end_s)
            if n <= 0:
                return (None, None)
            start = max(0, size - n)
            return (start, size - 1)
        start = int(start_s)
        end = int(end_s) if end_s else size - 1
    except ValueError:
        return (None, None)
    if start >= size or start > end:
        return (None, None)
    return (start, min(end, size - 1))
