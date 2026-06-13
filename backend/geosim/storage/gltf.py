"""Minimal binary glTF (``.glb``) triangle-mesh writer (doc 06 §5.3).

The M2/M4/M5 viewer loads geological **surfaces/faults/solids** as glTF meshes (doc 06
§5.3 — "surfaces/faults served as glTF the GLTFLoader can stream"). Horizons and faults
that ingest as GeoJSON grids/polygons have no mesh on disk, so the feature-serving API
(``geosim.api.features``) converts them **server-side** into a glTF the frontend can load.
This module is that converter: a self-contained, dependency-free writer for a single
triangle-mesh primitive in **Engineering metres** (the frame the rest of the stack works
in, doc 01 §1; Z-up positions, doc 06).

A ``.glb`` is the binary glTF container (glTF 2.0 §4.4.3): a 12-byte header followed by a
JSON chunk (the glTF document) and a BIN chunk (the packed vertex/index buffer). We emit
exactly one ``mesh`` / one ``primitive`` (mode 4 = TRIANGLES) with a ``POSITION`` accessor
(float32 ``VEC3``), an optional ``COLOR_0`` accessor (float32 ``VEC4`` per-vertex RGBA for
property-coloured surfaces, doc 06 §5.3), and a ``SCALAR`` index accessor
(``uint32``). The min/max bounds REQUIRED on the ``POSITION`` accessor (glTF 2.0 §5.1.1)
are filled so a loader can frame the mesh without scanning the buffer.

Kept intentionally tiny: no materials/animation/textures — just geometry the GLTFLoader
streams. :func:`triangulate_grid` turns a regular ``(ny, nx)`` height-grid of Engineering
points into the (vertices, triangles) a horizon/fault surface needs.
"""

from __future__ import annotations

import json
import struct
from typing import Any

import numpy as np

__all__ = ["write_glb", "triangulate_grid", "GLTF_TRIANGLES"]

# glTF 2.0 constants we use (glTF 2.0 spec §3.6.2.2 / §3.7.2).
GLTF_TRIANGLES = 4
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963
_FLOAT = 5126
_UNSIGNED_INT = 5125
_GLB_MAGIC = 0x46546C67  # "glTF"
_GLB_VERSION = 2
_JSON_CHUNK = 0x4E4F534A  # "JSON"
_BIN_CHUNK = 0x004E4942  # "BIN\0"


def _pad4(buf: bytes, pad: int = 0x00) -> bytes:
    """Pad ``buf`` to a 4-byte boundary (glTF 2.0 §4.4.3 chunk alignment)."""
    rem = (-len(buf)) % 4
    return buf + bytes([pad]) * rem


def triangulate_grid(
    points: np.ndarray, ny: int, nx: int
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a regular ``(ny, nx)`` grid of Engineering points → (verts, tris).

    ``points`` is the row-major ``(ny*nx, 3)`` array of Engineering XYZ grid nodes (the
    shape a horizon surface stores as a draped grid, doc 02 §5). Each grid cell becomes two
    triangles; returns the vertex array unchanged and an ``(n_tri, 3)`` ``uint32`` index
    array (CCW winding when viewed from +Z).
    """
    verts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    tris: list[tuple[int, int, int]] = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = j * nx + i
            b = j * nx + (i + 1)
            c = (j + 1) * nx + i
            d = (j + 1) * nx + (i + 1)
            tris.append((a, c, b))
            tris.append((b, c, d))
    return verts, np.asarray(tris, dtype=np.uint32).reshape(-1, 3)


def write_glb(
    vertices: np.ndarray,
    triangles: np.ndarray,
    *,
    colors: np.ndarray | None = None,
    extras: dict[str, Any] | None = None,
) -> bytes:
    """Pack a single triangle mesh into a binary glTF (``.glb``) blob (glTF 2.0 §4.4.3).

    Parameters
    ----------
    vertices : (V, 3) array
        Engineering-metre XYZ positions (Z-up, doc 01 §1 / doc 06).
    triangles : (T, 3) integer array
        Vertex indices per triangle (TRIANGLES, mode 4).
    colors : (V, 4) array, optional
        Per-vertex RGBA in ``[0, 1]`` → a ``COLOR_0`` accessor for a property-coloured
        surface (doc 06 §5.3).
    extras : dict, optional
        Free-form metadata copied to the glTF ``asset.extras`` (e.g. featureId, property).

    Returns the ``.glb`` bytes (header + JSON chunk + BIN chunk).
    """
    verts = np.ascontiguousarray(vertices, dtype="<f4").reshape(-1, 3)
    idx = np.ascontiguousarray(triangles, dtype="<u4").reshape(-1)
    if verts.shape[0] == 0 or idx.size == 0:
        raise ValueError("glb mesh requires at least one triangle and vertex")

    # ── pack the BIN buffer: positions, then (optional) colors, then indices ──
    bin_parts: list[bytes] = []
    buffer_views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []
    offset = 0

    pos_bytes = verts.tobytes(order="C")
    buffer_views.append({
        "buffer": 0, "byteOffset": offset, "byteLength": len(pos_bytes),
        "target": _ARRAY_BUFFER,
    })
    pos_view = len(buffer_views) - 1
    bin_parts.append(pos_bytes)
    offset += len(pos_bytes)
    vmin = verts.min(axis=0).astype(float).tolist()
    vmax = verts.max(axis=0).astype(float).tolist()
    accessors.append({
        "bufferView": pos_view, "componentType": _FLOAT, "count": int(verts.shape[0]),
        "type": "VEC3", "min": vmin, "max": vmax,
    })
    pos_accessor = len(accessors) - 1

    attributes: dict[str, int] = {"POSITION": pos_accessor}

    color_accessor: int | None = None
    if colors is not None:
        col = np.ascontiguousarray(colors, dtype="<f4").reshape(-1, 4)
        if col.shape[0] != verts.shape[0]:
            raise ValueError("colors must have one RGBA per vertex")
        col_bytes = col.tobytes(order="C")
        buffer_views.append({
            "buffer": 0, "byteOffset": offset, "byteLength": len(col_bytes),
            "target": _ARRAY_BUFFER,
        })
        bin_parts.append(col_bytes)
        offset += len(col_bytes)
        accessors.append({
            "bufferView": len(buffer_views) - 1, "componentType": _FLOAT,
            "count": int(col.shape[0]), "type": "VEC4",
        })
        color_accessor = len(accessors) - 1
        attributes["COLOR_0"] = color_accessor

    idx_bytes = idx.tobytes(order="C")
    buffer_views.append({
        "buffer": 0, "byteOffset": offset, "byteLength": len(idx_bytes),
        "target": _ELEMENT_ARRAY_BUFFER,
    })
    idx_view = len(buffer_views) - 1
    bin_parts.append(idx_bytes)
    offset += len(idx_bytes)
    accessors.append({
        "bufferView": idx_view, "componentType": _UNSIGNED_INT, "count": int(idx.size),
        "type": "SCALAR",
    })
    idx_accessor = len(accessors) - 1

    bin_buffer = _pad4(b"".join(bin_parts))

    asset: dict[str, Any] = {"version": "2.0", "generator": "geosim.storage.gltf"}
    if extras:
        asset["extras"] = extras

    gltf: dict[str, Any] = {
        "asset": asset,
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{
            "primitives": [{
                "attributes": attributes,
                "indices": idx_accessor,
                "mode": GLTF_TRIANGLES,
            }]
        }],
        "buffers": [{"byteLength": len(bin_buffer)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
    }

    json_bytes = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), pad=0x20)

    total = 12 + 8 + len(json_bytes) + 8 + len(bin_buffer)
    out = bytearray()
    out += struct.pack("<III", _GLB_MAGIC, _GLB_VERSION, total)
    out += struct.pack("<II", len(json_bytes), _JSON_CHUNK)
    out += json_bytes
    out += struct.pack("<II", len(bin_buffer), _BIN_CHUNK)
    out += bin_buffer
    return bytes(out)
