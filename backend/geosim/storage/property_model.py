"""Zarr v3 PropertyModel writer/reader — doc 02 §10.2 (THE authoritative layout).

A PropertyModel is **one Zarr v3 group**. Each property is a **multiscale subgroup**
whose members are the pyramid levels (``0`` = full resolution); the sibling 1σ array
lives in a parallel ``<property>_sigma`` subgroup with the SAME shape/levels (doc 02
§6, §10.2). Conventions, all binding (doc 02 §10.2):

- **Axis order** ``[z, y, x]`` (3D), Z-up — increasing index = increasing elevation.
- **Cubic chunks**, default ``64³`` (doc 02 §10.3, doc 04 §4.2).
- **Blosc(zstd, shuffle)** lossless compression on disk (doc 04 §4.3). *Browser
  Blosc/zstd decode maturity is an early-validation spike (doc 02 §10.3 critique
  #5/#17)* — recorded as a follow-up; on disk we use Blosc regardless.
- **fill_value = NaN** for floats; masked/outside-coverage cells are NaN (doc 02 §10.2).
- REQUIRED per-array attrs: ``origin`` + ``spacing`` (Engineering metres, z,y,x order),
  ``cellRef``, ``_ARRAY_DIMENSIONS``, plus ``propertyType``/``canonicalUnit``/
  ``scaling``/``colormap``/``displayRange`` (doc 02 §10.2).
- An **OME-Zarr ``multiscales``** block in each property subgroup describes the pyramid
  so any Zarr reader discovers levels identically (doc 02 §10.3).

Value arrays downsample by mean; ``_sigma`` arrays downsample variance-correct
(doc 02 §10.3) — see :mod:`geosim.storage.pyramid`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BloscCodec, BloscShuffle

from geosim.spatial import REGISTRY, PropertyType

from .pyramid import build_sigma_pyramid, build_value_pyramid

__all__ = [
    "DEFAULT_CHUNK",
    "SIGMA_SUFFIX",
    "GridSpec",
    "write_property_model",
    "open_property_model",
    "PropertyModelReader",
]

DEFAULT_CHUNK = 64  # cubic chunk edge (doc 02 §10.3, doc 04 §4.2)
SIGMA_SUFFIX = "_sigma"  # doc 02 §10.2 fixed naming


def _blosc() -> BloscCodec:
    """Blosc meta-codec: zstd level 3 + shuffle, lossless (doc 04 §4.3)."""
    return BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.shuffle)


@dataclass
class GridSpec:
    """Regular-grid geometry in the Engineering Frame (doc 01, doc 02 §10.2).

    ``origin``/``spacing`` are in Engineering metres in ``(z, y, x)`` order, matching
    the array axis order. ``cell_ref`` is ``"center"`` or ``"corner"``.
    """

    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)  # (z0, y0, x0)
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)  # (dz, dy, dx)
    cell_ref: str = "center"

    def level_spacing(self, level: int) -> tuple[float, float, float]:
        """Spacing at a pyramid ``level`` — each level doubles cell size (doc 04 §5)."""
        f = 2**level
        return (self.spacing[0] * f, self.spacing[1] * f, self.spacing[2] * f)


def _resolve_property_type(property_type: str | PropertyType) -> PropertyType:
    if isinstance(property_type, PropertyType):
        return property_type
    return REGISTRY.get(property_type)


def _value_attrs(pt: PropertyType, grid: GridSpec, level: int) -> dict:
    """Per-array attrs for a value level (doc 02 §10.2)."""
    return {
        "propertyType": pt.key,
        "canonicalUnit": pt.canonical_unit,
        "scaling": pt.default_scaling,
        "colormap": pt.default_colormap,
        "displayRange": list(pt.display_range) if pt.display_range is not None else None,
        "origin": list(grid.origin),  # Engineering m, (z,y,x)
        "spacing": list(grid.level_spacing(level)),
        "cellRef": grid.cell_ref,
        "_ARRAY_DIMENSIONS": ["z", "y", "x"],
        "categories": None,
    }


def _sigma_attrs(pt: PropertyType, grid: GridSpec, level: int) -> dict:
    """Per-array attrs for a ``_sigma`` level (same grid, unit = the value's unit)."""
    a = _value_attrs(pt, grid, level)
    a["propertyType"] = f"{pt.key}{SIGMA_SUFFIX}"
    a["scaling"] = "linear"  # an absolute 1σ in canonical units is linear
    return a


def _multiscales_block(name: str, levels: int, grid: GridSpec) -> list[dict]:
    """OME-Zarr ``multiscales`` block describing the pyramid (doc 02 §10.3).

    Each dataset entry carries a ``scale`` coordinate transformation (z,y,x metres)
    and a ``translation`` (the origin) so OME-Zarr tooling reads the pyramid.
    """
    datasets = []
    for lv in range(levels):
        sp = grid.level_spacing(lv)
        datasets.append(
            {
                "path": str(lv),
                "coordinateTransformations": [
                    {"type": "scale", "scale": [sp[0], sp[1], sp[2]]},
                    {"type": "translation", "translation": list(grid.origin)},
                ],
            }
        )
    return [
        {
            "version": "0.4",
            "name": name,
            "axes": [
                {"name": "z", "type": "space", "unit": "metre"},
                {"name": "y", "type": "space", "unit": "metre"},
                {"name": "x", "type": "space", "unit": "metre"},
            ],
            "datasets": datasets,
        }
    ]


def _write_subgroup(
    parent: zarr.Group,
    name: str,
    levels: list[np.ndarray],
    grid: GridSpec,
    attrs_fn,
    pt: PropertyType,
    chunk: int,
) -> None:
    """Write one multiscale subgroup: numbered level arrays + a multiscales block."""
    sub = parent.create_group(name)
    for lv, data in enumerate(levels):
        shape = data.shape
        chunks = tuple(min(chunk, s) for s in shape)  # cubic, clamped to shape
        arr = sub.create_array(
            name=str(lv),
            shape=shape,
            chunks=chunks,
            dtype="float32",
            fill_value=float("nan"),  # doc 02 §10.2
            compressors=[_blosc()],
            attributes=attrs_fn(pt, grid, lv),
        )
        arr[...] = np.asarray(data, dtype=np.float32)
    sub.attrs["multiscales"] = _multiscales_block(name, len(levels), grid)


def write_property_model(
    path: str | Path,
    property_type: str | PropertyType,
    values: np.ndarray,
    *,
    grid: GridSpec | None = None,
    sigma: np.ndarray | None = None,
    chunk: int = DEFAULT_CHUNK,
    overwrite: bool = False,
) -> Path:
    """Write a PropertyModel Zarr v3 group per doc 02 §10.2.

    ``values`` is a ``(z, y, x)`` float array (Z-up). A pyramid is built by mean
    downsampling; if ``sigma`` (same shape) is given, a variance-correct ``_sigma``
    pyramid is written as a sibling subgroup. Returns the group path.
    """
    pt = _resolve_property_type(property_type)
    grid = grid or GridSpec()
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"values must be 3D (z,y,x); got shape {values.shape}")

    path = Path(path)
    root = zarr.create_group(store=str(path), overwrite=overwrite)
    root.attrs["geosim"] = {
        "kind": "propertyModel",
        "layoutDoc": "02 §10.2",
        "axisOrder": ["z", "y", "x"],
        "properties": [pt.key],
    }

    value_levels = build_value_pyramid(values, chunk)
    _write_subgroup(root, pt.key, value_levels, grid, _value_attrs, pt, chunk)

    if sigma is not None:
        sigma = np.asarray(sigma, dtype=np.float32)
        if sigma.shape != values.shape:
            raise ValueError(
                f"sigma shape {sigma.shape} must match values shape {values.shape} (doc 02 §6)"
            )
        sigma_levels = build_sigma_pyramid(sigma, chunk)
        _write_subgroup(
            root, f"{pt.key}{SIGMA_SUFFIX}", sigma_levels, grid, _sigma_attrs, pt, chunk
        )

    return path


@dataclass
class PropertyModelReader:
    """Read-only accessor over a PropertyModel Zarr group (doc 02 §10.2)."""

    group: zarr.Group
    _path: Path = field(default=Path("."))

    @property
    def properties(self) -> list[str]:
        """The value property names in this model (excludes ``_sigma`` siblings)."""
        return list(self.group.attrs.get("geosim", {}).get("properties", []))

    def has_sigma(self, prop: str) -> bool:
        return f"{prop}{SIGMA_SUFFIX}" in self.group

    def level_count(self, prop: str) -> int:
        """Number of pyramid levels in property ``prop`` (doc 02 §10.3)."""
        ms = self.group[prop].attrs["multiscales"]
        return len(ms[0]["datasets"])

    def multiscales(self, prop: str) -> list[dict]:
        return list(self.group[prop].attrs["multiscales"])

    def read_level(self, prop: str, level: int = 0) -> np.ndarray:
        """Read a full pyramid level as a NumPy array (doc 02 §10.2)."""
        return self.group[f"{prop}/{level}"][...]

    def read_sigma_level(self, prop: str, level: int = 0) -> np.ndarray:
        return self.group[f"{prop}{SIGMA_SUFFIX}/{level}"][...]

    def attrs(self, prop: str, level: int = 0) -> dict:
        """Per-array attrs of a value level (origin/spacing/units/... doc 02 §10.2)."""
        return dict(self.group[f"{prop}/{level}"].attrs)

    def sigma_attrs(self, prop: str, level: int = 0) -> dict:
        return dict(self.group[f"{prop}{SIGMA_SUFFIX}/{level}"].attrs)


def open_property_model(path: str | Path) -> PropertyModelReader:
    """Open a PropertyModel Zarr group read-only (doc 02 §10.2)."""
    group = zarr.open_group(str(path), mode="r")
    return PropertyModelReader(group, Path(path))
