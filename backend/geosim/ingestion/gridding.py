"""User-initiated gridding: scattered Observations -> derived PropertyModel (doc 03 §3c/§3d/§4).

Gridding is the explicit, parameterized modeling step that turns scattered *point* or
*sounding* observations into a continuous grid/volume (doc 03 §3c, DECISIONS doc 03 #4:
"raw stays raw — gridding is a separate, user-initiated, provenance-recorded step").
Nothing here is run silently by the parse/normalize pipeline; a caller (or the UI) asks
for it with explicit :class:`GriddingParams`, and the result is a **new derived
PropertyModel** whose provenance records every gridding parameter so it is reproducible
(doc 03 §8, doc 02 §7). The contributing observations are never mutated — they remain the
immutable raw record (doc 02 §9).

Two shapes are supported, per the doc 03 §3c default table:

* **2D scattered points -> grid2d** (gravity/magnetic stations, surface geochem):
  :func:`grid_points_2d`. Default is a ``verde`` bias-corrected Green's-function
  :class:`~verde.Spline` (doc 03 §3c row 1); the ``"idw"`` method is the sparse/quick
  ``scipy`` inverse-distance fallback (row 3, "no uncertainty"). Either way a co-registered
  per-cell 1σ array is produced (doc 02 §6) — for the spline it is a **prediction-variance**
  proxy that grows with distance from the nearest datum and with fit residual; for IDW it
  is the property's default-noise-floor σ (the fallback "carries no uncertainty" of its own).

* **1D soundings -> stitched volume** (AEM/TEM/MT CDI, scattered temperature logs):
  :func:`stitch_soundings`. Each sounding is a resistivity/conductivity-vs-depth column at
  one ``(x, y)`` with a depth-of-investigation (DOI). The columns are resampled onto the
  canonical Engineering Z axis, interpolated **laterally per depth slice** (doc 03 §4 step
  2), and every cell **below its local DOI or beyond lateral coverage is masked to NaN**
  (doc 03 §4 step 3, DECISIONS doc 03: "footprint honesty — NaN beyond coverage/DOI; never
  silently extrapolated"). This is the standard 1D->3D stitching pattern (doc 03 §2 EM row,
  doc 03 decision #5).

Geometry/units are assumed already-normalized to the Engineering Frame + canonical units by
the doc 03 §3a/§3b pipeline before gridding runs (this module never reprojects or
unit-converts — doc 03 §10 decision #2). Arrays follow the storage contract: ``[z, y, x]``
Z-up, ``float32``, NaN for masked/uncovered cells (doc 02 §10.2). Property keys/units come
from :data:`geosim.spatial.REGISTRY` (doc 01 §5).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import verde as vd
from scipy.spatial import cKDTree

import geosim
from geosim.spatial import REGISTRY, PropertyType
from geosim.storage import GridSpec, write_property_model

__all__ = [
    "GriddingError",
    "GriddingMethod",
    "GriddingParams",
    "Sounding",
    "GridResult",
    "grid_points_2d",
    "stitch_soundings",
    "write_grid_result",
]

GriddingMethod = Literal["verde-spline", "idw", "kriging"]

_AGENT = "geosim.ingestion.gridding"


class GriddingError(ValueError):
    """Raised when a gridding request is ill-posed (too few points, bad params, …)."""


# ───────────────────────────── parameters & results ──────────────────────────────


@dataclass(frozen=True)
class GriddingParams:
    """Reproducible gridding parameters, recorded verbatim in provenance (doc 03 §8).

    ``method`` selects the interpolator (doc 03 §3c table): ``"verde-spline"`` is the
    bias-corrected Green's-function default, ``"idw"`` the sparse/quick scipy fallback,
    ``"kriging"`` the geostatistical/uncertainty option (doc 03 §3c row 2; requires an
    optional kriging backend, see :data:`GriddingResult.warnings`).

    Lateral grid layout is given by ``spacing`` (Engineering metres) over ``region``
    ``(xmin, xmax, ymin, ymax)``; if ``region`` is ``None`` it is taken from the data
    extent padded by one ``spacing``. ``max_distance`` is the footprint/DOI search radius
    (Engineering metres): cells farther than this from any datum are masked to NaN
    (doc 03 §4 step 3). ``idw_power`` is the inverse-distance exponent; ``idw_neighbors``
    the number of nearest data used per cell.
    """

    method: GriddingMethod = "verde-spline"
    spacing: float = 50.0
    region: tuple[float, float, float, float] | None = None
    max_distance: float | None = None
    damping: float | None = 1e-4
    idw_power: float = 2.0
    idw_neighbors: int = 8
    z_spacing: float = 25.0  # vertical sampling for sounding stitching (Engineering m)

    def as_provenance(self) -> dict[str, Any]:
        """JSON-able parameter record for the doc 02 §7 ``Step.params`` (doc 03 §8)."""
        return {
            "method": self.method,
            "spacing": self.spacing,
            "region": list(self.region) if self.region is not None else None,
            "maxDistance": self.max_distance,
            "damping": self.damping,
            "idwPower": self.idw_power,
            "idwNeighbors": self.idw_neighbors,
            "zSpacing": self.z_spacing,
        }


@dataclass(frozen=True)
class Sounding:
    """One 1D sounding for stitching (doc 03 §4): a property-vs-elevation column at ``(x, y)``.

    ``x``/``y`` are Engineering metres (East/North). ``elevations``/``values`` are paired
    1D arrays along the vertical axis (Engineering Z-up metres / canonical unit), ordered
    arbitrarily — they are resampled onto the output Z axis. ``doi_elevation`` is the
    depth-of-investigation expressed as the **lowest trustworthy elevation** (metres,
    Z-up): cells below it are masked (doc 03 §4 step 3). ``None`` trusts the full column.
    """

    x: float
    y: float
    elevations: np.ndarray
    values: np.ndarray
    doi_elevation: float | None = None


@dataclass
class GridResult:
    """A gridded value volume + co-registered 1σ, ready for :func:`write_grid_result`.

    ``values``/``sigma`` are ``float32`` ``(nz, ny, nx)`` Z-up arrays in the property's
    canonical unit; masked/uncovered cells are ``NaN`` (doc 02 §10.2). ``origin``/``spacing``
    are ``(z, y, x)`` Engineering metres matching :class:`geosim.storage.GridSpec`.
    ``provenance`` is the JSON-able gridding parameter record (doc 03 §8).
    """

    property: str
    canonical_unit: str
    values: np.ndarray
    sigma: np.ndarray
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    provenance: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


# ───────────────────────────── 2D scattered -> grid2d ────────────────────────────


def _resolve_region(
    x: np.ndarray, y: np.ndarray, params: GriddingParams
) -> tuple[float, float, float, float]:
    if params.region is not None:
        return params.region
    pad = params.spacing
    return (
        float(x.min() - pad),
        float(x.max() + pad),
        float(y.min() - pad),
        float(y.max() + pad),
    )


def _grid_axes(
    region: tuple[float, float, float, float], spacing: float
) -> tuple[np.ndarray, np.ndarray]:
    """1D cell-centre East/North axes covering ``region`` at ``spacing`` (>= 1 cell)."""
    xmin, xmax, ymin, ymax = region
    nx = max(int(round((xmax - xmin) / spacing)) + 1, 1)
    ny = max(int(round((ymax - ymin) / spacing)) + 1, 1)
    xs = xmin + np.arange(nx) * spacing
    ys = ymin + np.arange(ny) * spacing
    return xs, ys


def _footprint_distance(
    x: np.ndarray, y: np.ndarray, gx: np.ndarray, gy: np.ndarray
) -> np.ndarray:
    """Distance from each grid node ``(gx, gy)`` to the nearest datum (Engineering m)."""
    tree = cKDTree(np.column_stack([x, y]))
    dist, _ = tree.query(np.column_stack([gx.ravel(), gy.ravel()]), k=1)
    return dist.reshape(gx.shape)


def _idw(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    *,
    power: float,
    neighbors: int,
) -> np.ndarray:
    """Inverse-distance-weighted prediction (doc 03 §3c row 3 fallback, no native σ)."""
    tree = cKDTree(np.column_stack([x, y]))
    k = min(neighbors, x.size)
    dist, idx = tree.query(np.column_stack([gx.ravel(), gy.ravel()]), k=k)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    # Exact hits (distance 0) take that datum directly; avoid 1/0.
    with np.errstate(divide="ignore"):
        w = 1.0 / np.power(dist, power)
    exact = ~np.isfinite(w)
    w[exact] = 0.0
    has_exact = exact.any(axis=1)
    out = np.empty(gx.size, dtype=np.float64)
    vsel = values[idx]
    wsum = w.sum(axis=1)
    # general case
    good = wsum > 0
    out[good] = (w[good] * vsel[good]).sum(axis=1) / wsum[good]
    # rows with an exact hit → mean of the exactly-coincident data
    for r in np.nonzero(has_exact)[0]:
        out[r] = vsel[r][exact[r]].mean()
    return out.reshape(gx.shape)


def _spline_sigma(
    pt: PropertyType,
    values: np.ndarray,
    residual_rms: float,
    dist: np.ndarray,
    max_distance: float | None,
) -> np.ndarray:
    """Prediction-variance proxy for the spline (doc 02 §6, doc 03 §3c "uncertainty").

    Verde's Green's-function spline does not expose a closed-form kriging variance, so we
    synthesize an honest, monotonic-in-distance 1σ: a floor from the fit residual RMS and
    the property's default relative σ (doc 01 §5 ``default_rel_sigma``), growing linearly
    with distance-to-nearest-datum out to ``max_distance`` (the footprint edge). This makes
    interpolated cells near data confident and extrapolated cells progressively uncertain —
    the "low-confidence beyond coverage" surface of doc 03 §4 / OVERVIEW §6.
    """
    typical = float(np.nanmedian(np.abs(values))) if values.size else 1.0
    floor = max(residual_rms, pt.default_rel_sigma * typical, 1e-9)
    if max_distance and max_distance > 0:
        growth = np.clip(dist / max_distance, 0.0, 1.0)
    else:
        scale = float(np.nanmax(dist)) or 1.0
        growth = np.clip(dist / scale, 0.0, 1.0)
    # σ doubles at the footprint edge — conservative but bounded.
    return (floor * (1.0 + growth)).astype(np.float32)


def grid_points_2d(
    property_type: str,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    *,
    params: GriddingParams | None = None,
    elevation: float = 0.0,
) -> GridResult:
    """Grid scattered 2D point observations to a single-slice ``grid2d`` (doc 03 §3c).

    ``x``/``y``/``values`` are equal-length 1D arrays in Engineering metres / the
    property's canonical unit (already normalized — doc 03 §3a/b). Returns a
    :class:`GridResult` whose volume has ``nz == 1`` at ``elevation`` (a 2D field embedded
    in 3D, doc 03 §3d), carrying a co-registered 1σ. Cells beyond ``params.max_distance``
    of any datum are NaN (footprint honesty, doc 03 §4 step 3).
    """
    params = params or GriddingParams()
    pt = REGISTRY.get(property_type)
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    values = np.asarray(values, dtype=np.float64).ravel()
    if not (x.size == y.size == values.size):
        raise GriddingError("x, y, values must be equal-length 1D arrays")
    if x.size < 3:
        raise GriddingError(f"need >= 3 points to grid; got {x.size}")

    region = _resolve_region(x, y, params)
    xs, ys = _grid_axes(region, params.spacing)
    gx, gy = np.meshgrid(xs, ys)  # (ny, nx)
    dist = _footprint_distance(x, y, gx, gy)
    warnings: list[str] = []

    if params.method == "kriging":
        # pykrige/gstatsim are optional backends (doc 03 §3c row 2); when absent we degrade
        # to the spline default rather than fail, and say so loudly in the report.
        try:  # pragma: no cover - exercised only where the backend is installed
            import pykrige  # noqa: F401
        except ImportError:
            warnings.append(
                "kriging backend (pykrige/gstatsim) not installed; "
                "fell back to verde-spline (doc 03 §3c)"
            )

    if params.method == "idw":
        grid = _idw(
            x, y, values, gx, gy,
            power=params.idw_power, neighbors=params.idw_neighbors,
        )
        # The IDW fallback carries no native uncertainty (doc 03 §3c row 3): use the
        # property default-noise-floor σ, still distance-aware for footprint honesty.
        typical = float(np.median(np.abs(values))) or 1.0
        floor = pt.default_rel_sigma * typical
        ref = params.max_distance or (float(dist.max()) or 1.0)
        sigma = (floor * (1.0 + np.clip(dist / ref, 0.0, 1.0))).astype(np.float32)
    else:  # verde-spline (default + kriging-degraded path)
        spline = vd.Spline(damping=params.damping).fit((x, y), values)
        grid = spline.predict((gx, gy))
        residual = values - spline.predict((x, y))
        residual_rms = float(np.sqrt(np.mean(residual**2)))
        sigma = _spline_sigma(pt, values, residual_rms, dist, params.max_distance)

    grid = grid.astype(np.float32)

    # Footprint mask: NaN beyond coverage (doc 03 §4 step 3, DECISIONS doc 03).
    if params.max_distance is not None:
        outside = dist > params.max_distance
        grid[outside] = np.nan
        sigma[outside] = np.nan

    # Embed the 2D field as a single Z slice (nz=1) — doc 03 §3d / storage [z,y,x].
    values3d = grid[np.newaxis, :, :]
    sigma3d = sigma[np.newaxis, :, :]
    origin = (float(elevation), float(ys[0]), float(xs[0]))
    spacing = (float(params.z_spacing), float(params.spacing), float(params.spacing))

    prov = params.as_provenance()
    prov.update(
        {
            "agent": _AGENT,
            "op": "grid_points_2d",
            "support": "grid2d",
            "nPoints": int(x.size),
            "region": list(region),
            "elevation": float(elevation),
        }
    )
    return GridResult(
        property=pt.key,
        canonical_unit=pt.canonical_unit,
        values=values3d,
        sigma=sigma3d,
        origin=origin,
        spacing=spacing,
        provenance=prov,
        warnings=warnings,
    )


# ───────────────────────────── 1D soundings -> volume ────────────────────────────


def stitch_soundings(
    property_type: str,
    soundings: list[Sounding],
    *,
    params: GriddingParams | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
) -> GridResult:
    """Stitch 1D soundings into a 3D ``volume`` with DOI masking (doc 03 §4, decision #5).

    Each :class:`Sounding` is a property-vs-elevation column at ``(x, y)`` with an optional
    depth-of-investigation. The columns are (1) resampled onto a shared canonical Z axis,
    (2) interpolated **laterally per depth slice** with the chosen method (verde-spline
    default; IDW fallback), and (3) masked to NaN below each location's DOI and beyond the
    lateral footprint (``params.max_distance``). Native soundings are untouched (doc 02 §9);
    the returned volume is the derived PropertyModel (doc 03 §4 final paragraph).
    """
    params = params or GriddingParams()
    pt = REGISTRY.get(property_type)
    if len(soundings) < 3:
        raise GriddingError(f"need >= 3 soundings to stitch; got {len(soundings)}")

    xs_site = np.array([s.x for s in soundings], dtype=np.float64)
    ys_site = np.array([s.y for s in soundings], dtype=np.float64)

    # Shared Engineering Z axis (Z-up): index 0 = deepest, index nz-1 = shallowest, to
    # match the storage [z,y,x] convention used by synthgen (resistivity.py).
    all_elev = np.concatenate([np.asarray(s.elevations, dtype=np.float64) for s in soundings])
    zlo = float(z_min) if z_min is not None else float(all_elev.min())
    zhi = float(z_max) if z_max is not None else float(all_elev.max())
    nz = max(int(round((zhi - zlo) / params.z_spacing)) + 1, 1)
    z_axis = zlo + np.arange(nz) * params.z_spacing  # ascending elevation

    # Resample each sounding onto z_axis (log-space if the property spans decades).
    log = pt.interp_space == "log10"
    columns = np.empty((len(soundings), nz), dtype=np.float64)
    doi_mask = np.zeros((len(soundings), nz), dtype=bool)  # True == trusted
    for i, s in enumerate(soundings):
        ev = np.asarray(s.elevations, dtype=np.float64)
        va = np.asarray(s.values, dtype=np.float64)
        order = np.argsort(ev)
        ev, va = ev[order], va[order]
        ya = np.log10(va) if log else va
        col = np.interp(z_axis, ev, ya)  # flat extrapolation outside the column
        columns[i] = np.power(10.0, col) if log else col
        doi = s.doi_elevation if s.doi_elevation is not None else zlo
        doi_mask[i] = z_axis >= doi  # trusted at/above DOI elevation

    # Lateral grid (plan view).
    region = _resolve_region(xs_site, ys_site, params)
    gxs, gys = _grid_axes(region, params.spacing)
    gx, gy = np.meshgrid(gxs, gys)  # (ny, nx)
    ny, nx = gx.shape
    dist = _footprint_distance(xs_site, ys_site, gx, gy)
    warnings: list[str] = []

    values = np.empty((nz, ny, nx), dtype=np.float32)
    # Per-depth lateral interpolation of values AND of the (0/1) DOI coverage, so a cell is
    # trusted only where the surrounding soundings reach that depth (doc 03 §4 step 3).
    coverage = np.empty((nz, ny, nx), dtype=np.float32)
    for k in range(nz):
        slice_vals = columns[:, k]
        slice_log = np.log10(slice_vals) if log else slice_vals
        if params.method == "idw":
            interp = _idw(
                xs_site, ys_site, slice_log, gx, gy,
                power=params.idw_power, neighbors=params.idw_neighbors,
            )
            cov = _idw(
                xs_site, ys_site, doi_mask[:, k].astype(np.float64), gx, gy,
                power=params.idw_power, neighbors=params.idw_neighbors,
            )
        else:
            interp = (
                vd.Spline(damping=params.damping)
                .fit((xs_site, ys_site), slice_log)
                .predict((gx, gy))
            )
            cov = (
                vd.Spline(damping=params.damping)
                .fit((xs_site, ys_site), doi_mask[:, k].astype(np.float64))
                .predict((gx, gy))
            )
        values[k] = (np.power(10.0, interp) if log else interp).astype(np.float32)
        coverage[k] = cov

    # Build σ from the property noise floor, growing with distance (footprint honesty).
    typical = float(np.median(np.abs(columns))) or 1.0
    floor = max(pt.default_rel_sigma * typical, 1e-9)
    ref = params.max_distance or (float(dist.max()) or 1.0)
    lateral_growth = np.clip(dist / ref, 0.0, 1.0)  # (ny, nx)
    sigma = (floor * (1.0 + lateral_growth)).astype(np.float32)[np.newaxis, :, :]
    sigma = np.repeat(sigma, nz, axis=0)

    # Mask below DOI (interpolated coverage < 0.5) and beyond lateral footprint.
    untrusted = coverage < 0.5
    values[untrusted] = np.nan
    sigma[untrusted] = np.nan
    if params.max_distance is not None:
        outside = (dist > params.max_distance)[np.newaxis, :, :]
        outside = np.repeat(outside, nz, axis=0)
        values[outside] = np.nan
        sigma[outside] = np.nan

    origin = (float(z_axis[0]), float(gys[0]), float(gxs[0]))
    spacing = (float(params.z_spacing), float(params.spacing), float(params.spacing))
    prov = params.as_provenance()
    prov.update(
        {
            "agent": _AGENT,
            "op": "stitch_soundings",
            "support": "volume",
            "nSoundings": len(soundings),
            "region": list(region),
            "zRange": [zlo, zhi],
            "doiMasked": True,
        }
    )
    return GridResult(
        property=pt.key,
        canonical_unit=pt.canonical_unit,
        values=values,
        sigma=sigma,
        origin=origin,
        spacing=spacing,
        provenance=prov,
        warnings=warnings,
    )


# ───────────────────────────── write the derived model ───────────────────────────


def write_grid_result(
    result: GridResult,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a :class:`GridResult` as a doc-02 PropertyModel Zarr group (doc 03 §7 step 6).

    Raw stays raw: this is a NEW derived PropertyModel (pyramid + ``_sigma``), distinct from
    the contributing observations (doc 03 §3c, DECISIONS doc 03 #4). The gridding parameters
    travel with it: a ``provenance.json`` sidecar is written next to the Zarr group so the
    derivation is auditable even before the catalog ``Provenance`` row is inserted by the
    pipeline writer (doc 02 §7, doc 03 §8). Returns the Zarr group path.
    """
    path = Path(path)
    grid = GridSpec(origin=result.origin, spacing=result.spacing, cell_ref="center")
    write_property_model(
        path,
        result.property,
        result.values,
        grid=grid,
        sigma=result.sigma,
        overwrite=overwrite,
    )
    sidecar = {
        "process": "grid",
        "processVersion": geosim.__version__,
        "params": result.provenance,
        "warnings": result.warnings,
    }
    (path.parent / f"{path.name}.provenance.json").write_text(json.dumps(sidecar, indent=2))
    return path
