"""Uniform forward-model contract + the three universal degradations (doc 05 §4, §6 T0).

Every method's T0 ("plausible") forward is a small class implementing the
:class:`ForwardModel` protocol (doc 08 §4d): ``method``/``submethod`` canonical pair
(doc 02 §2), ``fidelity="plausible"`` (doc 05 §6 T0 tier), and ``simulate(truth,
acquisition, rng) -> list[Artifact]`` — each :class:`Artifact` a native-format file the
*same method's* doc-03 adapter could parse, plus a :class:`Provenance` record noting it
is synthetic (``source="synthgen"``, scene id, seed — doc 05 §5).

The T0 recipe is **degrade-the-truth** (doc 05 §6): take truth property volumes and apply
the three degradations applied in *every* model (doc 05 §4):

1. **acquisition geometry** — project the truth onto a simulated survey layout (station
   grid, flight lines, electrode array, well path, period band) that limits coverage;
2. **resolution / DOI** — each method only "sees what it physically could": smoothing
   kernels (depth/altitude-dependent low-pass), depth-of-investigation masks, footprint
   averaging, frequency→depth mapping (doc 05 §4.2 "only-sees-what-it-could");
3. **noise** — additive/multiplicative noise with method-appropriate statistics, drawn
   from the passed ``rng`` so a run is reproducible (doc 05 §4 noise column, §1 invariant).

This module owns the shared machinery: the :class:`Acquisition` spec (doc 05 §4.3), the
:class:`Artifact`/:class:`Provenance` DTOs, truth-volume sampling helpers (nearest /
trilinear in the Z-up ``[z,y,x]`` Engineering grid, doc 01 §1 / doc 02 §10.2), a separable
Gaussian low-pass (the resolution kernel), and noise helpers. The per-method physics lives
in the sibling modules (``potential_field``, ``electrical``, ``em_mt``, ``seismic``,
``borehole``, ``surface``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..truth import TruthEarth

__all__ = [
    "Acquisition",
    "Provenance",
    "Artifact",
    "ForwardModel",
    "T0Forward",
    "gaussian_lowpass",
    "depth_doi_mask",
    "sample_volume_at",
    "sample_column",
    "add_gaussian_noise",
    "add_percent_noise",
    "world_axes",
    "doi_floor_weight",
]


# --------------------------------------------------------------------------- acquisition


@dataclass(frozen=True)
class Acquisition:
    """Per-scenario acquisition spec — what gets collected (doc 05 §4.3).

    Decoupled from the earth so the *same* truth can be surveyed densely or sparsely.
    Every field is optional with a plausible default; a forward reads only the keys it
    needs. Spacings/altitudes are Engineering metres; periods seconds; the ``params``
    bag carries any extra per-method knobs from the authored ``acquisition.jsonc``.
    """

    # potential fields
    gravity_spacing: float = 500.0          # station grid spacing (m)
    mag_line_spacing: float = 400.0         # aeromag line spacing (m)
    mag_altitude: float = 80.0              # flight height above DEM (m)
    mag_heading: float = 90.0               # flight line azimuth (deg CW from N)
    # electrical (ert/ip)
    ert_n_electrodes: int = 32              # electrodes along the line
    ert_spacing: float = 50.0               # electrode spacing a (m)
    ert_array: str = "dipole-dipole"
    ert_line: tuple[tuple[float, float], tuple[float, float]] | None = None
    # em / mt
    em_n_soundings: int = 16                # TEM soundings on a grid
    mt_n_periods: int = 24                  # log-spaced periods
    mt_periods: tuple[float, float] = (1.0e-3, 1.0e3)  # s
    mt_grid_spacing: float = 1000.0
    # seismic
    seis_n_traces: int = 48
    seis_trace_spacing: float = 50.0        # CMP spacing (m)
    seis_dt: float = 0.002                  # sample interval (s)
    seis_n_samples: int = 512
    seis_wavelet_freq: float = 30.0         # Ricker peak frequency (Hz)
    seis_line: tuple[tuple[float, float], tuple[float, float]] | None = None
    # microseismic
    ms_n_events: int = 40
    ms_b_value: float = 1.0                 # Gutenberg-Richter b
    ms_mc: float = -1.0                     # magnitude of completeness
    # insar
    insar_los: tuple[float, float, float] = (0.6, -0.1, 0.79)  # (E, N, U) unit
    insar_n_epochs: int = 6
    insar_pixel: float = 100.0              # raster pixel size (m)
    insar_max_uplift_mm: float = 40.0
    # wells
    wells: tuple[dict[str, Any], ...] = ()  # {"id","path"(N,3 MD/inc/azi),"logs":[...]}
    # heat flow
    heat_n_points: int = 24
    # geology
    geology_n_samples: int = 64

    params: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- artifacts


@dataclass(frozen=True)
class Provenance:
    """Synthetic-data provenance stamped on every emitted artifact (doc 05 §5).

    Marks the file as ``source="synthgen"`` with the scene id + seed + the forward
    method/submethod/fidelity, so a measured file is never mistaken for real instrument
    data (doc 05 §5 "carries provenance noting it is synthetic").
    """

    source: str
    scene_id: str
    seed: int
    method: str
    submethod: str | None
    fidelity: str
    tool: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "sceneId": self.scene_id,
            "seed": self.seed,
            "method": self.method,
            "submethod": self.submethod,
            "fidelity": self.fidelity,
            "tool": self.tool,
            **self.extra,
        }


@dataclass(frozen=True)
class Artifact:
    """One native-format file emitted by a forward (doc 05 §4 contract / §5 measured/).

    ``path`` is the written file; ``fmt`` is the OVERVIEW §3 native format key
    (``csv``, ``geotiff``, ``segy``, ``las``, ``edi``, ``stg``, ``xyz``, ``quakeml``,
    ``geojson``); ``provenance`` records it is synthetic.
    """

    path: Path
    fmt: str
    method: str
    submethod: str | None
    provenance: Provenance


# --------------------------------------------------------------------------- protocol


@runtime_checkable
class ForwardModel(Protocol):
    """Uniform forward-model contract (doc 08 §4d, physics owned by doc 05 §4).

    A T0 forward sets ``fidelity="plausible"`` and emits native-format files the *same*
    method's adapter can ingest, closing the OVERVIEW §8 round-trip. ``simulate`` takes
    the compiled :class:`~geosim.synthgen.truth.TruthEarth`, an :class:`Acquisition`, and
    a seeded ``numpy.random.Generator`` and returns the list of written artifacts.
    """

    method: str
    submethod: str | None
    fidelity: str

    def simulate(
        self, truth: TruthEarth, acquisition: Acquisition, rng: np.random.Generator
    ) -> list[Artifact]:
        ...


class T0Forward:
    """Base for T0 ("plausible", doc 05 §6) forwards — shared provenance plumbing.

    Subclasses set ``method``/``submethod`` (a canonical doc 02 §2 pair) and implement
    :meth:`simulate`; :meth:`_prov` builds the synthetic provenance stamp.
    """

    method: str = ""
    submethod: str | None = None
    fidelity: str = "plausible"
    tool: str = "geosim.synthgen.forward"

    def _prov(self, truth: TruthEarth, **extra: Any) -> Provenance:
        return Provenance(
            source="synthgen",
            scene_id=truth.spec.id,
            seed=truth.spec.seed,
            method=self.method,
            submethod=self.submethod,
            fidelity=self.fidelity,
            tool=f"{self.tool}.{type(self).__name__}",
            extra=extra,
        )

    def simulate(  # pragma: no cover - overridden
        self, truth: TruthEarth, acquisition: Acquisition, rng: np.random.Generator
    ) -> list[Artifact]:
        raise NotImplementedError


# --------------------------------------------------------------------------- grid helpers


def world_axes(truth: TruthEarth) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-axis Engineering coordinate vectors ``(z, y, x)`` for cell centres (Z-up).

    ``origin``/``spacing`` are ``(z, y, x)`` (doc 02 §10.2); centres sit at
    ``origin + i*spacing`` (the compiler already offsets origin by half a cell, so the
    stored origin IS the first cell centre).
    """
    nz, ny, nx = truth.shape
    z0, y0, x0 = truth.origin
    dz, dy, dx = truth.spacing
    z = z0 + np.arange(nz) * dz
    y = y0 + np.arange(ny) * dy
    x = x0 + np.arange(nx) * dx
    return z, y, x


def gaussian_lowpass(field: np.ndarray, sigma_cells: float | tuple[float, ...]) -> np.ndarray:
    """Separable Gaussian low-pass — the universal *resolution kernel* (doc 05 §4 deg. 2).

    Smooths ``field`` so the forward only "sees" a blurred truth (the deep/smooth nature
    of potential-field & MT methods, the band-limited nature of seismic). ``sigma_cells``
    is per-axis in grid cells; reflect-padded so edges stay finite. NaNs are treated as
    holes via normalised convolution so masked cells don't bleed.
    """
    from scipy.ndimage import gaussian_filter

    if np.isscalar(sigma_cells):
        sigma = float(sigma_cells)  # type: ignore[arg-type]
    else:
        sigma = tuple(float(s) for s in sigma_cells)  # type: ignore[assignment]

    finite = np.isfinite(field)
    if finite.all():
        return gaussian_filter(field.astype(np.float64), sigma=sigma, mode="reflect")
    # normalised (NaN-aware) convolution
    vals = np.where(finite, field, 0.0).astype(np.float64)
    num = gaussian_filter(vals, sigma=sigma, mode="reflect")
    den = gaussian_filter(finite.astype(np.float64), sigma=sigma, mode="reflect")
    out = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1.0e-6)
    return out


def depth_doi_mask(
    z: np.ndarray, surface_elev: float, doi_depth: float, taper: float = 0.25
) -> np.ndarray:
    """Depth-of-investigation weight vs. elevation ``z`` (doc 05 §4 deg. 2 DOI mask).

    Returns a weight in ``[0, 1]`` per z-level: ~1 above the DOI depth (well-resolved),
    smoothly tapering to 0 below it (the method loses depth — ERT below ~0.2·array
    length, seismic below fold, etc.). ``doi_depth`` is metres below ``surface_elev``;
    ``taper`` is the fractional width of the roll-off.
    """
    depth = surface_elev - z  # +down
    edge = doi_depth
    width = max(edge * taper, 1.0)
    # logistic roll-off centred at the DOI depth
    return 1.0 / (1.0 + np.exp((depth - edge) / width))


def doi_floor_weight(value: np.ndarray, floor: float) -> np.ndarray:
    """Clamp small-signal values to a measurement floor (late-time/atmos floors)."""
    return np.where(np.abs(value) < floor, np.sign(value) * floor, value)


def sample_volume_at(
    vol: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    pts_zyx: np.ndarray,
) -> np.ndarray:
    """Trilinear-sample a Z-up ``[z,y,x]`` volume at world points ``pts_zyx`` (N,3).

    ``axes`` is ``(z, y, x)`` coordinate vectors; points outside the grid clamp to the
    nearest edge. Used to project truth onto well paths, station columns, etc.
    """
    z, y, x = axes
    p = np.atleast_2d(np.asarray(pts_zyx, dtype=np.float64))

    def _frac(coord: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = coord.size
        if n == 1:
            return np.zeros_like(q, dtype=int), np.zeros_like(q)
        idx = np.interp(q, coord, np.arange(n))
        i0 = np.clip(np.floor(idx).astype(int), 0, n - 2)
        return i0, idx - i0

    iz, fz = _frac(z, p[:, 0])
    iy, fy = _frac(y, p[:, 1])
    ix, fx = _frac(x, p[:, 2])
    out = np.zeros(p.shape[0], dtype=np.float64)
    for dz_ in (0, 1):
        for dy_ in (0, 1):
            for dx_ in (0, 1):
                w = (
                    (fz if dz_ else 1 - fz)
                    * (fy if dy_ else 1 - fy)
                    * (fx if dx_ else 1 - fx)
                )
                out += w * vol[
                    np.clip(iz + dz_, 0, z.size - 1),
                    np.clip(iy + dy_, 0, y.size - 1),
                    np.clip(ix + dx_, 0, x.size - 1),
                ]
    return out


def sample_column(
    vol: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    xy: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a vertical profile ``(z, values)`` through ``vol`` at plan point ``xy``.

    Returns the z-axis (ascending elevation) and the trilinearly-interpolated values
    along it — the basis for MT skin-depth mapping, TEM soundings, well sampling.
    """
    z, y, x = axes
    pts = np.column_stack([z, np.full_like(z, xy[1]), np.full_like(z, xy[0])])
    return z, sample_volume_at(vol, axes, pts)


# --------------------------------------------------------------------------- noise


def add_gaussian_noise(
    values: np.ndarray, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Additive Gaussian noise (doc 05 §4 noise column) — e.g. gravity mGal, mag nT."""
    return values + rng.normal(0.0, sigma, size=np.shape(values))


def add_percent_noise(
    values: np.ndarray, pct: float, rng: np.random.Generator
) -> np.ndarray:
    """Multiplicative '% of reading' noise (ERT/IP/EM/MT, doc 05 §4 noise column)."""
    return values * (1.0 + rng.normal(0.0, pct, size=np.shape(values)))
