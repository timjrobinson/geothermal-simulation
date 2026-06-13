"""Ground-truth earth result + bundle writer (doc 05 §1, §5).

:class:`TruthEarth` is the compiled scoring oracle: the shared lithology field ``L``,
the continuous state field ``S``, and every co-located geophysical property volume
derived from them by rock-physics — all on the fine truth grid in Engineering
coordinates, Z-up (doc 05 §2.1, §5; doc 01 §1). Truth is **retained for validation and
never ingested** (doc 05 §1, decision #6).

:func:`write_truth_bundle` serialises a ``truth/`` bundle (doc 05 §5) using the storage
conventions of doc 02 §10.2:

- one ``<property>.zarr`` PropertyModel per derived property (ρ, χ, res, η, Vp, Vs, T,
  φ) via :func:`geosim.storage.write_property_model`, with co-registered 1σ from the
  property registry's default relative σ (doc 02 §6);
- ``lithology.zarr`` (the integer ``L`` label volume) + the per-property state volumes;
- ``features.geojson`` with the true fault traces + anomaly footprints (doc 05 §5 "true
  faults/horizons/anomaly solids").

Property → registry key mapping (canonical units, doc 01 §5): ``density``,
``susceptibility``, ``resistivity``, ``chargeability_mv_v``, ``velocity_p``,
``velocity_s``, ``temperature`` (kelvin), ``porosity``. ``salinity``/``alteration``/
``fracture`` state fields are written as raw label/state arrays (no canonical registry
key for salinity/alteration in doc 01 §5; fracture_density/water_saturation/porosity
*are* registered).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from geosim.spatial import REGISTRY
from geosim.storage import GridSpec, write_property_model

from .rockphysics import RockPhysicsResult
from .scene import SceneSpec, UnitProps

__all__ = ["StateField", "TruthEarth", "write_truth_bundle"]


@dataclass(frozen=True)
class StateField:
    """The continuous state field ``S(x,y,z)`` (doc 05 §2.1).

    Co-located ``float`` ``(nz, ny, nx)`` volumes, Z-up. ``temperature`` is canonical
    kelvin (doc 01 §5); ``salinity_tds`` is ppm; the rest are fractions in ``[0, 1]``.
    ``porosity_boost`` is the additive porosity perturbation from anomalies (the matrix
    porosity comes from the per-unit library, doc 05 §3.2).
    """

    temperature: np.ndarray  # kelvin
    porosity_boost: np.ndarray  # fraction (additive)
    water_saturation: np.ndarray  # fraction
    salinity_tds: np.ndarray  # ppm
    alteration_fraction: np.ndarray  # fraction
    fracture_density: np.ndarray  # fraction


# property-volume attribute → property registry key (doc 01 §5 canonical units).
_PROPERTY_KEYS: dict[str, str] = {
    "density": "density",
    "susceptibility": "susceptibility",
    "resistivity": "resistivity",
    "chargeability_mv_v": "chargeability_mv_v",
    "velocity_p": "velocity_p",
    "velocity_s": "velocity_s",
    "temperature": "temperature",
    "porosity": "porosity",
}


@dataclass(frozen=True)
class TruthEarth:
    """Compiled ground-truth earth: ``L`` + ``S`` + all derived properties (doc 05 §2.1).

    ``origin``/``spacing`` are Engineering metres in ``(z, y, x)`` order (Z-up, doc 02
    §10.2). ``above_surface`` marks voxels above the synthetic DEM (air).
    """

    spec: SceneSpec
    lithology: np.ndarray  # int (nz,ny,nx)
    unit_names: list[str]
    units: list[UnitProps]
    state: StateField
    properties: RockPhysicsResult
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]
    above_surface: np.ndarray  # bool (nz,ny,nx)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.lithology.shape  # type: ignore[return-value]

    def property_volume(self, key: str) -> np.ndarray:
        """Return a derived property volume by registry key (doc 05 §2.2)."""
        for attr, k in _PROPERTY_KEYS.items():
            if k == key:
                return getattr(self.properties, attr)
        raise KeyError(f"no truth property volume for key {key!r}")

    def grid_spec(self) -> GridSpec:
        """Storage :class:`~geosim.storage.GridSpec` for the truth grid (doc 02 §10.2)."""
        return GridSpec(origin=self.origin, spacing=self.spacing, cell_ref="center")


# --------------------------------------------------------------------------- features


def _features_geojson(earth: TruthEarth) -> dict:
    """Build a GeoJSON FeatureCollection of true faults + anomaly footprints (doc 05 §5).

    Coordinates are Engineering metres (local frame); a CRS member is omitted because the
    bundle is Engineering-coordinate (doc 05 §5, doc 01 §1). Each feature carries the
    truth metadata a validation tool keys on (fault throw/dip/conduit; anomaly extent).
    """
    feats: list[dict] = []
    for f in earth.spec.faults:
        (x0, y0), (x1, y1) = f.trace
        feats.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[x0, y0], [x1, y1]]},
                "properties": {
                    "kind": "fault",
                    "id": f.id,
                    "faultKind": f.kind,
                    "dip": f.dip,
                    "dipAzimuth": f.dip_azimuth,
                    "throw": f.throw,
                    "isConduit": f.is_conduit,
                },
            }
        )
    for an in earth.spec.anomalies:
        cx, cy = an.footprint_center
        r = an.footprint_radius_xy
        # square the footprint as a polygon ring (plan-view extent solid, doc 05 §5)
        ring = [
            [cx - r, cy - r],
            [cx + r, cy - r],
            [cx + r, cy + r],
            [cx - r, cy + r],
            [cx - r, cy - r],
        ]
        feats.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "kind": "anomaly",
                    "id": an.id,
                    "anomalyKind": an.kind,
                    "topElev": an.top_elev,
                    "bottomElev": an.bottom_elev,
                    "tempPeak": an.temp_peak,
                    "controlledBy": an.controlled_by,
                },
            }
        )
    return {"type": "FeatureCollection", "name": earth.spec.id, "features": feats}


# --------------------------------------------------------------------------- writer


def write_truth_bundle(earth: TruthEarth, out_dir: str | Path, *, overwrite: bool = False) -> Path:
    """Write the ``truth/`` bundle for ``earth`` under ``out_dir`` (doc 05 §5).

    Layout (doc 05 §5)::

        <out_dir>/
          <property>.zarr        # one PropertyModel per derived property (+ _sigma)
          lithology.zarr         # integer L label volume
          state_<field>.zarr     # continuous S fields
          features.geojson       # true faults + anomaly footprints

    PropertyModels are written via :func:`geosim.storage.write_property_model` (doc 02
    §10.2 layout, mean/variance-correct pyramids, Engineering origin/spacing). Returns
    ``out_dir``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    grid = earth.grid_spec()

    # derived geophysical property volumes (+ co-registered 1σ, doc 02 §6).
    for attr, key in _PROPERTY_KEYS.items():
        values = getattr(earth.properties, attr)
        pt = REGISTRY.get(key)
        sigma = np.abs(values) * float(pt.default_rel_sigma)
        write_property_model(
            out / f"{key}.zarr",
            key,
            values,
            grid=grid,
            sigma=sigma.astype(np.float32),
            overwrite=overwrite,
        )

    # lithology label volume (categorical; written as a plain float PropertyModel using
    # the registered categorical `lithology_class` key, doc 02 §10.2).
    write_property_model(
        out / "lithology.zarr",
        "lithology_class",
        earth.lithology.astype(np.float32),
        grid=grid,
        overwrite=overwrite,
    )

    # continuous state fields that have a registered canonical key (doc 01 §5).
    state_keys = {
        "water_saturation": earth.state.water_saturation,
        "fracture_density": earth.state.fracture_density,
    }
    for key, vol in state_keys.items():
        write_property_model(
            out / f"state_{key}.zarr",
            key,
            vol.astype(np.float32),
            grid=grid,
            overwrite=overwrite,
        )

    # raw state fields without a canonical registry key (salinity ppm, alteration frac):
    # store as .npy alongside so validation can still consume them (doc 05 §2 note).
    np.save(out / "state_salinity_tds.npy", earth.state.salinity_tds.astype(np.float32))
    np.save(
        out / "state_alteration_fraction.npy",
        earth.state.alteration_fraction.astype(np.float32),
    )

    # true features (doc 05 §5).
    (out / "features.geojson").write_text(
        json.dumps(_features_geojson(earth), indent=2), encoding="utf-8"
    )

    # unit-name index for the lithology labels (so L integers map back to units).
    (out / "lithology_units.json").write_text(
        json.dumps({i: n for i, n in enumerate(earth.unit_names)}, indent=2),
        encoding="utf-8",
    )

    return out
