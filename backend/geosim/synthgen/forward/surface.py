"""InSAR + geology-map T0 forwards (doc 05 §4 rows 10, 13).

- :class:`InSARForward` (``insar``) — project a modeled surface uplift to the
  line-of-sight (doc 05 §4 row 10): a Mogi-like radially-symmetric uplift bowl centred on
  the plume footprint grows over time, projected onto the satellite LOS unit vector and
  emitted as a **GeoTIFF time-series** (one band-file per epoch, mm). Degradations: an
  atmospheric phase screen (correlated noise) + DEM-error tilt + temporal decorrelation
  floor (doc 05 §4 row 10). This is the "only-sees" surface deformation, not the source
  depth — fusion must infer the inflation source.
- :class:`GeologyMapForward` (``geology``) — export mapped *surface contacts + fault
  traces* from the lithology field ``L`` (doc 05 §4 row 13): trace the lithology-class
  boundaries on the top layer of ``L`` and the authored fault traces into a **GeoJSON**
  with an interpretive-uncertainty tag. Mapped-surface only (no depth, doc 05 §4 row 13).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..truth import TruthEarth
from .base import Acquisition, Artifact, T0Forward, world_axes
from .potential_field import write_local_geotiff

__all__ = ["InSARForward", "GeologyMapForward"]


class InSARForward(T0Forward):
    """T0 InSAR: modeled uplift → LOS GeoTIFF time-series (doc 05 §4 row 10)."""

    method = "insar"
    submethod = None

    def simulate(
        self, truth: TruthEarth, acq: Acquisition, rng: np.random.Generator
    ) -> list[Artifact]:
        z, y, x = world_axes(truth)
        # raster grid at the requested pixel size
        gx = np.arange(x[0], x[-1] + 1e-6, acq.insar_pixel)
        gy = np.arange(y[0], y[-1] + 1e-6, acq.insar_pixel)
        if gx.size < 2:
            gx = np.array([x[0], x[-1]])
        if gy.size < 2:
            gy = np.array([y[0], y[-1]])
        gxx, gyy = np.meshgrid(gx, gy, indexing="xy")

        # inflation source: the plume footprint (first anomaly) or ROI centre.
        if truth.spec.anomalies:
            an = truth.spec.anomalies[0]
            cx, cy = an.footprint_center
            radius = an.footprint_radius_xy
        else:
            cx, cy = float(np.mean(x)), float(np.mean(y))
            radius = float((x[-1] - x[0]) / 4.0)

        r2 = (gxx - cx) ** 2 + (gyy - cy) ** 2
        bowl = np.exp(-r2 / (2.0 * radius**2))  # radially symmetric uplift bowl (Mogi-ish)

        los = np.asarray(acq.insar_los, dtype=np.float64)
        los = los / (np.linalg.norm(los) or 1.0)
        # vertical uplift → LOS projection (vertical component los[2])
        los_vert = float(los[2])

        out_dir = Path(acq.params.get("out_dir", "."))
        ts_dir = out_dir / "insar"
        ts_dir.mkdir(parents=True, exist_ok=True)

        artifacts: list[Artifact] = []
        prov = self._prov(truth, units="mm", los=list(los), nEpochs=acq.insar_n_epochs)
        for epoch in range(acq.insar_n_epochs):
            frac = (epoch + 1) / acq.insar_n_epochs
            uplift_mm = acq.insar_max_uplift_mm * frac * bowl  # vertical (mm)
            los_mm = uplift_mm * los_vert
            # atmospheric phase screen: smooth correlated noise + DEM-error tilt
            aps = rng.normal(0, 2.0, los_mm.shape)
            from scipy.ndimage import gaussian_filter
            aps = gaussian_filter(aps, sigma=1.5)
            tilt = 0.5 * (gxx - cx) / (radius + 1.0)
            los_mm = los_mm + aps + tilt
            path = ts_dir / f"los_{epoch:02d}.tif"
            write_local_geotiff(path, los_mm.astype(np.float32), gx, gy)
            artifacts.append(Artifact(path, "geotiff", self.method, self.submethod, prov))
        return artifacts


class GeologyMapForward(T0Forward):
    """T0 geology map: surface contacts + fault traces from L → GeoJSON (doc 05 §4 row13)."""

    method = "geology"
    submethod = None

    def simulate(
        self, truth: TruthEarth, acq: Acquisition, rng: np.random.Generator
    ) -> list[Artifact]:
        z, y, x = world_axes(truth)
        L = truth.lithology  # (nz,ny,nx)
        # the mapped surface = topmost in-ground lithology label per (y,x):
        # scan from top (max z) down to the shallowest sub-surface (non-air) cell.
        air = truth.above_surface
        surf_label = np.full((L.shape[1], L.shape[2]), -1, dtype=int)
        for k in range(L.shape[0] - 1, -1, -1):
            mask = (~air[k]) & (surf_label < 0)
            surf_label[mask] = L[k][mask]
        surf_label[surf_label < 0] = L[0][surf_label < 0]

        features: list[dict] = []
        # contacts: cells where the surface label differs from a neighbour (E/N edges)
        ny, nx = surf_label.shape
        contact_pts: list[tuple[float, float, int, int]] = []
        for j in range(ny):
            for i in range(nx - 1):
                if surf_label[j, i] != surf_label[j, i + 1]:
                    contact_pts.append((
                        float((x[i] + x[i + 1]) / 2.0), float(y[j]),
                        int(surf_label[j, i]), int(surf_label[j, i + 1]),
                    ))
        # represent contacts as points (sampled, capped for size)
        cap = min(len(contact_pts), max(8, acq.geology_n_samples))
        for cxp, cyp, l0, l1 in contact_pts[:cap]:
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [cxp, cyp]},
                "properties": {
                    "kind": "contact",
                    "unitA": truth.unit_names[l0] if l0 < len(truth.unit_names) else str(l0),
                    "unitB": truth.unit_names[l1] if l1 < len(truth.unit_names) else str(l1),
                    "uncertainty": "interpretive",
                },
            })
        # fault traces from the scene (mapped at surface)
        for f in truth.spec.faults:
            (x0, y0), (x1, y1) = f.trace
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[x0, y0], [x1, y1]]},
                "properties": {
                    "kind": "fault", "id": f.id, "faultKind": f.kind,
                    "uncertainty": "interpretive",
                },
            })

        gj = {"type": "FeatureCollection", "name": f"{truth.spec.id}-geology",
              "features": features}
        out_dir = Path(acq.params.get("out_dir", "."))
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "geology_map.geojson"
        path.write_text(json.dumps(gj, indent=2), encoding="utf-8")
        prov = self._prov(truth, nFeatures=len(features))
        return [Artifact(path, "geojson", self.method, self.submethod, prov)]
