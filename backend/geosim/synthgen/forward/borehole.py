"""Well-log + heat-flow T0 forwards (doc 05 §4 rows 11–12).

- :class:`WellLogForward` (``welllog``) — the *cleanest* data (doc 05 §4 row 11): sample
  the truth property volumes along a deviated well path (a min-curvature integration of an
  MD/inc/azi survey, doc 01 §4 via :func:`geosim.spatial.min_curvature_positions`),
  emitting standard curves — ``RES`` (resistivity), ``GR`` (gamma proxy from alteration +
  lithology), ``DEN`` (bulk density), ``VP`` (velocity_p as sonic), ``TEMP`` (temperature,
  canonical kelvin / display °C). Degradations: tool vertical resolution (a short moving
  average) + small per-curve Gaussian noise (doc 05 §4 row 11). Emits **LAS** (``lasio``)
  + the deviation survey CSV.
- :class:`HeatFlowForward` (``heatflow``) — sparse bottom-hole-temperature / spring points
  (doc 05 §4 row 12): sample the truth temperature at scattered surface-to-depth points,
  store **kelvin** canonical with a display °C column, with a BHT-correction-style error.
  Emits a CSV of temperature points.
"""

from __future__ import annotations

from pathlib import Path

import lasio
import numpy as np
import pandas as pd

from geosim.spatial import convert, min_curvature_positions

from ..truth import TruthEarth
from .base import (
    Acquisition,
    Artifact,
    T0Forward,
    sample_volume_at,
    world_axes,
)

__all__ = ["WellLogForward", "HeatFlowForward"]


def _default_well_path(truth: TruthEarth) -> np.ndarray:
    """A simple deviated MD/inc/azi survey through the ROI centre (doc 01 §4)."""
    z, y, x = world_axes(truth)
    surf = float(np.max(z))
    bottom = float(np.min(z))
    total_md = (surf - bottom) * 1.15
    md = np.linspace(0.0, total_md, 8)
    inc = np.clip((md / total_md) * 35.0, 0.0, 35.0)  # build to 35° deviation
    azi = np.full_like(md, 90.0)  # heading east
    return np.column_stack([md, inc, azi])


def _wellhead(truth: TruthEarth) -> tuple[float, float, float]:
    z, y, x = world_axes(truth)
    return float(np.mean(x)), float(np.mean(y)), float(np.max(z))


def _moving_average(a: np.ndarray, w: int) -> np.ndarray:
    """Centred moving average — the tool vertical-resolution kernel (doc 05 §4 row 11)."""
    if w <= 1 or a.size < w:
        return a
    k = np.ones(w) / w
    return np.convolve(a, k, mode="same")


class WellLogForward(T0Forward):
    """T0 well logs: sample truth volumes along the well path → LAS (doc 05 §4 row 11)."""

    method = "welllog"
    submethod = None

    def simulate(
        self, truth: TruthEarth, acq: Acquisition, rng: np.random.Generator
    ) -> list[Artifact]:
        z, y, x = world_axes(truth)
        axes = (z, y, x)
        out_dir = Path(acq.params.get("out_dir", "."))
        wells_dir = out_dir / "wells"
        wells_dir.mkdir(parents=True, exist_ok=True)

        wells = acq.wells or ({"id": "GT-1", "logs": ["res", "gr", "den", "vp", "temp"]},)
        artifacts: list[Artifact] = []
        for well in wells:
            wid = well.get("id", "GT-1")
            survey = np.asarray(well["path"]) if "path" in well else _default_well_path(truth)
            head = _wellhead(truth)
            mc = min_curvature_positions(survey, (head[0], head[1]), kb_elev=head[2])
            enu = mc.enu  # (N,3) East,North,Up
            md = mc.md

            # densify to a fine MD sampling for log curves
            md_fine = np.arange(md[0], md[-1] + 0.1, max(truth.spacing[0] / 2.0, 1.0))
            ex = np.interp(md_fine, md, enu[:, 0])
            ny_ = np.interp(md_fine, md, enu[:, 1])
            ez = np.interp(md_fine, md, enu[:, 2])
            pts = np.column_stack([ez, ny_, ex])  # (z,y,x)

            res = sample_volume_at(truth.property_volume("resistivity"), axes, pts)
            den = sample_volume_at(truth.property_volume("density"), axes, pts)
            vp = sample_volume_at(truth.property_volume("velocity_p"), axes, pts)
            temp_k = sample_volume_at(truth.property_volume("temperature"), axes, pts)
            # gamma-ray proxy: alteration/clay raises GR (no canonical key → derived)
            alt = sample_volume_at(
                truth.state.alteration_fraction.astype(np.float64), axes, pts
            )
            gr = 40.0 + 120.0 * np.clip(alt, 0.0, 1.0)  # API-like

            # tool vertical resolution + per-curve Gaussian noise (doc 05 §4 row 11)
            res = _moving_average(res, 3) * (1.0 + rng.normal(0, 0.02, res.size))
            den = _moving_average(den, 3) + rng.normal(0, 10.0, den.size)
            vp = _moving_average(vp, 3) + rng.normal(0, 20.0, vp.size)
            gr = _moving_average(gr, 3) + rng.normal(0, 2.0, gr.size)
            temp_c = convert(_moving_average(temp_k, 3), "kelvin", "degC") + rng.normal(
                0, 0.5, temp_k.size
            )

            las = lasio.LASFile()
            las.well["WELL"] = lasio.HeaderItem("WELL", value=wid)
            las.well["FLD"] = lasio.HeaderItem("FLD", value=truth.spec.id)
            las.well["SRC"] = lasio.HeaderItem("SRC", value="synthgen")
            tvd = np.interp(md_fine, md, mc.tvd)
            las.append_curve("DEPT", tvd, unit="m", descr="TVD below KB")
            las.append_curve("MD", md_fine, unit="m", descr="measured depth")
            las.append_curve("RES", res, unit="ohm.m", descr="resistivity")
            las.append_curve("GR", gr, unit="gAPI", descr="gamma (alteration proxy)")
            las.append_curve("DEN", den, unit="kg/m3", descr="bulk density")
            las.append_curve("VP", vp, unit="m/s", descr="P velocity (sonic)")
            las.append_curve("TEMP", temp_c, unit="degC", descr="temperature")

            las_path = wells_dir / f"{wid}.las"
            las.write(str(las_path), version=2.0)

            dev_df = pd.DataFrame(survey, columns=["MD", "INC", "AZI"])
            dev_path = wells_dir / f"{wid}_deviation.csv"
            dev_df.to_csv(dev_path, index=False)

            prov = self._prov(truth, well=wid, curves=["RES", "GR", "DEN", "VP", "TEMP"])
            artifacts.append(Artifact(las_path, "las", self.method, self.submethod, prov))
            artifacts.append(Artifact(dev_path, "csv", self.method, self.submethod, prov))
        return artifacts


class HeatFlowForward(T0Forward):
    """T0 heat flow: sparse BHT / spring temperature points (kelvin) (doc 05 §4 row 12)."""

    method = "heatflow"
    submethod = None

    def simulate(
        self, truth: TruthEarth, acq: Acquisition, rng: np.random.Generator
    ) -> list[Artifact]:
        z, y, x = world_axes(truth)
        axes = (z, y, x)
        temp = truth.property_volume("temperature").astype(np.float64)

        n = acq.heat_n_points
        px = rng.uniform(x[0], x[-1], n)
        py = rng.uniform(y[0], y[-1], n)
        # measurement elevations: scattered between surface and mid-depth (BHT)
        surf = float(np.max(z))
        pz = surf - rng.uniform(0.0, (surf - float(np.min(z))) * 0.7, n)
        pts = np.column_stack([pz, py, px])
        temp_k = sample_volume_at(temp, axes, pts)
        # BHT correction error (doc 05 §4 row 12)
        temp_k = temp_k + rng.normal(0, 2.0, n)
        temp_c = convert(temp_k, "kelvin", "degC")

        out_dir = Path(acq.params.get("out_dir", "."))
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({
            "point": np.arange(n),
            "x": px, "y": py, "elev": pz,
            "temperature_k": temp_k,      # canonical kelvin (doc 01 §5)
            "temperature_degc": temp_c,   # display °C
        })
        path = out_dir / "temperature_points.csv"
        df.to_csv(path, index=False)
        prov = self._prov(truth, units="kelvin", nPoints=n)
        return [Artifact(path, "csv", self.method, self.submethod, prov)]
