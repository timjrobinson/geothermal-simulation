"""ERT + IP T0 forwards — apparent-resistivity / chargeability pseudosections (doc 05 §4).

Both run a **degrade-the-truth** sensitivity-kernel forward (doc 05 §6 T0) along a single
survey line of ``n`` electrodes at spacing ``a`` (dipole-dipole, doc 05 §4.3): for each
quadrupole the apparent property is the *sensitivity-weighted average* of the truth
property over the line's vertical section, where the sensitivity kernel is a depth-decaying
"banana" centred under the array midpoint at the pseudodepth ``≈ n·a/2`` and the array
loses depth past the DOI ``≈ 0.15-0.2 · array length`` (doc 05 §4 row 3/4). That is exactly
the §4.2 "only-sees-what-it-could": the shallow clay cap is sharply averaged, the deep
reservoir conductor falls below DOI and is invisible.

- :class:`ERTForward` (``ert/dc_resistivity``) → apparent **resistivity** pseudosection.
- :class:`IPForward` (``ip/ip_time``) → co-located apparent **chargeability** pseudosection.

Both write an **AGI-style ``.stg``** text file (custom writer matching the SuperSting
column layout the doc-03 adapter parses): a header block then one comma-separated record
per measurement (``A,B,M,N`` electrode XY + apparent value), so the same line round-trips
ingestion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..truth import TruthEarth
from .base import (
    Acquisition,
    Artifact,
    T0Forward,
    add_percent_noise,
    sample_volume_at,
    world_axes,
)

__all__ = ["ERTForward", "IPForward", "build_pseudosection"]


def _line_geometry(
    truth: TruthEarth, acq: Acquisition
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Electrode XY positions + the survey-line unit direction (doc 05 §4.3 ert line)."""
    z, y, x = world_axes(truth)
    if acq.ert_line is not None:
        (x0, y0), (x1, y1) = acq.ert_line
    else:  # default: a W-E line across the ROI centre
        x0, x1 = float(x[0]), float(x[-1])
        y0 = y1 = float(np.mean(y))
    n = acq.ert_n_electrodes
    t = np.linspace(0.0, 1.0, n)
    ex = x0 + t * (x1 - x0)
    ey = y0 + t * (y1 - y0)
    return ex, ey, np.array([x1 - x0, y1 - y0])


def build_pseudosection(
    truth: TruthEarth,
    acq: Acquisition,
    prop_key: str,
    *,
    log_average: bool,
) -> dict[str, np.ndarray]:
    """Sensitivity-kernel pseudosection of ``prop_key`` along the ERT line (doc 05 §4).

    Dipole-dipole sequence: current dipole ``A-B`` (adjacent electrodes), potential dipole
    ``M-N`` ``n`` electrodes away. The pseudodepth ``≈ n·a/2``; the apparent value is a
    Gaussian-sensitivity-weighted average of the truth column under the quadrupole midpoint
    (depth-decaying = DOI). ``log_average=True`` for resistivity (orders of magnitude),
    linear for chargeability.

    Returns dict of per-measurement arrays: ``ax,ay,bx,by,mx,my,nx,ny``, ``midx,midy``,
    ``pseudodepth``, ``apparent``.
    """
    z, y, x = world_axes(truth)
    axes = (z, y, x)
    ex, ey = _line_geometry(truth, acq)[:2]
    a = acq.ert_spacing
    n_elec = ex.size
    surf = float(np.max(z))
    array_len = float(np.hypot(ex[-1] - ex[0], ey[-1] - ey[0]))
    doi = 0.18 * array_len  # depth-of-investigation (doc 05 §4 row 3)

    vol = truth.property_volume(prop_key).astype(np.float64)

    recs: dict[str, list[float]] = {
        k: [] for k in
        ("ax", "ay", "bx", "by", "mx", "my", "nx_", "ny_", "midx", "midy",
         "pseudodepth", "apparent")
    }
    max_n = max(1, min(8, n_elec - 3))
    for ia in range(n_elec - 3):
        ib = ia + 1
        for sep in range(1, max_n + 1):
            im = ib + sep
            inn = im + 1
            if inn >= n_elec:
                break
            mid_x = 0.25 * (ex[ia] + ex[ib] + ex[im] + ex[inn])
            mid_y = 0.25 * (ey[ia] + ey[ib] + ey[im] + ey[inn])
            pdepth = sep * a / 2.0 + a / 2.0
            # sensitivity-weighted vertical average under the midpoint (DOI banana)
            elev_centre = surf - pdepth
            sigz = max(pdepth * 0.6, a)
            # sample a small vertical stack of points around the pseudodepth
            zsamp = elev_centre + np.linspace(-1.5 * sigz, 1.5 * sigz, 9)
            w = np.exp(-0.5 * ((surf - zsamp - pdepth) / sigz) ** 2)
            # DOI cutoff: kill sensitivity below the DOI
            w = w * (1.0 / (1.0 + np.exp(((surf - zsamp) - doi) / (0.25 * doi + 1.0))))
            pts = np.column_stack([
                zsamp, np.full_like(zsamp, mid_y), np.full_like(zsamp, mid_x)
            ])
            colv = sample_volume_at(vol, axes, pts)
            ws = w.sum()
            if ws <= 0:
                app = float(np.mean(colv))
            elif log_average:
                app = float(np.exp(np.sum(w * np.log(np.maximum(colv, 1e-6))) / ws))
            else:
                app = float(np.sum(w * colv) / ws)
            recs["ax"].append(ex[ia])
            recs["ay"].append(ey[ia])
            recs["bx"].append(ex[ib])
            recs["by"].append(ey[ib])
            recs["mx"].append(ex[im])
            recs["my"].append(ey[im])
            recs["nx_"].append(ex[inn])
            recs["ny_"].append(ey[inn])
            recs["midx"].append(mid_x)
            recs["midy"].append(mid_y)
            recs["pseudodepth"].append(pdepth)
            recs["apparent"].append(app)
    return {k: np.asarray(v, dtype=np.float64) for k, v in recs.items()}


def _write_stg(path: Path, ps: dict[str, np.ndarray], value_label: str) -> None:
    """Write an AGI SuperSting-style ``.stg`` text pseudosection (doc 05 §4 native out)."""
    n = ps["apparent"].size
    lines = [
        "AGI SuperSting Synthetic Pseudosection (geosim.synthgen)",
        "Type: dipole-dipole",
        f"records: {n}  value: {value_label}",
        "Idx,A_x,A_y,B_x,B_y,M_x,M_y,N_x,N_y,pseudodepth,value",
    ]
    for i in range(n):
        lines.append(
            f"{i+1},{ps['ax'][i]:.2f},{ps['ay'][i]:.2f},"
            f"{ps['bx'][i]:.2f},{ps['by'][i]:.2f},"
            f"{ps['mx'][i]:.2f},{ps['my'][i]:.2f},"
            f"{ps['nx_'][i]:.2f},{ps['ny_'][i]:.2f},"
            f"{ps['pseudodepth'][i]:.2f},{ps['apparent'][i]:.5f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class ERTForward(T0Forward):
    """T0 DC resistivity: apparent-resistivity pseudosection via sensitivity kernels."""

    method = "ert"
    submethod = "dc_resistivity"

    def simulate(
        self, truth: TruthEarth, acq: Acquisition, rng: np.random.Generator
    ) -> list[Artifact]:
        ps = build_pseudosection(truth, acq, "resistivity", log_average=True)
        # noise: 2-5 % of reading (doc 05 §4 row 3)
        ps["apparent"] = np.maximum(add_percent_noise(ps["apparent"], 0.03, rng), 1e-3)
        out_dir = Path(acq.params.get("out_dir", "."))
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "ert_lineAA.stg"
        _write_stg(path, ps, "apparent_resistivity_ohm_m")
        prov = self._prov(truth, units="ohm*m", array=acq.ert_array)
        return [Artifact(path, "stg", self.method, self.submethod, prov)]


class IPForward(T0Forward):
    """T0 IP: co-located apparent-chargeability pseudosection (same kernels as ERT)."""

    method = "ip"
    submethod = "ip_time"

    def simulate(
        self, truth: TruthEarth, acq: Acquisition, rng: np.random.Generator
    ) -> list[Artifact]:
        ps = build_pseudosection(truth, acq, "chargeability_mv_v", log_average=False)
        # noise: 5-10 % + worse at depth (doc 05 §4 row 4)
        depth_factor = 1.0 + ps["pseudodepth"] / (ps["pseudodepth"].max() + 1.0)
        ps["apparent"] = np.maximum(
            ps["apparent"] * (1.0 + rng.normal(0.0, 0.07, ps["apparent"].size) * depth_factor),
            0.0,
        )
        out_dir = Path(acq.params.get("out_dir", "."))
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "ip_lineAA.stg"
        _write_stg(path, ps, "apparent_chargeability_mv_v")
        prov = self._prov(truth, units="mV/V", array=acq.ert_array)
        return [Artifact(path, "stg", self.method, self.submethod, prov)]
