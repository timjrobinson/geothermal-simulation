"""Load an ``acquisition.jsonc`` survey plan into an :class:`Acquisition` (doc 05 §4.3).

The authored acquisition JSONC (doc 05 §4.3) is decoupled from the earth so the *same*
truth can be surveyed densely or sparsely. This module maps its per-method blocks onto
the flat :class:`~geosim.synthgen.forward.Acquisition` dataclass, reusing
:func:`~geosim.synthgen.scene.strip_jsonc` for comment/trailing-comma handling so the
authored JSONC parses as standard JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..forward import Acquisition
from ..scene import strip_jsonc

__all__ = ["load_acquisition", "acquisition_from_dict"]


def _line(v: Any) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if not v:
        return None
    (x0, y0), (x1, y1) = v
    return ((float(x0), float(y0)), (float(x1), float(y1)))


def acquisition_from_dict(d: dict[str, Any]) -> Acquisition:
    """Build an :class:`Acquisition` from a parsed acquisition dict (doc 05 §4.3)."""
    grav = d.get("gravity", {})
    mag = d.get("magnetics", {})
    mt = d.get("mt", {})
    em = d.get("em", {})
    ert = d.get("ert", {})
    seis = d.get("seismic", {})
    ms = d.get("microseismic", {})
    insar = d.get("insar", {})
    heat = d.get("heatflow", {})
    geol = d.get("geology", {})

    mt_periods = mt.get("periods", [1.0e-3, 1.0e3])
    insar_los = insar.get("los", [0.6, -0.1, 0.79])

    return Acquisition(
        gravity_spacing=float(grav.get("spacing", 500.0)),
        mag_line_spacing=float(mag.get("lineSpacing", 400.0)),
        mag_altitude=float(mag.get("altitude", 80.0)),
        mag_heading=float(mag.get("heading", 90.0)),
        ert_n_electrodes=int(ert.get("n", 32)),
        ert_spacing=float(ert.get("a", 50.0)),
        ert_array=ert.get("array", "dipole-dipole"),
        ert_line=_line(ert.get("line")),
        em_n_soundings=int(em.get("nSoundings", 16)),
        mt_n_periods=int(mt.get("nPeriods", 24)),
        mt_periods=(float(mt_periods[0]), float(mt_periods[1])),
        mt_grid_spacing=float(mt.get("gridSpacing", 1000.0)),
        seis_n_traces=int(seis.get("nTraces", 48)),
        seis_trace_spacing=float(seis.get("traceSpacing", 50.0)),
        seis_dt=float(seis.get("dt", 0.002)),
        seis_n_samples=int(seis.get("nSamples", 512)),
        seis_wavelet_freq=float(seis.get("waveletFreq", 30.0)),
        seis_line=_line(seis.get("line")),
        ms_n_events=int(ms.get("nEvents", 40)),
        ms_b_value=float(ms.get("bValue", 1.0)),
        ms_mc=float(ms.get("mc", -1.0)),
        insar_los=(float(insar_los[0]), float(insar_los[1]), float(insar_los[2])),
        insar_n_epochs=int(insar.get("nEpochs", 6)),
        insar_pixel=float(insar.get("pixel", 100.0)),
        insar_max_uplift_mm=float(insar.get("maxUpliftMm", 40.0)),
        wells=tuple(d.get("wells", ())),
        heat_n_points=int(heat.get("nPoints", 24)),
        geology_n_samples=int(geol.get("nSamples", 64)),
    )


def load_acquisition(source: str | Path | dict[str, Any]) -> Acquisition:
    """Load an :class:`Acquisition` from an acquisition JSONC path/string/dict (doc 05 §4.3)."""
    if isinstance(source, dict):
        return acquisition_from_dict(source)
    is_path = isinstance(source, Path) or (
        isinstance(source, str) and "\n" not in source and Path(source).exists()
    )
    text = Path(source).read_text(encoding="utf-8") if is_path else str(source)
    return acquisition_from_dict(json.loads(strip_jsonc(text)))
