"""Declarative scene specification (doc 05 §2.3).

A :class:`SceneSpec` is the authored, JSONC-loadable description of a synthetic earth:
the spatial :class:`FrameSpec` (ROI / depth range / truth-grid resolution), a
:class:`SurfaceSpec` DEM, a top→down :class:`LayerSpec` stack, optional
:class:`IntrusionSpec` bodies and :class:`FaultSpec` cuts (some of which are fluid
*conduits*), a background :class:`GeothermSpec`, hydrothermal :class:`AnomalySpec`
state perturbations, the named rock-physics ruleset, and a per-unit base property
:class:`UnitProps` library (doc 05 §3.2).

The compiler (:mod:`geosim.synthgen.compiler`) consumes this to build the shared
lithology field ``L`` and state field ``S`` (doc 05 §2.1), from which rock-physics
(:mod:`geosim.synthgen.rockphysics`) derives every geophysical property — the
"one geology → all properties" invariant (doc 05 §1, decision #1).

Everything is plain dataclasses so a scene round-trips to/from JSONC: :func:`load_scene`
strips ``//`` and ``/* */`` comments (JSONC) and builds the spec; nothing here is
stochastic — the *seed* lives on the spec and the compiler owns all randomness so a
scene is reproducible from ``(spec, seed)`` (doc 05 §1 invariant).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "FrameSpec",
    "SurfaceSpec",
    "LayerSpec",
    "IntrusionSpec",
    "FaultSpec",
    "GeothermSpec",
    "AnomalySpec",
    "UnitProps",
    "SceneSpec",
    "load_scene",
    "strip_jsonc",
]


@dataclass(frozen=True)
class FrameSpec:
    """ROI + depth range + truth-grid resolution (doc 05 §2.3 ``frame``).

    ``roi`` is Engineering metres ``(xmin, xmax, ymin, ymax)``; ``depth_range`` is
    Engineering elevation metres ``(zmin, zmax)`` (Z-up, zmax ≈ surface top); the
    truth-grid spacings ``dx/dy/dz`` are the fine resolution (doc 05 §2: 25–50 m
    laterally, 10–25 m vertically).
    """

    mode: str = "local"  # "local" | "georeferenced" (doc 01 §2)
    xmin: float = -6000.0
    xmax: float = 6000.0
    ymin: float = -6000.0
    ymax: float = 6000.0
    zmin: float = -6000.0
    zmax: float = 1700.0
    dx: float = 50.0
    dy: float = 50.0
    dz: float = 20.0

    @property
    def shape(self) -> tuple[int, int, int]:
        """Truth-grid shape ``(nz, ny, nx)`` (Z-up axis order, doc 02 §10.2)."""
        nx = max(1, int(round((self.xmax - self.xmin) / self.dx)))
        ny = max(1, int(round((self.ymax - self.ymin) / self.dy)))
        nz = max(1, int(round((self.zmax - self.zmin) / self.dz)))
        return (nz, ny, nx)


@dataclass(frozen=True)
class SurfaceSpec:
    """Synthetic DEM (doc 05 §2.3 ``surface``) → ``surfaceModel: synthetic:<id>``."""

    kind: str = "flat"  # "flat" | "fractal" | "tilted-block"
    base_elev: float = 1600.0
    relief: float = 0.0  # peak-to-trough amplitude (m) for fractal/tilted-block
    roughness: float = 0.7  # fractal persistence (0..1); higher = rougher
    # tilted-block: linear gradient over the ROI, metres of rise per metre of plan
    tilt_x: float = 0.0
    tilt_y: float = 0.0


@dataclass(frozen=True)
class LayerSpec:
    """One stratigraphic layer (doc 05 §2.3 ``layers``); filled top→down.

    ``thickness`` is either a scalar, a ``[min, max]`` range (noisy contact), or the
    sentinel ``"fill"`` for the bottom unit that fills to ``zmin``.
    """

    unit: str
    top: str = "conformable"  # "surface" | "conformable"
    thickness: float | tuple[float, float] | str = "fill"


@dataclass(frozen=True)
class IntrusionSpec:
    """An intrusive body inserted into ``L`` (doc 05 §2.3 ``intrusions``).

    ``center`` is Engineering ``(x, y, z)`` metres; ``shape`` is currently an
    ellipsoid ("stock") with horizontal radius ``radius_xy`` and vertical ``radius_z``.
    """

    unit: str
    center: tuple[float, float, float]
    radius_xy: float
    radius_z: float
    shape: str = "stock"


@dataclass(frozen=True)
class FaultSpec:
    """A planar normal/reverse fault that cuts ``L`` and offsets one side (doc 05 §2.3).

    ``trace`` is two surface points ``[[x0,y0],[x1,y1]]`` (Engineering m) defining the
    fault strike; ``dip``/``dip_azimuth`` orient the plane; ``throw`` is vertical offset
    (m). ``is_conduit`` marks the fault that focuses the hydrothermal upflow (doc 05
    §2.3, §7.1 — the range-front conduit).
    """

    id: str
    trace: tuple[tuple[float, float], tuple[float, float]]
    kind: str = "normal"  # "normal" | "reverse"
    dip: float = 60.0  # degrees from horizontal
    dip_azimuth: float = 90.0  # downdip direction, degrees CW from North (+Y)
    throw: float = 0.0  # vertical offset (m)
    is_conduit: bool = False


@dataclass(frozen=True)
class GeothermSpec:
    """Background conductive geotherm (doc 05 §2.3 ``geotherm``).

    ``surface_temp`` in °C at the surface; ``gradient`` in °C/km (Basin & Range is
    high, ~45). Stored canonical in kelvin by the compiler (doc 01 §5).
    """

    surface_temp: float = 15.0
    gradient: float = 45.0


@dataclass(frozen=True)
class AnomalySpec:
    """A hydrothermal-plume STATE perturbation (doc 05 §2.3 ``anomalies``, decision #2).

    The geothermal target is primarily a *state* perturbation (hot + altered +
    porous/fractured + saline), optionally focused by a conduit fault
    (``controlled_by``) and topped by a shallow clay-cap conductor. ``footprint`` is the
    plan-view ellipse ``(cx, cy, radius_xy)``; ``top_elev``/``bottom_elev`` bound it
    vertically (Engineering elevation m). The ``perturb_*`` fields set the peak values
    blended (gaussian halo) into ``S``.
    """

    id: str
    footprint_center: tuple[float, float]
    footprint_radius_xy: float
    top_elev: float
    bottom_elev: float
    kind: str = "hydrothermal-plume"
    controlled_by: str | None = None  # FaultSpec.id of the conduit
    temp_peak: float = 220.0  # °C at the plume core
    alteration_frac: float = 0.6  # clay cap up high / propylitic deep
    porosity_boost: float = 0.04
    salinity_tds: float = 8000.0  # ppm → conductive brine
    fracture_density: float = 0.5
    clay_cap_top_elev: float | None = None
    clay_cap_thickness: float = 0.0


@dataclass(frozen=True)
class UnitProps:
    """Per-lithology base property library entry (doc 05 §3.2).

    Base ``rho`` (kg/m³), ``chi`` (SI susceptibility), ``vp`` (m/s) and intrinsic
    ``phi`` (matrix porosity fraction). Resistivity / chargeability / Vs are DERIVED by
    rock-physics, never authored here (doc 05 §3.2 note). ``chargeable_frac`` optionally
    seeds an intrinsic sulphide/chargeable fraction; ``vp_vs_ratio`` sets the dry-rock
    Vp/Vs used to derive Vs.
    """

    rho: float  # kg/m³ grain/bulk density
    chi: float  # SI magnetic susceptibility
    vp: float  # m/s
    phi: float  # matrix porosity (fraction)
    chargeable_frac: float = 0.0
    vp_vs_ratio: float = 1.73  # ~√3, typical crystalline


@dataclass(frozen=True)
class SceneSpec:
    """A complete declarative synthetic-earth scene (doc 05 §2.3)."""

    id: str
    seed: int = 42
    frame: FrameSpec = field(default_factory=FrameSpec)
    surface: SurfaceSpec = field(default_factory=SurfaceSpec)
    layers: tuple[LayerSpec, ...] = ()
    intrusions: tuple[IntrusionSpec, ...] = ()
    faults: tuple[FaultSpec, ...] = ()
    geotherm: GeothermSpec = field(default_factory=GeothermSpec)
    anomalies: tuple[AnomalySpec, ...] = ()
    rock_physics: str = "default-v1"
    units: dict[str, UnitProps] = field(default_factory=dict)

    def conduit_fault(self, fault_id: str | None) -> FaultSpec | None:
        if fault_id is None:
            return None
        for f in self.faults:
            if f.id == fault_id:
                return f
        return None


# --------------------------------------------------------------------------- JSONC


_LINE_COMMENT = re.compile(r"(?m)//[^\n\r]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def strip_jsonc(text: str) -> str:
    """Strip ``//`` line + ``/* */`` block comments and trailing commas (JSONC → JSON).

    String literals containing ``//`` are protected: we only strip comments that are
    not inside a double-quoted string (doc 05 §2.3 authors scenes in JSONC).
    """
    out: list[str] = []
    i, n = 0, len(text)
    in_str = False
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            i = n if j == -1 else j
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            i = n if j == -1 else j + 2
            continue
        out.append(ch)
        i += 1
    cleaned = "".join(out)
    return _TRAILING_COMMA.sub(r"\1", cleaned)


def _frame_from(d: dict[str, Any]) -> FrameSpec:
    roi = d.get("roi", {})
    dr = d.get("depthRange", {})
    tg = d.get("truthGrid", {})
    return FrameSpec(
        mode=d.get("mode", "local"),
        xmin=float(roi.get("xmin", -6000.0)),
        xmax=float(roi.get("xmax", 6000.0)),
        ymin=float(roi.get("ymin", -6000.0)),
        ymax=float(roi.get("ymax", 6000.0)),
        zmin=float(dr.get("zmin", -6000.0)),
        zmax=float(dr.get("zmax", 1700.0)),
        dx=float(tg.get("dx", 50.0)),
        dy=float(tg.get("dy", 50.0)),
        dz=float(tg.get("dz", 20.0)),
    )


def _surface_from(d: dict[str, Any]) -> SurfaceSpec:
    return SurfaceSpec(
        kind=d.get("kind", "flat"),
        base_elev=float(d.get("baseElev", 1600.0)),
        relief=float(d.get("relief", 0.0)),
        roughness=float(d.get("roughness", 0.7)),
        tilt_x=float(d.get("tiltX", 0.0)),
        tilt_y=float(d.get("tiltY", 0.0)),
    )


def _thickness_from(v: Any) -> float | tuple[float, float] | str:
    if isinstance(v, (list, tuple)):
        return (float(v[0]), float(v[1]))
    if isinstance(v, str):
        return v
    return float(v)


def _layer_from(d: dict[str, Any]) -> LayerSpec:
    return LayerSpec(
        unit=d["unit"],
        top=d.get("top", "conformable"),
        thickness=_thickness_from(d.get("thickness", "fill")),
    )


def _intrusion_from(d: dict[str, Any]) -> IntrusionSpec:
    c = d["center"]
    return IntrusionSpec(
        unit=d["unit"],
        center=(float(c[0]), float(c[1]), float(c[2])),
        radius_xy=float(d["radiusXY"]),
        radius_z=float(d["radiusZ"]),
        shape=d.get("shape", "stock"),
    )


def _fault_from(d: dict[str, Any]) -> FaultSpec:
    t = d["trace"]
    return FaultSpec(
        id=d["id"],
        trace=((float(t[0][0]), float(t[0][1])), (float(t[1][0]), float(t[1][1]))),
        kind=d.get("kind", "normal"),
        dip=float(d.get("dip", 60.0)),
        dip_azimuth=float(d.get("dipAzimuth", 90.0)),
        throw=float(d.get("throw", 0.0)),
        is_conduit=bool(d.get("isConduit", False)),
    )


def _anomaly_from(d: dict[str, Any]) -> AnomalySpec:
    fp = d.get("footprint", {})
    c = fp.get("center", [0.0, 0.0])
    p = d.get("perturb", {})
    cc = d.get("clayCap", {})
    return AnomalySpec(
        id=d["id"],
        footprint_center=(float(c[0]), float(c[1])),
        footprint_radius_xy=float(fp.get("radiusXY", 1000.0)),
        top_elev=float(d.get("topElev", 0.0)),
        bottom_elev=float(d.get("bottomElev", -4000.0)),
        kind=d.get("kind", "hydrothermal-plume"),
        controlled_by=d.get("controlledBy"),
        temp_peak=float(p.get("tempPeak", 220.0)),
        alteration_frac=float(p.get("alterationFrac", 0.6)),
        porosity_boost=float(p.get("porosityBoost", 0.04)),
        salinity_tds=float(p.get("salinityTDS", 8000.0)),
        fracture_density=float(p.get("fractureDensity", 0.5)),
        clay_cap_top_elev=(float(cc["topElev"]) if "topElev" in cc else None),
        clay_cap_thickness=float(cc.get("thickness", 0.0)),
    )


def _units_from(d: dict[str, Any]) -> dict[str, UnitProps]:
    out: dict[str, UnitProps] = {}
    for name, u in d.items():
        out[name] = UnitProps(
            rho=float(u["rho"]),
            chi=float(u["chi"]),
            vp=float(u["Vp"]) if "Vp" in u else float(u["vp"]),
            phi=float(u["phi"]),
            chargeable_frac=float(u.get("chargeableFrac", 0.0)),
            vp_vs_ratio=float(u.get("vpVsRatio", 1.73)),
        )
    return out


def scene_from_dict(d: dict[str, Any]) -> SceneSpec:
    """Build a :class:`SceneSpec` from a parsed JSON dict (doc 05 §2.3 schema)."""
    return SceneSpec(
        id=d["id"],
        seed=int(d.get("seed", 42)),
        frame=_frame_from(d.get("frame", {})),
        surface=_surface_from(d.get("surface", {})),
        layers=tuple(_layer_from(x) for x in d.get("layers", [])),
        intrusions=tuple(_intrusion_from(x) for x in d.get("intrusions", [])),
        faults=tuple(_fault_from(x) for x in d.get("faults", [])),
        geotherm=GeothermSpec(
            surface_temp=float(d.get("geotherm", {}).get("surfaceTemp", 15.0)),
            gradient=float(d.get("geotherm", {}).get("gradient", 45.0)),
        ),
        anomalies=tuple(_anomaly_from(x) for x in d.get("anomalies", [])),
        rock_physics=d.get("rockPhysics", "default-v1"),
        units=_units_from(d.get("units", {})),
    )


def load_scene(source: str | Path | dict[str, Any]) -> SceneSpec:
    """Load a :class:`SceneSpec` from a JSONC file path, a JSONC string, or a dict.

    Comments and trailing commas are stripped first (:func:`strip_jsonc`) so the
    authored doc 05 §2.3 JSONC parses as standard JSON.
    """
    if isinstance(source, dict):
        return scene_from_dict(source)
    is_path = isinstance(source, Path) or (
        isinstance(source, str) and "\n" not in source and Path(source).exists()
    )
    if is_path:
        text = Path(source).read_text(encoding="utf-8")
    else:
        text = str(source)
    return scene_from_dict(json.loads(strip_jsonc(text)))
