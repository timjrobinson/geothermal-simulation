"""Scene compiler — build the shared lithology ``L`` and state ``S`` fields (doc 05 §2).

The compiler does **not** rasterise each geophysical property by hand. It builds two
intermediate fields on the fine truth grid, then maps them to all properties through
rock-physics (doc 05 §2.1, §3):

1. **Lithology field** ``L(x,y,z)`` — integer unit label per voxel, built by laying down
   a synthetic surface, stacking layers top→down, cutting + offsetting with faults, and
   inserting intrusion bodies (doc 05 §2.3 compilation order).
2. **State field** ``S(x,y,z)`` — continuous per-voxel ``temperature, porosity,
   water_saturation, salinity, alteration_fraction, fracture_density`` from a background
   conductive geotherm plus hydrothermal-plume / clay-cap blends (doc 05 §2.1, decision
   #2: the anomaly is *primarily a state perturbation*).

Seeded correlated-noise texture (doc 05 §2.3 "small correlated random fields") perturbs
contact depths and adds grain to ``S`` so volumes aren't cartoon-flat; all randomness is
drawn from ``numpy.random.SeedSequence(spec.seed)`` sub-streams so a scene is
byte-reproducible from ``(spec, seed)`` (doc 05 §1 invariant). Coordinates are
Engineering metres, Z-up (doc 01 §1); the truth grid uses storage axis order
``[z,y,x]`` (doc 02 §10.2).

Rock-physics (:mod:`geosim.synthgen.rockphysics`) then turns ``(L, S)`` into the
co-located property volumes, returned together as a :class:`~geosim.synthgen.truth.TruthEarth`.
"""

from __future__ import annotations

import numpy as np

from geosim.spatial import convert

from .rockphysics import DEFAULT_UNIT_LIBRARY, get_ruleset
from .scene import (
    FaultSpec,
    SceneSpec,
    SurfaceSpec,
    UnitProps,
)
from .truth import StateField, TruthEarth

__all__ = ["compile_scene", "CompiledFields"]

from dataclasses import dataclass


@dataclass(frozen=True)
class CompiledFields:
    """The intermediate ``L`` / ``S`` fields + grid metadata (doc 05 §2.1)."""

    lithology: np.ndarray  # int (nz,ny,nx) — index into `units`
    units: list[UnitProps]
    unit_names: list[str]
    state: StateField
    origin: tuple[float, float, float]  # (z0, y0, x0) Engineering m
    spacing: tuple[float, float, float]  # (dz, dy, dx) Engineering m


# --------------------------------------------------------------------------- helpers


def _grid_coords(spec: SceneSpec):
    """Return per-axis Engineering coordinate vectors (z, y, x) for cell centres.

    ``z`` is ascending elevation (Z-up): index 0 = deepest (``zmin``), last = shallowest.
    """
    f = spec.frame
    nz, ny, nx = f.shape
    z = f.zmin + (np.arange(nz) + 0.5) * f.dz
    y = f.ymin + (np.arange(ny) + 0.5) * f.dy
    x = f.xmin + (np.arange(nx) + 0.5) * f.dx
    return z, y, x


def _fractal_value_noise(
    shape2d: tuple[int, int], roughness: float, rng: np.random.Generator
) -> np.ndarray:
    """Seeded multi-octave value noise on a 2-D (ny,nx) plane, normalised to [-1, 1].

    A lightweight fBm-style sum of upsampled white-noise octaves (doc 05 §2.3 fractal
    surface / correlated texture). ``roughness`` is the per-octave persistence.
    """
    ny, nx = shape2d
    field = np.zeros((ny, nx), dtype=np.float64)
    amp = 1.0
    total = 0.0
    octaves = 5
    for o in range(octaves):
        cells = 2 ** (o + 1)
        cy = max(2, min(ny, cells))
        cx = max(2, min(nx, cells))
        coarse = rng.standard_normal((cy, cx))
        # bilinear upsample to (ny,nx)
        yi = np.linspace(0, cy - 1, ny)
        xi = np.linspace(0, cx - 1, nx)
        y0 = np.floor(yi).astype(int)
        x0 = np.floor(xi).astype(int)
        y1 = np.minimum(y0 + 1, cy - 1)
        x1 = np.minimum(x0 + 1, cx - 1)
        fy = (yi - y0)[:, None]
        fx = (xi - x0)[None, :]
        top = coarse[y0][:, x0] * (1 - fx) + coarse[y0][:, x1] * fx
        bot = coarse[y1][:, x0] * (1 - fx) + coarse[y1][:, x1] * fx
        field += amp * (top * (1 - fy) + bot * fy)
        total += amp
        amp *= roughness
    field /= max(total, 1e-9)
    m = np.max(np.abs(field)) or 1.0
    return field / m


def _surface_elevation(
    spec: SceneSpec, y: np.ndarray, x: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Build the synthetic DEM ``surface(y, x)`` (Engineering elevation m, doc 05 §2.3)."""
    s: SurfaceSpec = spec.surface
    ny, nx = y.size, x.size
    base = np.full((ny, nx), s.base_elev, dtype=np.float64)
    if s.kind == "flat":
        return base
    if s.kind == "tilted-block":
        yy, xx = np.meshgrid(y, x, indexing="ij")
        return base + s.tilt_x * xx + s.tilt_y * yy
    # "fractal"
    noise = _fractal_value_noise((ny, nx), s.roughness, rng)
    return base + 0.5 * s.relief * noise


def _ranged(thickness, rng: np.random.Generator, shape2d, base_noise) -> np.ndarray:
    """Resolve a layer thickness (scalar | [min,max] | 'fill') to a (ny,nx) field.

    A ``[min,max]`` range yields a smoothly-varying noisy contact (doc 05 §2.3 "ranged →
    noisy contact"); the returned array is the thickness in metres (``inf`` for 'fill').
    """
    ny, nx = shape2d
    if isinstance(thickness, str):  # "fill"
        return np.full((ny, nx), np.inf)
    if isinstance(thickness, (tuple, list)):
        lo, hi = thickness
        w = 0.5 * (base_noise + 1.0)  # base_noise in [-1,1] → [0,1]
        return lo + (hi - lo) * w
    return np.full((ny, nx), float(thickness))


def _fault_offset(
    fault: FaultSpec, xx: np.ndarray, yy: np.ndarray
) -> np.ndarray:
    """Per-(y,x) vertical offset (m) applied to the hanging wall of ``fault``.

    The fault trace defines a line in plan view; the downdip (``dip_azimuth``) side is
    the hanging wall, dropped by ``throw`` for a normal fault (raised for reverse). We
    return a signed elevation shift to add to contact depths so blocks are offset
    (doc 05 §2.3 "cut L, offset blocks").
    """
    (x0, y0), (x1, y1) = fault.trace
    # strike direction
    sdx, sdy = (x1 - x0), (y1 - y0)
    norm = np.hypot(sdx, sdy) or 1.0
    # left-normal of the strike, pointing to one side
    nx_, ny_ = -sdy / norm, sdx / norm
    # downdip azimuth as a plan vector (CW from North = +Y)
    az = np.radians(fault.dip_azimuth)
    ddx, ddy = np.sin(az), np.cos(az)
    side_sign = np.sign(nx_ * ddx + ny_ * ddy) or 1.0
    # signed distance of each point from the trace line
    sd = ((xx - x0) * ny_ - (yy - y0) * nx_)
    hanging = (np.sign(sd) == side_sign)
    drop = -fault.throw if fault.kind == "normal" else fault.throw
    return np.where(hanging, drop, 0.0)


# --------------------------------------------------------------------------- lithology


def _build_lithology(
    spec: SceneSpec, z, y, x, rng: np.random.Generator
) -> tuple[np.ndarray, list[str], list[UnitProps], np.ndarray]:
    """Build ``L(x,y,z)``: surface → layers → fault offsets → intrusions (doc 05 §2.3)."""
    nz, ny, nx = z.size, y.size, x.size
    yy, xx = np.meshgrid(y, x, indexing="ij")  # (ny,nx)

    # ordered unit list: layer units (in order) + intrusion units, deduped.
    unit_names: list[str] = []

    def _unit_index(name: str) -> int:
        if name not in unit_names:
            unit_names.append(name)
        return unit_names.index(name)

    # accumulate aggregate fault offset (sum of all faults) on the contact elevations.
    total_fault_offset = np.zeros((ny, nx), dtype=np.float64)
    for fault in spec.faults:
        total_fault_offset += _fault_offset(fault, xx, yy)

    surface = _surface_elevation(spec, y, x, rng) + total_fault_offset

    # Contacts top→down. Each layer occupies [contact_bottom, contact_top).
    contact_top = surface.copy()
    # per-layer index field
    label_2d_stack: list[tuple[np.ndarray, int]] = []  # (bottom_elev, unit_idx) top→down
    for layer in spec.layers:
        idx = _unit_index(layer.unit)
        noise = _fractal_value_noise((ny, nx), 0.6, rng)
        thick = _ranged(layer.thickness, rng, (ny, nx), noise)
        contact_bottom = np.where(np.isinf(thick), -np.inf, contact_top - thick)
        label_2d_stack.append((contact_bottom, idx))
        contact_top = np.where(np.isinf(thick), contact_top, contact_bottom)

    # rasterise: for each voxel, the shallowest layer whose bottom is below the voxel.
    L = np.zeros((nz, ny, nx), dtype=np.int32)
    # default to the last (deepest/fill) unit
    default_idx = label_2d_stack[-1][1] if label_2d_stack else 0
    L[...] = default_idx
    z3 = z[:, None, None]
    # assign from deepest layer up so shallower layers overwrite at their depths
    surf3 = surface[None, :, :]
    above_surface = z3 > surf3
    for bottom, idx in reversed(label_2d_stack):
        in_layer = z3 >= bottom[None, :, :]
        L = np.where(in_layer, idx, L)
    # voxels above the synthetic surface are "air" — represented as the top layer's unit
    # but flagged later; we keep them as the topmost layer so properties stay finite.

    # intrusions: ellipsoids overwrite L (doc 05 §2.3)
    zz3, yy3, xx3 = np.meshgrid(z, y, x, indexing="ij")
    for intr in spec.intrusions:
        idx = _unit_index(intr.unit)
        cx, cy, cz = intr.center
        r2 = (
            ((xx3 - cx) / intr.radius_xy) ** 2
            + ((yy3 - cy) / intr.radius_xy) ** 2
            + ((zz3 - cz) / intr.radius_z) ** 2
        )
        L = np.where(r2 <= 1.0, idx, L)

    # resolve unit properties from the scene's per-unit library, falling back to the
    # shipped default-v1 Basin-&-Range library (doc 05 §3.2).
    units: list[UnitProps] = []
    for name in unit_names:
        props = spec.units.get(name) or DEFAULT_UNIT_LIBRARY.get(name)
        if props is None:
            raise KeyError(
                f"unit {name!r} has no properties in scene.units or the default library "
                f"(doc 05 §3.2)"
            )
        units.append(props)

    return L, unit_names, units, above_surface


# --------------------------------------------------------------------------- state


def _gaussian_halo(d2: np.ndarray) -> np.ndarray:
    """Gaussian falloff weight from a normalised squared distance ``d2`` (1 at core)."""
    return np.exp(-d2)


def _build_state(
    spec: SceneSpec,
    z,
    y,
    x,
    above_surface: np.ndarray,
    rng: np.random.Generator,
) -> StateField:
    """Build ``S(x,y,z)``: geotherm + plume/clay-cap blends + texture (doc 05 §2.1)."""
    nz, ny, nx = z.size, y.size, x.size
    zz3, yy3, xx3 = np.meshgrid(z, y, x, indexing="ij")

    surf_elev = spec.surface.base_elev

    # ── background conductive geotherm (doc 05 §2.3) ─────────────────────────────
    # T(°C) = surfaceTemp + gradient(°C/km) * depth(km); depth = surf_elev - z.
    depth_km = np.maximum(surf_elev - zz3, 0.0) / 1000.0
    temp_c = spec.geotherm.surface_temp + spec.geotherm.gradient * depth_km

    # ── background state defaults ────────────────────────────────────────────────
    porosity_boost = np.zeros((nz, ny, nx), dtype=np.float64)
    water_saturation = np.full((nz, ny, nx), 0.8, dtype=np.float64)  # mostly saturated
    salinity = np.full((nz, ny, nx), 500.0, dtype=np.float64)  # ppm background
    alteration = np.zeros((nz, ny, nx), dtype=np.float64)
    fracture = np.zeros((nz, ny, nx), dtype=np.float64)

    # ── hydrothermal-plume anomalies (state perturbations, decision #2) ──────────
    for an in spec.anomalies:
        cx, cy = an.footprint_center
        conduit = spec.conduit_fault(an.controlled_by)
        # vertical extent gate
        in_z = (zz3 <= an.top_elev) & (zz3 >= an.bottom_elev)
        # plan-view footprint, optionally biased toward the conduit fault trace
        if conduit is not None:
            (fx0, fy0), (fx1, fy1) = conduit.trace
            sdx, sdy = (fx1 - fx0), (fy1 - fy0)
            n = np.hypot(sdx, sdy) or 1.0
            # distance from the fault line in plan
            dist_fault = np.abs((xx3 - fx0) * (sdy / n) - (yy3 - fy0) * (sdx / n))
            d2_plan = (dist_fault / an.footprint_radius_xy) ** 2
        else:
            d2_plan = (
                ((xx3 - cx) ** 2 + (yy3 - cy) ** 2) / (an.footprint_radius_xy**2)
            )
        w = _gaussian_halo(d2_plan) * in_z.astype(np.float64)

        # temperature: blend toward the plume peak (gaussian halo, doc 05 §2.3)
        temp_c = temp_c + w * (an.temp_peak - temp_c)
        alteration = np.maximum(alteration, w * an.alteration_frac)
        porosity_boost = np.maximum(porosity_boost, w * an.porosity_boost)
        salinity = np.maximum(salinity, w * an.salinity_tds)
        fracture = np.maximum(fracture, w * an.fracture_density)
        water_saturation = np.maximum(water_saturation, w * 1.0)

        # shallow clay-cap conductor (high alteration smile, doc 05 §2.3, §4.2)
        if an.clay_cap_top_elev is not None and an.clay_cap_thickness > 0.0:
            cap_bottom = an.clay_cap_top_elev - an.clay_cap_thickness
            in_cap = (zz3 <= an.clay_cap_top_elev) & (zz3 >= cap_bottom)
            cap_w = _gaussian_halo(d2_plan) * in_cap.astype(np.float64)
            alteration = np.maximum(alteration, cap_w * max(an.alteration_frac, 0.6))
            salinity = np.maximum(salinity, cap_w * an.salinity_tds)

    # ── seeded correlated texture (doc 05 §2.3) ──────────────────────────────────
    tex = _fractal_value_noise((ny, nx), 0.7, rng)[None, :, :]
    porosity_boost = np.clip(porosity_boost + 0.01 * tex, 0.0, 0.3)
    alteration = np.clip(alteration, 0.0, 1.0)
    fracture = np.clip(fracture, 0.0, 1.0)
    water_saturation = np.clip(water_saturation, 0.0, 1.0)
    salinity = np.maximum(salinity, 1.0)

    # above-surface voxels (air): force cold, dry, no anomaly so properties stay sane.
    temp_c = np.where(above_surface, spec.geotherm.surface_temp, temp_c)
    water_saturation = np.where(above_surface, 0.0, water_saturation)
    alteration = np.where(above_surface, 0.0, alteration)
    fracture = np.where(above_surface, 0.0, fracture)
    porosity_boost = np.where(above_surface, 0.0, porosity_boost)

    # canonical: temperature → kelvin (doc 01 §5).
    temperature_k = convert(temp_c, "degC", "kelvin")

    return StateField(
        temperature=np.asarray(temperature_k, dtype=np.float64),
        porosity_boost=porosity_boost,
        water_saturation=water_saturation,
        salinity_tds=salinity,
        alteration_fraction=alteration,
        fracture_density=fracture,
    )


# --------------------------------------------------------------------------- entry


def compile_scene(spec: SceneSpec) -> TruthEarth:
    """Compile a :class:`SceneSpec` into a :class:`TruthEarth` (doc 05 §2 pipeline).

    Surface → layers → fault offsets → intrusions build ``L``; geotherm → plume/clay-cap
    blends build ``S``; the named rock-physics ruleset (doc 05 §3) then derives every
    co-located property volume. Deterministic from ``(spec, spec.seed)`` via
    ``numpy.random.SeedSequence`` sub-streams (doc 05 §1).
    """
    ss = np.random.SeedSequence(spec.seed)
    rng_litho, rng_state = (np.random.default_rng(s) for s in ss.spawn(2))

    z, y, x = _grid_coords(spec)
    L, unit_names, units, above_surface = _build_lithology(spec, z, y, x, rng_litho)
    state = _build_state(spec, z, y, x, above_surface, rng_state)

    ruleset = get_ruleset(spec.rock_physics)
    props = ruleset.apply(
        unit_index=L,
        units=units,
        temperature_k=state.temperature,
        porosity_state=state.porosity_boost,
        water_saturation=state.water_saturation,
        salinity_tds=state.salinity_tds,
        alteration_frac=state.alteration_fraction,
        fracture_density=state.fracture_density,
    )

    origin = (float(z[0]), float(y[0]), float(x[0]))
    spacing = (float(spec.frame.dz), float(spec.frame.dy), float(spec.frame.dx))

    return TruthEarth(
        spec=spec,
        lithology=L,
        unit_names=unit_names,
        units=units,
        state=state,
        properties=props,
        origin=origin,
        spacing=spacing,
        above_surface=above_surface,
    )
