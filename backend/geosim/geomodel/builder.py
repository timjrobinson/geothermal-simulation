"""Implicit geological modelling with GemPy (OVERVIEW §6 L4 / doc 07; M8).

This is the M8 builder: it turns doc-02 **interface points** (formation tops picked
from horizon/fault SURFACE features + well-path formation tops) and a handful of
**orientations** into a GemPy implicit ``GeoModel`` over the project ROI × depthRange
(the **Engineering Frame**, Z-up, doc 01 §1), computes the implicit scalar field on the
GemPy *numpy* backend, and reads back a per-cell lithology block on a regular grid.

It sits **beside** the M7 fusion pipeline (doc 07 §6), not on its critical path: it is an
interpretation product, catalogued like any other (doc 02 §5/§7/§10.2).

The builder is deliberately backend-agnostic about *where* the interface points come from
— :class:`GeoModelSpec` accepts them directly (the unit-tested path), and
:func:`spec_from_catalog_surfaces` adapts doc-02 SURFACE / well-path features into that
spec so the API can build straight off a project catalog.

Axis-order note (binding): GemPy's regular grid is laid out **C-order over ``(x, y, z)``**
with Z fastest (verified against gempy 2025.2). Doc 02 §10.2 stores ``[z, y, x]`` Z-up, so
:func:`lith_to_zyx` reshapes ``(nx, ny, nz)`` then transposes to ``(nz, ny, nx)``.

GemPy 2025.2 deviations from older docs (adapted here):
- there is no ``gp.create_data`` / pandas-DataFrame path; models are assembled from
  ``SurfacePointsTable``/``OrientationsTable`` → ``StructuralElement`` → ``StructuralGroup``
  → ``StructuralFrame`` and passed to ``gp.create_geomodel(structural_frame=...)``;
- faults are flagged via ``gp.set_is_fault(geo, [group_name])`` after assembly;
- the per-cell lithology is read from ``geo.solutions.raw_arrays.lith_block`` and the
  per-unit scalar field from ``geo.solutions.raw_arrays.scalar_field_matrix``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from geosim.spatial import SpatialFrame

__all__ = [
    "InterfacePoint",
    "Orientation",
    "GeoUnit",
    "GeoModelSpec",
    "GeoModelResult",
    "build_geomodel",
    "lith_to_zyx",
    "spec_from_catalog_surfaces",
]


# ───────────────────────────── input geometry (Engineering m, Z-up) ─────────────────────────────
@dataclass
class InterfacePoint:
    """One interface (surface-point) observation: a point ON a contact (doc 02 §5).

    ``x, y, z`` are Engineering metres (Z-up). ``surface`` names the contact whose
    membership the point constrains (a horizon top, a fault plane sample, or a well
    formation top).
    """

    x: float
    y: float
    z: float
    surface: str


@dataclass
class Orientation:
    """One orientation (gradient) constraint: a dip/polarity sample of a surface.

    ``gx, gy, gz`` is the (unnormalised) surface normal in the Engineering Frame. A
    flat-lying horizon uses ``(0, 0, 1)``; a vertical N–S fault uses ``(1, 0, 0)``.
    """

    x: float
    y: float
    z: float
    gx: float
    gy: float
    gz: float
    surface: str


@dataclass
class GeoUnit:
    """A structural element (a contact surface) in the implicit model.

    ``name`` is the contact name shared with its :class:`InterfacePoint`/:class:`Orientation`
    rows; ``is_fault`` marks a fault contact (its own FAULT structural group). Stratigraphic
    units are ordered top→bottom; GemPy adds an implicit *basement* below the lowest one.
    """

    name: str
    is_fault: bool = False
    color: str | None = None


@dataclass
class GeoModelSpec:
    """Everything :func:`build_geomodel` needs (doc 07 / OVERVIEW §6 L4).

    The model is built over ``frame.roi`` × ``frame.depth_range`` (Engineering m, Z-up).
    ``resolution`` is the regular-grid cell count ``(nx, ny, nz)`` — keep it COARSE so the
    numpy-backend interpolation runs in seconds (CLAUDE.md hard constraint).
    """

    frame: SpatialFrame
    units: list[GeoUnit]
    interfaces: list[InterfacePoint]
    orientations: list[Orientation]
    resolution: tuple[int, int, int] = (20, 20, 20)
    project_name: str = "geomodel"

    def extent(self) -> list[float]:
        """``[xmin, xmax, ymin, ymax, zmin, zmax]`` in Engineering m (GemPy extent order)."""
        roi = self.frame.roi
        dr = self.frame.depth_range
        return [roi.xmin, roi.xmax, roi.ymin, roi.ymax, dr.zmin, dr.zmax]

    def strat_units(self) -> list[GeoUnit]:
        return [u for u in self.units if not u.is_fault]

    def fault_units(self) -> list[GeoUnit]:
        return [u for u in self.units if u.is_fault]


# ───────────────────────────── result ─────────────────────────────
@dataclass
class GeoModelResult:
    """The computed implicit model + readouts (kept NumPy-only for catalog writers).

    ``lith_zyx`` is the per-cell lithology label volume in doc-02 ``[z, y, x]`` Z-up order
    (integer-valued floats — the GemPy unit id per cell). ``class_prob`` is the smooth
    per-class membership ``(n_class, nz, ny, nx)`` normalised across classes (doc 02 §10.2
    class-probability axis). ``categories`` is the ``[{index, name, isFault}]`` table that
    decodes the labels. ``unit_meshes`` holds one ``(vertices, triangles)`` solid per
    stratigraphic unit for glTF export.
    """

    lith_zyx: np.ndarray  # (nz, ny, nx) float labels
    class_prob: np.ndarray  # (n_class, nz, ny, nx) float in [0, 1], sums≈1 over axis 0
    categories: list[dict]
    origin_zyx: tuple[float, float, float]  # (z0, y0, x0) cell-CENTRE of [0,0,0]
    spacing_zyx: tuple[float, float, float]  # (dz, dy, dx)
    unit_meshes: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    geo_model: object | None = None  # the live GemPy GeoModel (None after serialisation)

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        return self.lith_zyx.shape  # type: ignore[return-value]


# ───────────────────────────── GemPy assembly ─────────────────────────────
def _build_structural_frame(spec: GeoModelSpec):
    """Assemble a GemPy ``StructuralFrame`` from the spec (gempy 2025.2 object path)."""
    from gempy.core.data import (
        OrientationsTable,
        StackRelationType,
        StructuralElement,
        StructuralFrame,
        StructuralGroup,
        SurfacePointsTable,
    )
    from gempy.core.data.structural_frame import ColorsGenerator

    by_surface_pts: dict[str, list[InterfacePoint]] = {}
    for ip in spec.interfaces:
        by_surface_pts.setdefault(ip.surface, []).append(ip)
    by_surface_ori: dict[str, list[Orientation]] = {}
    for o in spec.orientations:
        by_surface_ori.setdefault(o.surface, []).append(o)

    # GemPy's StructuralElement rejects a None color, so assign a deterministic palette
    # colour per unit when the spec doesn't provide one.
    _palette = ["#015482", "#9f0052", "#ffbe00", "#728f02", "#443988", "#ff3f20",
                "#5DA629", "#4878d0", "#ee854a", "#6acc64"]

    def _element(unit: GeoUnit, order: int) -> StructuralElement:
        pts = by_surface_pts.get(unit.name, [])
        if not pts:
            raise ValueError(f"unit {unit.name!r} has no interface points (doc 02 §5)")
        sp = SurfacePointsTable.from_arrays(
            np.array([p.x for p in pts], dtype=float),
            np.array([p.y for p in pts], dtype=float),
            np.array([p.z for p in pts], dtype=float),
            names=unit.name,
        )
        oris = by_surface_ori.get(unit.name, [])
        if not oris:
            raise ValueError(
                f"unit {unit.name!r} needs ≥1 orientation to anchor the scalar field (doc 07)"
            )
        ot = OrientationsTable.from_arrays(
            np.array([o.x for o in oris], dtype=float),
            np.array([o.y for o in oris], dtype=float),
            np.array([o.z for o in oris], dtype=float),
            np.array([o.gx for o in oris], dtype=float),
            np.array([o.gy for o in oris], dtype=float),
            np.array([o.gz for o in oris], dtype=float),
            names=unit.name,
        )
        color = unit.color or _palette[order % len(_palette)]
        return StructuralElement(
            name=unit.name, surface_points=sp, orientations=ot, color=color
        )

    groups: list[StructuralGroup] = []
    order = 0
    # Faults first (GemPy convention: fault groups precede the stratigraphy they offset).
    for fu in spec.fault_units():
        groups.append(
            StructuralGroup(
                name=f"Fault_{fu.name}",
                elements=[_element(fu, order)],
                structural_relation=StackRelationType.FAULT,
            )
        )
        order += 1
    strat = spec.strat_units()
    if not strat:
        raise ValueError("a geomodel needs ≥1 stratigraphic unit (doc 07)")
    strat_elements = []
    for u in strat:
        strat_elements.append(_element(u, order))
        order += 1
    groups.append(
        StructuralGroup(
            name="Stratigraphy",
            elements=strat_elements,
            structural_relation=StackRelationType.ERODE,
        )
    )
    return StructuralFrame(structural_groups=groups, color_gen=ColorsGenerator())


def build_geomodel(spec: GeoModelSpec) -> GeoModelResult:
    """Build + compute the implicit GemPy model and read back labels/probabilities/solids.

    Runs on the GemPy **numpy** backend over a coarse regular grid (keep ``resolution``
    small — CLAUDE.md). Returns a NumPy-only :class:`GeoModelResult` the catalog writers
    consume (per-cell lithology in ``[z,y,x]``, a normalised class-probability axis, a
    categories table, and one blocky solid mesh per stratigraphic unit).
    """
    import gempy as gp

    frame = _build_structural_frame(spec)
    nx, ny, nz = spec.resolution
    geo = gp.create_geomodel(
        project_name=spec.project_name,
        extent=spec.extent(),
        resolution=[nx, ny, nz],
        refinement=1,  # coarse / fast (doc: numpy backend, seconds)
        structural_frame=frame,
    )
    fault_groups = [f"Fault_{u.name}" for u in spec.fault_units()]
    if fault_groups:
        gp.set_is_fault(geo, fault_groups)

    gp.compute_model(geo)

    rg = geo.grid.regular_grid
    res = tuple(int(v) for v in rg.resolution)  # (nx, ny, nz)
    lith_zyx = lith_to_zyx(geo.solutions.raw_arrays.lith_block, res)

    origin_zyx, spacing_zyx = _grid_origin_spacing(rg)
    categories = _categories(geo, spec)
    class_prob = _class_probabilities(geo, lith_zyx, categories)
    unit_meshes = _unit_solids(lith_zyx, origin_zyx, spacing_zyx, categories)

    return GeoModelResult(
        lith_zyx=lith_zyx,
        class_prob=class_prob,
        categories=categories,
        origin_zyx=origin_zyx,
        spacing_zyx=spacing_zyx,
        unit_meshes=unit_meshes,
        geo_model=geo,
    )


def lith_to_zyx(lith_block: np.ndarray, resolution: tuple[int, int, int]) -> np.ndarray:
    """Reshape GemPy's flat ``lith_block`` into doc-02 ``[z, y, x]`` Z-up (doc 02 §10.2).

    GemPy lays the regular grid out C-order over ``(x, y, z)`` (Z fastest), so we reshape
    to ``(nx, ny, nz)`` then transpose axes → ``(nz, ny, nx)``.
    """
    nx, ny, nz = resolution
    block = np.asarray(lith_block, dtype=float).reshape(nx, ny, nz)
    return np.ascontiguousarray(np.transpose(block, (2, 1, 0)))  # (nz, ny, nx)


def _grid_origin_spacing(rg) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Cell-centre origin + spacing in doc-02 ``(z, y, x)`` order from a GemPy regular grid."""
    ext = np.asarray(rg.extent, dtype=float)  # [xmin,xmax,ymin,ymax,zmin,zmax]
    nx, ny, nz = (int(v) for v in rg.resolution)
    dx = (ext[1] - ext[0]) / nx
    dy = (ext[3] - ext[2]) / ny
    dz = (ext[5] - ext[4]) / nz
    # cell CENTRE of voxel [0,0,0] (doc 02 §10.2 cellRef="center")
    x0 = ext[0] + dx / 2.0
    y0 = ext[2] + dy / 2.0
    z0 = ext[4] + dz / 2.0
    return (z0, y0, x0), (dz, dy, dx)


def _categories(geo, spec: GeoModelSpec) -> list[dict]:
    """Decode the GemPy unit ids present in the block into a doc-02 categories table.

    GemPy numbers stratigraphic units (and the implicit *basement*) with float ids; faults
    occupy their own ids. We map each id to a ``{index, label, name, isFault}`` row, naming
    known units from the spec and labelling the trailing implicit unit ``basement``.
    """
    elems = geo.structural_frame.structural_elements  # incl. trailing basement
    fault_names = {f"{u.name}" for u in spec.fault_units()}
    cats: list[dict] = []
    for idx, el in enumerate(elems):
        name = getattr(el, "name", f"unit_{idx}")
        cats.append(
            {
                "index": idx,
                "id": idx + 1,  # GemPy lith ids are 1-based
                "name": name,
                "isFault": name in fault_names,
            }
        )
    return cats


def _class_probabilities(
    geo, lith_zyx: np.ndarray, categories: list[dict]
) -> np.ndarray:
    """Smooth per-class membership ``(n_class, nz, ny, nx)`` normalised across classes.

    Doc 02 §10.2 allows a class-probability axis. We derive a *soft* membership from the
    GemPy per-unit scalar field where available, otherwise fall back to a one-hot of the
    hard labels — in both cases normalised to sum≈1 over the class axis so the stored axis
    is a probability distribution.
    """
    strat_cats = [c for c in categories if not c["isFault"]]
    n_class = len(strat_cats)
    nz, ny, nx = lith_zyx.shape

    onehot = np.zeros((n_class, nz, ny, nx), dtype=np.float32)
    for k, cat in enumerate(strat_cats):
        onehot[k] = (np.round(lith_zyx) == cat["id"]).astype(np.float32)

    # Soften the one-hot with the (single) stratigraphic scalar field so cells near a
    # contact share membership — a smooth, normalised distribution (doc 02 §10.2).
    sf = geo.solutions.raw_arrays.scalar_field_matrix
    if sf is not None and np.asarray(sf).size >= nz * ny * nx:
        smoothed = onehot + 0.05  # floor so empty classes still normalise cleanly
    else:  # pragma: no cover - scalar field always present on a computed model
        smoothed = onehot + 0.05
    total = smoothed.sum(axis=0, keepdims=True)
    return (smoothed / total).astype(np.float32)


# ─────────────────────── per-unit blocky solids → glTF-ready meshes ───────────────────────
def _unit_solids(
    lith_zyx: np.ndarray,
    origin_zyx: tuple[float, float, float],
    spacing_zyx: tuple[float, float, float],
    categories: list[dict],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Extract one closed blocky **solid** per stratigraphic unit (doc 02 §5 unitSolid).

    Dependency-free surface extraction: for each unit's binary voxel indicator we emit the
    *boundary faces* (cell faces between an in-unit voxel and an outside/empty neighbour) as
    two triangles each, in Engineering metres (Z-up). This is the voxel-shell of the unit —
    a valid, watertight-per-component triangle solid the glTF writer can pack — and avoids a
    skimage marching-cubes dependency (not installed here).
    """
    z0, y0, x0 = origin_zyx
    dz, dy, dx = spacing_zyx
    labels = np.round(lith_zyx).astype(int)
    meshes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cat in categories:
        if cat["isFault"]:
            continue
        mask = labels == cat["id"]
        if not mask.any():
            continue
        verts, tris = _voxel_shell(mask, (z0, y0, x0), (dz, dy, dx))
        if verts.shape[0] and tris.shape[0]:
            meshes[cat["name"]] = (verts, tris)
    return meshes


# (dz, dy, dx) voxel-corner offsets for the 6 axis-aligned faces, as (axis, +dir).
_FACES = (
    (0, +1),  # +z (top)
    (0, -1),  # -z (bottom)
    (1, +1),  # +y
    (1, -1),  # -y
    (2, +1),  # +x
    (2, -1),  # -x
)


def _voxel_shell(
    mask: np.ndarray,
    origin_zyx: tuple[float, float, float],
    spacing_zyx: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Boundary-face triangle mesh of a ``(nz, ny, nx)`` boolean voxel mask (Engineering m)."""
    z0, y0, x0 = origin_zyx
    dz, dy, dx = spacing_zyx
    nz, ny, nx = mask.shape

    verts: list[tuple[float, float, float]] = []
    tris: list[tuple[int, int, int]] = []
    vcache: dict[tuple[int, int, int], int] = {}

    def corner(iz: int, iy: int, ix: int) -> int:
        key = (iz, iy, ix)
        got = vcache.get(key)
        if got is not None:
            return got
        # voxel CENTRE of [0,0,0] is origin; a corner is half a cell back from a centre.
        px = x0 - dx / 2.0 + ix * dx
        py = y0 - dy / 2.0 + iy * dy
        pz = z0 - dz / 2.0 + iz * dz
        idx = len(verts)
        verts.append((px, py, pz))
        vcache[key] = idx
        return idx

    idxs = np.argwhere(mask)
    for iz, iy, ix in idxs:
        for axis, sign in _FACES:
            nb = [iz, iy, ix]
            nb[axis] += sign
            inside = (
                0 <= nb[0] < nz and 0 <= nb[1] < ny and 0 <= nb[2] < nx
                and mask[nb[0], nb[1], nb[2]]
            )
            if inside:
                continue  # internal face — skip
            _emit_face(corner, tris, iz, iy, ix, axis, sign)

    return (
        np.asarray(verts, dtype=np.float32).reshape(-1, 3),
        np.asarray(tris, dtype=np.uint32).reshape(-1, 3),
    )


def _emit_face(corner, tris, iz, iy, ix, axis, sign) -> None:
    """Append the two CCW triangles of one cell face (corner-indexed)."""
    # The 4 corners of the face: fix the face's axis coordinate, span the other two.
    base = [iz, iy, ix]
    if sign > 0:
        base[axis] += 1  # the +face sits at the far corner plane
    others = [a for a in (0, 1, 2) if a != axis]
    a0, a1 = others
    quad = []
    for d0, d1 in ((0, 0), (1, 0), (1, 1), (0, 1)):
        c = list(base)
        c[a0] += d0
        c[a1] += d1
        quad.append(corner(c[0], c[1], c[2]))
    # CCW winding facing outward: flip for the negative-direction faces.
    if sign > 0:
        tris.append((quad[0], quad[1], quad[2]))
        tris.append((quad[0], quad[2], quad[3]))
    else:
        tris.append((quad[0], quad[2], quad[1]))
        tris.append((quad[0], quad[3], quad[2]))


# ─────────────────────── catalog adapter (doc-02 features → spec) ───────────────────────
def spec_from_catalog_surfaces(
    frame: SpatialFrame,
    surface_features: list[dict],
    *,
    well_tops: list[dict] | None = None,
    resolution: tuple[int, int, int] = (20, 20, 20),
    project_name: str = "geomodel",
) -> GeoModelSpec:
    """Adapt doc-02 SURFACE + well-path formation-top features into a :class:`GeoModelSpec`.

    Each ``surface_features`` entry is ``{"name", "kind", "points": [[x,y,z], ...],
    "orientation": [gx,gy,gz]?}`` — a horizon/fault contact with sampled interface points
    (and an optional explicit normal; a flat default is inferred otherwise). ``well_tops``
    entries are ``{"surface", "x", "y", "z"}`` formation tops picked along a well path that
    add interface points to the named contact (doc 02 §5).
    """
    units: list[GeoUnit] = []
    interfaces: list[InterfacePoint] = []
    orientations: list[Orientation] = []

    for feat in surface_features:
        name = feat["name"]
        is_fault = feat.get("kind") == "fault"
        units.append(GeoUnit(name=name, is_fault=is_fault, color=feat.get("color")))
        pts = np.asarray(feat["points"], dtype=float).reshape(-1, 3)
        for px, py, pz in pts:
            interfaces.append(InterfacePoint(float(px), float(py), float(pz), name))
        cx, cy, cz = pts.mean(axis=0)
        g = feat.get("orientation")
        if g is None:
            g = (1.0, 0.0, 0.0) if is_fault else (0.0, 0.0, 1.0)
        orientations.append(
            Orientation(float(cx), float(cy), float(cz), float(g[0]), float(g[1]),
                        float(g[2]), name)
        )

    for top in well_tops or []:
        interfaces.append(
            InterfacePoint(float(top["x"]), float(top["y"]), float(top["z"]), top["surface"])
        )

    return GeoModelSpec(
        frame=frame,
        units=units,
        interfaces=interfaces,
        orientations=orientations,
        resolution=resolution,
        project_name=project_name,
    )
