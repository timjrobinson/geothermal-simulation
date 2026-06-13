"""ModelDomain — the inversion mesh over the Engineering Frame (doc 10 §4).

The :class:`ModelDomain` is the engine-agnostic description of *where* an inversion
solves: a mesh (``TensorMesh`` here; ``TreeMesh``/``SimplexMesh`` are declared in
:data:`MESH_TYPES` for later engines) expressed entirely in the **Engineering Frame**
(doc 01, Z-up, ``(z, y, x)`` axis order to match :mod:`geosim.storage` / doc 02 §10.2).

A domain is built from a **core region** (the resolved volume of interest) plus
**geometric padding** (cells that grow away from the core so boundary conditions sit far
from the target, doc 10 §4.2) plus **active cells** masked by the surface model
(topography — cells *above* ground are inactive air, doc 10 §4.3). The recovered model
on the *core* cells is what later resamples onto the fused grid (doc 10 §4.4); padding
and air cells never leave the engine.

Only :mod:`discretize` is imported here — it is the framework's meshing dependency, NOT a
solver. SimPEG/PyGIMLi survey containers are built *inside* an engine, never here (doc 10
§8).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from discretize import TensorMesh

__all__ = [
    "MESH_TYPES",
    "CoreRegion",
    "PaddingSpec",
    "ModelDomain",
    "build_tensor_domain",
]

# Mesh kinds an InversionEngine may declare it supports (doc 10 §4). Only ``tensor`` is
# built by the framework here; the others are reserved for engine-specific builders.
MESH_TYPES = ("tensor", "tree", "simplex")


@dataclass(frozen=True)
class CoreRegion:
    """The resolved core volume of an inversion, Engineering metres (doc 10 §4.1).

    ``origin`` is the ``(z, y, x)`` minimum corner and ``extent`` the ``(z, y, x)`` span;
    ``cell_size`` is the isotropic (or per-axis) core cell edge. The core is the region a
    recovered model is trusted over and the only part that resamples onto the fused grid
    (doc 10 §4.4).
    """

    origin: tuple[float, float, float]  # (z0, y0, x0) min corner, Engineering m
    extent: tuple[float, float, float]  # (dz, dy, dx) span, Engineering m
    cell_size: tuple[float, float, float]  # (cz, cy, cx) core cell edge, Engineering m

    def n_core(self) -> tuple[int, int, int]:
        """Number of CORE cells per axis ``(nz, ny, nx)`` (>= 1 on each axis)."""
        return tuple(  # type: ignore[return-value]
            max(1, int(round(e / c))) for e, c in zip(self.extent, self.cell_size, strict=True)
        )


@dataclass(frozen=True)
class PaddingSpec:
    """Geometric padding outside the core (doc 10 §4.2).

    ``n_pad`` padding cells are appended on each side of every axis, each ``factor``×
    larger than the previous (the standard expanding pad so the mesh boundary sits far
    from the target). ``factor`` 1.0 ⇒ uniform padding.
    """

    n_pad: int = 0
    factor: float = 1.3

    def __post_init__(self) -> None:
        if self.n_pad < 0:
            raise ValueError(f"n_pad must be >= 0; got {self.n_pad}")
        if self.factor < 1.0:
            raise ValueError(f"padding factor must be >= 1.0; got {self.factor}")


@dataclass
class ModelDomain:
    """An inversion mesh + the core/active bookkeeping (doc 10 §4).

    Holds the :mod:`discretize` ``mesh`` (full mesh incl. padding), a boolean
    ``active_cells`` mask over **all** mesh cells (False = air above topography, doc 10
    §4.3), and the ``core`` description. :meth:`core_slices` / :meth:`extract_core`
    recover the regular ``(nz, ny, nx)`` core sub-brick that resamples onto the fused grid
    (doc 10 §4.4).
    """

    mesh: TensorMesh
    active_cells: np.ndarray  # bool, shape (mesh.n_cells,)
    core: CoreRegion
    padding: PaddingSpec = field(default_factory=PaddingSpec)
    mesh_type: str = "tensor"

    @property
    def n_cells(self) -> int:
        return int(self.mesh.n_cells)

    @property
    def n_active(self) -> int:
        return int(np.count_nonzero(self.active_cells))

    def core_slices(self) -> tuple[slice, slice, slice]:
        """``(z, y, x)`` slices selecting the core block within the padded mesh grid.

        The discretize cell ordering is x-fastest; we reshape to ``(nz, ny, nx)`` and
        slice off the ``n_pad`` padding cells on each side.
        """
        nx, ny, nz = self.mesh.shape_cells  # discretize order is (x, y, z)
        npad = self.padding.n_pad
        return (
            slice(npad, nz - npad),
            slice(npad, ny - npad),
            slice(npad, nx - npad),
        )

    def extract_core(self, cell_values: np.ndarray) -> np.ndarray:
        """Reshape a per-cell vector to ``(nz, ny, nx)`` and return the CORE sub-brick.

        ``cell_values`` is length ``mesh.n_cells`` in discretize (x-fastest) order; the
        result is Z-up ``(z, y, x)`` to match :mod:`geosim.storage` (doc 02 §10.2).
        """
        nx, ny, nz = self.mesh.shape_cells
        # discretize ravels x-fastest → reshape (nz, ny, nx) with C order after transpose.
        cube = np.asarray(cell_values, dtype=float).reshape((nz, ny, nx), order="C")
        sz, sy, sx = self.core_slices()
        return cube[sz, sy, sx]

    def core_grid(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Core ``(origin, spacing)`` in ``(z, y, x)`` Engineering metres (cell-centred).

        This is the geometry a recovered-core PropertyModel is written with (doc 02
        §10.2) and that the fused-grid resampler reads (doc 10 §4.4).
        """
        cz, cy, cx = self.core.cell_size
        oz, oy, ox = self.core.origin
        # Cell-centre origin = min corner + half a cell.
        return ((oz + cz / 2.0, oy + cy / 2.0, ox + cx / 2.0), (cz, cy, cx))


def _padded_h(n_core: int, cell: float, pad: PaddingSpec) -> list:
    """A discretize ``h`` spec: ``n_pad`` expanding cells, ``n_core`` uniform, then pad.

    Returns the list-of-tuples discretize accepts: ``(cell, n_core)`` for the uniform core
    and ``(cell, n_pad, ±factor)`` for the expanding pads (negative = grow toward −, doc
    10 §4.2).
    """
    spec: list = []
    if pad.n_pad > 0:
        spec.append((cell, pad.n_pad, -pad.factor))
    spec.append((cell, n_core))
    if pad.n_pad > 0:
        spec.append((cell, pad.n_pad, pad.factor))
    return spec


def build_tensor_domain(
    core: CoreRegion,
    *,
    padding: PaddingSpec | None = None,
    surface_z: float | None = None,
) -> ModelDomain:
    """Build a :class:`ModelDomain` on a :mod:`discretize` ``TensorMesh`` (doc 10 §4).

    The mesh is a uniform core (``core.cell_size``) surrounded by ``padding`` expanding
    cells on each axis (doc 10 §4.2). ``active_cells`` masks out air cells whose centre is
    **above** ``surface_z`` (flat topography; per-cell DEMs are an engine concern) — doc
    10 §4.3. With ``surface_z=None`` every cell is active.

    The mesh ``origin`` is placed so the uniform-core block starts exactly at
    ``core.origin`` (Engineering metres), keeping the core sub-brick aligned to the
    PropertyModel grid that later resamples onto the fused grid (doc 10 §4.4).
    """
    padding = padding or PaddingSpec()
    nz, ny, nx = core.n_core()
    cz, cy, cx = core.cell_size

    hx = _padded_h(nx, cx, padding)
    hy = _padded_h(ny, cy, padding)
    hz = _padded_h(nz, cz, padding)

    # discretize axis order is (x, y, z).
    mesh = TensorMesh([hx, hy, hz])

    # Shift the mesh so the uniform CORE block min-corner lands at core.origin. The pad
    # cells occupy the total padded length before the core start on each axis.
    oz, oy, ox = core.origin

    def pad_len(cell: float, pad: PaddingSpec) -> float:
        if pad.n_pad == 0:
            return 0.0
        # Sum of the expanding pad widths: cell*factor + cell*factor^2 + ... (doc 10 §4.2).
        return float(sum(cell * pad.factor**k for k in range(1, pad.n_pad + 1)))

    mesh.origin = (
        ox - pad_len(cx, padding),
        oy - pad_len(cy, padding),
        oz - pad_len(cz, padding),
    )

    # Active cells: air above the (flat) surface is inactive (doc 10 §4.3). Engineering
    # Z-up — a cell is air if its centre elevation exceeds the surface.
    if surface_z is None:
        active = np.ones(mesh.n_cells, dtype=bool)
    else:
        cc_z = mesh.cell_centers[:, 2]  # (x, y, z) → z is column 2
        active = cc_z <= float(surface_z)

    return ModelDomain(
        mesh=mesh, active_cells=active, core=core, padding=padding, mesh_type="tensor"
    )
