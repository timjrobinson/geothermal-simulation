"""Project directory layout helper (doc 04 §3).

The **project directory holds the bulk stores** only — the catalog lives in
PostgreSQL (doc 04 §2.1), not in a file under this directory. Per doc 04 §3 the
on-disk tree under ``<storage_root>/<project_id>/`` is::

    arrays/   grids/   meshes/   vectors/   points/   raw/   cache/

with ``cache/`` itself holding ``slices/ isosurfaces/ tiles/`` (doc 04 §8). This
module owns *only* the directory names + creation; the Zarr-group-internal layout
is owned by doc 02 §10.2 (see :mod:`geosim.storage.property_model`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "BULK_STORES",
    "CACHE_SUBDIRS",
    "ProjectLayout",
    "ensure_project_layout",
]

# doc 04 §3 — the bulk stores under <storage_root>/<project_id>/
BULK_STORES = ("arrays", "grids", "meshes", "vectors", "points", "raw", "cache")
# doc 04 §8 — derivable cache subdirectories (safe to delete)
CACHE_SUBDIRS = ("slices", "isosurfaces", "tiles")


@dataclass(frozen=True)
class ProjectLayout:
    """Resolved paths for one project's bulk stores (doc 04 §3).

    The catalog is NOT here (it lives in PostgreSQL, doc 04 §2.1); this is purely
    the bulk-store directory tree.
    """

    storage_root: Path
    project_id: str

    @property
    def root(self) -> Path:
        return self.storage_root / self.project_id

    @property
    def frame_json(self) -> Path:
        # cached SpatialFrame (doc 01); the DB is canonical (doc 04 §3).
        return self.root / "frame.json"

    @property
    def arrays(self) -> Path:
        return self.root / "arrays"

    @property
    def grids(self) -> Path:
        return self.root / "grids"

    @property
    def meshes(self) -> Path:
        return self.root / "meshes"

    @property
    def vectors(self) -> Path:
        return self.root / "vectors"

    @property
    def points(self) -> Path:
        return self.root / "points"

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    def store(self, name: str) -> Path:
        """Path to a named bulk store (one of :data:`BULK_STORES`)."""
        if name not in BULK_STORES:
            raise ValueError(f"unknown bulk store {name!r}; expected one of {BULK_STORES}")
        return self.root / name

    def zarr_path(self, dataset_id: str) -> Path:
        """Path to a dataset's Zarr group, ``arrays/<dataset_id>.zarr`` (doc 04 §3)."""
        return self.arrays / f"{dataset_id}.zarr"


def ensure_project_layout(storage_root: str | Path, project_id: str) -> ProjectLayout:
    """Create ``<storage_root>/<project_id>/`` and all bulk-store dirs (doc 04 §3).

    Idempotent: existing directories are left untouched. The catalog is created
    elsewhere (PostgreSQL, doc 04 §2.1) — only the bulk-store tree is materialised
    here.
    """
    layout = ProjectLayout(Path(storage_root), project_id)
    layout.root.mkdir(parents=True, exist_ok=True)
    for name in BULK_STORES:
        (layout.root / name).mkdir(exist_ok=True)
    for sub in CACHE_SUBDIRS:
        (layout.cache / sub).mkdir(exist_ok=True)
    return layout
