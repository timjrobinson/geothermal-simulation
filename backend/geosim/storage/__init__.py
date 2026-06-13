"""Storage layer (doc 02 §10 + doc 04 §3–§5).

Three responsibilities, all per the design contract:

- **Project directory layout** (doc 04 §3): ``arrays/ grids/ meshes/ vectors/
  points/ raw/ cache/`` under ``<storage_root>/<project_id>/`` — :mod:`.layout`.
- **Zarr v3 PropertyModel writer/reader** (doc 02 §10.2 — THE authoritative layout):
  one group per dataset, per-property multiscale subgroups, sibling ``_sigma``, 64³
  cubic Blosc(zstd,shuffle) chunks, NaN fill, ``[z,y,x]`` Z-up axis order, REQUIRED
  ``origin``/``spacing`` + OME-Zarr ``multiscales`` — :mod:`.property_model`.
- **Multiresolution pyramids** (doc 02 §10.3, doc 04 §5): mean downsample for values,
  variance-correct for ``_sigma`` — :mod:`.pyramid`.
- **Content-addressed raw store** (doc 04 §3, §8.1): ``raw/<sha256>/<name>`` with
  byte-level de-dup — :mod:`.raw_store`.
"""

from .layout import BULK_STORES, CACHE_SUBDIRS, ProjectLayout, ensure_project_layout
from .property_model import (
    DEFAULT_CHUNK,
    SIGMA_SUFFIX,
    GridSpec,
    PropertyModelReader,
    open_property_model,
    write_property_model,
)
from .pyramid import (
    build_sigma_pyramid,
    build_value_pyramid,
    downsample_mean,
    downsample_sigma,
    pyramid_level_count,
)
from .raw_store import RawRef, RawStore, sha256_bytes

__all__ = [
    # layout
    "BULK_STORES", "CACHE_SUBDIRS", "ProjectLayout", "ensure_project_layout",
    # property model (zarr v3)
    "DEFAULT_CHUNK", "SIGMA_SUFFIX", "GridSpec", "PropertyModelReader",
    "write_property_model", "open_property_model",
    # pyramid
    "build_value_pyramid", "build_sigma_pyramid", "downsample_mean",
    "downsample_sigma", "pyramid_level_count",
    # raw store
    "RawRef", "RawStore", "sha256_bytes",
]
