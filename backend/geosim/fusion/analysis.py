"""Cross-plot / statistics / clustering over a fused grid (doc 07 §3).

Once N native :class:`~geosim.catalog.PropertyModel`\\s share a
:class:`~geosim.fusion.FusedGrid` as resampled :class:`~geosim.catalog.FusedLayer`\\s,
every fused cell is a **feature vector** ``[resistivity, density, Vp, …]`` (with nodata
where a method is absent). This module unlocks the multivariate analysis of doc 07 §3:

- **Co-located sampling** (§3.1): :func:`sample_fused` assembles a feature matrix from a
  subset of fused layers at the cells where **all** selected layers are non-NaN
  (listwise deletion by default; ``mode="any"`` keeps any-present cells for histograms).
  An optional ``bbox`` restricts the working set to a clipping box (doc 07 §3.1 "just the
  anomaly"). Along-path sampling reuses the same trilinear sampler (:func:`sample_path`).
- **Cross-plots / histograms / correlation** (§3.2): :func:`crossplot` returns a 2D/3D
  scatter (point set, optionally per-point coloured by depth/class/3rd prop) or — for big
  N — a 2D density/hexbin grid; :func:`histogram` returns bin edges + counts (+ a coarse
  KDE); :func:`correlation_matrix` returns the cross-correlation matrix.
- **Clustering** (§3.3): :func:`cluster_fused` standardizes the feature matrix
  (log-transforming log10-flagged properties first via :data:`geosim.spatial.REGISTRY`),
  fits **k-means** or a **GaussianMixture** (scikit-learn), predicts a label per valid
  cell, and **writes back** a categorical ``lithology_class`` :class:`PropertyModel` (the
  class volume) plus, for GMM, one probability volume per class — all via
  :mod:`geosim.storage` & :mod:`geosim.catalog`, exactly like any other derived volume
  (doc 07 §4.3). It returns cluster centroids / sizes / cross-plot ellipses.

A **selection mask** (linked brushing, §3.2) maps a set of cell indices back to a boolean
volume that is itself storable as a categorical derived volume (:func:`selection_to_mask`).

Compute placement (doc 07 §3.4): operations on a working set ``<= SYNC_CELL_LIMIT`` cells
run synchronously; whole-grid clustering on larger grids is job-based (the API layer wires
:func:`cluster_fused` into a :class:`geosim.jobs.InlineJobRunner`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import zarr
from scipy.interpolate import RegularGridInterpolator
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import Session

from geosim.catalog import (
    Dataset,
    FusedLayer,
    FusedModel,
    IdKind,
    PropertyModel,
    Provenance,
    ProvenanceInput,
    new_id,
)
from geosim.spatial import REGISTRY
from geosim.storage import GridSpec, ProjectLayout, write_property_model

from .grid import FusedGrid, fused_grid_from_row, open_fused_group

__all__ = [
    "SYNC_CELL_LIMIT",
    "BIG_N_SCATTER",
    "FusedSample",
    "ClusterResult",
    "sample_fused",
    "sample_path",
    "crossplot",
    "histogram",
    "correlation_matrix",
    "cluster_fused",
    "selection_to_mask",
]

# doc 07 §3.4 — sync vs job threshold is the WORKING-SET cell count (default 5 M).
SYNC_CELL_LIMIT = 5_000_000
# Above this many co-located points a 2D scatter is returned as a density/hexbin grid
# instead of a (too-heavy) raw point set (doc 07 §3.2).
BIG_N_SCATTER = 50_000


# ──────────────────────────────────────────────────────────────────────────
# layer resolution + array reads
# ──────────────────────────────────────────────────────────────────────────


def _layers_by_property(fem: FusedModel) -> dict[str, FusedLayer]:
    """Map ``property`` → its resampled :class:`FusedLayer` (last write wins)."""
    return {lay.property: lay for lay in fem.layers}


def _resolve_layers(fem: FusedModel, properties: list[str] | None) -> list[FusedLayer]:
    """The :class:`FusedLayer`\\s for ``properties`` (all, in catalog order, if None)."""
    by_prop = _layers_by_property(fem)
    if properties is None:
        return list(fem.layers)
    out = []
    for prop in properties:
        lay = by_prop.get(prop)
        if lay is None:
            raise ValueError(f"property {prop!r} has no resampled layer on fused grid {fem.id!r}")
        out.append(lay)
    return out


def _read_layer_value(group: zarr.Group, layer: FusedLayer) -> np.ndarray:
    """Read a layer's resampled value array (z,y,x) from the fused Zarr group."""
    return np.asarray(group[layer.id][...], dtype=float)


# ──────────────────────────────────────────────────────────────────────────
# co-located sampling (doc 07 §3.1)
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class FusedSample:
    """A co-located multivariate sample over a fused grid (doc 07 §3.1).

    ``features`` is ``(n_cells, n_props)`` in native (un-standardized) units; ``cell_index``
    is the **flattened** ``(z,y,x)`` index of each retained cell so a selection can be
    mapped back to a volume mask (linked brushing, §3.2). ``coords`` is ``(n_cells, 3)``
    Engineering ``(z,y,x)`` metres (e.g. for per-point colour-by-depth).
    """

    properties: list[str]
    features: np.ndarray  # (n, p) float
    cell_index: np.ndarray  # (n,) int — flat index into the (nz,ny,nx) grid
    coords: np.ndarray  # (n, 3) float — Engineering (z,y,x) metres
    grid_shape: tuple[int, int, int]
    mode: str  # "all" (listwise) | "any"

    @property
    def n(self) -> int:
        return int(self.features.shape[0])


def _bbox_cell_mask(grid: FusedGrid, bbox: dict[str, float] | None) -> np.ndarray:
    """Boolean (z,y,x) mask of cells whose centre lies inside ``bbox`` (or all True)."""
    nz, ny, nx = grid.shape
    if bbox is None:
        return np.ones((nz, ny, nx), dtype=bool)
    z, y, x = grid.axis_coords()
    mz = (z >= bbox["zmin"]) & (z <= bbox["zmax"])
    my = (y >= bbox["ymin"]) & (y <= bbox["ymax"])
    mx = (x >= bbox["xmin"]) & (x <= bbox["xmax"])
    return mz[:, None, None] & my[None, :, None] & mx[None, None, :]


def sample_fused(
    session: Session,
    fem: FusedModel,
    properties: list[str] | None = None,
    *,
    mode: str = "all",
    bbox: dict[str, float] | None = None,
    storage_root: str | Path | None = None,
) -> FusedSample:
    """Co-located multi-volume sampling over a fused grid (doc 07 §3.1).

    Stacks the selected fused layers and keeps only cells where **all** selected layers are
    non-NaN (``mode="all"``, listwise deletion — the default for joint analysis) or where
    **any** layer is present (``mode="any"``, for per-property histograms). An optional
    ``bbox`` (Engineering metres) clips the working set to a region of interest.
    """
    if mode not in ("all", "any"):
        raise ValueError(f"mode must be 'all' or 'any'; got {mode!r}")
    grid = fused_grid_from_row(fem)
    layers = _resolve_layers(fem, properties)
    if not layers:
        raise ValueError(f"fused grid {fem.id!r} has no resampled layers to sample")

    group = open_fused_group(fem, storage_root=storage_root)
    stack = np.stack([_read_layer_value(group, lay) for lay in layers], axis=-1)  # (z,y,x,p)
    finite = np.isfinite(stack)
    if mode == "all":
        keep = finite.all(axis=-1)
    else:
        keep = finite.any(axis=-1)
    keep &= _bbox_cell_mask(grid, bbox)

    flat_keep = keep.reshape(-1)
    cell_index = np.flatnonzero(flat_keep)
    feats = stack.reshape(-1, stack.shape[-1])[cell_index]

    z, y, x = grid.axis_coords()
    gz, gy, gx = np.meshgrid(z, y, x, indexing="ij")
    coords = np.column_stack([gz.reshape(-1), gy.reshape(-1), gx.reshape(-1)])[cell_index]

    return FusedSample(
        properties=[lay.property for lay in layers],
        features=feats,
        cell_index=cell_index,
        coords=coords,
        grid_shape=grid.shape,
        mode=mode,
    )


def sample_path(
    session: Session,
    fem: FusedModel,
    points_zyx: np.ndarray,
    properties: list[str] | None = None,
    *,
    storage_root: str | Path | None = None,
) -> FusedSample:
    """Sample the fused layers along an arbitrary path / well track (doc 07 §3.1).

    ``points_zyx`` is ``(m, 3)`` Engineering ``(z,y,x)`` metres (e.g. a well path's
    MD→TVD→Engineering points). Each fused layer is trilinearly sampled at those points
    (NaN outside the grid), giving co-located ``(log_value, volume_value)`` pairs — the
    key calibration view. ``cell_index`` is set to ``-1`` (the samples are off-grid points,
    not cells).
    """
    grid = fused_grid_from_row(fem)
    layers = _resolve_layers(fem, properties)
    group = open_fused_group(fem, storage_root=storage_root)
    pts = np.asarray(points_zyx, dtype=float).reshape(-1, 3)
    z, y, x = grid.axis_coords()

    cols = []
    for lay in layers:
        values = _read_layer_value(group, lay)
        interp = RegularGridInterpolator(
            (z, y, x), values, method="linear", bounds_error=False, fill_value=np.nan
        )
        cols.append(interp(pts))
    feats = np.column_stack(cols) if cols else np.empty((pts.shape[0], 0))

    return FusedSample(
        properties=[lay.property for lay in layers],
        features=feats,
        cell_index=np.full(pts.shape[0], -1, dtype=int),
        coords=pts,
        grid_shape=grid.shape,
        mode="path",
    )


# ──────────────────────────────────────────────────────────────────────────
# cross-plots / histograms / correlation (doc 07 §3.2)
# ──────────────────────────────────────────────────────────────────────────


def _color_values(sample: FusedSample, color_by: str | None, labels: np.ndarray | None):
    """Per-point colour channel: ``"depth"`` (the z coord), a property name, or a class."""
    if color_by is None:
        return None
    if color_by == "depth":
        return sample.coords[:, 0]
    if color_by == "class" and labels is not None:
        return labels.astype(float)
    if color_by in sample.properties:
        return sample.features[:, sample.properties.index(color_by)]
    return None


def crossplot(
    sample: FusedSample,
    axes: list[str],
    *,
    color_by: str | None = None,
    labels: np.ndarray | None = None,
    bins: int = 64,
    big_n: int = BIG_N_SCATTER,
) -> dict:
    """A 2D/3D cross-plot payload for the viewer (doc 07 §3.2).

    ``axes`` names 2 or 3 of ``sample.properties``. For ``n <= big_n`` (2D) a point set is
    returned (optionally per-point ``color`` by depth/class/3rd property); for ``n > big_n``
    a 2D ``density`` (hexbin-equivalent) grid is returned instead. 3D always returns a point
    set (a Three.js mini-scene; downsampled if huge).
    """
    if len(axes) not in (2, 3):
        raise ValueError("crossplot needs 2 or 3 axes")
    idx = []
    for ax in axes:
        if ax not in sample.properties:
            raise ValueError(f"axis {ax!r} not in sampled properties {sample.properties}")
        idx.append(sample.properties.index(ax))
    data = sample.features[:, idx]
    n = data.shape[0]

    if len(axes) == 2 and n > big_n:
        counts, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=bins)
        return {
            "kind": "density",
            "axes": axes,
            "n": int(n),
            "counts": counts.tolist(),
            "x_edges": xedges.tolist(),
            "y_edges": yedges.tolist(),
        }

    # Point set (downsample 3D if very large so the payload stays light).
    keep = np.arange(n)
    if len(axes) == 3 and n > big_n:
        rng = np.random.default_rng(0)
        keep = rng.choice(n, size=big_n, replace=False)
    color = _color_values(sample, color_by, labels)
    payload = {
        "kind": "scatter",
        "axes": axes,
        "n": int(n),
        "points": data[keep].tolist(),
    }
    if color is not None:
        payload["color"] = np.asarray(color)[keep].tolist()
        payload["color_by"] = color_by
    return payload


def histogram(
    sample: FusedSample, prop: str, *, bins: int = 64, kde: bool = False
) -> dict:
    """A 1D histogram (+ optional coarse KDE) of one property (doc 07 §3.2)."""
    if prop not in sample.properties:
        raise ValueError(f"property {prop!r} not in sampled properties {sample.properties}")
    col = sample.features[:, sample.properties.index(prop)]
    col = col[np.isfinite(col)]
    counts, edges = np.histogram(col, bins=bins)
    payload = {
        "property": prop,
        "n": int(col.size),
        "counts": counts.tolist(),
        "bin_edges": edges.tolist(),
    }
    if kde and col.size > 1:
        from scipy.stats import gaussian_kde

        centres = 0.5 * (edges[:-1] + edges[1:])
        payload["kde_x"] = centres.tolist()
        payload["kde_y"] = gaussian_kde(col)(centres).tolist()
    return payload


def correlation_matrix(sample: FusedSample) -> dict:
    """Cross-correlation matrix across the sampled properties (doc 07 §3.2 heatmap).

    log10-flagged properties (registry) are log-transformed first so the correlation
    reflects the space they are physically compared in (doc 07 §3.3 step 2).
    """
    feats = _log_transform_features(sample.features, sample.properties)
    if feats.shape[0] < 2:
        corr = np.full((feats.shape[1], feats.shape[1]), np.nan)
    else:
        corr = np.corrcoef(feats, rowvar=False)
    corr = np.atleast_2d(corr)
    return {
        "properties": sample.properties,
        "matrix": np.where(np.isfinite(corr), corr, None).tolist(),
    }


# ──────────────────────────────────────────────────────────────────────────
# feature standardization (doc 07 §3.3 step 2)
# ──────────────────────────────────────────────────────────────────────────


def _is_log10(prop: str) -> bool:
    try:
        return REGISTRY.get(prop).interp_space == "log10"
    except KeyError:
        return False


def _log_transform_features(feats: np.ndarray, properties: list[str]) -> np.ndarray:
    """log10-transform the columns whose property is log10-flagged (doc 07 §3.3)."""
    out = feats.astype(float).copy()
    for j, prop in enumerate(properties):
        if _is_log10(prop):
            col = out[:, j]
            pos = col > 0.0
            col = np.where(pos, col, np.nan)
            out[:, j] = np.log10(col)
    return out


def _standardize(feats: np.ndarray, properties: list[str]) -> tuple[np.ndarray, StandardScaler]:
    """Log-transform log10 props then z-score standardize (doc 07 §3.3 step 2)."""
    logged = _log_transform_features(feats, properties)
    scaler = StandardScaler()
    return scaler.fit_transform(logged), scaler


# ──────────────────────────────────────────────────────────────────────────
# clustering → class + probability volumes (doc 07 §3.3)
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class ClusterResult:
    """Result of a fused-grid clustering run (doc 07 §3.3 step 6)."""

    algorithm: str  # "kmeans" | "gmm"
    n_clusters: int
    properties: list[str]
    labels: np.ndarray  # (n,) int cluster id per sampled cell
    centroids: list[list[float]]  # (k, p) in standardized feature space
    centroids_native: list[list[float]]  # (k, p) back in native (post-log) feature space
    sizes: list[int]  # cells per cluster
    ellipses: list[dict]  # per-cluster 2D cross-plot ellipse (mean+cov of first 2 feats)
    class_model_id: str | None = None  # the written categorical class PropertyModel
    probability_model_ids: list[str] = field(default_factory=list)  # GMM per-class prob vols

    def to_payload(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "n_clusters": self.n_clusters,
            "properties": self.properties,
            "centroids": self.centroids,
            "centroids_native": self.centroids_native,
            "sizes": self.sizes,
            "ellipses": self.ellipses,
            "class_model_id": self.class_model_id,
            "probability_model_ids": self.probability_model_ids,
        }


def _ellipses(std_feats: np.ndarray, labels: np.ndarray, n_clusters: int) -> list[dict]:
    """Per-cluster 2D cross-plot ellipse (mean + 2×2 covariance of the first two feats)."""
    out = []
    if std_feats.shape[1] < 2:
        return out
    for c in range(n_clusters):
        pts = std_feats[labels == c][:, :2]
        if pts.shape[0] < 2:
            out.append({"cluster": int(c), "mean": None, "cov": None})
            continue
        out.append({
            "cluster": int(c),
            "mean": pts.mean(axis=0).tolist(),
            "cov": np.cov(pts, rowvar=False).tolist(),
        })
    return out


def cluster_fused(
    session: Session,
    layout: ProjectLayout,
    fem: FusedModel,
    *,
    properties: list[str] | None = None,
    algorithm: str = "kmeans",
    n_clusters: int = 3,
    bbox: dict[str, float] | None = None,
    write_volumes: bool = True,
    random_state: int = 0,
    storage_root: str | Path | None = None,
    progress=None,
) -> ClusterResult:
    """Cluster a fused grid into lithology classes + probability volumes (doc 07 §3.3).

    Pipeline (doc 07 §3.3): assemble the co-located feature matrix at all-present cells →
    log-transform log10-flagged props + standardize → fit ``kmeans`` or ``gmm`` (sklearn) →
    predict a label per valid cell → write back a categorical ``lithology_class``
    :class:`PropertyModel` (the class volume) and, for GMM, one probability volume per class
    — all via :mod:`geosim.storage` & :mod:`geosim.catalog` (doc 07 §4.3) — and return
    centroids / sizes / cross-plot ellipses.

    ``progress`` is an optional :class:`geosim.jobs.ProgressReporter` (whole-grid runs go
    through :class:`geosim.jobs.InlineJobRunner`, doc 07 §3.4).
    """
    if algorithm not in ("kmeans", "gmm"):
        raise ValueError(f"algorithm must be 'kmeans' or 'gmm'; got {algorithm!r}")

    if progress is not None:
        progress.report(0.05, "sampling fused layers")
    sample = sample_fused(
        session, fem, properties, mode="all", bbox=bbox, storage_root=storage_root
    )
    if sample.n < n_clusters:
        raise ValueError(
            f"only {sample.n} co-located cells for {n_clusters} clusters — too few"
        )

    if progress is not None:
        progress.report(0.25, "standardizing features")
    std, _scaler = _standardize(sample.features, sample.properties)

    if progress is not None:
        progress.report(0.45, f"fitting {algorithm}")
    proba: np.ndarray | None = None
    if algorithm == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = model.fit_predict(std)
        centroids = model.cluster_centers_
    else:
        model = GaussianMixture(n_components=n_clusters, random_state=random_state)
        model.fit(std)
        labels = model.predict(std)
        proba = model.predict_proba(std)
        centroids = model.means_

    sizes = [int(np.sum(labels == c)) for c in range(n_clusters)]
    # Native (post-log) centroids = mean of the (log-transformed) native features per cluster.
    logged = _log_transform_features(sample.features, sample.properties)
    centroids_native = [
        (logged[labels == c].mean(axis=0).tolist() if sizes[c] else [None] * logged.shape[1])
        for c in range(n_clusters)
    ]

    result = ClusterResult(
        algorithm=algorithm,
        n_clusters=n_clusters,
        properties=sample.properties,
        labels=labels,
        centroids=centroids.tolist(),
        centroids_native=centroids_native,
        sizes=sizes,
        ellipses=_ellipses(std, labels, n_clusters),
    )

    if write_volumes:
        if progress is not None:
            progress.report(0.75, "writing class + probability volumes")
        _write_cluster_volumes(session, layout, fem, sample, labels, proba, result)

    if progress is not None:
        progress.report(1.0, "done")
    return result


def _scatter_to_volume(
    cell_index: np.ndarray, values: np.ndarray, grid_shape: tuple[int, int, int]
) -> np.ndarray:
    """Scatter per-cell ``values`` back into a full (z,y,x) volume; NaN elsewhere."""
    vol = np.full(int(np.prod(grid_shape)), np.nan, dtype=np.float32)
    vol[cell_index] = values
    return vol.reshape(grid_shape)


def _write_cluster_volumes(
    session: Session,
    layout: ProjectLayout,
    fem: FusedModel,
    sample: FusedSample,
    labels: np.ndarray,
    proba: np.ndarray | None,
    result: ClusterResult,
) -> None:
    """Write the categorical class volume (+ GMM probability volumes) as derived
    :class:`PropertyModel`\\s on the fused-grid support (doc 07 §3.3 step 5, §4.3)."""
    grid = fused_grid_from_row(fem)
    grid_spec = GridSpec(origin=grid.origin, spacing=grid.spacing, cell_ref="center")
    bbox = json.loads(fem.bbox_json)

    class_vol = _scatter_to_volume(sample.cell_index, labels.astype(np.float32), grid.shape)
    result.class_model_id = _write_derived_pm(
        session, layout, fem, grid, grid_spec, bbox,
        prop="lithology_class", values=class_vol,
        process="fuse:cluster",
        params={
            "algorithm": result.algorithm, "n_clusters": result.n_clusters,
            "properties": result.properties, "derivation": "cluster",
        },
    )

    if proba is not None:
        for c in range(result.n_clusters):
            prob_vol = _scatter_to_volume(
                sample.cell_index, proba[:, c].astype(np.float32), grid.shape
            )
            pid = _write_derived_pm(
                session, layout, fem, grid, grid_spec, bbox,
                prop="lithology_class", values=prob_vol,
                process="fuse:cluster:probability",
                params={
                    "algorithm": result.algorithm, "n_clusters": result.n_clusters,
                    "cluster": c, "properties": result.properties,
                    "derivation": "cluster_probability",
                },
                name_suffix=f"-prob-class{c}",
            )
            result.probability_model_ids.append(pid)


def _write_derived_pm(
    session: Session,
    layout: ProjectLayout,
    fem: FusedModel,
    grid: FusedGrid,
    grid_spec: GridSpec,
    bbox: dict[str, float],
    *,
    prop: str,
    values: np.ndarray,
    process: str,
    params: dict,
    name_suffix: str = "",
) -> str:
    """Write one derived PropertyModel (Zarr + catalog rows) on the fused-grid support.

    Provenance records the fused grid + its source layers' native models as inputs so the
    derived volume is reproducible (doc 07 §4.3/§4.4).
    """
    pm_id = new_id(IdKind.PROPERTY_MODEL)
    ds_id = new_id(IdKind.DATASET)
    prov_id = new_id(IdKind.PROVENANCE)
    zarr_path = layout.zarr_path(pm_id)
    write_property_model(zarr_path, prop, values, grid=grid_spec, overwrite=True)

    bbox_json = json.dumps(bbox)
    prov = Provenance(
        id=prov_id, project_id=fem.project_id, target_kind="propertyModel",
        target_id=pm_id, process=process, process_version="1.0.0",
        params_json=json.dumps({**params, "fusedGridId": fem.id}),
    )
    session.add(prov)
    session.flush()
    # The fused grid + each source native model are inputs (reproducibility, doc 07 §4.4).
    session.add(ProvenanceInput(provenance_id=prov_id, input_kind="fusedModel", input_id=fem.id))
    for lay in fem.layers:
        session.add(ProvenanceInput(
            provenance_id=prov_id, input_kind="propertyModel",
            input_id=lay.source_property_model_id,
        ))
    session.add(Dataset(
        id=ds_id, project_id=fem.project_id, name=f"{prop}{name_suffix}", method="fusion",
        kind="propertyModel", status="ready", extent_json=bbox_json,
        spatial_frame_id=fem.project_id, provenance_id=prov_id,
        version_root_id=ds_id, version_seq=1, created_by="system:fusion",
    ))
    session.flush()
    pt = REGISTRY.get(prop)
    session.add(PropertyModel(
        id=pm_id, dataset_id=ds_id, project_id=fem.project_id, property=prop,
        canonical_unit=pt.canonical_unit, support="volume",
        store_uri=str(zarr_path), shape_json=json.dumps(list(grid.shape)),
        spacing_json=json.dumps(list(grid.spacing)), origin_json=json.dumps(list(grid.origin)),
        bbox_json=bbox_json, pyramid_levels=1,
    ))
    session.commit()
    return pm_id


# ──────────────────────────────────────────────────────────────────────────
# linked brushing — selection → cell-index mask (doc 07 §3.2)
# ──────────────────────────────────────────────────────────────────────────


def selection_to_mask(
    sample: FusedSample, selected_local_indices: list[int] | np.ndarray
) -> np.ndarray:
    """Map a cross-plot selection (positions into ``sample``) → a boolean volume mask.

    ``selected_local_indices`` are positions into the sample's rows (e.g. the points a
    lasso enclosed); the returned ``(nz,ny,nx)`` boolean volume is True at the selected
    cells — itself storable as a categorical derived volume for 3D linked brushing
    (doc 07 §3.2).
    """
    sel = np.asarray(selected_local_indices, dtype=int)
    flat = np.zeros(int(np.prod(sample.grid_shape)), dtype=bool)
    if sel.size:
        flat[sample.cell_index[sel]] = True
    return flat.reshape(sample.grid_shape)
