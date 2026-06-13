"""M5 exit criteria — full rock-physics + favorability + uncertainty (doc-ROADMAP M5).

The doc-ROADMAP M5 gate (design/ROADMAP.md §"M5 — Rock-physics … + favorability + uncertainty"):

    Exit: produce a "geothermal favorability" volume over ``great-basin-v1``, with a
    confidence volume, that highlights the known synthetic anomaly — and tune the weights
    live.

This test proves the headless half of that gate end-to-end (doc 07 §4):

1. **One earth** — compile the flagship ``great-basin-v1`` scene (doc 05 §7.1: a Basin-&-
   Range hydrothermal play with a fault-controlled upflow + shallow clay-cap conductor) at a
   deliberately **coarse** truth grid (~19×15×15) so the whole build + transform + favorability
   runs in ~1 s. The geology is unchanged, so the conductive / softened / fractured / hotter
   hydrothermal anomaly is still present and still co-varies across methods.
2. **Native geophysical models** — write the co-located truth resistivity (log10 interp),
   P-velocity and a binned microseismic event-count volume in as ordinary native
   :class:`PropertyModel`\\s (these stand in for the inverted/gridded survey models the fusion
   engine consumes, doc 07 §0). A **below-DOI** deep slab is masked NaN in the resistivity
   model — beyond an MT survey's depth-of-investigation there is no coverage (doc 07 §2.3).
3. **Fuse + resample** — :func:`build_fused_model` over the three native models, then
   :func:`resample_to_fused` each onto the shared support (doc 07 §1–§2), interpolating
   resistivity in **log10** space per :data:`geosim.spatial.REGISTRY`.
4. **Rock-physics transforms (doc 07 §4.1–§4.2)** — run the real shipped library transforms
   through the §4.5 harness:
   - ``rp.resistivity_to_temperature.arps`` → a **temperature likelihood** (kelvin, proxy);
   - ``rp.velocity_to_porosity`` → **porosity**;
   - ``rp.microseismic_density`` → a **fracture-density** index (the permeability proxy).
   Each is an *uncalibrated* transform, so the harness retitles it a likelihood + caps the
   tier to ``proxy`` (doc 07 §4.5 step 7) — exercising the uncertainty/honesty machinery.
5. **Fuzzy-conjunction favorability + confidence (THE M5 EXIT)** — combine the three derived
   volumes (hot AND porous AND fractured, all *required*) by the non-compensatory fuzzy-AND
   default (doc 07 §4.6) into a ``[0,1]`` favorability volume **plus its paired confidence
   volume**. Then **score against the retained synthetic ground truth**: the favorability
   hot-spot must COINCIDE with the known synthetic geothermal anomaly (the truth
   alteration/temperature body read off ``earth.state`` — an independent label never fed to
   the transforms) **well above chance**, and the **confidence volume must be lower in the
   below-DOI / low-coverage slab** than in the fully-covered region.

All I/O is to ``tmp_path`` with in-memory SQLite — no Docker/Postgres/Redis, coarse grids
throughout. Live weight-tuning + the confidence-faded 3-D render are the FRONTEND half of M5
(a browser check) and are intentionally out of scope here (see blockers).
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

# Importing the rock-physics package self-registers the whole §4.2 transform library AND its
# property types (permeability/microseismic/alteration), doc 08 §3.1 — must precede any
# REGISTRY.get on those keys.
import geosim.fusion.rockphys  # noqa: F401
from geosim.catalog import (
    Dataset,
    IdKind,
    Project,
    PropertyModel,
    Provenance,
    SpatialFrameRow,
    create_all,
    make_engine,
    new_id,
    session_factory,
)
from geosim.fusion import (
    Evidence,
    FavorabilitySpec,
    TransferFn,
    build_fused_model,
    compute_favorability,
    fused_grid_from_row,
    resample_to_fused,
    run_transform,
)
from geosim.fusion.rockphys import (
    MicroseismicDensity,
    ResistivityToTemperature,
    VelocityToPorosity,
)
from geosim.fusion.rockphys.fracture import events_to_count_volume
from geosim.spatial import REGISTRY, Aabb, DepthRange, SpatialFrame
from geosim.storage import (
    GridSpec,
    ensure_project_layout,
    open_property_model,
    write_property_model,
)
from geosim.synthgen import compile_scene
from geosim.synthgen.scenarios import get_scenario

# The native geophysical models the M5 rock-physics chain consumes (doc-ROADMAP M5).
_NATIVE_PROPS = ["resistivity", "velocity_p", "microseismic"]


# ─────────────────────────────── coarse great-basin earth ───────────────────────────────


def _coarse_great_basin():
    """Compile ``great-basin-v1`` at a coarse truth grid (same geology, ~19×15×15 cells).

    The shipped flagship truth grid is millions of cells; we only need the co-located
    property + state volumes, so we replace the fine spacings with coarse ones (doc 05 §2
    allows the truth-grid spacing to be chosen) to keep this gate ~1 s while preserving the
    hydrothermal anomaly's multi-method signature (low ρ, low Vp, high fracture, hot).
    """
    spec = get_scenario("great-basin-v1").scene
    coarse_frame = replace(spec.frame, dx=800.0, dy=800.0, dz=400.0)
    return compile_scene(replace(spec, frame=coarse_frame))


def _anomaly_truth_mask(earth) -> np.ndarray:
    """The known synthetic geothermal anomaly: the hydrothermal alteration + hot body.

    The flagship play's diagnostic anomaly is the fault-controlled hydrothermal upflow +
    clay cap — an **altered**, **hot** body (doc 05 §7.1). We take the truth
    alteration-fraction AND elevated-temperature state fields (the physical *cause*, not a
    property proxy) as the ground-truth anomaly the favorability volume must coincide with.
    This is an INDEPENDENT label: it is read from ``earth.state`` only, never from the
    resistivity / velocity / microseismic features fed to the rock-physics transforms.
    """
    sub = ~earth.above_surface
    hot = earth.state.temperature > 470.0  # ≈ 197 °C — the upflow signature
    altered = earth.state.alteration_fraction > 0.1
    return (altered & hot) & sub


# ─────────────────────────────────── native-model I/O ───────────────────────────────────


def _engineering_bbox(origin, spacing, shape) -> str:
    oz, oy, ox = origin
    dz, dy, dx = spacing
    nz, ny, nx = shape
    return json.dumps({
        "xmin": ox, "xmax": ox + dx * (nx - 1),
        "ymin": oy, "ymax": oy + dy * (ny - 1),
        "zmin": oz, "zmax": oz + dz * (nz - 1),
    })


def _write_native_pm(session, layout, pid, *, prop, values, origin, spacing, unit, method):
    """Persist a co-located native :class:`PropertyModel` (Zarr + catalog rows)."""
    ds_id = new_id(IdKind.DATASET)
    pm_id = new_id(IdKind.PROPERTY_MODEL)
    prov_id = new_id(IdKind.PROVENANCE)
    zarr_path = layout.zarr_path(pm_id)
    grid = GridSpec(origin=origin, spacing=spacing, cell_ref="center")
    write_property_model(zarr_path, prop, values.astype(np.float32), grid=grid, overwrite=True)

    bbox = _engineering_bbox(origin, spacing, values.shape)
    session.add(Provenance(id=prov_id, project_id=pid, target_kind="propertyModel",
                           target_id=pm_id, process="ingest:synthetic"))
    session.flush()
    session.add(Dataset(
        id=ds_id, project_id=pid, name=f"{prop}-native", method=method, kind="propertyModel",
        status="ready", extent_json=bbox, spatial_frame_id=pid, provenance_id=prov_id,
        version_root_id=ds_id, version_seq=1, created_by="m5@test",
    ))
    session.flush()
    session.add(PropertyModel(
        id=pm_id, dataset_id=ds_id, project_id=pid, property=prop,
        canonical_unit=unit, support="volume", store_uri=str(zarr_path),
        shape_json=json.dumps(list(values.shape)),
        spacing_json=json.dumps(list(spacing)), origin_json=json.dumps(list(origin)),
        bbox_json=bbox,
    ))
    session.commit()
    return session.get(PropertyModel, pm_id)


def _microseismic_count_volume(earth) -> np.ndarray:
    """A binned microseismic event cloud co-located with the true fractured upflow.

    Draws an event cloud whose density tracks the truth fracture-density state field (more
    active fracturing ⇒ more events, doc 07 §4.2), then bins it onto the truth grid with the
    real :func:`events_to_count_volume`. The transform later KDE-smooths this into a
    fracture-density (permeability) proxy. Deterministic RNG so the gate is reproducible.
    """
    sub = ~earth.above_surface
    frac = np.where(sub, earth.state.fracture_density, 0.0)
    nz, ny, nx = earth.shape
    oz, oy, ox = earth.origin
    dz, dy, dx = earth.spacing

    weights = frac.reshape(-1)
    weights = weights / weights.sum()
    rng = np.random.default_rng(0)
    n_events = 600
    cell_idx = rng.choice(weights.size, size=n_events, p=weights)
    iz, iy, ix = np.unravel_index(cell_idx, (nz, ny, nx))
    # cell-centre world coords + a little jitter so events fall inside their voxel.
    jitter = rng.uniform(-0.4, 0.4, size=(n_events, 3))
    xs = ox + (ix + jitter[:, 0]) * dx
    ys = oy + (iy + jitter[:, 1]) * dy
    zs = oz + (iz + jitter[:, 2]) * dz
    return np.column_stack([xs, ys, zs])


# ─────────────────────────────────────── fixture ───────────────────────────────────────


@pytest.fixture
def m5_fused_great_basin(tmp_path):
    """Coarse great-basin → native ρ/Vp/microseismic models → fused grid w/ all resampled in.

    A deep **below-DOI slab** is masked NaN in the resistivity model so the temperature
    likelihood (and thus a required favorability layer) has NO coverage there — the
    low-coverage region the confidence volume must penalise (doc 07 §2.3, §4.6/§5).

    Yields ``(session, layout, root, pid, fem, earth, anomaly, below_doi)``.
    """
    earth = _coarse_great_basin()
    sub = ~earth.above_surface

    storage_root = tmp_path / "store"
    engine = make_engine()  # in-memory SQLite (doc 04 §2.1 fallback)
    create_all(engine)
    Session = session_factory(engine)
    session = Session()

    pid = new_id(IdKind.PROJECT)
    layout = ensure_project_layout(storage_root, pid)

    origin, spacing, shape = earth.origin, earth.spacing, earth.shape
    bbox = json.loads(_engineering_bbox(origin, spacing, shape))
    frame = SpatialFrame(
        roi=Aabb(bbox["xmin"], bbox["xmax"], bbox["ymin"], bbox["ymax"]),
        depth_range=DepthRange(bbox["zmin"], bbox["zmax"]),
    )
    session.add(Project(id=pid, name="m5-great-basin", storage_root=str(storage_root)))
    session.add(SpatialFrameRow(
        project_id=pid, mode=frame.mode.value,
        roi_json=json.dumps({k: bbox[k] for k in ("xmin", "xmax", "ymin", "ymax")}),
        depth_range_json=json.dumps({"zmin": bbox["zmin"], "zmax": bbox["zmax"]}),
        frame_json=json.dumps({"mode": frame.mode.value}),
    ))
    session.commit()

    # The native geophysical models. Resistivity carries a below-DOI NaN slab (the deepest
    # 4 z-layers) — beyond the MT survey's depth-of-investigation there is no signal (§2.3).
    below_doi = np.zeros(shape, dtype=bool)
    below_doi[:4, :, :] = True  # z index 0..3 are the deepest cells (Z-up), beyond DOI
    below_doi &= sub

    resistivity = earth.property_volume("resistivity").astype(np.float32)
    resistivity[below_doi] = np.nan
    velocity_p = earth.property_volume("velocity_p").astype(np.float32)
    micro_events = _microseismic_count_volume(earth)

    native = {}
    native["resistivity"] = _write_native_pm(
        session, layout, pid, prop="resistivity", values=resistivity,
        origin=origin, spacing=spacing, unit="ohm*m", method="mt",
    )
    native["velocity_p"] = _write_native_pm(
        session, layout, pid, prop="velocity_p", values=velocity_p,
        origin=origin, spacing=spacing, unit="m/s", method="seismic",
    )

    fem, grid = build_fused_model(
        session, layout, pid,
        source_property_model_ids=[native["resistivity"].id, native["velocity_p"].id],
        spacing=spacing, name="m5-fused",
    )
    # Bin the microseismic event cloud onto the (shared) fused grid → a count volume, and
    # write it in as the native ``microseismic`` model (doc 07 §4.2 KDE pre-step).
    counts = events_to_count_volume(micro_events, grid)
    native["microseismic"] = _write_native_pm(
        session, layout, pid, prop="microseismic", values=counts,
        origin=origin, spacing=spacing, unit="dimensionless", method="microseismic",
    )

    for prop in _NATIVE_PROPS:
        resample_to_fused(session, fem, native[prop].id)
    session.refresh(fem)

    anomaly = _anomaly_truth_mask(earth)
    native_ids = {p: native[p].id for p in _NATIVE_PROPS}
    yield session, layout, storage_root, pid, fem, earth, anomaly, below_doi, native_ids
    session.close()


# ────────────────────────── rock-physics transforms (doc 07 §4.1–§4.2) ──────────────────────────


def _run_rockphysics(session, layout, fem, root):
    """Run the three shipped uncalibrated transforms → derived temperature/porosity/fracture.

    Returns ``(temp_id, poro_id, frac_id)`` derived-PropertyModel ids on the fused grid.
    """
    temp = run_transform(
        session, layout, fem, ResistivityToTemperature(),
        params={"porosity": 0.12, "fluid_salinity_ppm": 8000.0}, storage_root=root,
    )
    poro = run_transform(
        session, layout, fem, VelocityToPorosity(),
        params={"v_matrix_m_s": 6000.0, "v_fluid_m_s": 1500.0, "model": "wyllie"},
        storage_root=root,
    )
    frac = run_transform(
        session, layout, fem, MicroseismicDensity(),
        params={"bandwidth_cells": 1.0}, storage_root=root,
    )
    # Every transform here is uncalibrated ⇒ output capped to a proxy likelihood (§4.5 step 7).
    for r in (temp, poro, frac):
        assert r.calibration_status == "uncalibrated"
        assert r.tier == "proxy"
    assert temp.output_property == "temperature"
    assert poro.output_property == "porosity"
    assert frac.output_property == "fracture_density"
    return temp.model_id, poro.model_id, frac.model_id


def _read_fused(session, root, model_id, prop):
    pm = session.get(PropertyModel, model_id)
    return open_property_model(pm.store_uri).read_level(prop, 0)


def test_resistivity_resamples_in_log_space(m5_fused_great_basin):
    """Resistivity is interpolated in log10 space per the registry (doc 07 §2.3, doc 01 §5)."""
    assert REGISTRY.get("resistivity").interp_space == "log10"
    # canonical temperature is KELVIN end-to-end (doc 01 §5).
    assert REGISTRY.get("temperature").canonical_unit == "kelvin"


def test_rockphysics_chain_produces_proxy_derived_volumes(m5_fused_great_basin):
    """resistivity→temperature, velocity→porosity, microseismic→fracture run + flag proxy (§4)."""
    session, layout, root, _pid, fem, _earth, _anom, _doi, _native = m5_fused_great_basin
    temp_id, poro_id, frac_id = _run_rockphysics(session, layout, fem, root)

    shape = fused_grid_from_row(fem).shape
    temp = _read_fused(session, root, temp_id, "temperature")
    poro = _read_fused(session, root, poro_id, "porosity")
    frac = _read_fused(session, root, frac_id, "fracture_density")
    assert temp.shape == poro.shape == frac.shape == shape

    # Temperature out is canonical kelvin in a sane geothermal band where it is defined.
    tk = temp[np.isfinite(temp)]
    assert tk.size > 0 and tk.min() >= 273.0 and tk.max() <= 673.0
    # Porosity + fracture indices are bounded fractions where defined.
    assert np.nanmin(poro) >= 0.0 and np.nanmax(poro) <= 0.5
    assert np.nanmin(frac) >= 0.0 and np.nanmax(frac) <= 1.0


# ─────────────────────────── THE M5 EXIT: favorability + confidence ───────────────────────────


def _evidence(temp_id, poro_id, frac_id, conductor_id=None):
    """Hot AND porous AND fractured, all required — the geothermal fuzzy conjunction (§4.6).

    The three rock-physics proxies (temperature/porosity/fracture) are *uncalibrated* (proxy
    tier ⇒ they feed assumption-burden). Membership ramps are expressed in each evidence's
    canonical unit (temperature in KELVIN); thresholds bracket the synthetic anomaly's true
    derived range. When ``conductor_id`` is supplied, the **native** (calibrated, non-proxy)
    resistivity model is added as a fourth required conductor indicator — a low-resistivity
    membership (the clay-cap / brine signature, doc 07 §4.2). Because it is a primary
    measurement rather than a rock-physics guess it keeps the assumption-burden below 1, so
    the confidence volume (overlap·(1−burden)) is non-zero and varies with coverage.
    """
    ev = [
        Evidence(source=temp_id, target="temperature",
                 transfer=TransferFn("ramp", lo=410.0, hi=500.0), weight=0.35, role="required"),
        Evidence(source=poro_id, target="porosity",
                 transfer=TransferFn("ramp", lo=0.025, hi=0.07), weight=0.25, role="required"),
        Evidence(source=frac_id, target="fracture_density",
                 transfer=TransferFn("ramp", lo=0.15, hi=0.6), weight=0.25, role="required"),
    ]
    if conductor_id is not None:
        ev.append(Evidence(
            source=conductor_id, target="resistivity",
            transfer=TransferFn("ramp", lo=200.0, hi=20.0),  # descending: low ρ ⇒ favorable
            weight=0.15, role="required",
        ))
    return ev


def _hotspot_quality(fav: np.ndarray, anomaly: np.ndarray, threshold: float):
    """Coincidence of the favorability hot-spot with the known synthetic anomaly.

    Returns ``(recall, precision, lift)`` for the cells scoring ``fav >= threshold``:

    - ``recall``    — fraction of true-anomaly cells flagged as a hot-spot;
    - ``precision`` — fraction of hot-spot cells that are true anomaly;
    - ``lift``      — precision ÷ the anomaly's prevalence among *scored* cells; ``lift > 1``
      means the hot-spot is enriched in true anomaly **above chance** (the M5 "highlights the
      known synthetic anomaly" claim).
    """
    scored = np.isfinite(fav)
    anom_scored = anomaly & scored
    prevalence = float(anom_scored.sum()) / float(scored.sum())
    hot = scored & (fav >= threshold)
    tp = float((hot & anomaly).sum())
    recall = tp / float(max(anom_scored.sum(), 1))
    precision = tp / float(max(hot.sum(), 1))
    lift = precision / prevalence if prevalence else float("inf")
    return recall, precision, lift


def test_favorability_hotspot_coincides_with_synthetic_anomaly(m5_fused_great_basin):
    """THE M5 EXIT (part 1): the fuzzy favorability hot-spot lands on the true anomaly (§4.6).

    Combine the three derived rock-physics proxies (hot ∧ porous ∧ fractured) by the
    non-compensatory fuzzy-conjunction default into a ``[0,1]`` favorability volume, then
    score it against the RETAINED synthetic ground truth (the alteration+temperature body
    read off ``earth.state``, never fed to the transforms). The high-favorability cells must
    overlap the known geothermal anomaly well above chance.
    """
    session, layout, root, _pid, fem, _earth, anomaly, _doi, _native = m5_fused_great_basin
    temp_id, poro_id, frac_id = _run_rockphysics(session, layout, fem, root)

    spec = FavorabilitySpec(evidence=_evidence(temp_id, poro_id, frac_id), method="fuzzy")
    result = compute_favorability(session, layout, fem, spec, storage_root=root)
    assert result.method == "fuzzy" and result.n_required == 3
    assert result.confidence_model_id  # the paired confidence volume exists (§5)

    fav = _read_fused(session, root, result.model_id, "favorability")
    assert fav.shape == fused_grid_from_row(fem).shape
    finite = np.isfinite(fav)
    assert finite.any() and np.nanmin(fav) >= 0.0 and np.nanmax(fav) <= 1.0

    # Align the truth anomaly to the scored cells and require a real anomaly population.
    anomaly_scored = anomaly & finite
    assert anomaly_scored.sum() > 10, "need a meaningful anomaly population in the scored grid"

    # The favorability hot-spot (cells scoring ≥ 0.4) coincides with the synthetic anomaly
    # WELL ABOVE CHANCE: it captures most of the true anomaly AND is strongly enriched in it
    # (lift ≫ 1 ⇒ far better than the base rate). The non-compensatory fuzzy-AND deliberately
    # down-scores the anomaly fringes where one proxy is weak, so a strict majority recall
    # paired with a high lift is the honest "coincides above chance" signature.
    recall, precision, lift = _hotspot_quality(fav, anomaly, threshold=0.4)
    assert recall >= 0.6, f"favorability misses the true anomaly (recall={recall:.2f})"
    assert lift >= 2.5, f"favorability hot-spot not enriched above chance (lift={lift:.2f})"
    assert precision >= 0.6, f"favorability hot-spot too contaminated (precision={precision:.2f})"

    # And the anomaly is genuinely the favorable population: median favorability inside the
    # true anomaly is far higher than the background median.
    anom_med = float(np.median(fav[anomaly_scored]))
    bg_med = float(np.median(fav[finite & ~anomaly]))
    assert anom_med > bg_med + 0.3, (
        f"anomaly not the favorable population ({anom_med:.2f} vs {bg_med:.2f})"
    )


def test_confidence_lower_below_doi_low_coverage(m5_fused_great_basin):
    """THE M5 EXIT (part 2): the confidence volume is lower below DOI / in low coverage (§4.6/§5).

    The resistivity model (the temperature-likelihood input AND the native conductor evidence)
    has NO coverage in the deep below-DOI slab, so two required evidence layers are absent
    there. Confidence = evidence-overlap·(1−assumption-burden): a calibrated native conductor
    layer keeps burden < 1 (so confidence is non-zero where it is well covered), while below
    DOI the missing required layers drive evidence-overlap — and hence confidence — down. The
    confidence volume MUST therefore be markedly lower in the below-DOI / low-coverage slab.
    """
    session, layout, root, _pid, fem, _earth, _anom, below_doi, native = m5_fused_great_basin
    temp_id, poro_id, frac_id = _run_rockphysics(session, layout, fem, root)

    # Include the native (calibrated, non-proxy) resistivity conductor as a 4th required layer
    # so assumption-burden < 1 and confidence is a meaningful, coverage-sensitive number.
    # Neutral missing-policy keeps below-DOI cells scored — the confidence volume is what
    # flags their low coverage rather than the favorability score itself.
    spec = FavorabilitySpec(
        evidence=_evidence(temp_id, poro_id, frac_id, conductor_id=native["resistivity"]),
        method="fuzzy", missing_policy="neutral",
    )
    result = compute_favorability(session, layout, fem, spec, storage_root=root)
    assert result.n_required == 4

    conf = _read_fused(session, root, result.confidence_model_id, "confidence")
    overlap = _read_fused(session, root, result.overlap_model_id, "evidence_overlap")
    assert conf.shape == below_doi.shape

    well_covered = ~below_doi & np.isfinite(conf)
    low_coverage = below_doi & np.isfinite(conf)
    assert well_covered.sum() > 0 and low_coverage.sum() > 0

    # Below DOI the resistivity-derived required layers are absent ⇒ overlap < 1.
    assert float(np.nanmax(overlap[low_coverage])) < 1.0 - 1e-6
    # …and the well-covered region reaches full overlap somewhere.
    assert float(np.nanmax(overlap[well_covered])) == pytest.approx(1.0)

    # A non-proxy conductor keeps burden < 1 ⇒ confidence is genuinely positive where covered.
    assert float(np.nanmax(conf[well_covered])) > 0.0

    # THE confidence assertion: confidence is markedly LOWER in the below-DOI / low-coverage
    # slab than in the fully-covered region.
    conf_low = float(np.median(conf[low_coverage]))
    conf_high = float(np.median(conf[well_covered]))
    assert conf_low < conf_high, (
        f"confidence not lower below DOI ({conf_low:.2f} vs {conf_high:.2f})"
    )
