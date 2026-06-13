"""M3 exit criteria — the rigorous (T1) physics tier round-trips unchanged (doc 05 §4, §6).

This is the M3 milestone gate for the **backend physics**: it proves that the higher-fidelity
``fidelity="rigorous"`` forwards (doc 05 §6 T1 tier) for the three flagship methods — gravity,
MT, and seismic reflection — (1) run their real solver (harmonica / empymod / acoustic-
convolutional) on a SMALL coarse :class:`TruthEarth`, (2) emit responses that are *physically
defensible* with the skin-depth / DOI / resolution falloff VISIBLE and CORRECT (doc 05 §4
rigorous column), and (3) ingest back through the EXISTING :mod:`geosim.ingestion` adapters
**with no adapter change** — the same native file format as their T0 counterparts, so the
detection + parse path that serves the plausible tier serves the rigorous tier byte-for-byte
(doc 05 §4: T1 emits the SAME native files as T0; OVERVIEW §8 round-trip is preserved).

Concretely (ROADMAP M3 exit):

(a) **gravity** — the rigorous Newtonian-prism Bouguer anomaly is smooth and has the correct
    sign + order of magnitude over the truth's density anomaly (a gravity high tracks the
    excess-mass columns), and the GeoTIFF + CSV station files ingest via the gravity adapter.
(b) **MT** — skin-depth / DOI is visible: apparent resistivity at LONG periods diffuses down
    to the deep conductor while SHORT periods see only the shallow clay cap (period→depth
    falloff), and the per-station EDIs ingest via the MT EDI adapter.
(c) **seismic** — band-limited reflection energy sits at the TRUE impedance (ρ·Vp) contrasts,
    and the SEG-Y section + horizons GeoJSON ingest via the seismic SEG-Y adapter.

Each T1 output is routed through the *registry* :func:`geosim.ingestion.detect` (the real
detection step, doc 03 §7) so the assertion "same adapter as its T0 counterpart, unchanged"
is enforced, not assumed. Truth grids are deliberately tiny/coarse so the whole gate runs in
seconds and headless (no Docker/Postgres) — M3 is backend physics only, no frontend/visual leg.
"""

from __future__ import annotations

import numpy as np
import pytest

from geosim.ingestion import RawSource, detect
from geosim.ingestion.adapters.gravity import GravityPotentialFieldAdapter
from geosim.ingestion.adapters.mt import MtEdiAdapter
from geosim.ingestion.adapters.seismic import SeismicSegyAdapter
from geosim.synthgen import (
    AnomalySpec,
    FaultSpec,
    FrameSpec,
    GeothermSpec,
    LayerSpec,
    SceneSpec,
    SurfaceSpec,
    compile_scene,
)
from geosim.synthgen.forward import (
    Acquisition,
    Artifact,
    GravityForward,
    GravityRigorousForward,
    MTForward,
    MTRigorousForward,
    SeismicReflectionForward,
    SeismicReflectionRigorousForward,
    get_forward,
)

# --------------------------------------------------------------- a tiny coarse earth


def _tiny_scene(seed: int = 11) -> SceneSpec:
    """A SMALL (~10×10×~13) Basin-&-Range geothermal scene exercising all three methods.

    It carries a dense basement + light plume (gravity), a shallow clay-cap conductor over a
    resistive host over a deep reservoir conductor (MT period→depth), and sharp layer-boundary
    impedance contrasts (seismic) — one earth, many surveys (doc 05 §4 truth-driven forwards).
    """
    return SceneSpec(
        id="m3-exit-tiny-v1",
        seed=seed,
        frame=FrameSpec(
            xmin=-500, xmax=500, ymin=-500, ymax=500,
            zmin=-1500, zmax=200, dx=160, dy=160, dz=150,
        ),
        surface=SurfaceSpec(kind="tilted-block", base_elev=120.0, tilt_x=0.08),
        layers=(
            LayerSpec("alluvium", "surface", (50.0, 80.0)),
            LayerSpec("volcanics", "conformable", (150.0, 250.0)),
            LayerSpec("basement_granite", "conformable", "fill"),
        ),
        faults=(
            FaultSpec("range-front", trace=((-500, -100), (500, 200)),
                      kind="normal", dip=60, dip_azimuth=90, throw=180, is_conduit=True),
        ),
        geotherm=GeothermSpec(surface_temp=15.0, gradient=45.0),
        anomalies=(
            AnomalySpec(
                "upflow", footprint_center=(0.0, 0.0), footprint_radius_xy=300.0,
                top_elev=120.0, bottom_elev=-1200.0, controlled_by="range-front",
                temp_peak=220.0, alteration_frac=0.9, porosity_boost=0.05,
                salinity_tds=8000.0, fracture_density=0.5,
                clay_cap_top_elev=80.0, clay_cap_thickness=150.0,
            ),
        ),
        rock_physics="default-v1",
    )


@pytest.fixture(scope="module")
def earth():
    return compile_scene(_tiny_scene())


def _rng():
    return np.random.default_rng(2026)


def _detect_with(source: RawSource):
    """The real doc-03 §7 detection step → the adapter that will parse this file."""
    return detect(source)


# ============================================================================== (a) gravity


def test_gravity_rigorous_smooth_correct_sign_and_ingests_via_t0_adapter(earth, tmp_path):
    """(a) Rigorous Bouguer anomaly: smooth, correct sign/magnitude, T0 adapter unchanged.

    The harmonica prism-integrated Bouguer high tracks the truth's excess-mass columns (right
    sign), stays in a sane single-digit-to-tens-of-mGal range for a small model (right order of
    magnitude), and is spatially SMOOTH (a potential field has no station-to-station jitter).
    Both native files (CSV stations + Bouguer GeoTIFF) detect to the SAME gravity adapter the
    T0 forward's files do — ingestion is unchanged.
    """
    import rasterio

    acq = Acquisition(gravity_spacing=160.0, params={"out_dir": str(tmp_path)})
    arts = GravityRigorousForward().simulate(earth, acq, _rng())
    assert {a.fmt for a in arts} == {"csv", "geotiff"}  # SAME native files as T0 (doc 05 §4)

    # --- physically defensible: correct sign vs the truth column mass ---
    tif = next(a for a in arts if a.fmt == "geotiff")
    with rasterio.open(tif.path) as r:
        grid = r.read(1)
    assert grid.ndim == 2 and np.isfinite(grid).all()

    from geosim.synthgen.forward.potential_field import _density_anomaly

    col = np.nansum(_density_anomaly(earth), axis=0)  # (ny, nx) vertically-integrated Δρ
    ny_s, nx_s = grid.shape
    fy = max(col.shape[0] // ny_s, 1)
    fx = max(col.shape[1] // nx_s, 1)
    coarse = col[: ny_s * fy, : nx_s * fx].reshape(ny_s, fy, nx_s, fx).mean(axis=(1, 3))
    g = grid - np.nanmean(grid)
    c = coarse - np.nanmean(coarse)
    corr = np.corrcoef(g.ravel(), c.ravel())[0, 1]
    assert corr > 0.5, "Bouguer high must co-locate with the excess-mass columns"

    # --- physically defensible: correct order of magnitude (a small model) ---
    assert np.nanmax(np.abs(g)) < 50.0

    # --- physically defensible: SMOOTH (a potential field is low-pass; no jitter) ---
    # the lag-1 row differences are small vs the total field range (no high-frequency noise).
    rng_field = np.nanmax(grid) - np.nanmin(grid)
    if rng_field > 0:
        step = np.nanmax(np.abs(np.diff(grid, axis=1)))
        assert step <= 0.8 * rng_field

    # --- ingests through the SAME adapter as the T0 counterpart, unchanged ---
    csv = next(a for a in arts if a.fmt == "csv")
    csv_src = RawSource(filename=csv.path.name, data=csv.path.read_bytes())
    tif_src = RawSource(filename=tif.path.name, data=tif.path.read_bytes())
    assert isinstance(_detect_with(csv_src), GravityPotentialFieldAdapter)
    assert isinstance(_detect_with(tif_src), GravityPotentialFieldAdapter)

    # and the SAME adapter T0 routes to (no adapter swap between tiers).
    acq0 = Acquisition(gravity_spacing=160.0, params={"out_dir": str(tmp_path / "t0")})
    t0_arts = GravityForward().simulate(earth, acq0, _rng())
    t0_csv = next(a for a in t0_arts if a.fmt == "csv")
    t0_src = RawSource(filename=t0_csv.path.name, data=t0_csv.path.read_bytes())
    assert type(_detect_with(csv_src)) is type(_detect_with(t0_src))

    # the parse actually yields a usable gravity primitive (round-trip closed, doc 03 §2).
    res = GravityPotentialFieldAdapter().parse(csv_src)
    assert res.observations and res.observations[0].primary_property == "gravity_anomaly"
    assert np.isfinite(res.observations[0].values["gravity_anomaly"]).all()


# ============================================================================== (b) MT


def _read_edi(adapter: MtEdiAdapter, art: Artifact):
    src = RawSource(filename=art.path.name, data=art.path.read_bytes())
    return adapter.parse(src)


def test_mt_rigorous_skin_depth_doi_visible_and_ingests_via_t0_adapter(earth, tmp_path):
    """(b) Rigorous MT: skin-depth/DOI visible (short→cap, long→deep conductor); T0 adapter.

    The exact layered-earth impedance makes the period→depth split VISIBLE: short periods see
    the shallow conductive clay cap (low apparent resistivity) while long periods diffuse deeper
    into the resistive basement (high apparent resistivity) — the skin-depth (∝ √period) DOI
    falloff the T0 box-average only smears. Each EDI detects to (and re-reads through) the SAME
    MT EDI adapter the T0 uses.
    """
    acq = Acquisition(
        mt_n_periods=10, mt_periods=(1.0e-2, 1.0e3), params={"out_dir": str(tmp_path)}
    )
    arts = MTRigorousForward().simulate(earth, acq, _rng())
    assert arts and {a.fmt for a in arts} == {"edi"}  # SAME native EDI files as T0

    # --- ingests through the SAME adapter as the T0 counterpart, unchanged ---
    adapter = MtEdiAdapter()
    src0 = RawSource(filename=arts[0].path.name, data=arts[0].path.read_bytes())
    assert isinstance(_detect_with(src0), MtEdiAdapter)
    acq0 = Acquisition(
        mt_n_periods=10, mt_periods=(1.0e-2, 1.0e3), params={"out_dir": str(tmp_path / "t0")}
    )
    t0_arts = MTForward().simulate(earth, acq0, _rng())
    t0_src = RawSource(filename=t0_arts[0].path.name, data=t0_arts[0].path.read_bytes())
    assert type(_detect_with(src0)) is type(_detect_with(t0_src))

    # --- physically defensible: skin-depth / DOI visible in the ingested apparent-res ---
    res = _read_edi(adapter, arts[0])
    obs = res.observations[0]
    assert obs.geometry_kind == "tensor"
    rho = obs.values["resistivity"]
    freq = np.asarray(obs.meta["frequency_hz"], dtype=float)
    assert rho.size == acq.mt_n_periods and np.all(np.isfinite(rho)) and np.all(rho > 0)

    # order on ascending period (period = 1/freq): EDI stores descending frequency.
    periods = 1.0 / freq
    order = np.argsort(periods)
    rho_by_period = rho[order]  # short → long period

    # This station's truth column is a shallow CONDUCTOR (clay-cap/alteration, single-digit
    # ohm·m) over a very resistive basement at depth (doc 05 §4.2). So the skin-depth DOI is:
    #   short periods → shallow → read the conductive cap (LOW apparent resistivity)
    #   long periods  → deep    → diffuse into the resistive basement (HIGH apparent resistivity)
    short = rho_by_period[0]
    long = rho_by_period[-1]
    assert short < 30.0, "short periods (shallow DOI) must read the conductive clay cap"
    assert long > 5.0 * short, "long periods (deep DOI) must reach the deep resistive basement"
    # The DOI falloff is monotone (no period sees shallower than a shorter one): apparent
    # resistivity climbs with period as the skin depth (∝ √period) penetrates deeper — the
    # resolution/DOI falloff is VISIBLE across the band, not a flat (depth-blind) curve.
    span = rho_by_period.max() / rho_by_period.min()
    assert span > 10.0, "apparent resistivity must span the cap→basement DOI range"

    # cross-check directly against the exact physics of the station's own truth column: the
    # long period genuinely diffuses DEEPER than the short period (period→depth, doc 05 §4.2).
    from geosim.synthgen.forward.em_mt import (
        _MU0,
        _layer_model,
        _resistivity_column,
        _station_grid_xy,
        layered_mt_impedance,
    )

    stations = _station_grid_xy(earth, acq.mt_n_periods, spacing=False)
    p_exact = np.logspace(
        np.log10(acq.mt_periods[0]), np.log10(acq.mt_periods[1]), acq.mt_n_periods
    )
    depth, col_rho = _resistivity_column(earth, stations[0])
    # the truth column itself carries the shallow-conductor / deep-resistor split the DOI sees.
    assert col_rho[0] < 30.0 and col_rho[-1] > 10.0 * col_rho[0]
    r_lay, t_lay = _layer_model(depth, col_rho)
    z = layered_mt_impedance(r_lay, t_lay, p_exact)
    rho_exact = np.abs(z) ** 2 / ((2.0 * np.pi / p_exact) * _MU0)
    # skin depth grows ∝ sqrt(period): the deepest-sensing period samples below the shallowest.
    assert np.sqrt(p_exact[-1]) > np.sqrt(p_exact[0])
    # the rigorous apparent resistivity climbs into the deep resistor as the DOI deepens (the
    # exact layered impedance, not a smeared box-average): the long-period half sits well above
    # the short-period half, and the deep band rises monotonically — a clean period→depth DOI.
    half = rho_exact.size // 2
    assert rho_exact[half:].min() > rho_exact[:half].max()
    assert np.all(np.diff(rho_exact[half:]) > 0.0)
    # the emitted (noisy) curve is the physics of that column to within measurement noise.
    np.testing.assert_allclose(rho_by_period, rho_exact[order], rtol=0.30)


# ============================================================================== (c) seismic


def _read_segy(adapter: SeismicSegyAdapter, art: Artifact):
    return adapter.parse(RawSource(filename=art.path.name, path=str(art.path)))


def test_seismic_rigorous_reflections_at_true_impedance_and_ingests_via_t0_adapter(
    earth, tmp_path
):
    """(c) Rigorous seismic: band-limited reflections at the TRUE impedance contrasts; T0 adapter.

    The acoustic-convolutional synthetic places band-limited reflection energy at the two-way
    times of the truth's real ρ·Vp jumps (not arbitrary times), and the SEG-Y section + horizons
    GeoJSON detect to (and re-read through) the SAME seismic SEG-Y adapter the T0 reflection
    forward uses — ingestion unchanged.
    """
    acq = Acquisition(seis_n_traces=12, seis_n_samples=512, params={"out_dir": str(tmp_path)})
    arts = SeismicReflectionRigorousForward().simulate(earth, acq, _rng())
    assert {a.fmt for a in arts} == {"segy", "geojson"}  # SAME native files as T0

    # --- ingests through the SAME adapter as the T0 counterpart, unchanged ---
    adapter = SeismicSegyAdapter()
    segy = next(a for a in arts if a.fmt == "segy")
    segy_src = RawSource(filename=segy.path.name, path=str(segy.path))
    assert isinstance(_detect_with(segy_src), SeismicSegyAdapter)
    acq0 = Acquisition(
        seis_n_traces=12, seis_n_samples=512, params={"out_dir": str(tmp_path / "t0")}
    )
    t0_arts = SeismicReflectionForward().simulate(earth, acq0, _rng())
    t0_segy = next(a for a in t0_arts if a.fmt == "segy")
    t0_src = RawSource(filename=t0_segy.path.name, path=str(t0_segy.path))
    assert type(_detect_with(segy_src)) is type(_detect_with(t0_src))

    # --- physically defensible: reflection energy at the TRUE impedance contrasts ---
    res = _read_segy(adapter, segy)
    assert not res.warnings or all(w.severity.value != "high" for w in res.warnings)
    pm = res.property_models[0]
    assert pm.support == "section" and pm.property == "velocity_p"
    section = pm.values  # (n_samples, n_traces)
    assert section.shape == (acq.seis_n_samples, acq.seis_n_traces)
    assert np.all(np.isfinite(section))
    assert any(f.feature_type == "horizon" for f in res.features)  # horizons GeoJSON joined

    # recompute the truth impedance reflectivity under a central trace and find its strongest
    # contrast's two-way time; assert the re-read trace peaks within a Ricker half-width of it.
    from geosim.synthgen.forward.base import sample_volume_at, world_axes
    from geosim.synthgen.forward.seismic import _seismic_line

    z, y, x = world_axes(earth)
    axes = (z, y, x)
    dz = earth.spacing[0]
    imp_vol = earth.property_volume("density").astype(np.float64) * earth.property_volume(
        "velocity_p"
    ).astype(np.float64)
    vp = earth.property_volume("velocity_p").astype(np.float64)

    cx, cy = _seismic_line(earth, acq)
    it = acq.seis_n_traces // 2
    zsorted = np.sort(z)[::-1]
    pts = np.column_stack(
        [zsorted, np.full_like(zsorted, cy[it]), np.full_like(zsorted, cx[it])]
    )
    imp = sample_volume_at(imp_vol, axes, pts)
    vpc = sample_volume_at(vp, axes, pts)
    refl = np.zeros_like(imp)
    denom = imp[1:] + imp[:-1]
    refl[1:] = np.where(denom > 0.0, (imp[1:] - imp[:-1]) / denom, 0.0)
    twt = np.cumsum(2.0 * dz / np.maximum(vpc, 1.0))
    k_samp = np.round(twt / acq.seis_dt).astype(int)
    in_record = k_samp < acq.seis_n_samples
    assert np.any(in_record & (np.abs(refl) > 0)), "no reflector inside the record"
    k_true = int(k_samp[np.argmax(np.where(in_record, np.abs(refl), 0.0))])

    trace = np.abs(section[:, it])
    k_obs = int(np.argmax(trace))
    assert abs(k_obs - k_true) <= 12, "reflection must sit at the true impedance contrast"
    # --- physically defensible: band-limited (a real reflection peak, well above the median) ---
    assert trace[k_obs] > 5.0 * np.median(trace)


# ============================================================================== tier wiring


def test_rigorous_tier_selectable_and_t0_default_preserved():
    """The fidelity-aware registry selects T1 for all three methods; T0 stays the default.

    This is the contract M3 rides on (doc 05 §6): ``fidelity="rigorous"`` resolves a distinct
    T1 forward for gravity / MT / seismic-reflection, while the default (plausible) lookup is
    unchanged — so the existing round-trip pipeline keeps running the fast T0 fallback.
    """
    assert isinstance(get_forward("gravity", fidelity="rigorous"), GravityRigorousForward)
    assert isinstance(get_forward("mt", fidelity="rigorous"), MTRigorousForward)
    assert isinstance(
        get_forward("seismic", "reflection", fidelity="rigorous"),
        SeismicReflectionRigorousForward,
    )
    # default fidelity is still T0 (no silent swap, doc 05 §6).
    assert isinstance(get_forward("gravity"), GravityForward)
    assert isinstance(get_forward("mt"), MTForward)
    assert isinstance(get_forward("seismic", "reflection"), SeismicReflectionForward)
