"""Tests for the synthgen core ground-truth earth + rock physics (doc 05 §2–§3).

Everything runs against small truth grids and local temp dirs (``tmp_path``) — no
Docker/Postgres/Redis. The headline assertion is the doc-05 §1 invariant
"one geology → all properties": a single hydrothermal anomaly drives *every* property
consistently — its voxels are simultaneously hotter, more conductive, less dense, slower
(Vp), and magnetically suppressed than background.
"""

import json

import numpy as np
import pytest

from geosim.spatial import REGISTRY, convert
from geosim.storage import open_property_model
from geosim.synthgen import (
    AnomalySpec,
    FaultSpec,
    FrameSpec,
    GeothermSpec,
    LayerSpec,
    SceneSpec,
    SurfaceSpec,
    build_resistivity_volume,
    compile_scene,
    get_ruleset,
    load_scene,
    strip_jsonc,
    write_truth_bundle,
)

# --------------------------------------------------------------- a tiny scene


def _tiny_scene(seed: int = 7) -> SceneSpec:
    """A SMALL (24×24×16) Basin-&-Range-flavoured scene with a fault-controlled plume."""
    return SceneSpec(
        id="tiny-v1",
        seed=seed,
        frame=FrameSpec(
            xmin=-600, xmax=600, ymin=-600, ymax=600,
            zmin=-1200, zmax=400, dx=50, dy=50, dz=100,
        ),  # → (nz=16, ny=24, nx=24)
        surface=SurfaceSpec(kind="flat", base_elev=400.0),
        layers=(
            LayerSpec("alluvium", "surface", (100.0, 200.0)),
            LayerSpec("volcanics", "conformable", (200.0, 400.0)),
            LayerSpec("basement_granite", "conformable", "fill"),
        ),
        faults=(
            FaultSpec("range-front", trace=((-600, -300), (600, 100)),
                      kind="normal", dip=60, dip_azimuth=90, throw=300, is_conduit=True),
        ),
        geotherm=GeothermSpec(surface_temp=15.0, gradient=45.0),
        anomalies=(
            AnomalySpec(
                "upflow", footprint_center=(0.0, 0.0), footprint_radius_xy=400.0,
                top_elev=100.0, bottom_elev=-1000.0, controlled_by="range-front",
                temp_peak=220.0, alteration_frac=0.6, porosity_boost=0.04,
                salinity_tds=8000.0, fracture_density=0.5,
                clay_cap_top_elev=50.0, clay_cap_thickness=200.0,
            ),
        ),
        rock_physics="default-v1",
    )


def _anomaly_and_background_masks(earth):
    """High-alteration anomaly voxels vs. (sub-surface) background voxels."""
    alt = earth.state.alteration_fraction
    anomaly = alt > 0.3
    background = (~anomaly) & (~earth.above_surface)
    # ensure both populations are non-trivial for a meaningful comparison
    assert anomaly.sum() > 5
    assert background.sum() > 50
    return anomaly, background


# --------------------------------------------------------------- shape / grid


def test_truth_grid_shape_and_axis_order():
    earth = compile_scene(_tiny_scene())
    assert earth.shape == (16, 24, 24)  # (nz, ny, nx) Z-up (doc 02 §10.2)
    # origin/spacing are (z,y,x) Engineering m (doc 02 §10.2)
    assert earth.spacing == (100.0, 50.0, 50.0)
    assert earth.origin[0] == pytest.approx(-1200 + 50.0)  # zmin + dz/2 (cell centre)


def test_all_property_volumes_present_finite_and_colocated():
    earth = compile_scene(_tiny_scene())
    keys = ["density", "susceptibility", "resistivity", "chargeability_mv_v",
            "velocity_p", "velocity_s", "temperature", "porosity"]
    for k in keys:
        vol = earth.property_volume(k)
        assert vol.shape == earth.shape, k
        assert np.isfinite(vol).all(), f"{k} has non-finite values"
        assert vol.dtype == np.float32


# --------------------------------------------- THE invariant: one geology → all props


def test_one_geology_drives_all_properties_consistently():
    """doc 05 §1 decision #1 + §4.2: the SAME anomaly voxels are hotter, more
    conductive, less dense, slower (Vp), and magnetically suppressed."""
    earth = compile_scene(_tiny_scene())
    anomaly, background = _anomaly_and_background_masks(earth)

    res = earth.property_volume("resistivity")
    temp = earth.property_volume("temperature")
    rho = earth.property_volume("density")
    vp = earth.property_volume("velocity_p")
    chi = earth.property_volume("susceptibility")

    # conductive: anomaly resistivity well below background (doc 05 §3.1 Archie+clay)
    assert np.median(res[anomaly]) < 0.25 * np.median(res[background])
    # hotter (doc 05 §2.3 plume blend)
    assert np.median(temp[anomaly]) > np.median(temp[background])
    # less dense OR slower Vp — porosity/fracture softening (doc 05 §3.1)
    assert (np.median(rho[anomaly]) < np.median(rho[background])) or (
        np.median(vp[anomaly]) < np.median(vp[background])
    )
    # magnetic low: alteration destroys magnetite (doc 05 §3.1, §4.2)
    assert np.median(chi[anomaly]) < np.median(chi[background])


def test_temperature_is_canonical_kelvin():
    earth = compile_scene(_tiny_scene())
    temp = earth.property_volume("temperature")
    # background subsurface temps are absolute kelvin (≥ ~288 K = 15 °C surface)
    sub = temp[~earth.above_surface]
    assert sub.min() > 250.0  # kelvin, not °C (doc 01 §5)
    # plume core approaches ~220 °C = 493.15 K
    assert temp.max() > convert(150.0, "degC", "kelvin")


def test_units_are_registry_canonical(tmp_path):
    # every derived property is written under its registry-canonical unit (doc 01 §5)
    earth = compile_scene(_tiny_scene())
    out = write_truth_bundle(earth, tmp_path / "truth", overwrite=True)
    for k in ["density", "resistivity", "temperature", "velocity_p", "susceptibility"]:
        attrs = open_property_model(out / f"{k}.zarr").attrs(k)
        assert attrs["canonicalUnit"] == REGISTRY.get(k).canonical_unit
    # temperature canonical is kelvin; resistivity ohm*m (doc 01 §5)
    assert REGISTRY.get("temperature").canonical_unit == "kelvin"
    assert REGISTRY.get("resistivity").canonical_unit == "ohm*m"


# --------------------------------------------------------------- determinism


def test_deterministic_from_spec_and_seed():
    spec = _tiny_scene(seed=11)
    e1 = compile_scene(spec)
    e2 = compile_scene(spec)
    assert np.array_equal(e1.lithology, e2.lithology)
    for k in ["resistivity", "density", "temperature", "velocity_p"]:
        assert np.array_equal(e1.property_volume(k), e2.property_volume(k)), k


def test_different_seed_changes_texture_not_structure():
    e1 = compile_scene(_tiny_scene(seed=1))
    e2 = compile_scene(_tiny_scene(seed=2))
    # the resistivity realisations differ (seeded texture, doc 05 §2.3)
    assert not np.array_equal(e1.property_volume("resistivity"), e2.property_volume("resistivity"))


# --------------------------------------------------------------- lithology / faults


def test_lithology_layers_and_fault_offset():
    earth = compile_scene(_tiny_scene())
    # all three layer units appear in L (doc 05 §2.3 stacking)
    assert set(np.unique(earth.lithology)).issuperset({0, 1, 2})
    assert earth.unit_names[:3] == ["alluvium", "volcanics", "basement_granite"]
    # a normal fault with throw offsets blocks → the alluvium-top contact elevation
    # differs across the trace (i.e. L is not laterally constant at a fixed depth).
    mid_z = earth.shape[0] // 2
    layer_slice = earth.lithology[mid_z]
    assert layer_slice.min() != layer_slice.max()  # fault-offset structure present


def test_intrusion_overwrites_lithology():
    from geosim.synthgen import IntrusionSpec

    base = _tiny_scene()
    spec = SceneSpec(
        id=base.id, seed=base.seed, frame=base.frame, surface=base.surface,
        layers=base.layers, faults=base.faults, geotherm=base.geotherm,
        anomalies=base.anomalies, rock_physics=base.rock_physics,
        intrusions=(IntrusionSpec("young_intrusive", center=(0.0, 0.0, -800.0),
                                  radius_xy=300.0, radius_z=300.0),),
    )
    earth = compile_scene(spec)
    assert "young_intrusive" in earth.unit_names
    idx = earth.unit_names.index("young_intrusive")
    assert (earth.lithology == idx).any()


# --------------------------------------------------------------- rock physics rule


def test_ruleset_is_pure_and_named():
    rs = get_ruleset("default-v1")
    assert rs.name == "default-v1"
    with pytest.raises(KeyError):
        get_ruleset("does-not-exist")


def test_archie_resistivity_drops_with_salinity_and_temperature():
    """Hot + saline → more conductive (doc 05 §3.1 Arps ρw + Archie)."""
    from geosim.synthgen import UnitProps

    rs = get_ruleset("default-v1")
    shape = (1, 1, 1)
    base = dict(
        unit_index=np.zeros(shape, dtype=np.int32),
        units=[UnitProps(rho=2400, chi=0.01, vp=3000, phi=0.15)],
        porosity_state=np.zeros(shape),
        water_saturation=np.ones(shape),
        alteration_frac=np.zeros(shape),
        fracture_density=np.zeros(shape),
    )
    cold_fresh = rs.apply(temperature_k=np.full(shape, 290.0),
                          salinity_tds=np.full(shape, 500.0), **base)
    hot_saline = rs.apply(temperature_k=np.full(shape, 490.0),
                          salinity_tds=np.full(shape, 8000.0), **base)
    assert hot_saline.resistivity[0, 0, 0] < cold_fresh.resistivity[0, 0, 0]


# --------------------------------------------------------------- JSONC scene loading


def test_strip_jsonc_removes_comments_and_trailing_commas():
    src = """{
      // a line comment with // inside
      "a": 1, /* block */ "b": "http://keep//slashes",
      "c": [1, 2,],
    }"""
    parsed = json.loads(strip_jsonc(src))
    assert parsed == {"a": 1, "b": "http://keep//slashes", "c": [1, 2]}


def test_load_scene_from_jsonc_string_compiles():
    jsonc = """{
      "id": "jc-v1", "seed": 5,
      "frame": { "roi": {"xmin":-300,"xmax":300,"ymin":-300,"ymax":300},
                 "depthRange": {"zmin":-600,"zmax":200},
                 "truthGrid": {"dx":50,"dy":50,"dz":100} },
      "surface": { "kind": "fractal", "baseElev": 200, "relief": 120, "roughness": 0.7 },
      "layers": [
        { "unit":"alluvium", "top":"surface", "thickness":[100,150] },
        { "unit":"basement_granite", "thickness":"fill" }
      ],
      "geotherm": { "surfaceTemp": 15, "gradient": 45 },
      "anomalies": [
        { "id":"a", "kind":"hydrothermal-plume",
          "footprint": {"center":[0,0], "radiusXY":200}, "topElev":100, "bottomElev":-400,
          "perturb": {"tempPeak":210,"alterationFrac":0.6,"salinityTDS":8000} }
      ],
      "rockPhysics": "default-v1",
    }"""
    spec = load_scene(jsonc)
    assert spec.id == "jc-v1" and spec.seed == 5
    assert spec.frame.shape == (8, 12, 12)
    earth = compile_scene(spec)
    assert earth.shape == (8, 12, 12)
    assert np.isfinite(earth.property_volume("resistivity")).all()


# --------------------------------------------------------------- truth bundle writer


def test_write_truth_bundle_roundtrips_via_storage(tmp_path):
    earth = compile_scene(_tiny_scene())
    out = write_truth_bundle(earth, tmp_path / "truth", overwrite=True)

    # features.geojson holds the true fault + anomaly (doc 05 §5)
    fc = json.loads((out / "features.geojson").read_text())
    kinds = {f["properties"]["kind"] for f in fc["features"]}
    assert {"fault", "anomaly"} <= kinds

    # each derived property is a readable PropertyModel with co-registered sigma
    reader = open_property_model(out / "resistivity.zarr")
    assert reader.properties == ["resistivity"]
    assert reader.has_sigma("resistivity")
    vol = reader.read_level("resistivity", 0)
    assert vol.shape == earth.shape
    np.testing.assert_allclose(vol, earth.property_volume("resistivity"), rtol=1e-5)

    # canonical units + Engineering origin/spacing round-trip (doc 02 §10.2)
    attrs = reader.attrs("resistivity")
    assert attrs["canonicalUnit"] == "ohm*m"
    assert attrs["origin"] == list(earth.origin)
    assert attrs["spacing"] == list(earth.spacing)

    # temperature stored in canonical kelvin
    tattrs = open_property_model(out / "temperature.zarr").attrs("temperature")
    assert tattrs["canonicalUnit"] == "kelvin"

    # lithology label volume + unit index written
    assert (out / "lithology.zarr").exists()
    units_map = json.loads((out / "lithology_units.json").read_text())
    assert units_map["0"] == "alluvium"


# --------------------------------------------------------------- M1 helper still works


def test_m1_resistivity_helper_still_works():
    vr = build_resistivity_volume(shape=(8, 8, 8), seed=1)
    assert vr.values.shape == (8, 8, 8)
    assert np.isfinite(vr.values).all()
    assert vr.canonical_unit == "ohm*m"
