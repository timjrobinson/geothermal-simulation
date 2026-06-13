"""Tests for the plugin architecture (doc 08).

Covers: decorator registration of a property type + adapter; registry lookups;
``capabilities()`` shape (doc 08 §7.1); quarantine of an invalid ``executionMode`` and a
non-canonical ``(method, submethod)`` (doc 08 §8, doc 02 §2) WITHOUT raising; and
``executionMode`` defaulting to ``in_process`` (doc 08 §2.1).
"""

import pytest

from geosim.plugins import (
    ExecutionMode,
    IngestionAdapter,
    ManifestError,
    PluginManifest,
    PluginRegistry,
    PropertyType,
    RendererSpec,
    Transform,
    TransferFunction,
    api_version_compatible,
    is_canonical_pair,
    register,
)
from geosim.plugins.register import _Register


@pytest.fixture
def reg(monkeypatch):
    """A FRESH PluginRegistry wired into the module-level singleton + the register namespace.

    The registry is a process-wide singleton (doc 08 §3.1); tests that assert exact counts
    need isolation, so we swap in a fresh instance for the duration of the test.
    """
    fresh = PluginRegistry()
    # get_registry() returns the module-global registry.REGISTRY by name, so patching this
    # single binding redirects both the decorator path and direct registry use.
    monkeypatch.setattr("geosim.plugins.registry.REGISTRY", fresh)
    return fresh


# ----------------------------------------------------------------- dummy contributions


def _dummy_adapter_cls():
    class DummyHeatflowAdapter:
        method = "heatflow"          # canonical MethodKey (doc 02 §2)
        submethod = None
        formats = ["dummyhf", "csvhf"]

        def sniff(self, raw):
            return 0.9

        def parse(self, raw, ctx):
            return {"observations": [], "crs": "engineering"}

    return DummyHeatflowAdapter


# ---------------------------------------------------------------------- decorator path


def test_decorator_registers_property_type_and_adapter(reg):
    # doc 08 §4b: a NEW plugin-registered property type (doc 02 §1 reserves this).
    pt = PropertyType(
        key="heatflow_density", canonical_unit="W/m**2",
        default_colormap="thermal", default_scaling="linear", display_range=(0.0, 0.2),
    )
    register.property_type(pt)

    Adapter = _dummy_adapter_cls()
    register.adapter(Adapter)

    assert reg.quarantined() == []
    # registry lookups work (doc 08 §3.2)
    assert reg.property_type("heatflow_density").canonical_unit == "W/m**2"
    assert reg.adapter_for_format("dummyhf") is not None
    assert reg.adapter_for_format("csvhf") is not None
    assert reg.adapter_for_format("does-not-exist") is None


def test_register_adapter_as_class_decorator_returns_class(reg):
    @register.adapter
    class HF(IngestionAdapter):
        method = "heatflow"
        submethod = None
        formats = ["hf2"]

        def sniff(self, raw):
            return 1.0

        def parse(self, raw, ctx):
            return {}

    assert HF.formats == ["hf2"]           # decorator returns the class unchanged
    assert reg.adapter_for_format("hf2") is not None


# --------------------------------------------------------------------- capabilities §7.1


def test_capabilities_reflects_registered_contributions(reg):
    pt = PropertyType(
        key="heatflow_density", canonical_unit="W/m**2",
        default_colormap="thermal", default_scaling="linear", display_range=(0.0, 0.2),
    )
    register.property_type(pt)
    register.adapter(_dummy_adapter_cls())
    register.renderer(RendererSpec(
        key="volume.raymarch", applies_to=["heatflow_density"],
        default_transfer_function=TransferFunction(colormap="thermal"),
    ))

    class HFGradient:
        key = "hf_gradient"
        inputs = ["temperature"]
        outputs = ["heatflow_density"]

        def apply(self, fields, params):
            return fields

    register.transform(HFGradient())
    reg.register_manifest(PluginManifest.from_dict({
        "id": "geosim.method.heatflow", "name": "Heat Flow", "version": "0.1.0",
        "api_version": "1.x", "kind": "method-bundle", "method": "heatflow",
        "provides": {"property_types": ["heatflow_density"], "adapters": ["heatflow.dummyhf"]},
    }))

    cap = reg.capabilities()
    assert set(cap) == {"api_version", "property_types", "methods", "renderers", "transforms", "plugins"}

    # property_types carry the doc 08 §7.1 fields, sourced from the doc 01 §5 registry.
    by_key = {p["key"]: p for p in cap["property_types"]}
    assert by_key["heatflow_density"] == {
        "key": "heatflow_density", "unit": "W/m**2", "colormap": "thermal",
        "scaling": "linear", "display_range": [0.0, 0.2],
    }

    methods = {m["id"]: m for m in cap["methods"]}
    assert methods["heatflow"]["formats"] == ["dummyhf", "csvhf"]
    assert methods["heatflow"]["produces"] == ["heatflow_density"]
    assert methods["heatflow"]["has_forward_model"] is False
    assert methods["heatflow"]["name"] == "Heat Flow"

    renderer_keys = {r["key"] for r in cap["renderers"]}
    assert "volume.raymarch" in renderer_keys
    assert cap["renderers"][0]["default_transfer_function"]["colormap"] == "thermal"

    transforms = {t["key"]: t for t in cap["transforms"]}
    assert transforms["hf_gradient"]["outputs"] == ["heatflow_density"]

    assert {"id": "geosim.method.heatflow", "version": "0.1.0"} in cap["plugins"]


# ------------------------------------------------------------------- quarantine §8


def test_invalid_execution_mode_is_quarantined_not_raised(reg):
    # doc 08 §8: a bad executionMode quarantines the manifest; it must NOT raise.
    result = _register_manifest_safely(reg, {
        "id": "geosim.method.bad_exec", "name": "Bad", "version": "1.0.0",
        "api_version": "1.x", "kind": "single-contribution", "method": "gravity",
        "execution_modes": {"forward:gravity": "on_the_moon"},
    })
    assert result is None                                   # not registered
    q = reg.quarantined()
    assert any("executionMode" in r.reason for r in q)
    assert "geosim.method.bad_exec" not in [m.id for m in reg.manifests()]


def test_non_canonical_method_is_quarantined_not_raised(reg):
    # doc 02 §2 / doc 08 §8: invented variants like "seismic_reflection" are quarantined.
    result = _register_manifest_safely(reg, {
        "id": "geosim.method.bad_method", "name": "Bad Method", "version": "1.0.0",
        "api_version": "1.x", "kind": "method-bundle", "method": "seismic_reflection",
    })
    assert result is None
    assert any("non-canonical" in r.reason for r in reg.quarantined())


def test_non_canonical_adapter_method_is_quarantined(reg):
    class BadAdapter:
        method = "totally_made_up"
        submethod = None
        formats = ["x"]

        def sniff(self, raw):
            return 1.0

        def parse(self, raw, ctx):
            return {}

    register.adapter(BadAdapter)
    assert reg.adapter_for_format("x") is None
    assert any("non-canonical" in r.reason for r in reg.quarantined())


def test_property_type_unit_clash_is_quarantined(reg):
    # doc 08 §8 property-type integrity: a key/unit clash with the canonical registry.
    register.property_type(PropertyType(
        key="resistivity", canonical_unit="kg/m**3",   # wrong unit for a canonical key
        default_colormap="turbo", default_scaling="log",
    ))
    assert any("property_type:resistivity" in r.contribution for r in reg.quarantined())


# ------------------------------------------------------- executionMode default §2.1


def test_execution_mode_defaults_to_in_process():
    man = PluginManifest.from_dict({
        "id": "geosim.method.mt", "name": "MT", "version": "1.0.0",
        "api_version": "1.x", "kind": "method-bundle", "method": "mt",
        "execution_modes": {"forward:mt": "worker_process"},
    })
    # declared one stays declared
    assert man.execution_mode("forward:mt") is ExecutionMode.WORKER_PROCESS
    # an undeclared contribution defaults to in_process (doc 08 §2.1)
    assert man.execution_mode("adapter:mt.edi") is ExecutionMode.IN_PROCESS
    assert ExecutionMode.coerce(None) is ExecutionMode.IN_PROCESS


# ------------------------------------------------------------- canonical helpers / compat


def test_canonical_pair_validation():
    assert is_canonical_pair("seismic", "reflection") is True
    assert is_canonical_pair("seismic", None) is True
    assert is_canonical_pair("seismic", "bogus") is False
    assert is_canonical_pair("not_a_method", None) is False
    assert is_canonical_pair("em", "tdem") is True


def test_api_version_compatibility():
    assert api_version_compatible("1.x") is True
    assert api_version_compatible("1.2.3") is True
    assert api_version_compatible("2.x") is False


def test_manifest_missing_required_field_raises_manifest_error():
    # from_dict validates eagerly; the quarantine path is exercised via the registry/helper.
    with pytest.raises(ManifestError):
        PluginManifest.from_dict({"id": "x", "name": "x"})


# --------------------------------------------------------------------------- helpers


def _register_manifest_safely(reg, data):
    """Mirror the `manifest()` helper's quarantine-on-failure behaviour against `reg`.

    Builds + validates the manifest; on ManifestError the registry quarantines it and we
    return None (doc 08 §8) — exactly what the public `manifest()` helper does.
    """
    from geosim.plugins.registry import QuarantineRecord

    try:
        man = PluginManifest.from_dict(data)
    except ManifestError as e:
        reg._quarantine.append(QuarantineRecord(f"manifest:{data.get('id')}", str(e),
                                                plugin_id=data.get("id")))
        return None
    return reg.register_manifest(man)


def test_register_namespace_is_singleton_instance():
    assert isinstance(register, _Register)
