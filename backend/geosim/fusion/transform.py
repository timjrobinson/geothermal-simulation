"""Rock-physics transform engine — declarative spec + pure ``apply()`` (doc 07 §4).

A **transform** maps one or more input property volumes → an **output derived volume**,
applying a rock-physics relationship that moves from *what geophysics measures* toward
*what geothermal cares about* (temperature, fluid/permeability, alteration, fracture
density — doc 07 §4). A transform is a **declarative spec** (metadata, typed I/O contract,
tunable params, stated assumptions + calibration status) **+ a pure Python ``apply()``**
(the math); the spec makes it discoverable/parameterizable/versionable, the function does
the physics (doc 07 §4.1).

This module owns:

- :class:`InputSpec` / :class:`OutputSpec` / :class:`Param` — the typed I/O + parameter
  contract (doc 07 §4.1, §4.7); :class:`Transform` — the base class plugins subclass and
  register via :func:`geosim.plugins.register.transform` (doc 08 §4c). The base exposes
  the doc-08 registry shape (``key``/``inputs``/``outputs``/``apply``) AND the rich doc-07
  spec (``id``/``version``/``title``/``target``/``output``/``params``/``assumptions``/
  ``calibration_status``).
- :class:`TransformContext` — the ``ctx`` handed to ``apply()`` (carries the params and an
  :meth:`~TransformContext.as_output` convenience).
- :func:`run_transform` — the common **execution harness** (doc 07 §4.5): resolve inputs to
  a fused grid (auto-resample via :mod:`geosim.fusion` if needed), unit-check/convert each
  input to its declared unit (:mod:`geosim.spatial`), build the valid mask = AND of input
  coverage masks (NaN where any *required* input is missing — never zero-fill), apply
  vectorized over valid cells, clamp to ``valid_range`` (flagging out-of-range), **propagate
  σ** (delta-method finite-difference Jacobian by default; opt-in Monte-Carlo), stamp
  calibration honesty (uncalibrated ⇒ output retitled ``"… likelihood"`` + ``tier="proxy"``),
  and write the output + paired σ as a derived :class:`~geosim.catalog.PropertyModel` with
  the doc-07 §4.3 derivation-provenance block.

**Canonical temperature is KELVIN end-to-end** (doc 01 §5); °C is display-only. Derived
volumes are ordinary :class:`~geosim.catalog.PropertyModel`\\s (doc 07 §4.3) stored via
:mod:`geosim.storage` and catalogued with full provenance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from geosim.catalog import (
    Dataset,
    FusedModel,
    IdKind,
    PropertyModel,
    Provenance,
    ProvenanceInput,
    new_id,
)
from geosim.spatial import REGISTRY
from geosim.spatial.units import convert
from geosim.storage import GridSpec, ProjectLayout, write_property_model

from .grid import FusedGrid, fused_grid_from_row, open_fused_group
from .resample import resample_to_fused

__all__ = [
    "InputSpec",
    "OutputSpec",
    "Param",
    "Transform",
    "TransformContext",
    "TransformResult",
    "CALIBRATION_STATUSES",
    "TIER_ORDER",
    "run_transform",
]

# doc 07 §4.1 — every transform declares one of these calibration states; an uncalibrated
# transform's output is a LIKELIHOOD/PROXY field, never a deterministic measurement.
CALIBRATION_STATUSES = ("uncalibrated", "well_calibrated", "lab_calibrated")

# doc 02 §6 / doc 07 §5.1 — UncertaintySpec.tier ordered worst→best; the output tier is the
# MIN over inputs, then capped by the transform's calibration_status.
TIER_ORDER = ("unknown", "qualitative", "proxy", "quantitative")

# An uncalibrated transform can never output better than proxy (doc 07 §5.1).
_CALIBRATION_TIER_CAP = {
    "uncalibrated": "proxy",
    "well_calibrated": "quantitative",
    "lab_calibrated": "quantitative",
}


# ──────────────────────────────────────────────────────────────────────────
# declarative I/O + parameter contract (doc 07 §4.1)
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class InputSpec:
    """A typed transform input (doc 07 §4.1, §4.7).

    ``name`` is the property-type key consumed; ``unit`` is the unit ``apply()`` expects
    (the harness unit-checks + converts each input to it via :mod:`geosim.spatial`).
    ``required`` inputs gate the valid mask — a cell missing any required input is nodata.
    """

    name: str
    unit: str
    required: bool = True


@dataclass(frozen=True)
class OutputSpec:
    """The transform's output contract (doc 07 §4.1).

    ``valid_range`` (in ``unit``) clamps + flags out-of-range cells (doc 07 §4.5 step 5);
    ``proxy_when_uncalibrated`` declares that an uncalibrated run yields a likelihood/proxy
    field (doc 07 §4.1 — temperature etc.). ``colormap`` is a UI hint.
    """

    name: str
    unit: str
    valid_range: tuple[float, float] | None = None
    colormap: str = "viridis"
    proxy_when_uncalibrated: bool = True


@dataclass(frozen=True)
class Param:
    """A tunable transform parameter (doc 07 §4.1, first-class + user-tunable).

    ``sigma`` (optional 1σ) feeds the parameter-uncertainty term of σ propagation
    (doc 07 §5.2 — often the *dominant* term once calibrated, §4.8).
    """

    name: str
    type: type = float
    default: Any = None
    range: tuple[float, float] | None = None
    sigma: float | None = None


# ──────────────────────────────────────────────────────────────────────────
# transform base class (doc 07 §4.1; registers via doc 08 §4c)
# ──────────────────────────────────────────────────────────────────────────


class Transform:
    """Base class for a rock-physics transform (doc 07 §4.1).

    Subclasses set the class-level declarative spec (``id``/``version``/``title``/
    ``target``/``inputs``/``output``/``params``/``assumptions``/``calibration_status``)
    and implement the pure :meth:`apply`. The class also exposes the doc-08 plugin-registry
    shape so :func:`geosim.plugins.register.transform` accepts it directly:

    - ``key`` ← ``id`` (registry key, doc 08 §4c);
    - ``inputs`` is already a list of property-type keys (the :class:`InputSpec` ``name``s
      are surfaced via :meth:`input_keys`);
    - ``outputs`` ← ``[output.name]``.

    ``apply(self, ctx, **inputs, **params)`` is **pure math** — it receives each input as a
    NumPy array (already unit-converted by the harness) and each param as a scalar, and
    returns the output field array. All non-pure concerns (resampling, units, masking,
    clamping, σ, storage) live in :func:`run_transform`.
    """

    id: str = ""
    version: str = "0.0.0"
    title: str = ""
    target: str = ""
    inputs: list[InputSpec] = []
    output: OutputSpec = OutputSpec(name="", unit="dimensionless")
    params: list[Param] = []
    assumptions: list[str] = []
    calibration_status: str = "uncalibrated"

    # ---- doc 08 §4c registry shape (so register.transform accepts the class/instance) ----

    @property
    def key(self) -> str:
        """Registry key (doc 08 §4c) — the transform ``id``."""
        return self.id

    @property
    def input_keys(self) -> list[str]:
        """Property-type keys this transform consumes (doc 08 §4c ``inputs``)."""
        return [i.name for i in self.inputs]

    @property
    def outputs(self) -> list[str]:
        """Property-type keys this transform produces (doc 08 §4c ``outputs``)."""
        return [self.output.name]

    # ---- the pure physics (doc 07 §4.1) ----

    def apply(self, ctx: TransformContext, **fields: Any) -> np.ndarray:  # noqa: D401
        """Pure transform math: inputs + params → output field (doc 07 §4.1).

        ``fields`` carries each declared input (NumPy array, in its :class:`InputSpec`
        unit) and each declared param (scalar). Returns the output field array in the
        :class:`OutputSpec` unit. Subclasses override.
        """
        raise NotImplementedError

    def param_defaults(self) -> dict[str, Any]:
        """The declared default value of each param (doc 07 §4.1)."""
        return {p.name: p.default for p in self.params}

    def resolve_params(self, overrides: dict[str, Any] | None) -> dict[str, Any]:
        """Merge user ``overrides`` onto the declared defaults; clamp-check ranges.

        Unknown params are rejected; an out-of-range value is a hard error (a bad param is
        a common cause of out-of-range output, doc 07 §4.5 step 5).
        """
        merged = self.param_defaults()
        for name, value in (overrides or {}).items():
            if name not in merged:
                raise ValueError(f"unknown param {name!r} for transform {self.id!r}")
            merged[name] = value
        by_name = {p.name: p for p in self.params}
        for name, value in merged.items():
            spec = by_name[name]
            if spec.range is not None and isinstance(value, (int, float)):
                lo, hi = spec.range
                if not (lo <= float(value) <= hi):
                    raise ValueError(
                        f"param {name!r}={value} outside range [{lo}, {hi}] "
                        f"for transform {self.id!r}"
                    )
        return merged

    def describe(self) -> dict[str, Any]:
        """A registry-palette description of this transform (doc 07 §4.7, ``GET /transforms``)."""
        return {
            "id": self.id,
            "version": self.version,
            "title": self.title,
            "target": self.target,
            "inputs": [
                {"name": i.name, "unit": i.unit, "required": i.required} for i in self.inputs
            ],
            "output": {
                "name": self.output.name,
                "unit": self.output.unit,
                "valid_range": list(self.output.valid_range)
                if self.output.valid_range is not None
                else None,
                "colormap": self.output.colormap,
                "proxy_when_uncalibrated": self.output.proxy_when_uncalibrated,
            },
            "params": [
                {
                    "name": p.name,
                    "type": getattr(p.type, "__name__", str(p.type)),
                    "default": p.default,
                    "range": list(p.range) if p.range is not None else None,
                    "sigma": p.sigma,
                }
                for p in self.params
            ],
            "assumptions": list(self.assumptions),
            "calibration_status": self.calibration_status,
        }


# ──────────────────────────────────────────────────────────────────────────
# the ctx handed to apply() (doc 07 §4.1)
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class TransformContext:
    """The ``ctx`` an :meth:`Transform.apply` receives (doc 07 §4.1).

    Carries the resolved ``params`` (so a transform may read them off ``ctx`` as well as
    via kwargs) and the fused :class:`FusedGrid` for shape/coord-aware transforms.
    :meth:`as_output` is a convenience the example transforms use: it simply returns the
    array — units/masking/σ/tier are stamped by the harness, not the transform (doc 07
    §4.1 comment "carries units, masks nodata, σ + tier").
    """

    grid: FusedGrid
    params: dict[str, Any] = field(default_factory=dict)

    def as_output(self, field_array: np.ndarray) -> np.ndarray:
        """Return the output field unchanged (the harness owns units/mask/σ/tier)."""
        return np.asarray(field_array, dtype=float)


@dataclass
class TransformResult:
    """The result of a :func:`run_transform` run (doc 07 §4.3, §4.5).

    The derived value + paired σ are written as a :class:`~geosim.catalog.PropertyModel`;
    ``model_id`` / ``sigma_model_id`` reference them. ``title`` is the (possibly retitled)
    output name; ``tier`` is the displayed uncertainty tier; ``out_of_range_fraction``
    flags how much of the output was clamped (doc 07 §4.5 step 5).
    """

    transform_id: str
    transform_version: str
    output_property: str
    title: str
    tier: str
    calibration_status: str
    model_id: str
    sigma_model_id: str
    uncertainty_mode: str  # "delta" | "monte_carlo"
    out_of_range_fraction: float
    correlated_inputs: bool  # any input flagged correlated_unmodeled → σ is a lower bound
    n_valid: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "transform_id": self.transform_id,
            "transform_version": self.transform_version,
            "output_property": self.output_property,
            "title": self.title,
            "tier": self.tier,
            "calibration_status": self.calibration_status,
            "model_id": self.model_id,
            "sigma_model_id": self.sigma_model_id,
            "uncertainty_mode": self.uncertainty_mode,
            "out_of_range_fraction": self.out_of_range_fraction,
            "correlated_inputs": self.correlated_inputs,
            "n_valid": self.n_valid,
        }


# ──────────────────────────────────────────────────────────────────────────
# input resolution (auto-resample onto the fused grid, doc 07 §4.5 step 1)
# ──────────────────────────────────────────────────────────────────────────


def _layer_for_property(fem: FusedModel, prop: str):
    """The (last-written) resampled :class:`~geosim.catalog.FusedLayer` for ``prop``, if any."""
    layer = None
    for lay in fem.layers:
        if lay.property == prop:
            layer = lay
    return layer


def _resolve_input_layer(
    session: Session,
    fem: FusedModel,
    prop: str,
    requested: dict[str, str] | None,
    storage_root: str | Path | None,
):
    """Resolve an input property to a fused layer, auto-resampling if needed (doc 07 §4.5 step 1).

    If ``requested`` maps ``prop`` → a native PropertyModel id, that model is resampled in
    (idempotent via the §2.1 cache). Otherwise an already-resampled layer for ``prop`` is
    reused. Returns the :class:`~geosim.catalog.FusedLayer`.
    """
    if requested and prop in requested:
        resample_to_fused(session, fem, requested[prop], storage_root=storage_root)
        session.refresh(fem)
    layer = _layer_for_property(fem, prop)
    if layer is None:
        raise ValueError(
            f"input property {prop!r} is not resampled onto fused grid {fem.id!r}; "
            "pass its native propertyModel id in `inputs` so the harness can resample it"
        )
    return layer


def _read_array(group, name: str) -> np.ndarray:
    return np.asarray(group[name][...], dtype=float)


def _unit_convert(values: np.ndarray, src_unit: str, dst_unit: str) -> np.ndarray:
    """Unit-check + convert an input to its declared unit (doc 07 §4.5 step 2).

    A wrong-DIMENSION input raises (``pint`` ``DimensionalityError``) — a hard error per
    doc 07 §4.5. Same-dimension scale differences convert transparently.
    """
    if src_unit == dst_unit:
        return values
    return np.asarray(convert(values, src_unit, dst_unit), dtype=float)


# ──────────────────────────────────────────────────────────────────────────
# σ propagation (doc 07 §5.2)
# ──────────────────────────────────────────────────────────────────────────


def _delta_sigma(
    transform: Transform,
    ctx: TransformContext,
    flat_inputs: dict[str, np.ndarray],
    flat_sigmas: dict[str, np.ndarray],
    params: dict[str, Any],
    base_output: np.ndarray,
) -> np.ndarray:
    """First-order (delta-method) σ via a numeric finite-difference Jacobian (doc 07 §5.2).

    ``σ_y² ≈ Σᵢ (∂f/∂xᵢ)² · σ_xᵢ²  (+ Σ_θ param-σ terms)``. Each ∂f/∂xᵢ is a central
    finite difference so transform authors never hand-derive Jacobians — they just supply
    ``apply()`` (doc 07 §5.2). Param σ (declared on a :class:`Param`) contributes the same
    way and is often dominant once calibrated (doc 07 §4.8).
    """
    var = np.zeros_like(base_output, dtype=float)

    def _eval(over_inputs: dict[str, np.ndarray], over_params: dict[str, Any]) -> np.ndarray:
        return np.asarray(
            transform.apply(ctx, **over_inputs, **over_params), dtype=float
        )

    # Input terms.
    for name, x in flat_inputs.items():
        sx = flat_sigmas.get(name)
        if sx is None:
            continue
        h = _fd_step(x)
        plus = dict(flat_inputs)
        minus = dict(flat_inputs)
        plus[name] = x + h
        minus[name] = x - h
        dfdx = (_eval(plus, params) - _eval(minus, params)) / (2.0 * h)
        var += (dfdx * sx) ** 2

    # Parameter terms (doc 07 §5.2 — declared Param.sigma).
    for p in transform.params:
        if p.sigma is None or not isinstance(params.get(p.name), (int, float)):
            continue
        theta = float(params[p.name])
        h = _fd_step(np.asarray(theta))
        h = float(h) if np.ndim(h) == 0 else float(np.mean(h))
        if h == 0.0:
            continue
        plus = dict(params)
        minus = dict(params)
        plus[p.name] = theta + h
        minus[p.name] = theta - h
        dfdp = (_eval(flat_inputs, plus) - _eval(flat_inputs, minus)) / (2.0 * h)
        var += (dfdp * float(p.sigma)) ** 2

    return np.sqrt(var)


def _fd_step(x: np.ndarray) -> np.ndarray:
    """A relative central-difference step that is robust near zero (doc 07 §5.2)."""
    scale = np.maximum(np.abs(x), 1.0)
    return 1e-4 * scale


def _monte_carlo_sigma(
    transform: Transform,
    ctx: TransformContext,
    flat_inputs: dict[str, np.ndarray],
    flat_sigmas: dict[str, np.ndarray],
    params: dict[str, Any],
    *,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Monte-Carlo σ for strongly nonlinear transforms (doc 07 §5.2, opt-in / job-based).

    Samples each input ~ N(x, σ_x) and each σ-declared param ~ N(θ, σ_θ), pushes K samples
    through ``apply()``, and returns the per-cell output **mean** + **std**. This is the
    more-correct path for nonlinearities (e.g. Archie in φ); the delta method is the fast
    default (doc 07 §5.2, §5.5).
    """
    rng = np.random.default_rng(seed)
    any_input = next(iter(flat_inputs.values()))
    acc = np.zeros((n_samples, any_input.size), dtype=float)
    sigma_params = [p for p in transform.params if p.sigma is not None]
    for k in range(n_samples):
        sampled_inputs = {
            name: x + rng.normal(0.0, 1.0, size=x.shape) * flat_sigmas.get(name, 0.0)
            for name, x in flat_inputs.items()
        }
        sampled_params = dict(params)
        for p in sigma_params:
            if isinstance(params.get(p.name), (int, float)):
                sampled_params[p.name] = float(params[p.name]) + rng.normal(0.0, float(p.sigma))
        acc[k] = np.asarray(transform.apply(ctx, **sampled_inputs, **sampled_params), dtype=float)
    return acc.mean(axis=0), acc.std(axis=0)


# ──────────────────────────────────────────────────────────────────────────
# tier + honesty stamping (doc 07 §4.5 step 7, §5.1)
# ──────────────────────────────────────────────────────────────────────────


def _min_tier(tiers: list[str]) -> str:
    """The worst (minimum) tier over inputs (doc 07 §5.1 — tier is the min over inputs)."""
    if not tiers:
        return "unknown"
    return min(tiers, key=lambda t: TIER_ORDER.index(t) if t in TIER_ORDER else 0)


def _cap_tier(tier: str, calibration_status: str) -> str:
    """Cap the input-min tier by the transform's calibration_status (doc 07 §5.1)."""
    cap = _CALIBRATION_TIER_CAP.get(calibration_status, "proxy")
    if TIER_ORDER.index(tier) <= TIER_ORDER.index(cap):
        return tier
    return cap


def _layer_tier(layer) -> str:
    """The declared UncertaintySpec tier of a fused layer's source (doc 02 §6).

    The resample op may record a ``tier``; absent that we assume ``quantitative`` for a
    measured/ingested input (the transform's own calibration_status is the real gate).
    """
    try:
        op = json.loads(layer.resample_op_json)
    except (TypeError, ValueError):
        op = {}
    return op.get("tier", "quantitative")


def _layer_correlated(layer) -> bool:
    """Whether a layer's source declares ``independence='correlated_unmodeled'`` (doc 02 §6)."""
    try:
        op = json.loads(layer.resample_op_json)
    except (TypeError, ValueError):
        op = {}
    return op.get("independence") == "correlated_unmodeled"


# ──────────────────────────────────────────────────────────────────────────
# the execution harness (doc 07 §4.5)
# ──────────────────────────────────────────────────────────────────────────


def run_transform(
    session: Session,
    layout: ProjectLayout,
    fem: FusedModel,
    transform: Transform,
    *,
    inputs: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    uncertainty: str = "delta",
    mc_samples: int = 64,
    mc_seed: int = 0,
    created_by: str = "system:fusion",
    storage_root: str | Path | None = None,
    progress=None,
) -> TransformResult:
    """Run a transform over a fused grid → a derived PropertyModel + σ (doc 07 §4.5).

    The common harness so individual ``apply()`` functions stay pure math (doc 07 §4.5):

    1. **Resolve inputs** to the fused grid — auto-resample a named native model via §2 if
       it isn't on the grid yet (:func:`geosim.fusion.resample_to_fused`).
    2. **Unit-check & convert** each input to the transform's declared unit (``pint``,
       :mod:`geosim.spatial`); a wrong-dimension input is a hard error.
    3. **Build the valid mask** = AND of input coverage masks — a cell missing any
       *required* input is **nodata** (NaN), never zero-filled (doc 07 §2.3).
    4. **Vectorized apply** over the valid cells.
    5. **Clamp** to ``output.valid_range``; the out-of-range fraction is flagged.
    6. **Propagate σ** (doc 07 §5.2): delta-method finite-difference Jacobian by default, or
       opt-in Monte-Carlo (``uncertainty="monte_carlo"``) for nonlinear transforms.
    7. **Stamp calibration honesty** (doc 07 §4.5 step 7): if ``uncalibrated`` the output is
       retitled ``"<target> likelihood"`` and ``tier="proxy"``. Tier = min over inputs,
       capped by the transform's ``calibration_status``.
    8. **Write** the output + paired σ as a derived PropertyModel (doc 04) carrying the §4.3
       derivation-provenance block.
    """
    if uncertainty not in ("delta", "monte_carlo"):
        raise ValueError(f"uncertainty must be 'delta' or 'monte_carlo'; got {uncertainty!r}")
    if transform.calibration_status not in CALIBRATION_STATUSES:
        raise ValueError(
            f"transform {transform.id!r} has invalid calibration_status "
            f"{transform.calibration_status!r}"
        )

    grid = fused_grid_from_row(fem)
    resolved_params = transform.resolve_params(params)
    ctx = TransformContext(grid=grid, params=resolved_params)

    if progress is not None:
        progress.report(0.05, "resolving inputs")

    # 1) Resolve every declared input to a fused layer (auto-resample as needed).
    layers: dict[str, Any] = {}
    for spec in transform.inputs:
        layers[spec.name] = _resolve_input_layer(
            session, fem, spec.name, inputs, storage_root
        )
    session.refresh(fem)

    group = open_fused_group(fem, storage_root=storage_root)

    # 2) Read + unit-convert each input to its declared unit; gather σ.
    full_inputs: dict[str, np.ndarray] = {}
    full_sigmas: dict[str, np.ndarray] = {}
    input_tiers: list[str] = []
    correlated = False
    for spec in transform.inputs:
        layer = layers[spec.name]
        src_unit = REGISTRY.get(spec.name).canonical_unit
        raw = _read_array(group, layer.id)
        full_inputs[spec.name] = _unit_convert(raw, src_unit, spec.unit)
        if layer.sigma_array and layer.sigma_array in group:
            sig = _read_array(group, layer.sigma_array)
            full_sigmas[spec.name] = _unit_convert(sig, src_unit, spec.unit)
        if spec.required:
            input_tiers.append(_layer_tier(layer))
            correlated = correlated or _layer_correlated(layer)

    # 3) Valid mask = AND of required-input coverage (finite values), doc 07 §2.3/§4.5.
    mask = np.ones(grid.shape, dtype=bool)
    for spec in transform.inputs:
        if spec.required:
            mask &= np.isfinite(full_inputs[spec.name])

    flat_mask = mask.reshape(-1)
    valid = np.flatnonzero(flat_mask)
    if valid.size == 0:
        raise ValueError(
            f"no cells have all required inputs of transform {transform.id!r} present"
        )

    flat_inputs = {n: arr.reshape(-1)[valid] for n, arr in full_inputs.items()}
    flat_sigmas = {n: arr.reshape(-1)[valid] for n, arr in full_sigmas.items()}

    if progress is not None:
        progress.report(0.35, "applying transform")

    # 4) Vectorized apply over the valid cells.
    out_valid = np.asarray(
        transform.apply(ctx, **flat_inputs, **resolved_params), dtype=float
    )

    # 6) Propagate σ over the valid cells.
    if progress is not None:
        progress.report(0.55, f"propagating sigma ({uncertainty})")
    if uncertainty == "monte_carlo":
        mc_mean, sigma_valid = _monte_carlo_sigma(
            transform, ctx, flat_inputs, flat_sigmas, resolved_params,
            n_samples=mc_samples, seed=mc_seed,
        )
        out_valid = mc_mean  # MC mean is the reported value (doc 07 §5.2)
    else:
        sigma_valid = _delta_sigma(
            transform, ctx, flat_inputs, flat_sigmas, resolved_params, out_valid
        )

    # 5) Clamp to valid_range; flag the out-of-range fraction (doc 07 §4.5 step 5).
    out_of_range_fraction = 0.0
    vr = transform.output.valid_range
    if vr is not None:
        lo, hi = vr
        oor = (out_valid < lo) | (out_valid > hi)
        out_of_range_fraction = float(np.mean(oor)) if out_valid.size else 0.0
        out_valid = np.clip(out_valid, lo, hi)

    # Scatter the valid results back into full (NaN-elsewhere) volumes.
    value_vol = _scatter(valid, out_valid, grid.shape)
    sigma_vol = _scatter(valid, sigma_valid, grid.shape)

    # 7) Calibration honesty: tier = min(inputs) capped by calibration_status; retitle.
    tier = _cap_tier(_min_tier(input_tiers), transform.calibration_status)
    title = transform.title or transform.output.name
    out_property = transform.output.name
    if transform.calibration_status == "uncalibrated" and transform.output.proxy_when_uncalibrated:
        title = f"{transform.target or transform.output.name} likelihood"
        tier = _cap_tier(tier, "uncalibrated")  # never better than proxy

    if progress is not None:
        progress.report(0.8, "writing derived volume")

    # 8) Write value + σ as a derived PropertyModel with the §4.3 derivation block.
    model_id, sigma_model_id = _write_derived(
        session, layout, fem, grid,
        transform=transform,
        out_property=out_property,
        value=value_vol,
        sigma=sigma_vol,
        params=resolved_params,
        tier=tier,
        title=title,
        uncertainty_mode=uncertainty,
        correlated=correlated,
        inputs=inputs,
        created_by=created_by,
    )

    if progress is not None:
        progress.report(1.0, "done")

    return TransformResult(
        transform_id=transform.id,
        transform_version=transform.version,
        output_property=out_property,
        title=title,
        tier=tier,
        calibration_status=transform.calibration_status,
        model_id=model_id,
        sigma_model_id=sigma_model_id,
        uncertainty_mode=uncertainty,
        out_of_range_fraction=out_of_range_fraction,
        correlated_inputs=correlated,
        n_valid=int(valid.size),
    )


def _scatter(valid: np.ndarray, values: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Scatter per-cell ``values`` (at flat indices ``valid``) into a full volume; NaN else."""
    vol = np.full(int(np.prod(shape)), np.nan, dtype=np.float32)
    vol[valid] = values.astype(np.float32)
    return vol.reshape(shape)


def _write_derived(
    session: Session,
    layout: ProjectLayout,
    fem: FusedModel,
    grid: FusedGrid,
    *,
    transform: Transform,
    out_property: str,
    value: np.ndarray,
    sigma: np.ndarray,
    params: dict[str, Any],
    tier: str,
    title: str,
    uncertainty_mode: str,
    correlated: bool,
    inputs: dict[str, str] | None,
    created_by: str,
) -> tuple[str, str]:
    """Write the derived value (+ paired σ) PropertyModel + catalog rows (doc 07 §4.3).

    The output is an ordinary :class:`~geosim.catalog.PropertyModel` on the fused-grid
    support; its σ rides alongside via :func:`geosim.storage.write_property_model`'s
    ``sigma=`` argument (doc 02 §10.2 sibling ``_sigma`` array). The provenance
    ``params_json`` carries the full doc-07 §4.3 derivation block (transform id/version,
    fused grid, inputs+versions, params, calibration status + tier, assumptions) so the
    derived volume is fully reproducible (doc 07 §4.4).
    """
    pm_id = new_id(IdKind.PROPERTY_MODEL)
    ds_id = new_id(IdKind.DATASET)
    prov_id = new_id(IdKind.PROVENANCE)
    grid_spec = GridSpec(origin=grid.origin, spacing=grid.spacing, cell_ref="center")
    bbox_json = fem.bbox_json
    zarr_path = layout.zarr_path(pm_id)

    write_property_model(
        zarr_path, out_property, value, grid=grid_spec, sigma=sigma, overwrite=True
    )

    # The resolved input layers + their native source models (reproducibility, doc 07 §4.4).
    input_layers = {
        spec.name: _layer_for_property(fem, spec.name) for spec in transform.inputs
    }
    derivation = {
        "kind": "transform",
        "transformId": transform.id,
        "transformVersion": transform.version,
        "fusedGridId": fem.id,
        "inputs": [
            {
                "property": spec.name,
                "fusedLayerId": (input_layers[spec.name].id if input_layers[spec.name] else None),
                "propertyModelId": (
                    input_layers[spec.name].source_property_model_id
                    if input_layers[spec.name]
                    else None
                ),
                "version": (
                    input_layers[spec.name].source_version if input_layers[spec.name] else None
                ),
            }
            for spec in transform.inputs
        ],
        "params": params,
        "calibrationStatus": transform.calibration_status,
        "calibratedBy": None,
        "assumptions": list(transform.assumptions),
        "tier": tier,
        "title": title,
        "uncertaintyMode": uncertainty_mode,
        "independence": "correlated_unmodeled" if correlated else "assumed_independent",
        "createdBy": created_by,
    }

    prov = Provenance(
        id=prov_id, project_id=fem.project_id, target_kind="propertyModel",
        target_id=pm_id, process=f"transform:{transform.id}",
        process_version=transform.version,
        params_json=json.dumps({"derivation": derivation, "params": params}),
    )
    session.add(prov)
    session.flush()
    session.add(ProvenanceInput(provenance_id=prov_id, input_kind="fusedModel", input_id=fem.id))
    for layer in input_layers.values():
        if layer is not None:
            session.add(ProvenanceInput(
                provenance_id=prov_id, input_kind="propertyModel",
                input_id=layer.source_property_model_id,
            ))

    session.add(Dataset(
        id=ds_id, project_id=fem.project_id, name=title, method="fusion",
        kind="propertyModel", status="ready", extent_json=bbox_json,
        spatial_frame_id=fem.project_id, provenance_id=prov_id,
        version_root_id=ds_id, version_seq=1, created_by=created_by,
    ))
    session.flush()

    pt = REGISTRY.get(out_property)
    session.add(PropertyModel(
        id=pm_id, dataset_id=ds_id, project_id=fem.project_id, property=out_property,
        canonical_unit=pt.canonical_unit, support="volume", store_uri=str(zarr_path),
        shape_json=json.dumps(list(grid.shape)),
        spacing_json=json.dumps(list(grid.spacing)),
        origin_json=json.dumps(list(grid.origin)),
        bbox_json=bbox_json, pyramid_levels=1,
    ))
    session.commit()
    # The σ rides as the sibling _sigma array of the same PropertyModel (doc 02 §10.2);
    # we surface its model id as the value model id (the σ is read via the reader's
    # read_sigma_level). Keep a distinct handle for API symmetry.
    return pm_id, pm_id
