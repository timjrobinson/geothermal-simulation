"""Geothermal favorability — the headline fusion product (doc 07 §4.6).

Favorability is a **special transform** that combines *multiple evidence layers* into one
**targetable index in ``[0,1]``** — the thing a driller actually points at (doc 07 §4.6,
OVERVIEW §6.3). Each evidence layer (a native or derived volume already resampled onto the
fused grid) is mapped to ``[0,1]`` via a per-layer **transfer / fuzzy-membership function**
(ramp, sigmoid, gaussian-band), then the memberships are combined by one of three
user-selectable **methods**:

- **fuzzy-conjunction** *(DEFAULT, critique #11)* — fuzzy-AND (``min``/``product``) over the
  ``required`` evidence and fuzzy-OR (``max``) over ``supporting`` alternatives. A geothermal
  play needs heat **AND** fluid **AND** permeability *co-located*; an absent required layer
  pulls the cell toward 0. This is **non-compensatory** — a soaring temperature can never
  numerically mask absent permeability (doc 07 §4.6).
- **weighted-linear** *(EXPLORATORY)* — ``F = Σ wᵢ·eᵢ / Σ wᵢ``. Simple/transparent but
  **compensatory** (a strong layer offsets a missing one), so any cell missing a ``required``
  layer is **flagged** (``missing_required`` mask) and **excluded from top-targets**, never
  silently averaged away (doc 07 §4.6 exploratory-mode guard).
- **bayesian** *(DEFERRED per DECISIONS)* — raises ``NotImplementedError`` until known-occurrence
  training data exists (doc 07 §4.6 table).

The output is an ordinary favorability :class:`~geosim.catalog.PropertyModel` (``inferno``
colourmap, doc 07 §4.3) **plus three companion diagnostic volumes** (doc 07 §4.6, critique #4):

- a paired **confidence** volume (§5),
- **evidence-overlap** — per cell, the fraction of the *required* evidence layers that
  actually cover it (respecting each layer's footprint/DOI, §2.3); a high score over an
  overlap of 1-of-3 is a warning, not a target,
- **assumption-burden** — per cell, the fraction of contributing evidence whose source is an
  *uncalibrated*/``proxy``-tier transform (where the hotspot is essentially "the rock-physics
  guessed", so calibration §4.8 can be prioritised there).

Canonical temperature is KELVIN end-to-end (doc 01 §5); membership thresholds are expressed in
each evidence's canonical unit. Derived volumes are ordinary PropertyModels stored via
:mod:`geosim.storage` and catalogued with the doc-07 §4.3 derivation block.
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
from geosim.spatial.property_types import PropertyType
from geosim.storage import GridSpec, ProjectLayout, write_property_model

from .grid import FusedGrid, fused_grid_from_row, open_fused_group
from .resample import resample_to_fused

__all__ = [
    "TransferFn",
    "Evidence",
    "FavorabilitySpec",
    "FavorabilityResult",
    "membership",
    "compute_favorability",
    "FAVORABILITY_METHODS",
    "MISSING_POLICIES",
    "TRANSFER_TYPES",
]

# doc 07 §4.6 table — fuzzy-conjunction is the DEFAULT; weighted is exploratory; bayesian deferred.
FAVORABILITY_METHODS = ("fuzzy", "weighted", "bayesian")

# doc 07 §4.6 — how a cell missing one evidence layer is treated (interacts with footprints §2.3).
MISSING_POLICIES = ("nodata", "neutral", "drop")

# doc 07 §4.6 — per-evidence fuzzy-membership / transfer curves (user-editable, doc 06).
TRANSFER_TYPES = ("ramp", "sigmoid", "gaussian-band")

# An evidence source whose calibration tier is one of these contributes to assumption-burden
# (doc 07 §4.6 — "the rock-physics guessed"); native/quantitative sources do not.
_PROXY_TIERS = ("unknown", "qualitative", "proxy")


# ──────────────────────────────────────────────────────────────────────────
# companion diagnostic property types (doc 07 §4.6, doc 08 §4b extension point)
# ──────────────────────────────────────────────────────────────────────────

# Favorability ships three companion [0,1] volumes; register their property types so they
# store/serve/colour like any other (doc 01 §5 / doc 08 §4b). Idempotent (replace=True).
for _pt in (
    PropertyType("confidence", "dimensionless", "viridis", "linear", (0.0, 1.0), "linear",
                 description="favorability confidence (doc 07 §4.6/§5)"),
    PropertyType("evidence_overlap", "dimensionless", "YlGnBu", "linear", (0.0, 1.0), "linear",
                 description="fraction of required evidence covering each cell (doc 07 §4.6)"),
    PropertyType("assumption_burden", "dimensionless", "OrRd", "linear", (0.0, 1.0), "linear",
                 description="fraction of contributing evidence uncalibrated/proxy (doc 07 §4.6)"),
):
    REGISTRY.register(_pt, replace=True)


# ──────────────────────────────────────────────────────────────────────────
# declarative spec (doc 07 §4.6 FavorabilitySpec sketch)
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TransferFn:
    """A per-evidence transfer / fuzzy-membership curve mapping raw → ``[0,1]`` (doc 07 §4.6).

    Three shapes (``TRANSFER_TYPES``), each user-parameterised in the evidence's canonical unit:

    - ``ramp`` — linear ramp from ``lo`` (membership 0) to ``hi`` (membership 1); ``hi<lo``
      gives a *descending* ramp (favorable = low values).
    - ``sigmoid`` — logistic centred at ``center`` with steepness ``k``; ``k<0`` descends.
    - ``gaussian-band`` — a band peaking at ``center`` with half-width ``width`` (favorable
      *around* a value, e.g. an optimal alteration index).
    """

    type: str = "ramp"
    lo: float | None = None
    hi: float | None = None
    center: float | None = None
    width: float | None = None
    k: float | None = None

    def __post_init__(self) -> None:
        if self.type not in TRANSFER_TYPES:
            raise ValueError(f"transferFn.type must be one of {TRANSFER_TYPES}; got {self.type!r}")


@dataclass(frozen=True)
class Evidence:
    """One favorable-indicator layer (doc 07 §4.6 ``FavorabilitySpec.evidence[]``).

    ``source`` is the native/derived PropertyModel id supplying the evidence (resampled onto
    the grid on demand); ``target`` is its property-type key (e.g. ``temperature``). ``transfer``
    maps the raw field → ``[0,1]``. ``weight`` is used by the weighted-linear method; ``role``
    is ``required`` (fuzzy-AND conjunct / guarded in weighted mode) or ``supporting``
    (fuzzy-OR alternative).
    """

    source: str
    target: str
    transfer: TransferFn = field(default_factory=TransferFn)
    weight: float = 1.0
    role: str = "required"

    def __post_init__(self) -> None:
        if self.role not in ("required", "supporting"):
            raise ValueError(f"evidence.role must be 'required'|'supporting'; got {self.role!r}")
        if self.weight < 0:
            raise ValueError(f"evidence.weight must be >= 0; got {self.weight}")


@dataclass(frozen=True)
class FavorabilitySpec:
    """The favorability configuration (doc 07 §4.6 ``FavorabilitySpec``).

    ``method`` selects the combination rule (``FAVORABILITY_METHODS``; default fuzzy-conjunction).
    ``fuzzy_and`` chooses the conjunction operator (``min`` or ``product``) for the fuzzy method.
    ``missingPolicy`` (``MISSING_POLICIES``) decides how a cell lacking one evidence layer is
    treated — strict ``nodata``, ``neutral`` (0.5), or ``drop`` (re-normalise over present
    evidence). Favorability is a research instrument: method/weights/curves are all user-set.
    """

    evidence: list[Evidence]
    method: str = "fuzzy"
    fuzzy_and: str = "min"  # "min" | "product"
    missing_policy: str = "nodata"

    def __post_init__(self) -> None:
        if self.method not in FAVORABILITY_METHODS:
            raise ValueError(f"method must be one of {FAVORABILITY_METHODS}; got {self.method!r}")
        if self.fuzzy_and not in ("min", "product"):
            raise ValueError(f"fuzzy_and must be 'min'|'product'; got {self.fuzzy_and!r}")
        if self.missing_policy not in MISSING_POLICIES:
            raise ValueError(
                f"missingPolicy must be one of {MISSING_POLICIES}; got {self.missing_policy!r}"
            )
        if not self.evidence:
            raise ValueError("favorability needs at least one evidence layer (doc 07 §4.6)")

    @property
    def required(self) -> list[Evidence]:
        return [e for e in self.evidence if e.role == "required"]

    @property
    def supporting(self) -> list[Evidence]:
        return [e for e in self.evidence if e.role == "supporting"]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> FavorabilitySpec:
        """Build a spec from the wire JSON (doc 07 §4.6 sketch; tolerant of camelCase keys)."""
        ev = []
        for item in payload.get("evidence", []):
            tf = item.get("transferFn") or item.get("transfer") or {}
            ev.append(Evidence(
                source=item["source"],
                target=item["target"],
                transfer=TransferFn(
                    type=tf.get("type", "ramp"),
                    lo=tf.get("lo"), hi=tf.get("hi"),
                    center=tf.get("center"), width=tf.get("width"), k=tf.get("k"),
                ),
                weight=float(item.get("weight", 1.0)),
                role=item.get("role", "required"),
            ))
        return cls(
            evidence=ev,
            method=payload.get("method", "fuzzy"),
            fuzzy_and=payload.get("fuzzyAnd", payload.get("fuzzy_and", "min")),
            missing_policy=payload.get("missingPolicy", payload.get("missing_policy", "nodata")),
        )


@dataclass
class FavorabilityResult:
    """The result of :func:`compute_favorability` (doc 07 §4.6, §4.3).

    ``model_id`` is the favorability PropertyModel; the three companions
    (``confidence_model_id`` / ``overlap_model_id`` / ``burden_model_id``) are its honesty
    diagnostics. ``n_valid`` is the cell count scored; ``n_missing_required`` counts cells
    flagged as missing a required layer (the weighted-mode guard).
    """

    method: str
    output_property: str
    model_id: str
    confidence_model_id: str
    overlap_model_id: str
    burden_model_id: str
    n_valid: int
    n_missing_required: int
    n_required: int
    n_supporting: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "output_property": self.output_property,
            "model_id": self.model_id,
            "confidence_model_id": self.confidence_model_id,
            "overlap_model_id": self.overlap_model_id,
            "burden_model_id": self.burden_model_id,
            "n_valid": self.n_valid,
            "n_missing_required": self.n_missing_required,
            "n_required": self.n_required,
            "n_supporting": self.n_supporting,
        }


# ──────────────────────────────────────────────────────────────────────────
# membership curves (doc 07 §4.6 — raw → [0,1])
# ──────────────────────────────────────────────────────────────────────────


def membership(values: np.ndarray, tf: TransferFn) -> np.ndarray:
    """Map a raw evidence field → fuzzy membership in ``[0,1]`` (doc 07 §4.6).

    NaN (no-coverage) cells stay NaN — membership never invents evidence where a layer's
    footprint does not reach (doc 07 §2.3). Out-of-band cells clamp to ``[0,1]``.
    """
    x = np.asarray(values, dtype=float)
    out = np.full(x.shape, np.nan, dtype=float)
    finite = np.isfinite(x)
    xf = x[finite]

    if tf.type == "ramp":
        if tf.lo is None or tf.hi is None:
            raise ValueError("ramp transferFn needs lo and hi")
        lo, hi = float(tf.lo), float(tf.hi)
        if hi == lo:
            raise ValueError("ramp transferFn needs lo != hi")
        m = (xf - lo) / (hi - lo)  # hi<lo ⇒ descending ramp automatically
        out[finite] = np.clip(m, 0.0, 1.0)
    elif tf.type == "sigmoid":
        if tf.center is None or tf.k is None:
            raise ValueError("sigmoid transferFn needs center and k")
        center, k = float(tf.center), float(tf.k)
        out[finite] = 1.0 / (1.0 + np.exp(-k * (xf - center)))
    elif tf.type == "gaussian-band":
        if tf.center is None or tf.width is None:
            raise ValueError("gaussian-band transferFn needs center and width")
        center, width = float(tf.center), float(tf.width)
        if width <= 0:
            raise ValueError("gaussian-band transferFn needs width > 0")
        out[finite] = np.exp(-0.5 * ((xf - center) / width) ** 2)
    else:  # pragma: no cover - guarded by TransferFn.__post_init__
        raise ValueError(f"unknown transferFn type {tf.type!r}")
    return out


# ──────────────────────────────────────────────────────────────────────────
# evidence resolution (auto-resample onto the fused grid, doc 07 §4.5 step 1)
# ──────────────────────────────────────────────────────────────────────────


def _layer_for_source(fem: FusedModel, source_pm_id: str, target: str):
    """The fused layer for ``source_pm_id`` (its ``target`` property), if already resampled."""
    layer = None
    for lay in fem.layers:
        if lay.source_property_model_id == source_pm_id and lay.property == target:
            layer = lay
    return layer


def _resolve_evidence_layer(
    session: Session,
    fem: FusedModel,
    ev: Evidence,
    storage_root: str | Path | None,
):
    """Resolve one evidence to a fused layer, auto-resampling its source if needed (§4.5 step 1)."""
    layer = _layer_for_source(fem, ev.source, ev.target)
    if layer is None:
        resample_to_fused(session, fem, ev.source, storage_root=storage_root)
        session.refresh(fem)
        layer = _layer_for_source(fem, ev.source, ev.target)
    if layer is None:
        raise ValueError(
            f"evidence source {ev.source!r} did not produce a {ev.target!r} fused layer"
        )
    return layer


def _source_is_proxy(session: Session, source_pm_id: str) -> bool:
    """Whether an evidence source is an *uncalibrated*/proxy-tier transform (doc 07 §4.6).

    A derived (transform-output) PropertyModel carries a §4.3 derivation block with
    ``calibrationStatus``/``tier``; an ``uncalibrated`` or ``proxy``-tier source contributes
    to assumption-burden. A native ingested model (no derivation block) is treated as
    calibrated/quantitative — its honesty is its own ingest tier, not a rock-physics guess.
    """
    pm = session.get(PropertyModel, source_pm_id)
    if pm is None:
        return False
    ds = session.get(Dataset, pm.dataset_id)
    if ds is None or ds.provenance_id is None:
        return False
    prov = session.get(Provenance, ds.provenance_id)
    if prov is None or not prov.params_json:
        return False
    try:
        deriv = json.loads(prov.params_json).get("derivation", {})
    except (TypeError, ValueError):
        return False
    if not isinstance(deriv, dict):
        return False
    status = deriv.get("calibrationStatus")
    tier = deriv.get("tier")
    return status == "uncalibrated" or tier in _PROXY_TIERS


# ──────────────────────────────────────────────────────────────────────────
# combination methods (doc 07 §4.6)
# ──────────────────────────────────────────────────────────────────────────


def _fuzzy_and(stack: np.ndarray, op: str) -> np.ndarray:
    """Fuzzy-AND over a (k, ...) membership stack (NaN-aware), doc 07 §4.6.

    Treats a NaN (missing required) conjunct as 0 — an absent required layer pulls the cell
    toward 0 (non-compensatory). ``min`` is the classic Zadeh AND; ``product`` is the soft
    probabilistic AND.
    """
    s = np.where(np.isfinite(stack), stack, 0.0)
    if op == "product":
        return np.prod(s, axis=0)
    return np.min(s, axis=0)


def _fuzzy_or(stack: np.ndarray) -> np.ndarray:
    """Fuzzy-OR (``max``) over a (k, ...) membership stack, ignoring NaN (doc 07 §4.6)."""
    s = np.where(np.isfinite(stack), stack, 0.0)
    return np.max(s, axis=0)


# ──────────────────────────────────────────────────────────────────────────
# the favorability computation (doc 07 §4.6)
# ──────────────────────────────────────────────────────────────────────────


def compute_favorability(
    session: Session,
    layout: ProjectLayout,
    fem: FusedModel,
    spec: FavorabilitySpec,
    *,
    created_by: str = "system:fusion",
    storage_root: str | Path | None = None,
    progress=None,
) -> FavorabilityResult:
    """Compute a ``[0,1]`` favorability volume + honesty diagnostics on a fused grid (doc 07 §4.6).

    Pipeline:

    1. **Resolve evidence** — each evidence's ``source`` model is resampled onto the grid (§2)
       if not already present; its ``target`` field is read off the fused group.
    2. **Membership** — each field → ``[0,1]`` via its :class:`TransferFn` (NaN where the
       layer's footprint does not cover the cell, §2.3).
    3. **Combine** by ``method``:
       - ``fuzzy`` *(default)* — fuzzy-AND over required + fuzzy-OR over supporting,
         non-compensatory (an absent required layer pulls the cell toward 0).
       - ``weighted`` *(exploratory)* — ``Σ wᵢeᵢ / Σ wᵢ`` with the missing-required guard:
         cells missing a required layer are flagged + excluded from top-targets, not averaged.
       - ``bayesian`` — deferred (``NotImplementedError``).
    4. **Diagnostics** — confidence, evidence-overlap (fraction of required layers covering the
       cell), assumption-burden (fraction of contributing evidence that is uncalibrated/proxy).
    5. **Write** the favorability PropertyModel + the three companion diagnostic volumes (doc 04)
       with the §4.3 derivation-provenance block.
    """
    if spec.method == "bayesian":
        raise NotImplementedError(
            "Bayesian (weights-of-evidence) favorability is deferred per DECISIONS / doc 07 §4.6 "
            "until known-occurrence training data exists; use 'fuzzy' (default) or 'weighted'."
        )

    grid = fused_grid_from_row(fem)
    if progress is not None:
        progress.report(0.05, "resolving evidence")

    # 1) Resolve + read each evidence field, 2) compute membership.
    layers: dict[int, Any] = {}
    for i, ev in enumerate(spec.evidence):
        layers[i] = _resolve_evidence_layer(session, fem, ev, storage_root)
    session.refresh(fem)
    group = open_fused_group(fem, storage_root=storage_root)

    if progress is not None:
        progress.report(0.3, "computing memberships")

    members: list[np.ndarray] = []   # per-evidence membership (NaN where no coverage)
    coverage: list[np.ndarray] = []  # per-evidence boolean coverage (footprint, §2.3)
    proxy_flags: list[bool] = []     # per-evidence: source is uncalibrated/proxy
    for i, ev in enumerate(spec.evidence):
        layer = layers[i]
        raw = np.asarray(group[layer.id][...], dtype=float)
        m = membership(raw, ev.transfer)
        members.append(m)
        coverage.append(np.isfinite(raw))
        proxy_flags.append(_source_is_proxy(session, ev.source))

    req_idx = [i for i, ev in enumerate(spec.evidence) if ev.role == "required"]
    sup_idx = [i for i, ev in enumerate(spec.evidence) if ev.role == "supporting"]

    # Cells reached by at least one evidence layer's footprint — the region where the
    # honesty diagnostics are meaningful (a cell with NO evidence at all is plain nodata).
    any_coverage = np.any(np.stack(coverage, axis=0), axis=0)

    # 4a) Evidence-overlap: fraction of REQUIRED layers covering each cell (doc 07 §4.6).
    if req_idx:
        overlap_count = np.sum([coverage[i] for i in req_idx], axis=0).astype(float)
        overlap_frac = overlap_count / float(len(req_idx))
    else:
        overlap_frac = np.where(any_coverage, 1.0, 0.0)
    missing_required = any_coverage & (overlap_frac < 1.0)  # covered but a required layer absent

    if progress is not None:
        progress.report(0.55, f"combining ({spec.method})")

    # 3) Combine memberships → favorability.
    if spec.method == "fuzzy":
        favorability, valid = _combine_fuzzy(members, req_idx, sup_idx, spec, grid.shape)
    else:  # weighted-linear (exploratory)
        favorability, valid = _combine_weighted(members, spec, grid.shape, missing_required)

    # 4b) Assumption-burden: fraction of CONTRIBUTING (covered) evidence that is proxy/uncal.
    burden = _assumption_burden(coverage, proxy_flags, grid.shape)

    # 4c) Confidence: down-weighted by missing required evidence AND assumption burden (§5/§4.6).
    confidence = np.where(any_coverage, np.clip(overlap_frac * (1.0 - burden), 0.0, 1.0), np.nan)

    # Diagnostics are shown over the whole covered region — INCLUDING the missing-required
    # cells excluded from favorability — so overlap<1 / burden are visible exactly where the
    # favorability score was withheld (doc 07 §4.6 honesty indicators).
    overlap_out = np.where(any_coverage, overlap_frac, np.nan)
    burden_out = np.where(any_coverage, burden, np.nan)

    if progress is not None:
        progress.report(0.8, "writing favorability + diagnostics")

    grid_spec = GridSpec(origin=grid.origin, spacing=grid.spacing, cell_ref="center")
    bbox_json = fem.bbox_json
    n_valid = int(np.count_nonzero(valid))
    n_missing = int(np.count_nonzero(missing_required))

    common = dict(
        session=session, layout=layout, fem=fem, grid=grid, grid_spec=grid_spec,
        bbox_json=bbox_json, spec=spec, layers=layers, proxy_flags=proxy_flags,
        created_by=created_by, storage_root=storage_root,
    )
    model_id = _write_volume(prop="favorability", values=favorability, sigma=None,
                             name="favorability", role="value", **common)
    confidence_model_id = _write_volume(prop="confidence", values=confidence, sigma=None,
                                        name="favorability-confidence", role="confidence", **common)
    overlap_model_id = _write_volume(prop="evidence_overlap", values=overlap_out, sigma=None,
                                     name="favorability-evidence-overlap", role="overlap", **common)
    burden_model_id = _write_volume(prop="assumption_burden", values=burden_out, sigma=None,
                                    name="favorability-assumption-burden", role="burden", **common)

    if progress is not None:
        progress.report(1.0, "done")

    return FavorabilityResult(
        method=spec.method,
        output_property="favorability",
        model_id=model_id,
        confidence_model_id=confidence_model_id,
        overlap_model_id=overlap_model_id,
        burden_model_id=burden_model_id,
        n_valid=n_valid,
        n_missing_required=n_missing,
        n_required=len(req_idx),
        n_supporting=len(sup_idx),
    )


def _combine_fuzzy(
    members: list[np.ndarray],
    req_idx: list[int],
    sup_idx: list[int],
    spec: FavorabilitySpec,
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Fuzzy-conjunction combine (doc 07 §4.6 default).

    F = (fuzzy-AND over required) combined with (fuzzy-OR over supporting). The required AND
    is the non-compensatory gate: an absent required layer (NaN→0) drives F toward 0 — a
    dry-hot cell scores LOW. Supporting evidence can only *add* via a soft OR boost (it never
    rescues a missing required conjunct). With ``missingPolicy='nodata'`` a cell missing a
    required layer is NaN (not scored); ``neutral``/``drop`` keep it scored but the required
    gate still penalises it.
    """
    valid = np.ones(shape, dtype=bool)

    if req_idx:
        req_members = []
        for i in req_idx:
            m = members[i]
            if spec.missing_policy == "nodata":
                valid &= np.isfinite(m)
                req_members.append(m)
            elif spec.missing_policy == "neutral":
                req_members.append(np.where(np.isfinite(m), m, 0.5))
            else:  # drop → a missing required conjunct contributes neutrally to the AND
                req_members.append(np.where(np.isfinite(m), m, 1.0))
        and_score = _fuzzy_and(np.stack(req_members, axis=0), spec.fuzzy_and)
    else:
        and_score = np.ones(shape, dtype=float)

    favorability = and_score
    if sup_idx:
        sup_stack = np.stack([members[i] for i in sup_idx], axis=0)
        or_score = _fuzzy_or(sup_stack)
        # Supporting OR boosts the conjunction toward 1 without compensating a missing required
        # conjunct: F = AND + (1-AND)*OR  (a bounded, monotone soft boost).
        favorability = and_score + (1.0 - and_score) * or_score

    favorability = np.where(valid, np.clip(favorability, 0.0, 1.0), np.nan)
    return favorability, valid


def _combine_weighted(
    members: list[np.ndarray],
    spec: FavorabilitySpec,
    shape: tuple[int, int, int],
    missing_required: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted-linear combine — EXPLORATORY, COMPENSATORY (doc 07 §4.6).

    ``F = Σ wᵢeᵢ / Σ wᵢ`` over the evidence present in each cell. This is the compensatory
    mode: a strong layer can offset a weak/absent one. The **missing-required guard** still
    applies — cells missing a ``required`` layer are flagged (returned ``valid=False`` so they
    are excluded from top-targets and rendered with the missing-required overlay), never
    silently averaged into a favorable score.
    """
    weights = np.array([ev.weight for ev in spec.evidence], dtype=float)
    stack = np.stack(members, axis=0)  # (k, z, y, x)
    present = np.isfinite(stack)
    w = weights[:, None, None, None] * present
    num = np.sum(np.where(present, stack, 0.0) * w, axis=0)
    den = np.sum(w, axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        favorability = np.where(den > 0, num / den, np.nan)

    # Missing-required guard: exclude (do not silently average) cells lacking a required layer.
    if spec.missing_policy == "nodata":
        valid = (den > 0) & (~missing_required)
    else:
        valid = den > 0
    favorability = np.where(np.isfinite(favorability), np.clip(favorability, 0.0, 1.0), np.nan)
    # Cells missing a required layer keep their computed value but are flagged via valid=False;
    # under nodata they are also NaN'd so they never appear as targets.
    if spec.missing_policy == "nodata":
        favorability = np.where(valid, favorability, np.nan)
    return favorability, valid


def _assumption_burden(
    coverage: list[np.ndarray],
    proxy_flags: list[bool],
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Per-cell fraction of CONTRIBUTING evidence whose source is uncalibrated/proxy (doc 07 §4.6).

    Only evidence that actually covers the cell counts toward the denominator (a layer that does
    not reach the cell is not "riding on a guess" there). 0 where no evidence covers the cell.
    """
    cov_stack = np.stack(coverage, axis=0).astype(float)          # (k, ...)
    proxy = np.array(proxy_flags, dtype=float)[:, None, None, None]
    contributing = np.sum(cov_stack, axis=0)
    proxy_contrib = np.sum(cov_stack * proxy, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        burden = np.where(contributing > 0, proxy_contrib / contributing, 0.0)
    return burden


def _write_volume(
    *,
    session: Session,
    layout: ProjectLayout,
    fem: FusedModel,
    grid: FusedGrid,
    grid_spec: GridSpec,
    bbox_json: str,
    spec: FavorabilitySpec,
    layers: dict[int, Any],
    proxy_flags: list[bool],
    prop: str,
    values: np.ndarray,
    sigma: np.ndarray | None,
    name: str,
    role: str,
    created_by: str,
    storage_root: str | Path | None,
) -> str:
    """Write one favorability output (value/confidence/overlap/burden) as a derived PropertyModel.

    Each carries the doc-07 §4.3 derivation block (``kind="favorability"``, method, the evidence
    spec + resolved source models/versions, proxy flags) so the volume is fully reproducible
    (doc 07 §4.4) and the diagnostics are traceable to the exact evidence that produced them.
    """
    pm_id = new_id(IdKind.PROPERTY_MODEL)
    ds_id = new_id(IdKind.DATASET)
    prov_id = new_id(IdKind.PROVENANCE)
    zarr_path = layout.zarr_path(pm_id)
    write_property_model(zarr_path, prop, values, grid=grid_spec, sigma=sigma, overwrite=True)

    derivation = {
        "kind": "favorability",
        "role": role,  # value | confidence | overlap | burden
        "method": spec.method,
        "fuzzyAnd": spec.fuzzy_and,
        "missingPolicy": spec.missing_policy,
        "fusedGridId": fem.id,
        "evidence": [
            {
                "source": ev.source,
                "target": ev.target,
                "role": ev.role,
                "weight": ev.weight,
                "transferFn": {
                    "type": ev.transfer.type, "lo": ev.transfer.lo, "hi": ev.transfer.hi,
                    "center": ev.transfer.center, "width": ev.transfer.width, "k": ev.transfer.k,
                },
                "fusedLayerId": layers[i].id,
                "sourceVersion": layers[i].source_version,
                "proxy": bool(proxy_flags[i]),
            }
            for i, ev in enumerate(spec.evidence)
        ],
        "createdBy": created_by,
    }

    prov = Provenance(
        id=prov_id, project_id=fem.project_id, target_kind="propertyModel",
        target_id=pm_id, process="fusion:favorability", process_version="1.0.0",
        params_json=json.dumps({"derivation": derivation}),
    )
    session.add(prov)
    session.flush()
    session.add(ProvenanceInput(provenance_id=prov_id, input_kind="fusedModel", input_id=fem.id))
    for ev in spec.evidence:
        session.add(ProvenanceInput(
            provenance_id=prov_id, input_kind="propertyModel", input_id=ev.source,
        ))

    session.add(Dataset(
        id=ds_id, project_id=fem.project_id, name=name, method="fusion",
        kind="propertyModel", status="ready", extent_json=bbox_json,
        spatial_frame_id=fem.project_id, provenance_id=prov_id,
        version_root_id=ds_id, version_seq=1, created_by=created_by,
    ))
    session.flush()

    pt = REGISTRY.get(prop)
    session.add(PropertyModel(
        id=pm_id, dataset_id=ds_id, project_id=fem.project_id, property=prop,
        canonical_unit=pt.canonical_unit, support="volume", store_uri=str(zarr_path),
        shape_json=json.dumps(list(grid.shape)),
        spacing_json=json.dumps(list(grid.spacing)),
        origin_json=json.dumps(list(grid.origin)),
        bbox_json=bbox_json, pyramid_levels=1,
    ))
    session.commit()
    return pm_id
