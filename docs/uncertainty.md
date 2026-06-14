# Uncertainty & scientific honesty

> **What you'll learn / why it matters.** Every number in this platform is an *estimate of
> something you cannot see*. A resistivity cube from an inversion, a temperature derived from
> [rock physics](rock-physics.md), a favorability score — none of them is a measurement of
> the rock; they are inferences with error bars. The single most important engineering
> principle in the whole system is this: **never let the type system or the UI imply more
> certainty than the data supports.** This page is the theory and the mechanics of keeping
> that promise. As a programmer, read it as: *uncertainty is a first-class field that rides
> alongside every value, and the renderer literally shows you less where the model knows
> less.*

!!! abstract "The one-sentence version"
    A value without its uncertainty is a lie of omission; this platform refuses to ship one.

---

## 1. Why this is non-optional

The survey methods feeding the model have **wildly different sensitivity and
non-uniqueness**. A magnetotelluric (MT) inversion produces a smooth, deep, blurry
resistivity field; a seismic reflection survey produces sharp, shallow boundaries; gravity is
notoriously **non-unique** (many different density distributions produce the *same* surface
gravity). If you fuse these into one grid and treat every cell as equally trustworthy, you
have laundered the weakest data into looking as good as the strongest.

!!! note "A CS framing: this is an inverse problem"
    Geophysics is an **inverse problem** — recovering hidden parameters (the 3-D earth) from
    indirect observations (surface readings). Inverse problems are generically
    *ill-posed*: many models fit the data equally well, and small data noise maps to large
    model swings. The honest output of an inverse problem is therefore never a single answer
    but *an answer plus how constrained it is*. Uncertainty fields are how we carry that.

---

## 2. The two *different* things people call "uncertainty"

This distinction is the conceptual heart of the page. There are **two** independent ways a
value can be untrustworthy, and conflating them is the classic mistake.

### 2.1 σ (sigma) — noise

Per-cell **standard deviation** $\sigma(x)$: how much the value would wiggle if you re-ran the
measurement. Comes from the inversion posterior or model covariance. This is "noisy."

### 2.2 Resolution / DOI — blur

**Resolution** is how *sharp* the model can possibly be, regardless of noise. A smooth method
like MT or gravity might report a *tight* σ on every cell and still be unable to resolve
anything smaller than a kilometre — it is **confidently blurry**. The
**depth of investigation (DOI)** is the depth below which a method simply *cannot see*;
beneath it the "model" is just the inversion's smoothness prior, not data.

!!! warning "Low σ does NOT mean well-resolved"
    This is the trap. A gravity inversion can hand you a 3-D density volume with small per-cell
    σ everywhere — and it would be *deeply misleading* to render that as high-confidence,
    because the method has almost no vertical resolution. **σ tells you about noise; the
    resolution kernel / DOI tells you about blur.** You need both. The data model captures
    them separately (`UncertaintySpec` for σ, `ResolutionSpec` for the DOI surface + kernel
    length) precisely so fusion can distinguish *noisy* from *blurry*.

!!! tip "The analogy that makes it stick"
    Think of two images. Image A is **sharp but grainy** (high resolution, high σ): you can
    see fine structure under the noise. Image B is **clean but out of focus** (low σ, poor
    resolution): every pixel is confident and yet you can't make out any detail. A gravity or
    MT model is Image B. Reporting only σ would call Image B "high quality." Or in
    compression terms: σ is the *quantization noise*; resolution is the *downsampling
    factor*. A heavily downsampled image can be noise-free and still carry almost no
    information.

| | σ (sigma) | Resolution / DOI |
|---|---|---|
| Answers | "how noisy?" | "how sharp / how deep can it see?" |
| Schema | `UncertaintySpec` (per-cell `<property>_sigma`) | `ResolutionSpec` (DOI surface, kernel length) |
| Failure if ignored | trust a noisy value | trust a confidently blurry value |
| Flag it triggers | high-σ → faint render | low-sensitivity → faint render + excluded from clustering |

---

## 3. Uncertainty tiers — and why `null` means *unknown*, not *zero*

Beyond *how much* uncertainty, the platform tracks *what kind of basis* the uncertainty
number even has. Each `UncertaintySpec` carries a **`tier`** (defined in
[the data model](data-model.md) §6):

| Tier | Meaning | How it's displayed |
|---|---|---|
| `quantitative` | from a real posterior / propagated calibrated errors | a real σ number is fine |
| `proxy` | magnitude is *indicative only* (rule-of-thumb σ, uncalibrated transform) | shown as low/med/high, **not** a spurious decimal |
| `qualitative` | ordinal confidence only | low/med/high, never a number |
| `unknown` | no basis at all | **un-weightable**, *not* high confidence |

!!! danger "`uncertainty = null` means UNKNOWN, not zero"
    The most dangerous default in any data system is to treat "missing" as "fine." A field
    with no uncertainty representation is **not perfectly certain** — it is *unknown*. The
    platform treats a `null`-uncertainty field as **un-weightable** and attaches a
    conservative default relative σ from the property registry so propagation can still run,
    rather than letting a missing error bar read as a perfect one. (In CS terms: `null` is not
    `0`; conflating them is a null-pointer bug with scientific consequences.)

The tier also governs *display precision*. A precise-looking decimal σ printed on top of a
rule-of-thumb input is **false precision** — so when any input to a derived volume is `proxy`
or `qualitative` tier, the resulting confidence is shown **qualitatively (low/med/high)**, not
as a fake number. In code, the output tier is the **minimum over the inputs**, then **capped
by the transform's calibration status** — an uncalibrated transform can never emit better than
`proxy`:

```python
# backend/geosim/fusion/transform.py
TIER_ORDER = ("unknown", "qualitative", "proxy", "quantitative")  # worst → best
_CALIBRATION_TIER_CAP = {"uncalibrated": "proxy",
                         "well_calibrated": "quantitative",
                         "lab_calibrated":  "quantitative"}
tier = _cap_tier(_min_tier(input_tiers), transform.calibration_status)
```

---

## 4. Propagating uncertainty through the pipeline

Uncertainty rides through the **same** operations as the values — resampling, then
transforms.

### 4.1 Through resampling

When a native model is [resampled](fusion.md) onto the fused grid, its σ is interpolated by
the **same interpolator** as the value. Two honesty rules:

- **Interpolation inflates σ.** Upsampling a coarse model onto a finer grid *fabricates*
  detail the method never had, so an interpolation-variance term is added that grows with
  distance from the native nodes. Faked detail therefore reads as **low confidence** — you
  see the upsampled cells, but faintly. (Gridding scattered points, e.g. gravity stations,
  adds the gridder's own prediction variance.)
- **Outside the footprint/DOI → nodata, never high σ.** Absence of data is *NaN*, not a large
  error bar. There's a difference between "we measured this badly" and "we never measured this
  here," and the platform keeps them distinct.

### 4.2 Through transforms — the delta method

By default the transform [harness](rock-physics.md#4-the-transform-engine) propagates σ by
the **delta method** (first-order Taylor / linearized error propagation). For an output
$y = f(x_1,\dots,x_n;\theta)$:

$$
\sigma_y^2 \;\approx\; \sum_i \left(\frac{\partial f}{\partial x_i}\right)^{2} \sigma_{x_i}^2
\;+\; \sum_\theta \left(\frac{\partial f}{\partial \theta}\right)^{2} \sigma_\theta^2
$$

- $\partial f/\partial x_i$ — sensitivity of the output to input $i$, evaluated **numerically**
  by central finite difference so transform authors never hand-derive a Jacobian. They just
  supply `apply()`; the harness differentiates it.
- $\sigma_{x_i}$ — the input field's 1σ (itself propagated from upstream).
- The second sum is **parameter uncertainty**: declare a `sigma` on a `Param` (e.g. porosity
  $\sigma = 0.03$ in Archie) and it contributes — and after [calibration](rock-physics.md#7-calibration-turning-proxies-into-measurements-honestly)
  it is frequently the *dominant* term.

```python
# backend/geosim/fusion/transform.py — _delta_sigma (central finite-difference Jacobian)
dfdx = (_eval(plus, params) - _eval(minus, params)) / (2.0 * h)   # ∂f/∂xᵢ, numeric
var += (dfdx * sx) ** 2                                            # Σ (∂f/∂xᵢ)² σ_xᵢ²
# ... plus the same construction over each Param that declares a sigma
```

!!! warning "The delta method assumes the inputs are independent"
    The formula above **drops the cross-terms**
    $2\,(\partial f/\partial x_i)(\partial f/\partial x_j)\,\mathrm{cov}(x_i,x_j)$. When two
    inputs are actually correlated (e.g. two fields derived from the *same* resistivity
    inversion), the propagated σ is an **under-estimate** — falsely tight. The data model
    flags this with an `independence` field (`assumed_independent` |
    `correlated_unmodeled`); when any input is `correlated_unmodeled`, the harness records the
    caveat rather than reporting a falsely confident number, and downstream display collapses
    to qualitative low/med/high.

### 4.3 When linearization isn't enough — Monte-Carlo

The delta method is a *linear* approximation; it's fast but wrong for strongly nonlinear
transforms. Archie's law is the canonical case — it's $\propto \phi^{-m}$, so a modest
porosity error explodes nonlinearly into saturation. For these you opt into **Monte-Carlo**:
sample each input $\sim \mathcal{N}(x,\sigma_x)$ and each uncertain param
$\sim \mathcal{N}(\theta,\sigma_\theta)$, push $K$ samples through `apply()`, and take the
output **mean and std** empirically.

```python
# transform.py — _monte_carlo_sigma (push K samples through the pure apply())
sampled_inputs = {name: x + rng.normal(0,1,x.shape) * sigma.get(name, 0.0)
                  for name, x in flat_inputs.items()}
acc[k] = transform.apply(ctx, **sampled_inputs, **sampled_params)
return acc.mean(axis=0), acc.std(axis=0)
```

| | Delta method | Monte-Carlo |
|---|---|---|
| Cost | cheap (a few extra `apply()` calls) | $K\times$ the work — always **job-based** |
| Accuracy | first-order; fine for mild nonlinearity | correct for strong nonlinearity |
| Default? | **yes** | **opt-in** per transform |

The rule of thumb the design fixes: *delta-method everywhere, Monte-Carlo opt-in per
nonlinear transform.*

---

## 5. Confidence volumes — "you see less where the model knows less"

Every resampled layer and every derived volume ships a **paired confidence volume**
(`sigmaRef` / a `_sigma` array, stored exactly like any other field). It can be:

1. **rendered directly** as its own layer (a map of where the model is shaky), or
2. **bound to the value layer as an opacity/desaturation modulator** — low-confidence cells
   render faint or greyed.

Option 2 is the **default honest view**: the 3-D model literally fades out where the data
thins. A driller flying through the [viewer](visualization.md) sees a crisp, bright anomaly
where methods overlap and agree, and a ghostly haze in the deep, under-constrained regions —
no caption required.

[Favorability](rock-physics.md#5-geothermal-favorability-the-drill-here-index) makes this
concrete by deriving its confidence from its own honesty diagnostics:

```python
# backend/geosim/fusion/favorability.py
confidence = np.where(any_coverage,
                      np.clip(overlap_frac * (1.0 - burden), 0.0, 1.0),
                      np.nan)
# overlap_frac : fraction of REQUIRED evidence layers actually covering the cell
# burden       : fraction of contributing evidence that is uncalibrated/proxy
```

So a favorability hotspot is bright **only** where (a) all the required ingredients were
actually measured there, and (b) they came from calibrated, not guessed, sources.

---

## 6. The honesty guards — a checklist

These are the mechanical rules the engine enforces so honesty isn't left to discipline:

| Guard | What it does | Where |
|---|---|---|
| **"likelihood" relabel** | an `uncalibrated` transform's output is retitled `"<target> likelihood"` and stamped `tier="proxy"` — it cannot present as a measurement | `transform.py` step 7 |
| **footprint NaNs** | cells outside a method's coverage/DOI are NaN (nodata), never 0 or edge-bleed | `transform.py` step 3, [fusion](fusion.md) |
| **`null` ≠ certain** | missing uncertainty ⇒ un-weightable + conservative default σ, never "perfect" | [data model](data-model.md) §6 |
| **tier capping** | output tier = min(inputs), capped by calibration status; proxy inputs collapse confidence to low/med/high | `transform.py` `_cap_tier` |
| **independence caveat** | `correlated_unmodeled` inputs flag the propagated σ as a lower bound | `transform.py` `_layer_correlated` |
| **low-sensitivity flag** | tight σ but resolution kernel ≫ fused spacing ⇒ flagged, faint render, excluded from clustering | doc 07 §5.4 |
| **evidence-overlap / assumption-burden** | favorability ships diagnostics showing where a score rides on absent or guessed evidence | `favorability.py` |
| **spatially-honest calibration** | a well promotes only its neighbourhood to `quantitative`; far cells stay `proxy` | `calibration.py` `promote_spatial` |
| **`synthetic_only` truth scoring** | scoring against a truth field is flagged so it can never masquerade as a real-data metric | `calibration.py` `score_against_truth` |

### Non-uniqueness & low-sensitivity flags

Smooth, non-unique methods and below-DOI regions are **flagged, not silently trusted** — and
the flags are themselves toggleable diagnostic volumes (nothing is ever deleted):

| Flag | Trigger | Effect |
|---|---|---|
| **Below-DOI / unconstrained** | cell below a model's DOI surface, or outside footprint | nodata for that input |
| **Low-sensitivity** | confidence below threshold, or resolution kernel ≫ fused spacing | faint render; excluded from cluster fit |
| **High-disagreement** | two methods that *should* correlate (via a transform) don't | a "tension" volume — interesting, not necessarily wrong |
| **Extrapolated-detail** | upsampling-dominated cells (interp variance ≫ native σ) | high-σ, faint render |

---

## 7. The mindset, for a programmer

If you remember nothing else from this page, remember this reframing:

> **Treat certainty as a capability your types must *earn*, not a default they get for free.**

A bare `float` temperature volume is a type that *claims* "this is the temperature." That
claim is almost always false — it's a *likelihood* until a well proves otherwise, and even
then only locally. The platform's design makes the honest representation the *easy* one: every
value drags its σ, its tier, its footprint, and its provenance along with it, and the renderer
defaults to fading the uncertain parts out. You have to go *out of your way* to present a
number as more certain than it is — and the `synthetic_only` flag, the "likelihood" relabel,
and the tier cap exist to stop you even then.

---

## Key takeaways

- Uncertainty is **non-optional** because every value is the output of an ill-posed inverse
  problem — many earths fit the same data.
- **σ ≠ resolution.** σ is *noise*; resolution/DOI is *blur*. A method can be **confidently
  blurry** (low σ, poor resolution) — so a low σ does **not** mean a cell is well resolved.
  The platform tracks both (`UncertaintySpec` + `ResolutionSpec`).
- Uncertainty has **tiers** (`quantitative`/`proxy`/`qualitative`/`unknown`), and
  **`null` means unknown, not zero**. Proxy/qualitative inputs collapse display to
  low/med/high to avoid false precision.
- Propagation rides through the pipeline: σ inflates through upsampling; transforms use the
  **delta method** ($\sigma_y^2 \approx \sum (\partial f/\partial x_i)^2 \sigma_{x_i}^2$) by
  default — which **assumes input independence** — and **Monte-Carlo** opt-in for nonlinear
  transforms like Archie.
- **Confidence volumes modulate render opacity**: you literally see less where the model knows
  less. Favorability derives its confidence from evidence-overlap and assumption-burden.
- A wall of **honesty guards** (likelihood relabel, footprint NaNs, tier capping, independence
  caveat, low-sensitivity flags, spatially-honest calibration, `synthetic_only` scoring) makes
  honesty mechanical, not a matter of discipline.

## Where this lives in the code

| Concern | Path |
|---|---|
| σ propagation (delta method + Monte-Carlo), tier capping, honesty stamping | `backend/geosim/fusion/transform.py` |
| Confidence, evidence-overlap, assumption-burden diagnostics | `backend/geosim/fusion/favorability.py` |
| Spatially-honest calibration + `synthetic_only` truth scoring | `backend/geosim/fusion/calibration.py` |
| Uncertainty resampling / interpolation-variance inflation | `backend/geosim/fusion/resample.py` |
| `UncertaintySpec` / `ResolutionSpec` schema (tier, independence, DOI) | `backend/geosim/catalog/` (per [the data model](data-model.md) §6) |

See also: [rock physics & favorability](rock-physics.md), [fusion](fusion.md),
[the data model](data-model.md), [the 3-D viewer](visualization.md),
[the synthetic data generator](synthetic-data.md), and the [glossary](glossary.md).
