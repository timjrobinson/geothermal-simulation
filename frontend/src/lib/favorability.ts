// Favorability — the headline fusion product (doc 07 §4.6; doc 06 §3.2/§9.2). This is the
// PURE logic + API client for the Favorability panel: the membership-curve evaluator (raw →
// [0,1]), the FavorabilitySpec builder (the wire payload POSTed to the backend), the
// transform-palette + favorability fetch wrappers, and an offline mock. Everything here is
// free of THREE / DOM / Zustand so the membership eval + spec builder are unit-testable
// headlessly (npm test bundles src/lib/*.test.ts).
//
// Favorability is a *research instrument* (doc 07 §4.6): the user picks evidence layers,
// shapes a per-evidence fuzzy-membership curve (ramp / sigmoid / gaussian-band, doc 06 §3.2
// transfer functions), sets weights + a role (required ⇒ fuzzy-AND conjunct / supporting ⇒
// fuzzy-OR alternative), and chooses a combination method (fuzzy default, weighted
// exploratory) + a missingPolicy. The wire shapes here mirror geosim/api/fusion.py
// FavorabilityRequest + geosim.fusion.favorability.FavorabilitySpec.from_payload EXACTLY
// (it is tolerant of camelCase transferFn keys), and the membership() math mirrors
// geosim.fusion.favorability.membership so the curve preview matches the backend result.

// ── transfer / fuzzy-membership curves (doc 07 §4.6 TransferFn) ──────────────────────────
// Three shapes, each parameterised in the evidence's canonical unit (KELVIN for temperature,
// doc 01 §5). Match TRANSFER_TYPES on the backend.

export type TransferType = "ramp" | "sigmoid" | "gaussian-band";
export const TRANSFER_TYPES: readonly TransferType[] = ["ramp", "sigmoid", "gaussian-band"];

export interface TransferFnSpec {
  type: TransferType;
  // ramp: linear lo→hi (hi<lo ⇒ descending — favorable = low values).
  lo?: number | null;
  hi?: number | null;
  // sigmoid: logistic centred at `center`, steepness `k` (k<0 descends).
  // gaussian-band: peak at `center`, half-width `width` (favorable AROUND a value).
  center?: number | null;
  width?: number | null;
  k?: number | null;
}

// Combination methods (doc 07 §4.6 table). fuzzy = non-compensatory DEFAULT; weighted =
// compensatory EXPLORATORY (with the missing-required guard); bayesian is deferred (→400).
export type FavorabilityMethod = "fuzzy" | "weighted" | "bayesian";
export const FAVORABILITY_METHODS: readonly FavorabilityMethod[] = [
  "fuzzy",
  "weighted",
  "bayesian",
];

export type FuzzyAnd = "min" | "product";
export const FUZZY_AND_OPS: readonly FuzzyAnd[] = ["min", "product"];

// How a cell missing one evidence layer is treated (doc 07 §4.6, interacts with footprints).
export type MissingPolicy = "nodata" | "neutral" | "drop";
export const MISSING_POLICIES: readonly MissingPolicy[] = ["nodata", "neutral", "drop"];

export type EvidenceRole = "required" | "supporting";
export const EVIDENCE_ROLES: readonly EvidenceRole[] = ["required", "supporting"];

// One favorable-indicator layer (doc 07 §4.6 FavorabilitySpec.evidence[]). `source` is the
// native/derived PropertyModel id; `target` is its property-type key (e.g. "temperature");
// `transferFn` maps the raw field → [0,1]; `weight` is used by the weighted method; `role`
// selects required (fuzzy-AND / guarded) vs supporting (fuzzy-OR).
export interface EvidenceSpec {
  source: string;
  target: string;
  transferFn: TransferFnSpec;
  weight: number;
  role: EvidenceRole;
}

// The wire body for POST /fused/{id}/favorability (mirrors FavorabilityRequest). The backend
// FavorabilitySpec.from_payload reads `evidence[].transferFn`, `method`, `fuzzyAnd`,
// `missingPolicy` — we pass both snake + the camelCase it prefers.
export interface FavorabilityBody {
  project_id: string;
  method: FavorabilityMethod;
  fuzzy_and: FuzzyAnd;
  missing_policy: MissingPolicy;
  evidence: Array<{
    source: string;
    target: string;
    transferFn: TransferFnSpec;
    weight: number;
    role: EvidenceRole;
  }>;
  force_job?: boolean;
}

// The favorability response (FavorabilityResult.to_payload + the route's {mode} wrapper).
export interface FavorabilityResult {
  mode: "sync" | "job";
  job_id?: string;
  method?: string;
  output_property?: string;
  model_id?: string;
  confidence_model_id?: string;
  overlap_model_id?: string;
  burden_model_id?: string;
  n_valid?: number;
  n_missing_required?: number;
  n_required?: number;
  n_supporting?: number;
}

// GET /transforms palette entry (Transform.describe()). Used to seed the evidence target +
// the suggested membership defaults; read defensively (only id/target/output needed here).
export interface TransformDescriptor {
  id: string;
  version?: string;
  title?: string;
  target?: string;
  output?: { name?: string; unit?: string; colormap?: string | null };
  params?: Array<{ name: string; type?: string; default?: unknown }>;
  assumptions?: string[];
  calibration_status?: string;
}

// ── membership evaluation (doc 07 §4.6 — raw → [0,1]) ────────────────────────────────────
// Mirrors geosim.fusion.favorability.membership EXACTLY so the panel's curve preview matches
// the backend volume. A non-finite (no-coverage) input stays NaN — membership never invents
// evidence where a layer's footprint does not reach (doc 07 §2.3). In-band values clamp to
// [0,1]. Invalid params throw (same guards as the backend) so the editor can surface them.

export function membership(value: number, tf: TransferFnSpec): number {
  if (!Number.isFinite(value)) return NaN;
  if (tf.type === "ramp") {
    if (tf.lo == null || tf.hi == null) throw new Error("ramp transferFn needs lo and hi");
    const lo = tf.lo;
    const hi = tf.hi;
    if (hi === lo) throw new Error("ramp transferFn needs lo != hi");
    const m = (value - lo) / (hi - lo); // hi<lo ⇒ descending ramp automatically
    return clamp01(m);
  }
  if (tf.type === "sigmoid") {
    if (tf.center == null || tf.k == null) throw new Error("sigmoid transferFn needs center and k");
    return 1 / (1 + Math.exp(-tf.k * (value - tf.center)));
  }
  if (tf.type === "gaussian-band") {
    if (tf.center == null || tf.width == null)
      throw new Error("gaussian-band transferFn needs center and width");
    if (tf.width <= 0) throw new Error("gaussian-band transferFn needs width > 0");
    const z = (value - tf.center) / tf.width;
    return Math.exp(-0.5 * z * z);
  }
  throw new Error(`unknown transferFn type ${tf.type}`);
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

// Sample a membership curve across [x0,x1] into `n` (x, m) pairs — the curve-editor preview.
// Skips throwing on incomplete params (returns an empty array) so the editor can render a
// blank plot while the user is still filling the curve in.
export function sampleMembershipCurve(
  tf: TransferFnSpec,
  x0: number,
  x1: number,
  n = 64,
): Array<[number, number]> {
  const out: Array<[number, number]> = [];
  if (!(n > 1) || !Number.isFinite(x0) || !Number.isFinite(x1) || x1 <= x0) return out;
  for (let i = 0; i < n; i++) {
    const x = x0 + ((x1 - x0) * i) / (n - 1);
    let m: number;
    try {
      m = membership(x, tf);
    } catch {
      return [];
    }
    out.push([x, m]);
  }
  return out;
}

// Suggest sensible default membership parameters for a transfer type over an observed value
// range [lo,hi] (doc 06 §3.2 registry-seeded defaults). An ASCENDING ramp by default
// (favorable = high), a sigmoid centred mid-range, a gaussian band peaking mid-range. This is
// only a seed — the user re-shapes the curve afterwards.
export function defaultTransferFn(type: TransferType, lo: number, hi: number): TransferFnSpec {
  const span = hi > lo ? hi - lo : 1;
  const mid = (lo + hi) / 2;
  if (type === "ramp") return { type, lo, hi };
  if (type === "sigmoid") return { type, center: mid, k: 4 / span };
  return { type, center: mid, width: span / 4 };
}

// ── the FavorabilitySpec builder (the load-bearing wire logic) ───────────────────────────
// Build + VALIDATE the POST body from the panel state. Mirrors the backend
// FavorabilitySpec.__post_init__ guards so we fail fast in the UI with a useful message
// rather than round-tripping a 400: ≥1 evidence layer; valid method / fuzzy_and /
// missing_policy enums; non-negative weights; each evidence's transferFn params present for
// its type. Returns the exact JSON the route accepts (camelCase transferFn keys included).

export interface FavorabilityFormState {
  projectId: string;
  method: FavorabilityMethod;
  fuzzyAnd: FuzzyAnd;
  missingPolicy: MissingPolicy;
  evidence: EvidenceSpec[];
  forceJob?: boolean;
}

export function buildFavorabilitySpec(state: FavorabilityFormState): FavorabilityBody {
  if (!FAVORABILITY_METHODS.includes(state.method))
    throw new Error(`method must be one of ${FAVORABILITY_METHODS.join("|")}`);
  if (state.method === "bayesian")
    throw new Error("bayesian favorability is deferred (doc 07 §4.6) — use fuzzy or weighted");
  if (!FUZZY_AND_OPS.includes(state.fuzzyAnd))
    throw new Error(`fuzzyAnd must be one of ${FUZZY_AND_OPS.join("|")}`);
  if (!MISSING_POLICIES.includes(state.missingPolicy))
    throw new Error(`missingPolicy must be one of ${MISSING_POLICIES.join("|")}`);
  if (!state.evidence || state.evidence.length === 0)
    throw new Error("favorability needs at least one evidence layer (doc 07 §4.6)");

  const evidence = state.evidence.map((ev, i) => {
    if (!ev.source) throw new Error(`evidence[${i}] needs a source layer`);
    if (!ev.target) throw new Error(`evidence[${i}] needs a target property`);
    if (!(ev.weight >= 0)) throw new Error(`evidence[${i}].weight must be >= 0`);
    if (!EVIDENCE_ROLES.includes(ev.role))
      throw new Error(`evidence[${i}].role must be required|supporting`);
    // Validate the curve params for the chosen type (same guards as membership()).
    validateTransferFn(ev.transferFn, i);
    return {
      source: ev.source,
      target: ev.target,
      transferFn: normalizeTransferFn(ev.transferFn),
      weight: ev.weight,
      role: ev.role,
    };
  });

  return {
    project_id: state.projectId,
    method: state.method,
    fuzzy_and: state.fuzzyAnd,
    missing_policy: state.missingPolicy,
    evidence,
    ...(state.forceJob ? { force_job: true } : {}),
  };
}

function validateTransferFn(tf: TransferFnSpec, i: number): void {
  if (!TRANSFER_TYPES.includes(tf.type))
    throw new Error(`evidence[${i}].transferFn.type must be one of ${TRANSFER_TYPES.join("|")}`);
  if (tf.type === "ramp") {
    if (tf.lo == null || tf.hi == null)
      throw new Error(`evidence[${i}] ramp transferFn needs lo and hi`);
    if (tf.lo === tf.hi) throw new Error(`evidence[${i}] ramp transferFn needs lo != hi`);
  } else if (tf.type === "sigmoid") {
    if (tf.center == null || tf.k == null)
      throw new Error(`evidence[${i}] sigmoid transferFn needs center and k`);
  } else {
    if (tf.center == null || tf.width == null)
      throw new Error(`evidence[${i}] gaussian-band transferFn needs center and width`);
    if (!(tf.width > 0)) throw new Error(`evidence[${i}] gaussian-band transferFn needs width > 0`);
  }
}

// Emit only the fields relevant to the type (the backend tolerates extra null keys, but a
// clean payload is easier to read in provenance / DevTools).
function normalizeTransferFn(tf: TransferFnSpec): TransferFnSpec {
  if (tf.type === "ramp") return { type: "ramp", lo: tf.lo, hi: tf.hi };
  if (tf.type === "sigmoid") return { type: "sigmoid", center: tf.center, k: tf.k };
  return { type: "gaussian-band", center: tf.center, width: tf.width };
}

// ── API client (mirrors geosim/api/fusion.py) ────────────────────────────────────────────

async function postJSON<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`;
    try {
      const j = (await r.json()) as { detail?: string };
      if (j && j.detail) detail = j.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`${url} failed: ${detail}`);
  }
  return (await r.json()) as T;
}

// GET /transforms — the transform-registry palette (doc 07 §6). Tolerant of {transforms:[…]}.
export async function listTransforms(): Promise<TransformDescriptor[]> {
  const r = await fetch(`/transforms`);
  if (!r.ok) throw new Error(`transforms fetch failed: ${r.status} ${r.statusText}`);
  const body = (await r.json()) as unknown;
  if (Array.isArray(body)) return body as TransformDescriptor[];
  if (body && typeof body === "object") {
    const o = body as Record<string, unknown>;
    if (Array.isArray(o.transforms)) return o.transforms as TransformDescriptor[];
  }
  return [];
}

// POST /fused/{id}/favorability — compute the favorability volume + honesty diagnostics.
export async function computeFavorability(
  gridId: string,
  body: FavorabilityBody,
): Promise<FavorabilityResult> {
  return postJSON<FavorabilityResult>(
    `/fused/${encodeURIComponent(gridId)}/favorability`,
    body,
  );
}
