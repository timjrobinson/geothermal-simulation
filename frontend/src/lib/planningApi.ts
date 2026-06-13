// Planning API client (doc 09 §10 backend surface). Thin fetch wrappers over the planning
// router (geosim/api/planning.py); all request-body SHAPING lives in lib/planning.ts (pure +
// unit-tested) so this module only does the network I/O. Response payloads are the camelCase
// shapes the router emits (DrillTarget.to_payload / _solve_payload / PredictedLog.to_payload).

import {
  wellCreateBody,
  solveBody,
  predictBody,
  type DesignParams,
  type RiskWeights,
  type SolveResult,
  type WellPositions,
  type PredictedLog,
} from "./planning";

// ── DrillTarget create + enrich (POST /projects/{pid}/targets, doc 09 §3.3) ────────────────

export interface TargetEnrichment {
  temperatureC: { value: number | null; sigma?: number | null; confidence?: number | null } | null;
  favorability: { value: number | null; sigma?: number | null } | null;
  lithology: string | null;
  depthTVD_m: number | null;
  modelVersion: string | null;
  properties: Record<string, { value: number | null; sigma?: number | null }>;
}

export interface DrillTargetOut {
  id: string;
  name: string;
  projectId: string;
  kind: string;
  location: { x: number; y: number; z: number };
  tolerance: { radius_m: number; tvd_window_m: number };
  desiredTemperatureC: number | null;
  minTemperatureC: number | null;
  geologicalUnit: string | null;
  rationale: string | null;
  sampled: TargetEnrichment | null;
}

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    let detail = `${r.status} ${r.statusText}`;
    try {
      const j = await r.json();
      if (j && j.detail) detail = String(j.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return (await r.json()) as T;
}

export async function createTarget(
  pid: string,
  fusedModelId: string,
  location: [number, number, number],
  opts: {
    name?: string;
    kind?: "point" | "zone";
    toleranceRadius_m?: number;
    tvdWindow_m?: number;
    desiredTemperatureC?: number | null;
    minTemperatureC?: number | null;
    kbElev_m?: number;
  } = {},
): Promise<DrillTargetOut> {
  return postJson<DrillTargetOut>(
    `/projects/${encodeURIComponent(pid)}/targets`,
    {
      fused_model_id: fusedModelId,
      name: opts.name ?? "target",
      kind: opts.kind ?? "point",
      location,
      tolerance_radius_m: opts.toleranceRadius_m ?? 50,
      tvd_window_m: opts.tvdWindow_m ?? 25,
      desired_temperature_c: opts.desiredTemperatureC ?? null,
      min_temperature_c: opts.minTemperatureC ?? null,
      kb_elev_m: opts.kbElev_m ?? 0,
    },
  );
}

// ── PlannedWell create (intent mode) + solve + positions + predict ─────────────────────────

export interface WellOut {
  id: string;
  projectId: string;
  props: Record<string, unknown>;
  solve?: SolveResult;
}

export async function createWell(
  pid: string,
  name: string,
  wellhead: [number, number],
  design: DesignParams,
  opts: { kbElev_m?: number; targetIds?: string[] } = {},
): Promise<WellOut> {
  const body = wellCreateBody(name, wellhead, design, opts);
  return postJson<WellOut>(`/projects/${encodeURIComponent(pid)}/wells`, body);
}

export async function solveWell(
  wid: string,
  design: DesignParams,
): Promise<WellOut> {
  return postJson<WellOut>(`/wells/${encodeURIComponent(wid)}/solve`, solveBody(design));
}

export async function fetchWellPositions(wid: string): Promise<WellPositions> {
  const r = await fetch(`/wells/${encodeURIComponent(wid)}/positions`);
  if (!r.ok) throw new Error(`positions fetch failed: ${r.status} ${r.statusText}`);
  return (await r.json()) as WellPositions;
}

export async function predictWell(
  wid: string,
  fusedModelId: string,
  opts: {
    mdStep_m?: number;
    targetId?: string | null;
    riskWeights?: RiskWeights;
    favorabilityThreshold?: number;
    fractureThreshold?: number;
  } = {},
): Promise<PredictedLog> {
  const body = predictBody(fusedModelId, opts);
  return postJson<PredictedLog>(`/wells/${encodeURIComponent(wid)}/predict`, body);
}

// ── Export (GET /wells/{wid}/export, doc 09 §9) — returns a download URL ────────────────────

export type ExportFormat = "csv-survey" | "csv-log" | "witsml";

// Build the export URL (the GET endpoint streams the file as an attachment). The panel opens
// this in a new tab / anchor so the browser handles the download.
export function exportUrl(
  wid: string,
  fmt: ExportFormat,
  opts: {
    version?: string;
    units?: "metric" | "field";
    fusedModelId?: string | null;
    mdStep_m?: number;
    targetId?: string | null;
  } = {},
): string {
  const q = new URLSearchParams({ fmt });
  if (opts.version) q.set("version", opts.version);
  if (opts.units) q.set("units", opts.units);
  if (opts.fusedModelId) q.set("fused_model_id", opts.fusedModelId);
  if (opts.mdStep_m != null) q.set("md_step_m", String(opts.mdStep_m));
  if (opts.targetId) q.set("target_id", opts.targetId);
  return `/wells/${encodeURIComponent(wid)}/export?${q.toString()}`;
}
