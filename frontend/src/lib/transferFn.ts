// Transfer function → 256×1 RGBA DataTexture (doc 06 §3.2).
//
// A transfer function = 1D colormap LUT × opacity curve, baked to a small 256×1 RGBA
// DataTexture the shader samples (doc 06 §3.2). The colour comes from a named ramp
// (lib/colormaps.ts); the alpha is a simple ramp here (linear from 0→max across the
// domain) which the M1 control panel scales by an opacity gain. The shader maps a raw
// scalar to a domain-normalized t (log/linear, doc 06 §3.1 applyScaling) and looks up
// this LUT — so changing the colormap/domain only re-bakes a 256-texel texture, no
// volume refetch (doc 06 §9.2 "edits push a new LUT texture, no refetch").

import * as THREE from "three";
import { bakeColormapRGB, resolveColormap } from "./colormaps";

export type ScalingMode = "linear" | "log";

export interface TransferFnSpec {
  colormap: string; // ramp name (lib/colormaps.ts)
  domainMin: number; // raw value mapped to t=0 (pre-scaling)
  domainMax: number; // raw value mapped to t=1
  scaling: ScalingMode; // log vs linear value→t (doc 06 §3.1)
  opacity: number; // global opacity gain 0..1 (uOpacityGain)
  invert: boolean; // flip the colour ramp
  // "Isolate band" (doc 06 §9.2): when enabled, only the normalized-t window
  // [bandMin, bandMax] keeps its opacity; everything outside is forced transparent so a
  // single value range (e.g. the conductive anomaly) is isolated. Optional/back-compat:
  // absent => no band (full ramp opaque as before).
  bandEnabled?: boolean;
  bandMin?: number; // normalized t in [0,1]
  bandMax?: number; // normalized t in [0,1]
}

export const LUT_SIZE = 256;

// Bake the colour+opacity LUT into an RGBA Uint8 buffer (length LUT_SIZE*4). The alpha
// channel rises linearly across the LUT (0 at t=0 → opacity*255 at t=1) so low-domain
// "background" stays transparent and high-domain anomalies are opaque — a sane default
// for the conductive/thermal blob without an opacity-curve editor (deferred to M2, §9.2).
export function bakeTransferFnRGBA(spec: TransferFnSpec): Uint8Array {
  const cm = resolveColormap(spec.colormap);
  const rgb = bakeColormapRGB(cm, LUT_SIZE);
  const out = new Uint8Array(LUT_SIZE * 4);
  // "Isolate band" window (doc 06 §9.2). Tolerant of swapped/absent bounds.
  const bandOn = spec.bandEnabled === true;
  const b0 = Math.min(spec.bandMin ?? 0, spec.bandMax ?? 1);
  const b1 = Math.max(spec.bandMin ?? 0, spec.bandMax ?? 1);
  for (let i = 0; i < LUT_SIZE; i++) {
    const src = spec.invert ? LUT_SIZE - 1 - i : i;
    out[i * 4 + 0] = rgb[src * 3 + 0];
    out[i * 4 + 1] = rgb[src * 3 + 1];
    out[i * 4 + 2] = rgb[src * 3 + 2];
    const t = LUT_SIZE > 1 ? i / (LUT_SIZE - 1) : 0;
    let a = t * spec.opacity;
    if (bandOn && (t < b0 || t > b1)) a = 0; // outside the isolate band -> transparent
    out[i * 4 + 3] = Math.round(a * 255);
  }
  return out;
}

// Build (or update) the 256×1 RGBA DataTexture for a transfer function.
export function makeTransferFnTexture(spec: TransferFnSpec): THREE.DataTexture {
  const tex = new THREE.DataTexture(
    bakeTransferFnRGBA(spec) as unknown as BufferSource,
    LUT_SIZE,
    1,
    THREE.RGBAFormat,
    THREE.UnsignedByteType,
  );
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.needsUpdate = true;
  return tex;
}

// Re-bake an existing texture in place (avoids reallocating the GPU texture on edits).
export function updateTransferFnTexture(
  tex: THREE.DataTexture,
  spec: TransferFnSpec,
): void {
  const data = tex.image.data as Uint8Array;
  const baked = bakeTransferFnRGBA(spec);
  data.set(baked);
  tex.needsUpdate = true;
}
