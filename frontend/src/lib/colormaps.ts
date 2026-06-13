// Colormap LUTs for the transfer function (doc 06 §3.2).
//
// A transfer function = 1D colormap LUT × opacity curve baked to a 256×1 RGBA
// DataTexture the shader samples (doc 06 §3.2). This module owns the pure RGB colour
// ramps; the opacity curve and the actual DataTexture are assembled in lib/transferFn.ts.
//
// Ramps are sparse control points (t in [0,1] → [r,g,b] in [0,1]) linearly interpolated
// into the 256-entry LUT. Names match the property-type registry's default colormap
// strings (doc 01 §5) where possible so meta.colormap can be honoured directly.

export type ColorStop = readonly [number, readonly [number, number, number]];

export interface Colormap {
  name: string;
  stops: readonly ColorStop[];
}

// viridis — perceptually uniform; the registry default for many properties.
const viridis: Colormap = {
  name: "viridis",
  stops: [
    [0.0, [0.267, 0.005, 0.329]],
    [0.25, [0.229, 0.322, 0.545]],
    [0.5, [0.127, 0.567, 0.551]],
    [0.75, [0.369, 0.789, 0.383]],
    [1.0, [0.993, 0.906, 0.144]],
  ],
};

// inferno — high-contrast dark→bright; good for thermal fields.
const inferno: Colormap = {
  name: "inferno",
  stops: [
    [0.0, [0.001, 0.0, 0.014]],
    [0.25, [0.258, 0.039, 0.406]],
    [0.5, [0.578, 0.148, 0.404]],
    [0.75, [0.865, 0.317, 0.226]],
    [1.0, [0.988, 0.998, 0.645]],
  ],
};

// turbo — rainbow-like; common for resistivity / velocity.
const turbo: Colormap = {
  name: "turbo",
  stops: [
    [0.0, [0.19, 0.072, 0.232]],
    [0.25, [0.122, 0.633, 0.904]],
    [0.5, [0.473, 0.988, 0.448]],
    [0.75, [0.965, 0.66, 0.155]],
    [1.0, [0.48, 0.016, 0.011]],
  ],
};

// jet — legacy geophysics rainbow (resistivity sections often expect it).
const jet: Colormap = {
  name: "jet",
  stops: [
    [0.0, [0.0, 0.0, 0.5]],
    [0.125, [0.0, 0.0, 1.0]],
    [0.375, [0.0, 1.0, 1.0]],
    [0.625, [1.0, 1.0, 0.0]],
    [0.875, [1.0, 0.0, 0.0]],
    [1.0, [0.5, 0.0, 0.0]],
  ],
};

// gray — neutral.
const gray: Colormap = {
  name: "gray",
  stops: [
    [0.0, [0.0, 0.0, 0.0]],
    [1.0, [1.0, 1.0, 1.0]],
  ],
};

export const COLORMAPS: Record<string, Colormap> = {
  viridis,
  inferno,
  turbo,
  jet,
  gray,
};

export const COLORMAP_NAMES = Object.keys(COLORMAPS);

export const DEFAULT_COLORMAP = "viridis";

// Resolve a registry colormap string to a known ramp, falling back to the default.
export function resolveColormap(name: string | null | undefined): Colormap {
  if (name && name in COLORMAPS) return COLORMAPS[name];
  return COLORMAPS[DEFAULT_COLORMAP];
}

// Sample a colormap at t∈[0,1] (clamped), linearly interpolating between stops.
export function sampleColormap(
  cm: Colormap,
  t: number,
): [number, number, number] {
  const x = Math.min(1, Math.max(0, t));
  const stops = cm.stops;
  if (x <= stops[0][0]) return [...stops[0][1]] as [number, number, number];
  for (let i = 1; i < stops.length; i++) {
    const [t1, c1] = stops[i];
    if (x <= t1) {
      const [t0, c0] = stops[i - 1];
      const span = t1 - t0;
      const f = span > 0 ? (x - t0) / span : 0;
      return [
        c0[0] + (c1[0] - c0[0]) * f,
        c0[1] + (c1[1] - c0[1]) * f,
        c0[2] + (c1[2] - c0[2]) * f,
      ];
    }
  }
  return [...stops[stops.length - 1][1]] as [number, number, number];
}

// Bake a colormap into an N-entry RGB Uint8 ramp (no alpha; opacity added separately).
export function bakeColormapRGB(cm: Colormap, n = 256): Uint8Array {
  const out = new Uint8Array(n * 3);
  for (let i = 0; i < n; i++) {
    const t = n > 1 ? i / (n - 1) : 0;
    const [r, g, b] = sampleColormap(cm, t);
    out[i * 3 + 0] = Math.round(r * 255);
    out[i * 3 + 1] = Math.round(g * 255);
    out[i * 3 + 2] = Math.round(b * 255);
  }
  return out;
}
