// Client-side mock volume generator (doc 06 §1.3 — self-contained dev path).
//
// For local dev with NO backend, this synthesizes the same "conductive blob" shape the
// synthetic generator (doc 05) produces: a smooth Gaussian anomaly embedded in a
// background field, with a NaN no-data shell so the no-data skip path (doc 06 §3.1) is
// exercised. Enabled via the ?mock URL param so `npm run dev` shows something and the
// build is self-contained (no backend required to see a render).
//
// The generated DecodedVolume + PropertyModelMeta match the real fetch* shapes so the
// rest of the viewer is identical whether the volume is mock or server-decoded.

import type { PropertyModelMeta } from "./api";
import { type DecodedVolume } from "./volume";

export interface MockVolume {
  meta: PropertyModelMeta;
  volume: DecodedVolume;
}

// Generate a conductive-blob volume of (nz, ny, nx) samples. Values are a log-resistivity-
// like field: a low-resistivity (conductive) Gaussian anomaly dropped into a higher
// background, plus a NaN shell around the outer voxels (no-data) to test masking.
export function makeMockVolume(
  nx = 96,
  ny = 96,
  nz = 64,
): MockVolume {
  const data = new Float32Array(nx * ny * nz);

  // Engineering frame: origin near a floating-origin anchor (doc 01); Z is elevation
  // (up positive), so the volume sits below ground (negative-ish Z) — values arbitrary
  // metres for the mock.
  const ox = 0,
    oy = 0,
    oz = -2000;
  const dx = 25,
    dy = 25,
    dz = 25;

  // Anomaly centre (in sample index space) and radius.
  const cx = nx * 0.45;
  const cy = ny * 0.55;
  const cz = nz * 0.5;
  const sigma = Math.min(nx, ny, nz) * 0.22;

  const bg = 2.5; // background log10(Ω·m)
  const anomalyDepth = 2.0; // how far the conductor drops below background

  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const idx = (k * ny + j) * nx + i;
        // No-data shell on the outermost voxel ring (tests NaN skip).
        if (
          i === 0 ||
          j === 0 ||
          k === 0 ||
          i === nx - 1 ||
          j === ny - 1 ||
          k === nz - 1
        ) {
          data[idx] = NaN;
          continue;
        }
        const r2 =
          (i - cx) * (i - cx) + (j - cy) * (j - cy) + (k - cz) * (k - cz);
        const blob = anomalyDepth * Math.exp(-r2 / (2 * sigma * sigma));
        // Mild vertical gradient so slices read as non-flat.
        const grad = 0.3 * (k / nz);
        data[idx] = bg + grad - blob;
      }
    }
  }

  const volume: DecodedVolume = {
    shape: [nz, ny, nx],
    origin: [oz, oy, ox],
    spacing: [dz, dy, dx],
    data,
  };

  const meta: PropertyModelMeta = {
    id: "mock",
    property: "resistivity",
    canonicalUnit: "log10(ohm.m)",
    scaling: "linear", // values already log10 in the mock
    colormap: "turbo",
    displayRange: [bg - anomalyDepth, bg + 0.5],
    shape: [nz, ny, nx],
    origin: [oz, oy, ox],
    spacing: [dz, dy, dx],
    levels: 1,
    stats: {
      min: bg - anomalyDepth,
      max: bg + 0.5,
      p1: bg - anomalyDepth + 0.1,
      p99: bg + 0.4,
    },
    frame: null,
    hasSigma: false,
  };

  return { meta, volume };
}
