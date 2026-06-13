// Linked-brushing core (doc 06 §10.3, doc 07 §3.2). The viewer owns ONLY the linking and
// brushing (not the statistics): it maps a cross-plot selection on to the 3D scene and back.
//
// Two directions, both pure (no THREE / DOM / fetch) so they are unit-testable headlessly:
//   1. cross-plot brush → 3D highlight: a set of selected sample ROWS (local indices into
//      FusedSampleOut.features) → a boolean voxel mask over the fused grid, then a
//      DecodedVolume the viewer renders as a "selection mask" overlay layer. This mirrors
//      the backend geosim.fusion.selection_to_mask EXACTLY (cell_index[row] → True).
//   2. 3D pick → cross-plot inspector: an Engineering (x,y,z) pick point → the nearest
//      sampled cell → that cell's multi-property feature vector (doc 04 sample).

import type { FusedSampleOut } from "./fusion";
import type { DecodedVolume } from "./volume";

// Selected sample ROWS → flat cell indices over the (nz,ny,nx) grid. `cell_index[row]` is
// the backend-supplied flattened (z,y,x) index of each retained cell, so this is the exact
// frontend twin of geosim.fusion.selection_to_mask (which sets mask[cell_index[sel]] = True).
export function selectionToCellIndices(
  sample: FusedSampleOut,
  selectedRows: number[],
): number[] {
  const out: number[] = [];
  for (const row of selectedRows) {
    if (row >= 0 && row < sample.cell_index.length) out.push(sample.cell_index[row]);
  }
  return out;
}

// Selected sample ROWS → boolean voxel mask (flattened C-order (z,y,x), x fastest), the
// frontend twin of selection_to_mask's returned (nz,ny,nx) boolean volume.
export function selectionToMask(
  sample: FusedSampleOut,
  selectedRows: number[],
): Uint8Array {
  const [nz, ny, nx] = sample.grid_shape;
  const mask = new Uint8Array(nz * ny * nx);
  for (const flat of selectionToCellIndices(sample, selectedRows)) {
    if (flat >= 0 && flat < mask.length) mask[flat] = 1;
  }
  return mask;
}

// Build a DecodedVolume from a selection so the viewer can render it as a highlight overlay
// (doc 06 §10.3 "a selection mask layer"). Selected cells get value 1.0; everything else is
// NaN (the no-data sentinel the volume shader skips), so the overlay paints ONLY the brushed
// voxels. Geometry (origin/spacing/shape) comes from the fused grid so the overlay registers
// exactly with the source volumes in the Engineering Frame.
export function selectionToVolume(
  sample: FusedSampleOut,
  selectedRows: number[],
  grid: { origin: [number, number, number]; spacing: [number, number, number] },
): DecodedVolume {
  const shape = sample.grid_shape;
  const [nz, ny, nx] = shape;
  const data = new Float32Array(nz * ny * nx).fill(NaN);
  for (const flat of selectionToCellIndices(sample, selectedRows)) {
    if (flat >= 0 && flat < data.length) data[flat] = 1.0;
  }
  return { shape, origin: grid.origin, spacing: grid.spacing, data };
}

// A picked voxel's multi-property readout (3D pick → cross-plot inspector, doc 06 §10.3).
export interface VoxelReadout {
  row: number; // local sample row, or -1 if the pick hit no sampled cell
  cellIndex: number; // flat (z,y,x) index, or -1
  coords: [number, number, number]; // Engineering (z,y,x) metres of the matched cell
  values: { property: string; value: number }[];
}

// Find the sampled row nearest (in Engineering metres) to a picked Engineering point, and
// return its multi-property values. `pointXYZ` is THREE/scene order (x,y,z); the sample's
// coords are (z,y,x), so we compare in a common metric. Returns row=-1 if the sample is
// empty. A `maxDist` (metres) gate rejects picks that fall outside the sampled region.
export function pickNearestVoxel(
  sample: FusedSampleOut,
  pointXYZ: [number, number, number],
  maxDist = Infinity,
): VoxelReadout | null {
  const [px, py, pz] = pointXYZ;
  let best = -1;
  let bestD = Infinity;
  for (let i = 0; i < sample.coords.length; i++) {
    const [cz, cy, cx] = sample.coords[i];
    const dx = cx - px;
    const dy = cy - py;
    const dz = cz - pz;
    const d = dx * dx + dy * dy + dz * dz;
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  if (best < 0 || Math.sqrt(bestD) > maxDist) {
    return { row: -1, cellIndex: -1, coords: [pz, py, px], values: [] };
  }
  const coords = sample.coords[best] as [number, number, number];
  const values = sample.properties.map((property, c) => ({
    property,
    value: sample.features[best][c],
  }));
  return { row: best, cellIndex: sample.cell_index[best], coords, values };
}
