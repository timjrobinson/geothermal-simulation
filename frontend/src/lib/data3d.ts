// Build a THREE.Data3DTexture from a decoded (z,y,x) volume (doc 06 §1.3, §3.1).
//
// The M1 single-resident path: one Data3DTexture, RedFormat + FloatType, NaN preserved
// as the no-data sentinel (the shader skips it via isnan). The buffer is C-contiguous
// (z,y,x) — i.e. x is the fastest-varying axis — which is exactly the memory order a
// Data3DTexture of (width=nx, height=ny, depth=nz) expects, so no transpose is needed and
// texcoord .x↔X, .y↔Y, .z↔Z line up with the Engineering AABB mapping in shaders.ts.

import * as THREE from "three";
import type { DecodedVolume } from "./volume";

export function makeData3DTexture(vol: DecodedVolume): THREE.Data3DTexture {
  const [nz, ny, nx] = vol.shape;
  // Cast through BufferSource: the lib types parameterize TypedArray over ArrayBuffer
  // specifically, but our Float32Array is backed by a plain ArrayBuffer at runtime.
  const tex = new THREE.Data3DTexture(
    vol.data as unknown as BufferSource,
    nx,
    ny,
    nz,
  );
  tex.format = THREE.RedFormat;
  tex.type = THREE.FloatType;
  // Linear filtering for smooth volume/slice sampling; clamp so out-of-box reads are safe.
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.wrapR = THREE.ClampToEdgeWrapping;
  tex.unpackAlignment = 1;
  tex.needsUpdate = true;
  return tex;
}
