// WebGL2 GLSL shaders for the M1 volume ray-marcher + orthogonal slice (doc 06 §3.1, §4).
//
// Both passes sample the SAME Data3DTexture (uVolume) through the SAME transfer-function
// LUT (uTransferFn) so slice and volume colours stay locked (doc 06 §4.1). WebGL2 only
// (GLSL ES 3.00, sampler3D); R3F/THREE sets the GLSL3 preamble when glslVersion=GLSL3.
//
// Coordinate convention: the proxy box geometry spans the volume's Engineering-frame AABB
// (XYZ metres, Z-up). The vertex shader passes world position; the fragment shader builds
// the ray in WORLD (Engineering) space, intersects it with both the volume AABB
// (uBoxMin/uBoxMax) and the user clip box (uClipMin/uClipMax), then marches, converting
// each world sample point to the volume's [0,1]^3 texture coordinate. Texture coord t maps
// linearly across the AABB: t = (p - boxMin) / (boxMax - boxMin). Because the buffer is
// (z,y,x) C-contiguous uploaded as a Data3DTexture sized (nx, ny, nz), texcoord.x↔X,
// .y↔Y, .z↔Z directly (we upload with width=nx, height=ny, depth=nz).

export const VOLUME_VERT = /* glsl */ `
out vec3 vWorldPos;

void main() {
  // Box proxy vertices are already in Engineering (world) metres via the mesh transform.
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vWorldPos = wp.xyz;
  gl_Position = projectionMatrix * viewMatrix * wp;
}
`;

export const VOLUME_FRAG = /* glsl */ `
precision highp float;
precision highp sampler3D;

in vec3 vWorldPos;
out vec4 fragColor;

uniform sampler3D uVolume;
uniform sampler2D uTransferFn;   // 256x1 RGBA LUT

uniform vec3  uBoxMin;           // volume AABB (Engineering m)
uniform vec3  uBoxMax;
uniform vec3  uClipMin;          // user clip box (Engineering m)
uniform vec3  uClipMax;
uniform vec3  uCameraPos;        // world-space camera (Engineering m)

uniform float uDomainMin;        // raw value -> t domain
uniform float uDomainMax;
uniform float uLog;              // 1.0 = log scaling, 0.0 = linear
uniform float uOpacityGain;
uniform int   uSteps;            // max ray-march steps
uniform float uRefStep;          // reference step (m) for opacity correction
uniform int   uBlend;            // 0=over (alpha), 1=additive, 2=MIP, 3=minIP (doc 06 §3.3)

// Ray/box intersection (slab method). Returns vec2(tNear, tFar); tFar<tNear => miss.
vec2 intersectBox(vec3 ro, vec3 rd, vec3 bmin, vec3 bmax) {
  vec3 inv = 1.0 / rd;
  vec3 t0 = (bmin - ro) * inv;
  vec3 t1 = (bmax - ro) * inv;
  vec3 tsmall = min(t0, t1);
  vec3 tbig   = max(t0, t1);
  float tNear = max(max(tsmall.x, tsmall.y), tsmall.z);
  float tFar  = min(min(tbig.x, tbig.y), tbig.z);
  return vec2(tNear, tFar);
}

// raw value -> normalized t in [0,1] over the (optionally log) domain (doc 06 §3.1).
float applyScaling(float raw) {
  float lo = uDomainMin;
  float hi = uDomainMax;
  float v = raw;
  if (uLog > 0.5) {
    // Log domain: guard non-positive values; domain bounds are linear-space values.
    float eps = 1e-12;
    v  = log(max(raw, eps));
    lo = log(max(uDomainMin, eps));
    hi = log(max(uDomainMax, eps));
  }
  float denom = max(hi - lo, 1e-12);
  return clamp((v - lo) / denom, 0.0, 1.0);
}

void main() {
  vec3 ro = uCameraPos;
  vec3 rd = normalize(vWorldPos - uCameraPos);

  // Intersect the ray with the volume AABB and the user clip box; march their overlap.
  vec2 tv = intersectBox(ro, rd, uBoxMin, uBoxMax);
  vec2 tc = intersectBox(ro, rd, uClipMin, uClipMax);
  float t0 = max(max(tv.x, tc.x), 0.0);
  float t1 = min(tv.y, tc.y);
  if (t1 <= t0) discard;

  vec3 span = uBoxMax - uBoxMin;
  float maxDim = max(span.x, max(span.y, span.z));
  float dt = maxDim / float(uSteps);

  vec4 acc = vec4(0.0);
  // MIP / minIP track the extremum sample (doc 06 §3.3); seed minIP high so any sample wins.
  float extN = (uBlend == 3) ? 1.0 : 0.0;     // normalized-t extremum
  vec4  extC = vec4(0.0);                      // colour at the extremum
  bool  hit  = false;
  float t = t0;
  for (int i = 0; i < 4096; ++i) {
    if (i >= uSteps) break;
    if (t > t1) break;
    vec3 p = ro + rd * t;
    vec3 uvw = (p - uBoxMin) / span;          // -> [0,1]^3 texcoord
    float raw = texture(uVolume, uvw).r;
    t += dt;
    if (isnan(raw)) continue;                 // no-data skip (doc 06 §3.1)
    float vn = applyScaling(raw);
    vec4 c = texture(uTransferFn, vec2(vn, 0.5));
    if (uBlend == 2) {                        // MIP — keep the max value
      if (!hit || vn > extN) { extN = vn; extC = c; }
      hit = true;
    } else if (uBlend == 3) {                 // minIP — keep the min value
      if (!hit || vn < extN) { extN = vn; extC = c; }
      hit = true;
    } else {                                  // over / additive — accumulate front-to-back
      float a = 1.0 - pow(1.0 - clamp(c.a * uOpacityGain, 0.0, 1.0), dt / uRefStep);
      if (uBlend == 1) {                      // additive — emission, no occlusion
        acc.rgb += a * c.rgb;
        acc.a   = max(acc.a, a);
      } else {                                // over — front-to-back alpha
        acc.rgb += (1.0 - acc.a) * a * c.rgb;
        acc.a   += (1.0 - acc.a) * a;
        if (acc.a > 0.98) break;              // early ray termination
      }
    }
  }
  if (uBlend == 2 || uBlend == 3) {
    if (!hit) discard;
    fragColor = vec4(extC.rgb, clamp(extC.a * uOpacityGain, 0.0, 1.0));
    return;
  }
  if (acc.a <= 0.0) discard;
  fragColor = vec4(acc.rgb, clamp(acc.a, 0.0, 1.0));
}
`;

// Orthogonal slice: a quad placed in the scene whose fragment shader samples the SAME
// 3D texture + same transfer fn (doc 06 §4.1). The quad carries a per-vertex 3D texcoord
// (uvw) interpolated across its face; we look it up directly.
export const SLICE_VERT = /* glsl */ `
in vec3 uvw;            // per-vertex volume texcoord [0,1]^3
out vec3 vUVW;

void main() {
  vUVW = uvw;
  gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(position, 1.0);
}
`;

export const SLICE_FRAG = /* glsl */ `
precision highp float;
precision highp sampler3D;

in vec3 vUVW;
out vec4 fragColor;

uniform sampler3D uVolume;
uniform sampler2D uTransferFn;
uniform vec3  uClipMin;          // clip box in [0,1]^3 volume space
uniform vec3  uClipMax;
uniform float uDomainMin;
uniform float uDomainMax;
uniform float uLog;
uniform float uSliceOpacity;

float applyScaling(float raw) {
  float lo = uDomainMin;
  float hi = uDomainMax;
  float v = raw;
  if (uLog > 0.5) {
    float eps = 1e-12;
    v  = log(max(raw, eps));
    lo = log(max(uDomainMin, eps));
    hi = log(max(uDomainMax, eps));
  }
  float denom = max(hi - lo, 1e-12);
  return clamp((v - lo) / denom, 0.0, 1.0);
}

void main() {
  vec3 uvw = vUVW;
  // Respect the clip box (in volume [0,1]^3 space).
  if (any(lessThan(uvw, uClipMin)) || any(greaterThan(uvw, uClipMax))) discard;
  if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0)))) discard;
  float raw = texture(uVolume, uvw).r;
  if (isnan(raw)) discard;                    // no-data
  float vn = applyScaling(raw);
  vec4 c = texture(uTransferFn, vec2(vn, 0.5));
  fragColor = vec4(c.rgb, uSliceOpacity);
}
`;
