// WebGL2 GLSL for the M2+ brick-streaming ray-marcher (doc 06 §3.4). The shader walks a
// PAGE TABLE (node -> atlas slot) per sample, samples the resident brick from the brick-pool
// ATLAS Data3DTexture, and FALLS BACK to a coarser resident level on a miss so there are NO
// HOLES (doc 06 §3.4 "falls back to a coarser resident level on miss"). The coarsest level is
// always resident (pinned in the pool) so the volume is never blank.
//
// Encoding (matches lib/brickPool.ts):
//   - uAtlas: the brick-pool atlas, a sampler3D of (sx*edge, sy*edge, sz*edge) f32 voxels.
//   - uPageTable: a 2D R32I-equivalent lookup. We encode it as a FLOAT texture (RedFormat,
//     FloatType) of width=uPageW, height=ceil(totalPages/uPageW); each texel holds the atlas
//     SLOT index for one brick (or -1.0 when not resident). The flat page index per brick is
//     pageOffset[level] + ((bz*gy+by)*gx+bx) — uLevelOffset/uLevelGrid carry the per-level
//     blocks (mirrors PageTable.levelOffset/levelGrids). MAX_LEVELS caps the uniform arrays.
//   - To resolve a sample at world p: pick the FINEST level whose brick is resident, walk down
//     from the finest selected level to the coarsest until getPage != -1, then sample the
//     atlas slot at the brick-local uvw.
//
// The volume/clip intersection, scaling, transfer-fn lookup, blend modes and opacity
// correction are identical to the single-resident shader (lib/shaders.ts) so streamed and
// resident volumes look the same.

export const MAX_LEVELS = 12;

export const BRICK_VOLUME_VERT = /* glsl */ `
out vec3 vWorldPos;
void main() {
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vWorldPos = wp.xyz;
  gl_Position = projectionMatrix * viewMatrix * wp;
}
`;

export const BRICK_VOLUME_FRAG = /* glsl */ `
precision highp float;
precision highp sampler3D;

in vec3 vWorldPos;
out vec4 fragColor;

uniform sampler3D uAtlas;        // brick-pool atlas (f32, RedFormat)
uniform sampler2D uPageTable;    // flat page table: texel.r = atlas slot (or -1)
uniform sampler2D uTransferFn;   // 256x1 RGBA LUT

uniform vec3  uBoxMin;           // volume AABB (Engineering m)
uniform vec3  uBoxMax;
uniform vec3  uClipMin;          // user clip box (Engineering m)
uniform vec3  uClipMax;
uniform vec3  uCameraPos;

uniform float uDomainMin;
uniform float uDomainMax;
uniform float uLog;
uniform float uOpacityGain;
uniform int   uSteps;
uniform float uRefStep;
uniform int   uBlend;            // 0=over 1=additive 2=MIP 3=minIP

// Atlas layout.
uniform int   uBrickEdge;        // voxels per brick edge
uniform ivec3 uAtlasGrid;        // slots per axis [sx,sy,sz]
uniform vec3  uAtlasDim;         // atlas voxel dims [w,h,d] = grid*edge (float for division)

// Page-table layout.
uniform int   uPageW;            // page-table texture width (texels)
uniform int   uPageH;            // page-table texture height (texels)
uniform int   uMaxLevel;         // coarsest level index (== levels-1)
uniform int   uLevelOffset[${MAX_LEVELS}]; // flat page offset of each level block
uniform ivec3 uLevelGrid[${MAX_LEVELS}];   // [gz,gy,gx] brick grid per level
uniform vec3  uLevelVoxels[${MAX_LEVELS}]; // [nz,ny,nx] voxel extent per level (for uvw)

const float NAN_VAL = 0.0 / 0.0;

vec2 intersectBox(vec3 ro, vec3 rd, vec3 bmin, vec3 bmax) {
  vec3 inv = 1.0 / rd;
  vec3 t0 = (bmin - ro) * inv;
  vec3 t1 = (bmax - ro) * inv;
  vec3 tsmall = min(t0, t1);
  vec3 tbig   = max(t0, t1);
  return vec2(max(max(tsmall.x, tsmall.y), tsmall.z),
              min(min(tbig.x, tbig.y), tbig.z));
}

float applyScaling(float raw) {
  float lo = uDomainMin, hi = uDomainMax, v = raw;
  if (uLog > 0.5) {
    float eps = 1e-12;
    v  = log(max(raw, eps));
    lo = log(max(uDomainMin, eps));
    hi = log(max(uDomainMax, eps));
  }
  return clamp((v - lo) / max(hi - lo, 1e-12), 0.0, 1.0);
}

// Read the page table at a flat brick index -> atlas slot (or -1.0).
float pageLookup(int flatIndex) {
  int px = flatIndex % uPageW;
  int py = flatIndex / uPageW;
  vec2 uv = (vec2(float(px), float(py)) + 0.5) / vec2(float(uPageW), float(uPageH));
  return texture(uPageTable, uv).r;
}

// Sample the atlas for the given slot at brick-local uvw in [0,1]^3.
float sampleAtlas(int slot, vec3 luvw) {
  int gx = uAtlasGrid.x;
  int gy = uAtlasGrid.y;
  int sx = slot % gx;
  int sy = (slot / gx) % gy;
  int sz = slot / (gx * gy);
  float e = float(uBrickEdge);
  // Brick voxel origin in the atlas; sample at the brick-local position (clamped inside the
  // brick to avoid bleeding into neighbour slots under linear filtering).
  vec3 originVox = vec3(float(sx), float(sy), float(sz)) * e;
  vec3 inBrick = clamp(luvw, 0.5 / e, 1.0 - 0.5 / e) * e; // voxel-space, half-texel inset
  vec3 atlasVox = originVox + inBrick;
  vec3 atlasUVW = atlasVox / uAtlasDim;
  return texture(uAtlas, atlasUVW).r;
}

// Resolve the property value at world position p by walking the page table from the FINEST
// level (0) up to the coarsest, sampling the first RESIDENT brick (no holes, doc 06 §3.4).
// Returns the raw value (NaN if no resident brick covers p — caller skips it).
float sampleVolume(vec3 p) {
  vec3 frac = (p - uBoxMin) / (uBoxMax - uBoxMin); // [0,1]^3 over the whole volume
  if (any(lessThan(frac, vec3(0.0))) || any(greaterThan(frac, vec3(1.0)))) return NAN_VAL;

  for (int level = 0; level <= ${MAX_LEVELS - 1}; ++level) {
    if (level > uMaxLevel) break;
    ivec3 grid = uLevelGrid[level];          // [gz,gy,gx]
    vec3  vox  = uLevelVoxels[level];         // [nz,ny,nx]
    float e = float(uBrickEdge);
    // Voxel coordinate at this level, then brick index + brick-local uvw.
    vec3 voxCoord = frac * vox;               // [0,nz/ny/nx]
    int bz = int(floor(voxCoord.z / e));
    int by = int(floor(voxCoord.y / e));
    int bx = int(floor(voxCoord.x / e));
    bz = clamp(bz, 0, grid.x - 1);
    by = clamp(by, 0, grid.y - 1);
    bx = clamp(bx, 0, grid.z - 1);
    int flat = uLevelOffset[level] + (bz * grid.y + by) * grid.z + bx;
    float slotF = pageLookup(flat);
    if (slotF >= 0.0) {
      // brick-local uvw within this brick (z,y,x -> we sample atlas in (x,y,z) order below).
      vec3 local = vec3(
        voxCoord.x - float(bx) * e,
        voxCoord.y - float(by) * e,
        voxCoord.z - float(bz) * e
      ) / e;
      return sampleAtlas(int(slotF + 0.5), local);
    }
    // miss -> try the next coarser level (no holes)
  }
  return NAN_VAL;
}

void main() {
  vec3 ro = uCameraPos;
  vec3 rd = normalize(vWorldPos - uCameraPos);
  vec2 tv = intersectBox(ro, rd, uBoxMin, uBoxMax);
  vec2 tc = intersectBox(ro, rd, uClipMin, uClipMax);
  float t0 = max(max(tv.x, tc.x), 0.0);
  float t1 = min(tv.y, tc.y);
  if (t1 <= t0) discard;

  vec3 span = uBoxMax - uBoxMin;
  float dt = max(span.x, max(span.y, span.z)) / float(uSteps);

  vec4 acc = vec4(0.0);
  float extN = (uBlend == 3) ? 1.0 : 0.0;
  vec4  extC = vec4(0.0);
  bool  hit = false;
  float t = t0;
  for (int i = 0; i < 4096; ++i) {
    if (i >= uSteps) break;
    if (t > t1) break;
    vec3 p = ro + rd * t;
    float raw = sampleVolume(p);
    t += dt;
    if (isnan(raw)) continue;
    float vn = applyScaling(raw);
    vec4 c = texture(uTransferFn, vec2(vn, 0.5));
    if (uBlend == 2) {
      if (!hit || vn > extN) { extN = vn; extC = c; }
      hit = true;
    } else if (uBlend == 3) {
      if (!hit || vn < extN) { extN = vn; extC = c; }
      hit = true;
    } else {
      float a = 1.0 - pow(1.0 - clamp(c.a * uOpacityGain, 0.0, 1.0), dt / uRefStep);
      if (uBlend == 1) {
        acc.rgb += a * c.rgb;
        acc.a = max(acc.a, a);
      } else {
        acc.rgb += (1.0 - acc.a) * a * c.rgb;
        acc.a   += (1.0 - acc.a) * a;
        if (acc.a > 0.98) break;
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
