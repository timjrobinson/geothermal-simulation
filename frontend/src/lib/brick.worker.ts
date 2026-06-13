// Brick-decode Web Worker (doc 06 §3.4, §7.5 "decode bricks in Web Workers (transferable
// buffers) to keep the main thread responsive"). It fetches a server-decoded level via the
// doc-06 §1.3 fallback path (GET /property-models/{id}/volume?level=L) ONCE per level, caches
// it, then carves requested 64³ bricks out of it and posts the brick buffers back as
// TRANSFERABLES (zero-copy). Empty (all-NaN) bricks are reported so the renderer skips them.
//
// Vite compiles this with `new Worker(new URL('./brick.worker.ts', import.meta.url),
// { type: 'module' })`. It imports only the PURE brickDecode helpers (no THREE/DOM) so the
// bundle is tiny. The level cache is bounded (LRU over a couple of levels) to cap worker RAM.

/// <reference lib="webworker" />
import { extractBrick, isEmptyBrick } from "./brickDecode";
import { levelVolumeUrl } from "./bricks";
import type { Shape3 } from "./volume";

interface DecodeRequest {
  type: "decode";
  reqId: number;
  id: string; // property-model id
  property: string;
  level: number;
  t: number;
  bz: number;
  by: number;
  bx: number;
  brickEdge: number;
}
interface ConfigMessage {
  type: "config";
  maxCachedLevels?: number;
}
type InMessage = DecodeRequest | ConfigMessage;

interface DecodedLevel {
  data: Float32Array;
  shape: Shape3;
}

const levelCache = new Map<string, DecodedLevel>();
const inflight = new Map<string, Promise<DecodedLevel>>();
let maxCachedLevels = 4;

function levelKey(id: string, property: string, level: number): string {
  return `${id}::${property}::${level}`;
}

function parseHeaderShape(h: Headers): Shape3 | null {
  const raw = h.get("X-Volume-Shape");
  if (!raw) return null;
  try {
    const arr = JSON.parse(raw);
    if (Array.isArray(arr) && arr.length === 3) return [arr[0], arr[1], arr[2]] as Shape3;
  } catch {
    /* fall through */
  }
  return null;
}

async function getLevel(id: string, property: string, level: number): Promise<DecodedLevel> {
  const key = levelKey(id, property, level);
  const cached = levelCache.get(key);
  if (cached) {
    // refresh LRU position
    levelCache.delete(key);
    levelCache.set(key, cached);
    return cached;
  }
  const pending = inflight.get(key);
  if (pending) return pending;

  const p = (async () => {
    const r = await fetch(levelVolumeUrl(id, property, level));
    if (!r.ok) throw new Error(`level ${level} fetch failed: ${r.status}`);
    const buf = await r.arrayBuffer();
    const shape = parseHeaderShape(r.headers) ?? guessShape(buf);
    const data = new Float32Array(buf);
    const decoded: DecodedLevel = { data, shape };
    levelCache.set(key, decoded);
    // Evict oldest levels beyond the cap (LRU — Map preserves insertion/refresh order).
    while (levelCache.size > maxCachedLevels) {
      const oldest = levelCache.keys().next().value as string | undefined;
      if (oldest === undefined) break;
      levelCache.delete(oldest);
    }
    inflight.delete(key);
    return decoded;
  })();
  inflight.set(key, p);
  return p;
}

// Fallback if the X-Volume-Shape header is missing: assume a cube (best effort). Should not
// happen against the real backend, which always sets the header.
function guessShape(buf: ArrayBuffer): Shape3 {
  const n = Math.round(Math.cbrt(buf.byteLength / 4));
  return [n, n, n];
}

self.onmessage = (ev: MessageEvent<InMessage>) => {
  const msg = ev.data;
  if (msg.type === "config") {
    if (msg.maxCachedLevels && msg.maxCachedLevels > 0) maxCachedLevels = msg.maxCachedLevels;
    return;
  }
  if (msg.type === "decode") {
    const { reqId, id, property, level, t, bz, by, bx, brickEdge } = msg;
    getLevel(id, property, level)
      .then((lvl) => {
        const brick = extractBrick(lvl.data, lvl.shape, brickEdge, bz, by, bx);
        const empty = isEmptyBrick(brick);
        const payload = {
          type: "brick" as const,
          reqId,
          level,
          t,
          bz,
          by,
          bx,
          empty,
          brickEdge,
          data: empty ? null : brick,
        };
        // Transfer the brick buffer (zero-copy) unless it's empty (no buffer sent).
        if (empty) {
          (self as unknown as Worker).postMessage(payload);
        } else {
          (self as unknown as Worker).postMessage(payload, [brick.buffer]);
        }
      })
      .catch((err: unknown) => {
        (self as unknown as Worker).postMessage({
          type: "error",
          reqId,
          message: err instanceof Error ? err.message : String(err),
        });
      });
  }
};
