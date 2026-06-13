// M1 viewer app (doc 06 §1.3, §11 M1 row). Replaces the M0 stub with the real Z-up scene:
// it loads a single resident volume (server-decoded /volume, or a client-side mock for a
// self-contained `npm run dev`) into a Data3DTexture, ray-marches it, cuts it with one
// orthogonal slice from the same texture, clips it with a draggable box, and drives a
// transfer function from a control panel. The M0 capabilities fetch is kept minor (a
// footer badge).
//
// URL params:
//   ?id=<propertyModelId>[&level=N]  — load that PropertyModel from the backend
//   ?mock                            — synthesize a conductive-blob volume (no backend)
// Default (no params) uses ?mock so the build is self-contained and shows something.

import { useEffect, useState } from "react";
import { useViewer, selectedLayer, type Capabilities } from "./store";
import { fetchMeta, fetchVolume } from "./lib/api";
import { makeMockVolume } from "./lib/mockVolume";
import { Scene } from "./scene/Scene";
import { ControlPanel } from "./ui/ControlPanel";

export function App() {
  const loadData = useViewer((s) => s.loadData);
  const addTerrainLayer = useViewer((s) => s.addTerrainLayer);
  const setLoading = useViewer((s) => s.setLoading);
  const setError = useViewer((s) => s.setError);
  const loading = useViewer((s) => s.loading);
  const error = useViewer((s) => s.error);
  const layerCount = useViewer((s) => s.layerOrder.length);
  const layer = useViewer(selectedLayer);
  const meta = layer?.meta ?? null;
  const sceneAABB = useViewer((s) => s.sceneAABB);
  const layers = useViewer((s) => s.layers);
  const layerOrder = useViewer((s) => s.layerOrder);
  const setCapabilities = useViewer((s) => s.setCapabilities);
  const capabilities = useViewer((s) => s.capabilities);

  const [mode, setMode] = useState<string>("");

  // Load the volume (mock or backend) once on mount per URL params.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const id = params.get("id");
    const useMock = params.has("mock") || !id;
    const level = parseInt(params.get("level") ?? "0", 10);

    let cancelled = false;
    setLoading(true);
    if (useMock) {
      setMode("mock");
      const { meta, volume } = makeMockVolume();
      loadData(meta, volume);
    } else {
      setMode(`model ${id}`);
      (async () => {
        try {
          const m = await fetchMeta(id!);
          const v = await fetchVolume(id!, level, m);
          if (!cancelled) loadData(m, v);
        } catch (e) {
          if (!cancelled) setError(e instanceof Error ? e.message : String(e));
        }
      })();
    }
    return () => {
      cancelled = true;
    };
  }, [loadData, setLoading, setError]);

  // Auto-add the ground-surface terrain layer once a volume scene AABB is known (doc 06
  // §6): the surface spans the data footprint at the project's surfaceModel elevation, and
  // the subsurface volume hangs beneath it. Runs once (skipped if a terrain layer exists).
  useEffect(() => {
    if (!sceneAABB) return;
    const hasTerrain = layerOrder.some((id) => layers[id]?.kind === "terrain");
    const hasVolume = layerOrder.some((id) => layers[id]?.kind === "volume");
    if (hasTerrain || !hasVolume) return;
    const frame = (meta?.frame as Record<string, unknown> | undefined) ?? undefined;
    const surfaceModelSpec =
      (frame?.surfaceModel as string | null | undefined) ?? "flat:0";
    addTerrainLayer({
      surfaceModelSpec,
      extent: {
        xmin: sceneAABB.min[0],
        xmax: sceneAABB.max[0],
        ymin: sceneAABB.min[1],
        ymax: sceneAABB.max[1],
      },
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sceneAABB]);

  // Minor M0 capabilities fetch (footer badge); failure is non-fatal.
  useEffect(() => {
    fetch("/api/capabilities")
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((c: Capabilities) => setCapabilities(c))
      .catch(() => {});
  }, [setCapabilities]);

  return (
    <div style={{ position: "absolute", inset: 0, overflow: "hidden" }}>
      <Scene />
      <ControlPanel />

      {/* Title + status */}
      <div
        style={{
          position: "absolute",
          top: 12,
          left: 12,
          zIndex: 10,
          fontFamily: "ui-sans-serif, system-ui, sans-serif",
        }}
      >
        <div style={{ fontSize: 16, fontWeight: 600, color: "#cdd6f4" }}>
          Geothermal Underground Simulator
        </div>
        <div style={{ fontSize: 12, opacity: 0.7, color: "#cdd6f4" }}>
          Multi-layer viewer — {layerCount} layer{layerCount === 1 ? "" : "s"}{" "}
          {mode && `(${mode})`}
          {loading && " — loading…"}
          {meta && ` — ${meta.shape.join("×")} (z,y,x)`}
        </div>
        {error && (
          <div style={{ fontSize: 12, color: "#f38ba8", marginTop: 4 }}>
            load error: {error}
          </div>
        )}
      </div>

      {/* Minor M0 capabilities badge */}
      {capabilities && (
        <div
          style={{
            position: "absolute",
            bottom: 8,
            left: 12,
            zIndex: 10,
            fontSize: 11,
            opacity: 0.55,
            color: "#cdd6f4",
            fontFamily: "ui-monospace, monospace",
          }}
        >
          backend api {capabilities.api_version} ·{" "}
          {capabilities.property_types.length} property types ·{" "}
          {capabilities.plugins.length} plugins
        </div>
      )}
    </div>
  );
}
