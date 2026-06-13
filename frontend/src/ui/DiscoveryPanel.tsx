// Dataset discovery — the "open project" flow (doc 06 §9.1 datasets→layers). Fetches
// GET /projects then GET /projects/{pid}/artifacts and lists the ingested property models
// as available layers to ADD. On add it fetches that model's meta + decoded volume and
// pushes a new volume layer (registry-seeded transfer function, doc 06 §9.1 zero-config).
// When no backend is reachable the project list is empty and the user falls back to the
// in-memory mock layer (the app still works offline).

import { useState } from "react";
import { useViewer } from "../store";
import {
  fetchProjects,
  fetchProjectArtifacts,
  fetchMeta,
  fetchVolume,
  type ProjectSummary,
  type PropertyModelArtifact,
} from "../lib/api";
import { makeMockVolume } from "../lib/mockVolume";

const btn: React.CSSProperties = {
  background: "#313244",
  color: "#cdd6f4",
  border: "none",
  borderRadius: 4,
  padding: "3px 10px",
  cursor: "pointer",
  fontSize: 12,
};

export function DiscoveryPanel({ onClose }: { onClose: () => void }) {
  const addVolumeLayer = useViewer((s) => s.addVolumeLayer);
  const setError = useViewer((s) => s.setError);

  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  const [artifacts, setArtifacts] = useState<PropertyModelArtifact[]>([]);
  const [activePid, setActivePid] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string>("");

  const loadProjects = async () => {
    setBusy(true);
    setNote("");
    try {
      const ps = await fetchProjects();
      setProjects(ps);
      if (ps.length === 0) setNote("no projects (offline?) — use Add mock layer");
    } catch {
      setProjects([]);
      setNote("backend unavailable — use Add mock layer");
    } finally {
      setBusy(false);
    }
  };

  const openProject = async (pid: string) => {
    setActivePid(pid);
    setBusy(true);
    setNote("");
    try {
      const arts = await fetchProjectArtifacts(pid);
      setArtifacts(arts);
      if (arts.length === 0) setNote("no property models in this project");
    } catch {
      setArtifacts([]);
      setNote("could not list artifacts");
    } finally {
      setBusy(false);
    }
  };

  const addArtifact = async (a: PropertyModelArtifact) => {
    setBusy(true);
    try {
      const meta = await fetchMeta(a.id);
      const vol = await fetchVolume(a.id, 0, meta);
      addVolumeLayer(meta, vol, { name: a.property ?? meta.property });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const addMock = () => {
    const { meta, volume } = makeMockVolume();
    addVolumeLayer(
      { ...meta, id: `mock-${Math.random().toString(36).slice(2, 7)}` },
      volume,
      { name: "mock resistivity" },
    );
  };

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
        <span style={{ fontWeight: 600 }}>Open project</span>
        <button style={btn} onClick={onClose}>
          close
        </button>
      </div>

      <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
        <button style={btn} onClick={loadProjects} disabled={busy}>
          {projects == null ? "Fetch projects" : "Refresh"}
        </button>
        <button style={btn} onClick={addMock}>
          Add mock layer
        </button>
      </div>

      {projects && projects.length > 0 && (
        <div style={{ marginBottom: 8 }}>
          <div style={{ fontSize: 11, opacity: 0.6, marginBottom: 4 }}>Projects</div>
          {projects.map((p) => (
            <button
              key={p.id}
              style={{
                ...btn,
                display: "block",
                width: "100%",
                textAlign: "left",
                marginBottom: 3,
                background: activePid === p.id ? "#45475a" : "#313244",
              }}
              onClick={() => openProject(p.id)}
            >
              {p.name}
            </button>
          ))}
        </div>
      )}

      {activePid && artifacts.length > 0 && (
        <div>
          <div style={{ fontSize: 11, opacity: 0.6, marginBottom: 4 }}>
            Property models
          </div>
          {artifacts.map((a) => (
            <div
              key={a.id}
              style={{ display: "flex", justifyContent: "space-between", marginBottom: 3 }}
            >
              <span style={{ fontSize: 12 }}>
                {a.property}
                {a.canonicalUnit ? ` (${a.canonicalUnit})` : ""}
              </span>
              <button style={btn} onClick={() => addArtifact(a)} disabled={busy}>
                + add
              </button>
            </div>
          ))}
        </div>
      )}

      {note && (
        <div style={{ fontSize: 11, opacity: 0.6, marginTop: 6 }}>{note}</div>
      )}
    </div>
  );
}
