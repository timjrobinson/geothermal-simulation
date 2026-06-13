import { useEffect } from "react";
import { useApp } from "./store";

// M0 stub page: fetch /api/capabilities and show the backend's declared property types,
// methods, and plugins. The Z-up R3F scene + volume ray-marcher arrive in M1.
export function App() {
  const { capabilities, setCapabilities } = useApp();

  useEffect(() => {
    fetch("/api/capabilities")
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then(setCapabilities)
      .catch(() => {});
  }, [setCapabilities]);

  return (
    <div style={{ padding: "2rem", maxWidth: 900, margin: "0 auto" }}>
      <h1>Geothermal Underground Simulator</h1>
      <p style={{ opacity: 0.7 }}>
        M0 walking skeleton. The 3D viewer (volume ray-marching in the Engineering Frame)
        lands in M1.
      </p>
      {!capabilities ? (
        <p>Connecting to backend at <code>/api/capabilities</code>…</p>
      ) : (
        <>
          <h2>Property types ({capabilities.property_types.length})</h2>
          <ul>
            {capabilities.property_types.map((p) => (
              <li key={p.key}>
                <code>{p.key}</code> — {p.unit} ({p.scaling}, {p.colormap})
              </li>
            ))}
          </ul>
          <h2>Methods ({capabilities.methods.length})</h2>
          <ul>
            {capabilities.methods.map((m) => (
              <li key={m.id}>{m.name}</li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
