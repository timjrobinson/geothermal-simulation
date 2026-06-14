import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The backend FastAPI serves on :8000; proxy /api and /jobs (WS) during dev.
export default defineConfig({
  plugins: [react()],
  server: {
    // Bind 0.0.0.0 (not loopback) so devcontainer / Codespaces / docker port-forwarding
    // can reach the dev server — otherwise the forwarded :5173 connects to nothing and
    // the browser "loads forever". strictPort keeps it on the forwarded port.
    host: true,
    port: 5173,
    strictPort: true,
    // Proxy API/WS to the backend on the SAME container's loopback (works regardless of
    // what host uvicorn binds). The browser only ever talks to Vite on :5173.
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/property-models": "http://127.0.0.1:8000",
      "/projects": "http://127.0.0.1:8000",
      "/features": "http://127.0.0.1:8000",
      "/fused": "http://127.0.0.1:8000",
      "/wells": "http://127.0.0.1:8000",
      "/inversion-engines": "http://127.0.0.1:8000",
      "/jobs": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
});
