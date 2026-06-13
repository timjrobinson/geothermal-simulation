import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The backend FastAPI serves on :8000; proxy /api and /jobs (WS) during dev.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:8000",
      "/property-models": "http://localhost:8000",
      "/projects": "http://localhost:8000",
      "/jobs": { target: "ws://localhost:8000", ws: true },
    },
  },
});
