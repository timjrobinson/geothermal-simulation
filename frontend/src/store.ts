import { create } from "zustand";

// Viewer store (doc 06 §10). All stored positions are Engineering metres; vertical
// exaggeration and basemap UVs are render-time transforms, never written back into data.
// This is the M0 stub; the full shape (layers, clipBox, time, selection) lands in M1/M2.
export interface Capabilities {
  api_version: string;
  property_types: { key: string; unit: string; colormap: string; scaling: string }[];
  methods: { id: string; name: string }[];
  plugins: { id: string; version: string }[];
}

interface AppState {
  capabilities: Capabilities | null;
  setCapabilities: (c: Capabilities) => void;
}

export const useApp = create<AppState>((set) => ({
  capabilities: null,
  setCapabilities: (c) => set({ capabilities: c }),
}));
