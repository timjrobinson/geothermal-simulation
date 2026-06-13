// Global 4-D time slider (doc 06 §9.4). A bottom-docked playhead + window control built from
// the UNION of every time-bearing layer's epochs (the store's timeAxis, fed by
// GET /projects/{pid}/time-extent or merged in as 4-D layers are added). Play/pause/scrub/
// speed drive the store playhead; the window MODE (instant / cumulative / rolling) expands it
// to the interval the microseismic cloud (uTimeWindow) and InSAR raster (frame select) filter
// against — NO per-tick geometry rebuild (the scene's TimePlayer advances the playhead and
// the layers read uniforms). Hidden when the project has no time-bearing data (empty axis).

import { useViewer, timePlayheadFraction } from "../store";
import { TIME_WINDOW_MODES, type TimeWindowMode } from "../lib/time";

const bar: React.CSSProperties = {
  position: "absolute",
  bottom: 12,
  left: "50%",
  transform: "translateX(-50%)",
  width: "min(720px, calc(100vw - 320px))",
  padding: "8px 14px",
  background: "rgba(17,22,33,0.94)",
  border: "1px solid #313244",
  borderRadius: 8,
  color: "#cdd6f4",
  fontFamily: "ui-sans-serif, system-ui, sans-serif",
  fontSize: 12,
  zIndex: 12,
  display: "flex",
  flexDirection: "column",
  gap: 6,
};
const btn: React.CSSProperties = {
  background: "#313244",
  color: "#cdd6f4",
  border: "none",
  borderRadius: 4,
  padding: "3px 10px",
  cursor: "pointer",
  fontSize: 12,
};
const sel: React.CSSProperties = {
  fontSize: 12,
  background: "#1e2230",
  color: "#cdd6f4",
  border: "1px solid #313244",
  borderRadius: 4,
  padding: "2px 4px",
};

const DAY_MS = 24 * 3600 * 1000;
const SPEEDS: { label: string; msPerSec: number }[] = [
  { label: "1d/s", msPerSec: DAY_MS },
  { label: "7d/s", msPerSec: 7 * DAY_MS },
  { label: "30d/s", msPerSec: 30 * DAY_MS },
  { label: "1y/s", msPerSec: 365 * DAY_MS },
];

function fmt(ms: number): string {
  if (!Number.isFinite(ms)) return "—";
  return new Date(ms).toISOString().replace("T", " ").replace(".000Z", "Z");
}

export function TimeSlider() {
  const axis = useViewer((s) => s.timeAxis);
  const playheadMs = useViewer((s) => s.timePlayheadMs);
  const fraction = useViewer(timePlayheadFraction);
  const mode = useViewer((s) => s.timeWindowMode);
  const rollingWidthMs = useViewer((s) => s.timeRollingWidthMs);
  const playing = useViewer((s) => s.timePlaying);
  const speed = useViewer((s) => s.timeSpeed);
  const setTimePlayheadFraction = useViewer((s) => s.setTimePlayheadFraction);
  const setTimeWindowMode = useViewer((s) => s.setTimeWindowMode);
  const setTimeRollingWidthMs = useViewer((s) => s.setTimeRollingWidthMs);
  const setTimePlaying = useViewer((s) => s.setTimePlaying);
  const setTimeSpeed = useViewer((s) => s.setTimeSpeed);

  // No time-bearing data -> no slider.
  if (axis.epochs.length === 0) return null;

  const togglePlay = () => {
    if (!playing && speed <= 0) setTimeSpeed(SPEEDS[1].msPerSec); // default speed on first play
    setTimePlaying(!playing);
  };

  return (
    <div style={bar}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <button style={btn} onClick={togglePlay} title="play / pause the time animation">
          {playing ? "⏸ Pause" : "▶ Play"}
        </button>
        <span style={{ fontFamily: "ui-monospace, monospace", minWidth: 168 }}>
          {fmt(playheadMs)}
        </span>
        <span style={{ opacity: 0.55, marginLeft: "auto" }}>
          {axis.epochs.length} epoch{axis.epochs.length === 1 ? "" : "s"}
        </span>
      </div>

      {/* Scrub */}
      <input
        type="range"
        min={0}
        max={1}
        step={0.001}
        value={fraction}
        onChange={(e) => {
          if (playing) setTimePlaying(false);
          setTimePlayheadFraction(parseFloat(e.target.value));
        }}
        style={{ width: "100%" }}
        title="scrub the playhead"
      />

      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
        <label>
          Window{" "}
          <select
            style={sel}
            value={mode}
            onChange={(e) => setTimeWindowMode(e.target.value as TimeWindowMode)}
            title="instant: one frame · cumulative: history accretes · rolling: trailing window"
          >
            {TIME_WINDOW_MODES.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>

        {mode === "rolling" && (
          <label title="rolling-window width (days)">
            ±
            <input
              type="number"
              min={1}
              step={1}
              value={Math.max(1, Math.round(rollingWidthMs / DAY_MS))}
              onChange={(e) =>
                setTimeRollingWidthMs(Math.max(1, parseInt(e.target.value || "1", 10)) * DAY_MS)
              }
              style={{ ...sel, width: 56 }}
            />
            d
          </label>
        )}

        <label style={{ marginLeft: "auto" }}>
          Speed{" "}
          <select
            style={sel}
            value={speed}
            onChange={(e) => setTimeSpeed(parseFloat(e.target.value))}
            title="playback speed (axis time per real second)"
          >
            {speed > 0 && !SPEEDS.some((s) => s.msPerSec === speed) && (
              <option value={speed}>custom</option>
            )}
            {SPEEDS.map((s) => (
              <option key={s.label} value={s.msPerSec}>
                {s.label}
              </option>
            ))}
          </select>
        </label>
      </div>
    </div>
  );
}
