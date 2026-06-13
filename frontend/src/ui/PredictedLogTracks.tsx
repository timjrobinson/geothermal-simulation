// Predicted-log 2D tracks (doc 09 §5.2, doc 06 §10.3). Renders the along-path predicted log
// as vertical tracks vs MD: temperature, favorability, risk (with ±σ uncertainty bands) and a
// categorical lithology fill column. A horizontal marker tracks the hovered MD so the 3D tube
// hover and the 2D tracks stay in lockstep (the LogTrackPanel pattern, doc 06 §5.3). Pure
// shaping lives in lib/planning (predictedTrack / riskTrack / lithologyIntervals / trackToSvg).

import { useMemo } from "react";
import { useViewer } from "../store";
import {
  predictedTrack,
  riskTrack,
  lithologyIntervals,
  trackToSvg,
  type LogTrack,
  type PredictedLog,
} from "../lib/planning";

const W = 120;
const H = 320;

function mdRange(log: PredictedLog): [number, number] {
  if (log.stations.length === 0) return [0, 1];
  const first = log.stations[0].md;
  const last = log.stations[log.stations.length - 1].md;
  return [first, last === first ? first + 1 : last];
}

function Track({
  track,
  color,
  bandColor,
  mdMin,
  mdMax,
  markerY,
}: {
  track: LogTrack;
  color: string;
  bandColor: string;
  mdMin: number;
  mdMax: number;
  markerY: number | null;
}) {
  const { line, band } = useMemo(
    () => trackToSvg(track, W, H, mdMin, mdMax),
    [track, mdMin, mdMax],
  );
  return (
    <div style={{ textAlign: "center" }}>
      <div style={{ fontSize: 11, marginBottom: 2, opacity: 0.85 }}>
        {track.property}
        {track.unit ? ` (${track.unit})` : ""}
      </div>
      <svg width={W} height={H} style={{ background: "#11131c", borderRadius: 4 }}>
        {band && <polygon points={band} fill={bandColor} stroke="none" />}
        {line && <polyline points={line} fill="none" stroke={color} strokeWidth={1.5} />}
        {markerY != null && (
          <line x1={0} x2={W} y1={markerY} y2={markerY} stroke="#f9e2af" strokeWidth={1} />
        )}
      </svg>
      <div style={{ fontSize: 9, opacity: 0.6, fontFamily: "ui-monospace, monospace" }}>
        {track.min.toFixed(1)} – {track.max.toFixed(1)}
      </div>
    </div>
  );
}

function LithologyTrack({
  log,
  mdMin,
  mdMax,
  markerY,
}: {
  log: PredictedLog;
  mdMin: number;
  mdMax: number;
  markerY: number | null;
}) {
  const ivals = useMemo(() => lithologyIntervals(log), [log]);
  const span = mdMax - mdMin || 1;
  const yOf = (md: number) => ((md - mdMin) / span) * (H - 4) + 2;
  return (
    <div style={{ textAlign: "center" }}>
      <div style={{ fontSize: 11, marginBottom: 2, opacity: 0.85 }}>lithology</div>
      <svg width={40} height={H} style={{ background: "#11131c", borderRadius: 4 }}>
        {ivals.map((iv, i) => {
          const y0 = yOf(iv.mdTop);
          const y1 = yOf(iv.mdBottom);
          const [r, g, b] = iv.color;
          return (
            <rect
              key={i}
              x={2}
              y={y0}
              width={36}
              height={Math.max(1, y1 - y0)}
              fill={`rgb(${(r * 255) | 0},${(g * 255) | 0},${(b * 255) | 0})`}
            >
              <title>{iv.lithology}</title>
            </rect>
          );
        })}
        {markerY != null && (
          <line x1={0} x2={40} y1={markerY} y2={markerY} stroke="#f9e2af" strokeWidth={1} />
        )}
      </svg>
    </div>
  );
}

export function PredictedLogTracks() {
  const log = useViewer((s) => s.predictedLog);
  const readout = useViewer((s) => s.wellReadout);
  const activeWellId = useViewer((s) => s.activeWellId);

  const tracks = useMemo(() => {
    if (!log) return null;
    const temp = predictedTrack(log, "temperatureC", "degC");
    const fav = predictedTrack(log, "favorability");
    const risk = riskTrack(log);
    return { temp, fav, risk };
  }, [log]);

  if (!log || !tracks) {
    return (
      <div style={{ opacity: 0.6, fontSize: 11 }}>
        run Predict to see the along-path log
      </div>
    );
  }

  const [mdMin, mdMax] = mdRange(log);
  const span = mdMax - mdMin || 1;
  const markerY =
    readout && readout.wellId === activeWellId
      ? ((readout.md - mdMin) / span) * (H - 4) + 2
      : null;

  return (
    <div style={{ display: "flex", gap: 6, overflowX: "auto", paddingBottom: 4 }}>
      <Track
        track={tracks.temp}
        color="#f38ba8"
        bandColor="rgba(243,139,168,0.18)"
        mdMin={mdMin}
        mdMax={mdMax}
        markerY={markerY}
      />
      <Track
        track={tracks.fav}
        color="#a6e3a1"
        bandColor="rgba(166,227,161,0.18)"
        mdMin={mdMin}
        mdMax={mdMax}
        markerY={markerY}
      />
      <Track
        track={tracks.risk}
        color="#fab387"
        bandColor="rgba(250,179,135,0.18)"
        mdMin={mdMin}
        mdMax={mdMax}
        markerY={markerY}
      />
      <LithologyTrack log={log} mdMin={mdMin} mdMax={mdMax} markerY={markerY} />
    </div>
  );
}
