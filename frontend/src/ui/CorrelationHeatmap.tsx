// Correlation-matrix heatmap (doc 07 §3.2). A small SVG grid of the cross-correlation
// between all sampled properties, diverging blue↔white↔red. Computed client-side from the
// resident fused sample (lib/crossplot.correlationMatrix) so it works offline and matches
// the backend correlation_matrix() payload shape exactly.

import { useMemo } from "react";
import { useViewer } from "../store";
import { correlationMatrix, correlationColor } from "../lib/crossplot";

export function CorrelationHeatmap() {
  const sample = useViewer((s) => s.fusedSample);
  const { properties, matrix } = useMemo(
    () => (sample ? correlationMatrix(sample) : { properties: [], matrix: [] }),
    [sample],
  );

  if (!sample || properties.length === 0) return null;

  const p = properties.length;
  const cell = Math.max(22, Math.min(46, Math.floor(220 / p)));
  const labelW = 64;

  return (
    <div style={{ overflowX: "auto" }}>
      <svg width={labelW + p * cell + 4} height={labelW + p * cell + 4}>
        {/* column labels (top, rotated) */}
        {properties.map((name, j) => (
          <text
            key={`c${j}`}
            x={labelW + j * cell + cell / 2}
            y={labelW - 4}
            fill="#cdd6f4"
            fontSize={10}
            textAnchor="start"
            transform={`rotate(-45 ${labelW + j * cell + cell / 2} ${labelW - 4})`}
          >
            {name}
          </text>
        ))}
        {properties.map((rowName, i) => (
          <g key={`r${i}`}>
            <text x={labelW - 4} y={labelW + i * cell + cell / 2 + 3} fill="#cdd6f4" fontSize={10} textAnchor="end">
              {rowName}
            </text>
            {properties.map((_, j) => {
              const r = matrix[i]?.[j] ?? null;
              return (
                <g key={`${i}-${j}`}>
                  <rect
                    x={labelW + j * cell}
                    y={labelW + i * cell}
                    width={cell - 1}
                    height={cell - 1}
                    fill={correlationColor(r)}
                  />
                  <text
                    x={labelW + j * cell + cell / 2}
                    y={labelW + i * cell + cell / 2 + 3}
                    fill="#11161f"
                    fontSize={9}
                    textAnchor="middle"
                  >
                    {r == null ? "·" : r.toFixed(2)}
                  </text>
                </g>
              );
            })}
          </g>
        ))}
      </svg>
    </div>
  );
}
