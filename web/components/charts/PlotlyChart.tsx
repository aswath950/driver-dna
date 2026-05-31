"use client";

import dynamic from "next/dynamic";
import { useMemo } from "react";

// Plotly is heavy (~3 MB minified). Dynamic-import disables SSR so it
// only ships to the client when actually needed.
const Plot = dynamic(() => import("react-plotly.js"), {
  ssr: false,
  loading: () => <div className="h-64 grid place-items-center text-white/50">Loading chart…</div>,
});

export function PlotlyChart({
  figureJson,
  height = 420,
}: {
  figureJson: string;
  height?: number;
}) {
  const fig = useMemo(() => {
    try {
      return JSON.parse(figureJson) as { data: unknown[]; layout: Record<string, unknown> };
    } catch {
      return { data: [], layout: {} };
    }
  }, [figureJson]);

  return (
    <Plot
      data={fig.data as never}
      layout={{
        ...fig.layout,
        autosize: true,
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "#fff", family: "Space Grotesk, system-ui" },
        margin: { l: 50, r: 20, t: 40, b: 40 },
      }}
      useResizeHandler
      style={{ width: "100%", height }}
      config={{ displayModeBar: false, responsive: true }}
    />
  );
}
