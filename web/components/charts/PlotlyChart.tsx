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
  margin,
}: {
  figureJson: string;
  height?: number;
  margin?: { l?: number; r?: number; t?: number; b?: number };
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
        // Default hover styling: a light box with dark text so the hover data
        // stays legible against the dark dashboard (otherwise Plotly's default
        // light box + the white global font above renders white-on-white).
        // Charts that set their own hoverlabel keep it via the spread below.
        hoverlabel: {
          bgcolor: "rgba(230, 230, 235, 0.97)",
          bordercolor: "rgba(190, 190, 200, 0.9)",
          font: { color: "rgba(15, 15, 20, 1)", size: 12, family: "Space Grotesk, system-ui" },
          ...(fig.layout?.hoverlabel as Record<string, unknown> | undefined),
        },
        margin: margin ?? { l: 50, r: 20, t: 40, b: 40 },
      }}
      useResizeHandler
      style={{ width: "100%", height }}
      config={{ displayModeBar: false, responsive: true }}
    />
  );
}
