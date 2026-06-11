"use client";

import { useState, useMemo } from "react";
import { useRouter } from "next/navigation";
import { Card } from "@/components/ui/Card";
import { PlotlyChart } from "@/components/charts/PlotlyChart";
import { AIPanel } from "@/components/radar/AIPanel";
import { ArchetypeCards } from "@/components/radar/ArchetypeCards";
import type { SeasonOut, DriverOut } from "@/lib/api";

// ── Radar axis labels ──────────────────────────────────────────────────────

const STANDARD_AXES = [
  "Brake Precision",
  "Throttle Control",
  "Top Speed",
  "Lap Consistency",
  "Steering Smoothness",
  "Cornering",
];

const EXTENDED_AXES = [
  ...STANDARD_AXES,
  "Trail Braking",
  "Zone 1 Pace",
  "Zone 2 Pace",
  "Zone 3 Pace",
  "DRS Usage",
  "Oversteer Tendency",
];

type RadarMode = "Standard (6)" | "Extended (12)";

function buildEmptyPolar(mode: RadarMode): string {
  const axes = mode === "Extended (12)" ? EXTENDED_AXES : STANDARD_AXES;
  const closed = [...axes, axes[0]!];
  return JSON.stringify({
    data: [
      {
        type: "scatterpolar",
        r: Array<number>(closed.length).fill(0),
        theta: closed,
        opacity: 0,
        showlegend: false,
        hoverinfo: "none",
      },
    ],
    layout: {
      polar: { radialaxis: { visible: true, range: [0, 1] } },
      title: "Driver Style Radar (normalised 0–1)",
      legend_title: "Driver",
    },
  });
}

// Static empty figures for the chart strip
const EMPTY_LINE = JSON.stringify({
  data: [],
  layout: {
    height: 360,
    xaxis: { title: "Lap Distance (normalised)" },
    yaxis: { title: "Speed (km/h)" },
    hovermode: "x unified",
  },
});

const EMPTY_HEATMAP = JSON.stringify({
  data: [],
  layout: { height: 300, margin: { l: 10, r: 80, t: 20, b: 60 } },
});

function buildEmptyBarFig(yLabel: string): string {
  return JSON.stringify({
    data: [],
    layout: {
      barmode: "group",
      height: 360,
      xaxis: { title: "Lap Zone" },
      yaxis: { title: yLabel },
    },
  });
}

const ZONE_CHANNELS: { label: string; yLabel: string }[] = [
  { label: "Speed (km/h)", yLabel: "Speed (km/h)" },
  { label: "Throttle (%)", yLabel: "Throttle (%)" },
  { label: "Braking (0–1)", yLabel: "Braking Intensity" },
];

// ── Component ──────────────────────────────────────────────────────────────

interface Props {
  seasons: SeasonOut[];
  drivers: DriverOut[];
  initialSeason: number;
}

export function RadarClient({ seasons, drivers, initialSeason }: Props) {
  const router = useRouter();

  // Driver multiselect — default to first 3
  const [selectedIds, setSelectedIds] = useState<number[]>(() =>
    drivers.slice(0, Math.min(3, drivers.length)).map((d) => d.id),
  );

  const [radarMode, setRadarMode] = useState<RadarMode>("Standard (6)");
  const [zoneChannel, setZoneChannel] = useState(ZONE_CHANNELS[0]!.label);

  const selectedDrivers = useMemo(
    () => drivers.filter((d) => selectedIds.includes(d.id)),
    [drivers, selectedIds],
  );

  const tooFew = selectedIds.length < 2;
  const tooMany = selectedIds.length > 4;
  const valid = !tooFew && !tooMany;

  const radarFigJson = useMemo(() => buildEmptyPolar(radarMode), [radarMode]);

  const zoneBarFig = useMemo(() => {
    const ch = ZONE_CHANNELS.find((c) => c.label === zoneChannel)!;
    return buildEmptyBarFig(ch.yLabel);
  }, [zoneChannel]);

  function toggleDriver(id: number) {
    setSelectedIds((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id);
      if (prev.length >= 4) return prev; // silent max-4 guard; warning text covers this
      return [...prev, id];
    });
  }

  function handleSeasonChange(year: number) {
    router.push(`/radar?season=${year}`);
  }

  return (
    <div className="flex flex-col gap-6">
      <h2 className="text-xl font-bold tracking-tight">Driver Style Fingerprint</h2>

      {/* ── Selectors ─────────────────────────────────────────────────── */}
      <div className="flex flex-wrap gap-6 items-start">
        {/* Season picker */}
        <div className="flex flex-col gap-1">
          <label className="text-xs uppercase tracking-widest text-white/50">
            Season
          </label>
          <select
            value={initialSeason}
            onChange={(e) => handleSeasonChange(Number(e.target.value))}
            className="border border-white/20 bg-[var(--bg-2)] px-3 py-2 text-sm text-white focus:border-[var(--f1-red)] focus:outline-none"
          >
            {seasons.map((s) => (
              <option key={s.id} value={s.year}>
                {s.year}
              </option>
            ))}
          </select>
        </div>

        {/* Driver multiselect */}
        <div className="flex flex-col gap-1 flex-1 min-w-0">
          <label className="text-xs uppercase tracking-widest text-white/50">
            Select 2–4 drivers
          </label>
          <div className="flex flex-wrap gap-2">
            {drivers.map((d) => {
              const on = selectedIds.includes(d.id);
              return (
                <button
                  key={d.id}
                  onClick={() => toggleDriver(d.id)}
                  className={`border px-3 py-1 text-sm font-semibold transition-colors ${
                    on
                      ? "border-[var(--f1-red)] bg-[var(--f1-red)]/20 text-white"
                      : "border-white/20 text-white/50 hover:border-white/50 hover:text-white/80"
                  }`}
                >
                  {d.code}
                </button>
              );
            })}
          </div>
        </div>

        {/* Radar mode */}
        <div className="flex flex-col gap-1">
          <label className="text-xs uppercase tracking-widest text-white/50">
            Radar dimensions
          </label>
          <div className="flex gap-3">
            {(["Standard (6)", "Extended (12)"] as RadarMode[]).map((m) => (
              <label key={m} className="flex items-center gap-2 text-sm text-white/70 cursor-pointer">
                <input
                  type="radio"
                  name="radar_mode"
                  checked={radarMode === m}
                  onChange={() => setRadarMode(m)}
                  className="accent-[var(--f1-red)]"
                />
                {m}
              </label>
            ))}
          </div>
        </div>
      </div>

      {/* Validation warnings */}
      {tooFew && (
        <div className="border border-yellow-500/40 bg-yellow-900/10 px-3 py-2 text-sm text-yellow-300">
          Select at least 2 drivers.
        </div>
      )}
      {tooMany && (
        <div className="border border-yellow-500/40 bg-yellow-900/10 px-3 py-2 text-sm text-yellow-300">
          Select at most 4 drivers.
        </div>
      )}

      {/* ── Radar chart (always shown, degraded) ─────────────────────── */}
      <Card>
        <div className="relative">
          <PlotlyChart figureJson={radarFigJson} height={420} />
          <p className="mt-2 text-center text-xs text-white/40 italic">
            Radar data not available in this build
          </p>
        </div>
      </Card>

      {/* ── Archetype cards ───────────────────────────────────────────── */}
      {valid && selectedDrivers.length > 0 && (
        <ArchetypeCards drivers={selectedDrivers} />
      )}

      {/* ── AI Analysis ───────────────────────────────────────────────── */}
      <hr className="border-white/10" />
      <h2 className="text-lg font-bold">AI Analysis</h2>
      <AIPanel drivers={valid ? selectedDrivers : []} season={initialSeason} />

      <hr className="border-white/10" />

      {/* ── Chart strip ───────────────────────────────────────────────── */}

      {/* 1. Speed Profile Comparison */}
      <div>
        <h3 className="text-base font-semibold mb-1">Speed Profile Comparison</h3>
        <p className="text-xs text-white/50 mb-3">
          Average speed at each point along the lap — reveals where each driver
          goes faster.
        </p>
        <Card>
          <PlotlyChart figureJson={EMPTY_LINE} height={360} />
          <p className="mt-2 text-center text-xs text-white/40 italic">
            Not available in this build
          </p>
        </Card>
      </div>

      {/* 2. Lap-by-Lap Consistency */}
      <div>
        <h3 className="text-base font-semibold mb-1">Lap-by-Lap Consistency</h3>
        <p className="text-xs text-white/50 mb-3">
          How variable each driver is across laps — red means less consistent.
        </p>
        <Card>
          <PlotlyChart figureJson={EMPTY_HEATMAP} height={300} />
          <p className="mt-2 text-center text-xs text-white/40 italic">
            Not available in this build
          </p>
        </Card>
      </div>

      {/* 3. Lap Zone Performance */}
      <div>
        <h3 className="text-base font-semibold mb-1">Lap Zone Performance</h3>
        <p className="text-xs text-white/50 mb-2">
          Lap split into three equal distance zones — shows where each driver
          excels.
        </p>
        <div className="mb-3">
          <label className="text-xs uppercase tracking-widest text-white/50 mr-2">
            Channel
          </label>
          <select
            value={zoneChannel}
            onChange={(e) => setZoneChannel(e.target.value)}
            className="border border-white/20 bg-[var(--bg-2)] px-3 py-1.5 text-sm text-white focus:border-[var(--f1-red)] focus:outline-none"
          >
            {ZONE_CHANNELS.map((c) => (
              <option key={c.label} value={c.label}>
                {c.label}
              </option>
            ))}
          </select>
        </div>
        <Card>
          <PlotlyChart figureJson={zoneBarFig} height={360} />
          <p className="mt-2 text-center text-xs text-white/40 italic">
            Not available in this build
          </p>
        </Card>
      </div>

      {/* 4. Throttle Application Map */}
      <div>
        <h3 className="text-base font-semibold mb-1">Throttle Application Map</h3>
        <p className="text-xs text-white/50 mb-3">
          Track outline coloured by average throttle — green: full throttle,
          red: braking / coasting.
        </p>
        <Card>
          <div className="flex items-center justify-center h-48 text-white/30 text-sm italic">
            Track map unavailable — telemetry data not available in this build
          </div>
        </Card>
      </div>
    </div>
  );
}
