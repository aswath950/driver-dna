"use client";

import { useState, useCallback } from "react";
import { Card } from "@/components/ui/Card";
import { PlotlyChart } from "@/components/charts/PlotlyChart";
import {
  apiClient,
  type TeamOut,
  type CornerPerformancePayload,
  type ClassSummary,
  type SingleCorner,
} from "@/lib/api-client";

interface Props {
  sessionId: number;
  teams: TeamOut[];
}

type View =
  | "summary"
  | "per-corner"
  | "track-map"
  | "straight-performance"
  | "hybrid-map";

const VIEW_LABELS: { key: View; label: string }[] = [
  { key: "summary", label: "Summary" },
  { key: "per-corner", label: "Per Corner" },
  { key: "track-map", label: "Track Map" },
  { key: "straight-performance", label: "Straight Performance" },
  { key: "hybrid-map", label: "Hybrid Map" },
];

const CLASS_ORDER = ["slow", "medium", "high"] as const;
const CLASS_LABEL: Record<string, string> = {
  slow: "Slow",
  medium: "Medium",
  high: "High",
};

const SELECT_CLASS =
  "border border-white/20 bg-[var(--bg-2)] px-3 py-2 text-sm text-white focus:outline-none";

function hexToRgba(hex: string, alpha: number): string {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

const CLASS_BADGE: Record<string, string> = {
  slow: "text-yellow-400/70",
  medium: "text-blue-300/70",
  high: "text-red-400/70",
};

function DeltaPill({
  label,
  summary,
  nameA,
  nameB,
}: {
  label: string;
  summary: ClassSummary;
  nameA: string;
  nameB: string;
}) {
  const delta = summary.team_a.v_min - summary.team_b.v_min;
  const faster = Math.abs(delta) < 0.5 ? null : delta > 0 ? nameA : nameB;
  const sign = delta >= 0 ? "+" : "";
  return (
    <div className="flex flex-1 min-w-[180px] flex-col gap-3 rounded border border-white/10 px-6 py-5">
      <span className="text-xs uppercase tracking-widest text-white/50">{label}</span>
      <span className="text-3xl font-semibold">
        {sign}{delta.toFixed(1)}{" "}
        <span className="text-base font-normal text-white/50">km/h</span>
      </span>
      <span className="text-sm text-white/40">
        {faster ? `${faster} faster` : "Equal"}
      </span>
      <span className="text-xs text-white/30">{summary.corner_count} corners</span>
    </div>
  );
}

function CornerCard({
  corner,
  teamAName,
  teamBName,
  teamAColor,
  teamBColor,
}: {
  corner: SingleCorner;
  teamAName: string;
  teamBName: string;
  teamAColor: string;
  teamBColor: string;
}) {
  const delta = corner.team_a.v_min - corner.team_b.v_min;
  const tied = Math.abs(delta) < 0.5;
  const fasterColor = tied ? null : delta > 0 ? teamAColor : teamBColor;
  const fasterName = tied ? null : delta > 0 ? teamAName : teamBName;

  return (
    <div
      className="flex flex-col gap-2 rounded border border-white/10 px-4 py-3 transition-colors"
      style={fasterColor ? { backgroundColor: hexToRgba(fasterColor, 0.12) } : undefined}
    >
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold">C{corner.corner_number}</span>
        <span className={`text-xs uppercase tracking-widest ${CLASS_BADGE[corner.corner_class]}`}>
          {corner.corner_class}
        </span>
      </div>
      <div className="mt-1 flex flex-col gap-1">
        <div className="flex items-center justify-between text-xs">
          <span className="text-white/50 truncate pr-2">{teamAName}</span>
          <span className="font-mono text-white/90">{corner.team_a.v_min.toFixed(1)}</span>
        </div>
        <div className="flex items-center justify-between text-xs">
          <span className="text-white/50 truncate pr-2">{teamBName}</span>
          <span className="font-mono text-white/90">{corner.team_b.v_min.toFixed(1)}</span>
        </div>
      </div>
      <div className="mt-1 border-t border-white/8 pt-2 text-xs text-white/40">
        {fasterName
          ? `${fasterName} +${Math.abs(delta).toFixed(1)} km/h`
          : "Equal"}
      </div>
    </div>
  );
}

export function CornerPerformance({ sessionId, teams }: Props) {
  const [teamAId, setTeamAId] = useState<number>(teams[0]?.id ?? 0);
  const [teamBId, setTeamBId] = useState<number>(teams[1]?.id ?? 0);
  const [view, setView] = useState<View>("summary");
  const [payload, setPayload] = useState<CornerPerformancePayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(
    async (aId: number, bId: number) => {
      if (!aId || !bId || aId === bId) return;
      setLoading(true);
      setError(null);
      try {
        const data = await apiClient.cornerPerformance(sessionId, aId, bId);
        setPayload(data);
      } catch (e: unknown) {
        const msg =
          e instanceof Error ? e.message : "Corner performance data unavailable.";
        setError(msg);
        setPayload(null);
      } finally {
        setLoading(false);
      }
    },
    [sessionId],
  );

  function onTeamAChange(id: number) {
    setTeamAId(id);
    void fetchData(id, teamBId);
  }

  function onTeamBChange(id: number) {
    setTeamBId(id);
    void fetchData(teamAId, id);
  }

  if (teams.length < 2) {
    return (
      <Card>
        <p className="py-8 text-center text-sm text-white/40">
          Not enough teams in this session to compare.
        </p>
      </Card>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Controls */}
      <div className="flex flex-wrap items-end gap-4">
        <div className="flex flex-col gap-1">
          <label className="text-xs uppercase tracking-widest text-white/50">
            Team A
          </label>
          <select
            value={teamAId}
            onChange={(e) => onTeamAChange(Number(e.target.value))}
            className={SELECT_CLASS}
          >
            {teams.map((t) => (
              <option key={t.id} value={t.id}>
                {t.name}
              </option>
            ))}
          </select>
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs uppercase tracking-widest text-white/50">
            Team B
          </label>
          <select
            value={teamBId}
            onChange={(e) => onTeamBChange(Number(e.target.value))}
            className={SELECT_CLASS}
          >
            {teams.map((t) => (
              <option key={t.id} value={t.id}>
                {t.name}
              </option>
            ))}
          </select>
        </div>

        {/* View toggle */}
        {payload && (
          <div className="flex gap-1">
            {VIEW_LABELS.map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setView(key)}
                className={
                  "px-3 py-2 text-xs font-medium transition-colors " +
                  (view === key
                    ? "bg-[var(--f1-red)] text-white"
                    : "border border-white/20 text-white/60 hover:text-white")
                }
              >
                {label}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Content */}
      <Card>
        {loading ? (
          <div className="flex h-64 items-center justify-center text-white/50">
            Analysing corners…
          </div>
        ) : error ? (
          <div className="flex h-64 items-center justify-center text-center text-sm text-white/50 px-4">
            {error}
          </div>
        ) : payload && view === "summary" ? (
          <div className="flex flex-wrap gap-4 py-2">
            {CLASS_ORDER.filter((c) => payload.summary[c]).map((cls) => (
              <DeltaPill
                key={cls}
                label={CLASS_LABEL[cls]}
                summary={payload.summary[cls]}
                nameA={payload.team_a.name}
                nameB={payload.team_b.name}
              />
            ))}
          </div>
        ) : payload && view === "per-corner" ? (
          <div className="grid grid-cols-2 gap-3 py-1 sm:grid-cols-3 lg:grid-cols-4">
            {payload.corners.map((corner) => (
              <CornerCard
                key={corner.corner_number}
                corner={corner}
                teamAName={payload.team_a.name}
                teamBName={payload.team_b.name}
                teamAColor={payload.team_a.color_hex}
                teamBColor={payload.team_b.color_hex}
              />
            ))}
          </div>
        ) : payload && view === "track-map" ? (
          <PlotlyChart
            figureJson={payload.track_map_figure}
            height={520}
            margin={{ l: 0, r: 0, t: 40, b: 0 }}
          />
        ) : payload && view === "straight-performance" ? (
          <PlotlyChart
            figureJson={payload.straight_map_figure}
            height={520}
            margin={{ l: 0, r: 0, t: 40, b: 0 }}
          />
        ) : payload && view === "hybrid-map" ? (
          <PlotlyChart
            figureJson={payload.hybrid_map_figure}
            height={520}
            margin={{ l: 0, r: 0, t: 40, b: 0 }}
          />
        ) : (
          <div className="flex h-64 items-center justify-center text-center text-sm text-white/40">
            Select two teams to compare corner performance.
          </div>
        )}
      </Card>
    </div>
  );
}
