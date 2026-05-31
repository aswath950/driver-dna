"use client";

import { useState, useCallback } from "react";
import { Card } from "@/components/ui/Card";
import { PlotlyChart } from "@/components/charts/PlotlyChart";
import { apiClient } from "@/lib/api-client";

interface Driver {
  id: number;
  code: string;
  full_name: string;
}

const CHANNELS = ["Speed", "Throttle", "Brake"] as const;
type Channel = (typeof CHANNELS)[number];

interface Props {
  sessionId: number;
  drivers: Driver[];
}

const SELECT_CLASS =
  "border border-white/20 bg-[var(--bg-2)] px-3 py-2 text-sm text-white focus:outline-none";

export function TelemetryCompare({ sessionId, drivers }: Props) {
  const [driverAId, setDriverAId] = useState<number>(drivers[0]?.id ?? 0);
  const [driverBId, setDriverBId] = useState<number>(drivers[1]?.id ?? 0);
  const [channel, setChannel] = useState<Channel>("Speed");
  const [figureJson, setFigureJson] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchCompare = useCallback(
    async (aId: number, bId: number, ch: Channel) => {
      if (!aId || !bId) return;
      setLoading(true);
      setError(null);
      try {
        const payload = await apiClient.compare(sessionId, aId, bId, ch);
        setFigureJson(payload.figure_json);
      } catch {
        setError("Comparison data unavailable.");
        setFigureJson(null);
      } finally {
        setLoading(false);
      }
    },
    [sessionId],
  );

  function onDriverAChange(id: number) {
    setDriverAId(id);
    void fetchCompare(id, driverBId, channel);
  }

  function onDriverBChange(id: number) {
    setDriverBId(id);
    void fetchCompare(driverAId, id, channel);
  }

  function onChannelChange(ch: Channel) {
    setChannel(ch);
    void fetchCompare(driverAId, driverBId, ch);
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Controls */}
      <div className="flex flex-wrap items-end gap-4">
        <div className="flex flex-col gap-1">
          <label className="text-xs uppercase tracking-widest text-white/50">
            Driver A
          </label>
          <select
            value={driverAId}
            onChange={(e) => onDriverAChange(Number(e.target.value))}
            className={SELECT_CLASS}
          >
            {drivers.map((d) => (
              <option key={d.id} value={d.id}>
                {d.code} — {d.full_name}
              </option>
            ))}
          </select>
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs uppercase tracking-widest text-white/50">
            Driver B
          </label>
          <select
            value={driverBId}
            onChange={(e) => onDriverBChange(Number(e.target.value))}
            className={SELECT_CLASS}
          >
            {drivers.map((d) => (
              <option key={d.id} value={d.id}>
                {d.code} — {d.full_name}
              </option>
            ))}
          </select>
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs uppercase tracking-widest text-white/50">
            Channel
          </label>
          <select
            value={channel}
            onChange={(e) => onChannelChange(e.target.value as Channel)}
            className={SELECT_CLASS}
          >
            {CHANNELS.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Chart */}
      <Card>
        {loading ? (
          <div className="flex h-64 items-center justify-center text-white/50">
            Loading…
          </div>
        ) : error ? (
          <div className="flex h-64 items-center justify-center text-sm text-white/50">
            {error}
          </div>
        ) : figureJson ? (
          <PlotlyChart figureJson={figureJson} height={320} />
        ) : (
          <div className="flex h-64 items-center justify-center text-center text-sm text-white/40">
            Select two drivers then change any control to load the comparison.
          </div>
        )}
      </Card>
    </div>
  );
}
