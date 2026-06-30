"use client";

import { useState, useCallback, useEffect } from "react";
import { Card } from "@/components/ui/Card";
import { PlotlyChart } from "@/components/charts/PlotlyChart";
import { apiClient, type SectorTimesPayload } from "@/lib/api-client";

interface Driver {
  id: number;
  code: string;
  full_name: string;
}

const CHANNELS = [
  "Speed", "Throttle", "Brake", "RPM", "nGear", "DRS",
  "TimeDelta", "SpeedTimeDelta", "Sector Times", "Track Map",
] as const;
type Channel = (typeof CHANNELS)[number];

// Display labels for channels whose API value isn't user-friendly. Falls back
// to the raw channel value when no override is present.
const CHANNEL_LABELS: Partial<Record<Channel, string>> = {
  SpeedTimeDelta: "Speed + Time Delta",
};

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
  const [sectorPayload, setSectorPayload] = useState<SectorTimesPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [telemetryCached, setTelemetryCached] = useState<boolean | null>(null);

  useEffect(() => {
    setTelemetryCached(null);
    apiClient.telemetryStatus(sessionId)
      .then((s) => setTelemetryCached(s.fetched_at !== null))
      .catch(() => setTelemetryCached(null));
  }, [sessionId]);

  const fetchCompare = useCallback(
    async (aId: number, bId: number, ch: Channel) => {
      if (!aId || !bId) return;
      setLoading(true);
      setError(null);
      setSectorPayload(null);
      try {
        if (ch === "Sector Times") {
          const payload = await apiClient.sectorTimes(sessionId, aId, bId);
          setSectorPayload(payload);
          setFigureJson(payload.figure_json);
        } else if (ch === "Track Map") {
          const payload = await apiClient.trackMap(sessionId, aId, bId);
          setFigureJson(payload.figure_json);
        } else {
          const payload = await apiClient.compare(sessionId, aId, bId, ch);
          setFigureJson(payload.figure_json);
        }
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
                {CHANNEL_LABELS[c] ?? c}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Cache status banner */}
      {telemetryCached === false && (
        <p className="text-xs text-white/40 border border-white/10 px-3 py-2">
          Telemetry not pre-cached — first comparison may take a few seconds.
          Visit the <a href="/pipeline" className="underline text-white/60">Pipeline page</a> to
          pre-download all session telemetry for instant loads.
        </p>
      )}

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
          <>
            <PlotlyChart
            figureJson={figureJson}
            // Stacked Speed + Time-Delta needs room for two panels and a
            // two-line title; other channels fit a single 320px panel.
            height={channel === "SpeedTimeDelta" ? 540 : 320}
            margin={
              channel === "Track Map"
                ? { l: 5, r: 5, t: 30, b: 5 }
                : channel === "SpeedTimeDelta"
                  ? { l: 65, r: 45, t: 75, b: 45 }
                  : undefined
            }
          />
            {sectorPayload &&
              [sectorPayload.driver_a, sectorPayload.driver_b].some(
                (d) =>
                  d.sector1_ms === null ||
                  d.sector2_ms === null ||
                  d.sector3_ms === null,
              ) && (
                <p className="text-xs text-white/40 mt-1">
                  Some sector splits unavailable for this driver/lap.
                </p>
              )}
          </>
        ) : (
          <div className="flex h-64 items-center justify-center text-center text-sm text-white/40">
            Select two drivers then change any control to load the comparison.
          </div>
        )}
      </Card>
    </div>
  );
}
