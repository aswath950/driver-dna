"use client";

import { useState } from "react";
import { Card } from "@/components/ui/Card";
import { StyleAnalystPanel } from "./StyleAnalystPanel";
import { DNAMatchPanel } from "./DNAMatchPanel";
import { ReportCardPanel } from "./ReportCardPanel";
import { useAIMode } from "@/lib/preferences";
import type { DriverOut } from "@/lib/api";

const FEATURES = [
  "Driver Style Analyst",
  "Historical DNA Matching",
  "Driver DNA Report Card",
] as const;

type Feature = (typeof FEATURES)[number];

interface Props {
  drivers: DriverOut[];
  season: number;
}

export function AIPanel({ drivers, season }: Props) {
  const { mode } = useAIMode();
  const [feature, setFeature] = useState<Feature>("Driver Style Analyst");
  const [driverId, setDriverId] = useState<number | null>(
    drivers[0]?.id ?? null,
  );

  if (drivers.length === 0) {
    return (
      <Card>
        <p className="text-sm text-white/50">
          Select at least one driver above to use AI features.
        </p>
      </Card>
    );
  }

  const resolvedDriverId = driverId ?? drivers[0]!.id;

  return (
    <div className="flex flex-col gap-4">
      {/* Selectors row */}
      <div className="flex flex-wrap gap-4">
        <div className="flex flex-col gap-1">
          <label className="text-xs uppercase tracking-widest text-white/50">
            AI Feature
          </label>
          <select
            value={feature}
            onChange={(e) => setFeature(e.target.value as Feature)}
            className="border border-white/20 bg-[var(--bg-2)] px-3 py-2 text-sm text-white focus:border-[var(--f1-red)] focus:outline-none"
          >
            {FEATURES.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </select>
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs uppercase tracking-widest text-white/50">
            Driver to analyse
          </label>
          <select
            value={resolvedDriverId}
            onChange={(e) => setDriverId(Number(e.target.value))}
            className="border border-white/20 bg-[var(--bg-2)] px-3 py-2 text-sm text-white focus:border-[var(--f1-red)] focus:outline-none"
          >
            {drivers.map((d) => (
              <option key={d.id} value={d.id}>
                {d.code} — {d.full_name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Sub-panel */}
      <Card>
        {feature === "Driver Style Analyst" && (
          <StyleAnalystPanel
            driverId={resolvedDriverId}
            season={season}
            aiMode={mode}
          />
        )}
        {feature === "Historical DNA Matching" && (
          <DNAMatchPanel
            driverId={resolvedDriverId}
            season={season}
            aiMode={mode}
          />
        )}
        {feature === "Driver DNA Report Card" && (
          <ReportCardPanel
            driverId={resolvedDriverId}
            season={season}
            aiMode={mode}
          />
        )}
      </Card>
    </div>
  );
}
