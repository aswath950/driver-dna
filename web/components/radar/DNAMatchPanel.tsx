"use client";

import { useState } from "react";
import { Button } from "@/components/ui/Button";
import { PlotlyChart } from "@/components/charts/PlotlyChart";
import { useCooldown } from "@/components/ui/Cooldown";
import { apiClient, type DNAMatchResponse } from "@/lib/api-client";
import type { AIMode } from "@/lib/preferences";

interface Props {
  driverId: number;
  season: number;
  aiMode: AIMode;
}

export function DNAMatchPanel({ driverId, season, aiMode }: Props) {
  const { secondsLeft, startCooldown, onCooldown } = useCooldown();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DNAMatchResponse | null>(null);

  async function run() {
    if (onCooldown || loading) return;
    setLoading(true);
    setError(null);
    try {
      const data = await apiClient.dnaMatch(driverId, season);
      setResult(data);
      startCooldown();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  const btnLabel = onCooldown
    ? `Find historical matches (${secondsLeft}s)`
    : loading
      ? "Matching…"
      : "Find historical matches";

  function buildBarFig(res: DNAMatchResponse): string {
    return JSON.stringify({
      data: [
        {
          type: "bar",
          x: res.matches.map((m) => m.similarity),
          y: res.matches.map((m) => m.name),
          orientation: "h",
          marker: { color: "#f1c40f" },
          hovertemplate: "%{y}<br>Similarity: %{x:.4f}<extra></extra>",
        },
      ],
      layout: {
        title: `Cosine Similarity to Historical F1 Drivers — ${res.driver_code}`,
        xaxis: { range: [0.8, 1.0], title: "Cosine Similarity" },
        yaxis_title: "",
        height: 200,
      },
    });
  }

  return (
    <div className="flex flex-col gap-4">
      <p className="text-xs text-white/50">
        <strong className="text-white/70">RAG pattern</strong>: the driver&apos;s
        normalised radar vector is compared to legendary F1 drivers by cosine
        similarity. The most stylistically similar historical profiles are
        retrieved and used to augment the LLM explanation.
      </p>

      <Button
        variant="primary"
        onClick={run}
        disabled={onCooldown || loading}
        className="self-start"
      >
        {btnLabel}
      </Button>

      {error && (
        <div className="border border-red-500/50 bg-red-900/20 px-3 py-2 text-sm text-red-300">
          {error}
        </div>
      )}

      {result && (
        <div className="flex flex-col gap-4">
          <PlotlyChart figureJson={buildBarFig(result)} height={220} />

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {result.matches.map((match, ci) => (
              <div
                key={match.name}
                className="border border-white/10 bg-[var(--bg-2)] p-3 flex flex-col gap-1"
              >
                <div className="text-xs uppercase tracking-wider text-white/40">
                  #{ci + 1} DNA Match
                </div>
                <div className="font-bold text-white">{match.name}</div>
                <div className="text-sm text-[var(--f1-red)] font-semibold">
                  Similarity: {(match.similarity * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-white/50">Era: {match.era}</div>
                {aiMode !== "Concise" && (
                  <p className="text-xs text-white/60 mt-1">{match.description}</p>
                )}
              </div>
            ))}
          </div>

          <div>
            <h4 className="text-sm font-semibold text-white/80 mb-2">
              Why the DNA aligns
            </h4>
            <div className="border-l-2 border-[var(--f1-red)] pl-3 py-1 text-sm text-white/70">
              {result.narrative}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
