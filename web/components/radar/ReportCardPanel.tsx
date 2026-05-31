"use client";

import { useState } from "react";
import { Button } from "@/components/ui/Button";
import { useCooldown } from "@/components/ui/Cooldown";
import { apiClient, type ReportCardResponse } from "@/lib/api-client";
import type { AIMode } from "@/lib/preferences";

interface Props {
  driverId: number;
  season: number;
  aiMode: AIMode;
}

export function ReportCardPanel({ driverId, season, aiMode }: Props) {
  const { secondsLeft, startCooldown, onCooldown } = useCooldown();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ReportCardResponse | null>(null);

  async function run() {
    if (onCooldown || loading) return;
    setLoading(true);
    setError(null);
    try {
      const data = await apiClient.reportCard(driverId, season);
      setResult(data);
      startCooldown();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  const btnLabel = onCooldown
    ? `Generate report (${secondsLeft}s)`
    : loading
      ? "Generating…"
      : "Generate report";

  return (
    <div className="flex flex-col gap-4">
      <p className="text-xs text-white/50">
        <strong className="text-white/70">Structured output pattern</strong>: the
        model is called in JSON mode. The schema is enforced via prompt
        engineering and validated server-side. If validation fails, the prompt is
        retried once with the specific error injected.
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
          {/* Headline + grade */}
          <div>
            <h3 className="text-lg font-bold text-white">{result.headline}</h3>
            <span className="inline-block mt-1 border-2 border-[var(--f1-red)] px-3 py-0.5 text-sm font-bold text-white shadow-[2px_2px_0_var(--f1-red)]">
              Grade: {result.grade}
            </span>
          </div>

          <hr className="border-white/10" />

          {/* Three columns: Strengths / Weaknesses / Verdict */}
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            {/* Strengths */}
            <div>
              <h4 className="text-xs uppercase tracking-widest text-white/60 mb-2">
                Strengths
              </h4>
              <div className="flex flex-col gap-2">
                {result.strengths.map((s, i) => (
                  <div
                    key={i}
                    className="border border-green-500/30 border-l-[3px] border-l-green-500 bg-[var(--bg-2)] px-3 py-2 text-xs text-white/70"
                  >
                    ✅ {s}
                  </div>
                ))}
              </div>
            </div>

            {/* Weaknesses */}
            <div>
              <h4 className="text-xs uppercase tracking-widest text-white/60 mb-2">
                Weaknesses
              </h4>
              <div className="flex flex-col gap-2">
                {result.weaknesses.map((w, i) => (
                  <div
                    key={i}
                    className="border border-orange-500/30 border-l-[3px] border-l-orange-500 bg-[var(--bg-2)] px-3 py-2 text-xs text-white/70"
                  >
                    ⚠️ {w}
                  </div>
                ))}
              </div>
            </div>

            {/* Verdict */}
            <div>
              <h4 className="text-xs uppercase tracking-widest text-white/60 mb-2">
                Verdict
              </h4>
              {aiMode !== "Concise" ? (
                <div className="border border-blue-500/30 border-l-[3px] border-l-blue-500 bg-[var(--bg-2)] px-3 py-2 text-xs text-white/70">
                  {result.verdict}
                </div>
              ) : (
                <p className="text-xs text-white/40 italic">
                  Switch to Detailed mode for full verdict.
                </p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
