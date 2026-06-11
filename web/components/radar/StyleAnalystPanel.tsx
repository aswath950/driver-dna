"use client";

import { useState } from "react";
import { Button } from "@/components/ui/Button";
import { useCooldown } from "@/components/ui/Cooldown";
import { apiClient, type StyleAnalystResponse, type Critique } from "@/lib/api-client";
import type { AIMode } from "@/lib/preferences";

interface Props {
  driverId: number;
  season: number;
  aiMode: AIMode;
}

export function StyleAnalystPanel({ driverId, season, aiMode }: Props) {
  const { secondsLeft, startCooldown, onCooldown } = useCooldown();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<StyleAnalystResponse | null>(null);

  async function run() {
    if (onCooldown || loading) return;
    setLoading(true);
    setError(null);
    try {
      const data = await apiClient.styleAnalyst(driverId, season);
      setResult(data);
      startCooldown();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Request failed");
    } finally {
      setLoading(false);
    }
  }

  const btnLabel = onCooldown
    ? `Analyse style (${secondsLeft}s)`
    : loading
      ? "Analysing…"
      : "Analyse style";

  // Build synthetic iteration list from the flat backend response
  function getIterations() {
    if (!result) return [];
    const iters: { roundNum: number; narrative: string; critique?: Critique }[] = [
      {
        roundNum: 1,
        narrative: result.analysis,
        critique: result.critique as Critique | undefined,
      },
    ];
    if (result.revised) {
      iters.push({ roundNum: 2, narrative: result.revised });
    }
    return iters;
  }

  const iterations = getIterations();
  const finalNarrative = result?.revised ?? result?.analysis ?? null;

  return (
    <div className="flex flex-col gap-4">
      <p className="text-xs text-white/50">
        <strong className="text-white/70">Reflexion pattern</strong> (Shinn et al.
        2023): an Analyst LLM generates a driving style narrative → a Critic LLM
        evaluates it and returns a confidence score + critique in JSON → if
        confidence &lt; 7/10 the Analyst revises using the critique as context.
        Max 2 rounds.
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

      {iterations.length > 0 && (
        <div className="flex flex-col gap-3">
          {iterations.map((iter, idx) => {
            const isLast = idx === iterations.length - 1;
            const label =
              iter.roundNum === 1
                ? "Round 1 — Initial Draft"
                : `Round ${iter.roundNum} — Revised Narrative`;
            const crit = iter.critique;
            const conf = crit?.confidence ?? result?.confidence;

            return (
              <details key={iter.roundNum} open={isLast}>
                <summary className="cursor-pointer select-none border border-white/10 bg-[var(--bg-2)] px-3 py-2 text-sm font-semibold text-white/80 hover:text-white">
                  {label}
                </summary>
                <div className="border border-white/10 border-t-0 px-3 py-3 flex flex-col gap-3">
                  <p className="text-sm text-white/70 whitespace-pre-wrap">
                    {iter.narrative}
                  </p>

                  {crit && aiMode !== "Concise" && (
                    <div className="flex flex-col gap-2">
                      <div className="flex items-center gap-3">
                        <div className="border border-white/20 px-3 py-1 text-sm">
                          <span className="text-white/50 text-xs uppercase tracking-wider mr-2">
                            Critic Confidence
                          </span>
                          <span className="font-bold text-white">
                            {conf ?? "—"}/10
                          </span>
                        </div>
                        {conf !== undefined && conf < 7 ? (
                          <span className="text-xs text-yellow-400">
                            Below 7/10 — Analyst will revise.
                          </span>
                        ) : (
                          <span className="text-xs text-green-400">
                            Confidence meets threshold — narrative accepted.
                          </span>
                        )}
                      </div>

                      {crit.factual_errors && crit.factual_errors.length > 0 && (
                        <div>
                          <p className="text-xs font-semibold text-white/60 mb-1">
                            Factual errors flagged:
                          </p>
                          <ul className="list-disc pl-4 text-xs text-white/50 space-y-0.5">
                            {crit.factual_errors.map((e, i) => (
                              <li key={i}>{e}</li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {crit.suggested_improvements && crit.suggested_improvements.length > 0 && (
                        <div>
                          <p className="text-xs font-semibold text-white/60 mb-1">
                            Suggested improvements:
                          </p>
                          <ul className="list-disc pl-4 text-xs text-white/50 space-y-0.5">
                            {crit.suggested_improvements.map((s, i) => (
                              <li key={i}>{s}</li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {crit.parse_note && (
                        <p className="text-xs text-white/40 italic">{crit.parse_note}</p>
                      )}
                    </div>
                  )}
                </div>
              </details>
            );
          })}

          {/* Final accepted narrative */}
          <div className="border border-green-500/40 bg-green-900/10 px-3 py-2 text-sm text-white/80">
            <span className="text-xs uppercase tracking-wider text-green-400/70 mr-2">
              Final:
            </span>
            {finalNarrative}
          </div>
        </div>
      )}
    </div>
  );
}
