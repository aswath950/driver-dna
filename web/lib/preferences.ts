"use client";

import { useState, useEffect, useCallback } from "react";

export const AI_MODE_STORAGE_KEY = "dna_ai_mode";

// Internal custom-event name for same-tab broadcast
const AI_MODE_EVENT = "dna_ai_mode_change";

export const CHART_LABELS = [
  "Rolling pace",
  "Gap to leader",
  "Undercuts",
  "Tyre degradation",
  "Telemetry compare",
  "Corner performance",
] as const;

export type ChartLabel = (typeof CHART_LABELS)[number];

// Charts applicable per session type.  Sessions not listed here default to all charts.
// This is a domain-correctness rule (e.g. a Qualifying session has no race-pace data),
// not an availability switch — operator section visibility lives in lib/features.ts.
export const CHARTS_FOR_SESSION: Record<string, ChartLabel[]> = {
  R:   [...CHART_LABELS],
  S:   [...CHART_LABELS],
  FP1: ["Tyre degradation", "Telemetry compare", "Corner performance"],
  FP2: ["Tyre degradation", "Telemetry compare", "Corner performance"],
  FP3: ["Tyre degradation", "Telemetry compare", "Corner performance"],
  Q:   ["Telemetry compare", "Corner performance"],
  SQ:  ["Telemetry compare", "Corner performance"],
};

/**
 * Whether a chart applies to a given session type. Sessions not listed in
 * CHARTS_FOR_SESSION default to allowing every chart. Pure — safe to call from
 * server components with the server-known session type.
 */
export function chartAllowedForSession(
  label: ChartLabel,
  sessionType: string,
): boolean {
  const allowed = CHARTS_FOR_SESSION[sessionType];
  return allowed ? allowed.includes(label) : true;
}

export type AIMode = "Concise" | "Detailed" | "Critique loop";
const DEFAULT_AI_MODE: AIMode = "Detailed";

function readLS<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

function writeLS(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // ignore quota / security errors
  }
}

// ── useAIMode ──────────────────────────────────────────────────────────────

export function useAIMode() {
  const [mode, setModeState] = useState<AIMode>(DEFAULT_AI_MODE);

  useEffect(() => {
    // Hydrate on mount
    setModeState(readLS<AIMode>(AI_MODE_STORAGE_KEY, DEFAULT_AI_MODE));

    // Same-tab sync: another caller called setMode()
    const onCustom = (e: Event) => {
      setModeState((e as CustomEvent<AIMode>).detail);
    };

    // Cross-tab sync
    const onStorage = (e: StorageEvent) => {
      if (e.key === AI_MODE_STORAGE_KEY && e.newValue) {
        try {
          setModeState(JSON.parse(e.newValue) as AIMode);
        } catch {
          /* malformed */
        }
      }
    };

    window.addEventListener(AI_MODE_EVENT, onCustom);
    window.addEventListener("storage", onStorage);
    return () => {
      window.removeEventListener(AI_MODE_EVENT, onCustom);
      window.removeEventListener("storage", onStorage);
    };
  }, []);

  const setMode = useCallback((m: AIMode) => {
    setModeState(m);
    writeLS(AI_MODE_STORAGE_KEY, m);
    window.dispatchEvent(new CustomEvent<AIMode>(AI_MODE_EVENT, { detail: m }));
  }, []);

  return { mode, setMode };
}
