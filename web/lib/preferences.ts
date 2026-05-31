"use client";

import { useState, useEffect, useCallback } from "react";

export const VISIBLE_CHARTS_STORAGE_KEY = "dna_visible_charts";
export const AI_MODE_STORAGE_KEY = "dna_ai_mode";

// Internal custom-event names for same-tab broadcast
const AI_MODE_EVENT = "dna_ai_mode_change";
const CHARTS_EVENT = "dna_charts_change";

export const CHART_LABELS = [
  "Rolling pace",
  "Gap to leader",
  "Undercuts",
  "Tyre degradation",
  "Telemetry compare",
] as const;

export type ChartLabel = (typeof CHART_LABELS)[number];
export type VisibleCharts = Record<ChartLabel, boolean>;

const DEFAULT_VISIBLE: VisibleCharts = {
  "Rolling pace": true,
  "Gap to leader": true,
  Undercuts: true,
  "Tyre degradation": true,
  "Telemetry compare": true,
};

export type AIMode = "Concise" | "Detailed" | "Critique loop";
const DEFAULT_AI_MODE: AIMode = "Detailed";

function readLS<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
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

// ── useVisibleCharts ───────────────────────────────────────────────────────

export function useVisibleCharts() {
  const [charts, setCharts] = useState<VisibleCharts>(DEFAULT_VISIBLE);

  useEffect(() => {
    // Hydrate on mount
    setCharts(readLS(VISIBLE_CHARTS_STORAGE_KEY, DEFAULT_VISIBLE));

    // Same-tab sync: another caller called toggle()
    const onCustom = (e: Event) => {
      setCharts((e as CustomEvent<VisibleCharts>).detail);
    };

    // Cross-tab sync
    const onStorage = (e: StorageEvent) => {
      if (e.key === VISIBLE_CHARTS_STORAGE_KEY && e.newValue) {
        try {
          setCharts(JSON.parse(e.newValue) as VisibleCharts);
        } catch {
          /* malformed */
        }
      }
    };

    window.addEventListener(CHARTS_EVENT, onCustom);
    window.addEventListener("storage", onStorage);
    return () => {
      window.removeEventListener(CHARTS_EVENT, onCustom);
      window.removeEventListener("storage", onStorage);
    };
  }, []);

  const toggle = useCallback((label: ChartLabel) => {
    setCharts((prev) => {
      const next = { ...prev, [label]: !prev[label] };
      writeLS(VISIBLE_CHARTS_STORAGE_KEY, next);
      window.dispatchEvent(
        new CustomEvent<VisibleCharts>(CHARTS_EVENT, { detail: next }),
      );
      return next;
    });
  }, []);

  return { charts, toggle };
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
