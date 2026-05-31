"use client";

import { useState, useEffect } from "react";
import {
  useVisibleCharts,
  useAIMode,
  CHART_LABELS,
  type AIMode,
  type ChartLabel,
} from "@/lib/preferences";

const AI_MODES: AIMode[] = ["Concise", "Detailed", "Critique loop"];
const RACE_MODE_KEY = "dna_rp_mode";

// ── Custom radio indicator ─────────────────────────────────────────────────
// The native <input> is opacity-0 and fills the container; peer-checked
// drives the ring + dot siblings that follow it in the DOM.

function RadioOption({
  name,
  value,
  checked,
  onChange,
}: {
  name: string;
  value: string;
  checked: boolean;
  onChange: () => void;
}) {
  return (
    <label className="flex items-center gap-2 cursor-pointer group">
      <span className="relative inline-flex h-4 w-4 shrink-0">
        <input
          type="radio"
          name={name}
          checked={checked}
          onChange={onChange}
          className="peer absolute inset-0 h-full w-full cursor-pointer opacity-0"
        />
        {/* Outer ring */}
        <span className="h-4 w-4 rounded-full border-2 border-white/25 transition-colors peer-checked:border-[var(--f1-red)]" />
        {/* Inner filled dot */}
        <span className="absolute left-1/2 top-1/2 h-2 w-2 -translate-x-1/2 -translate-y-1/2 scale-0 rounded-full bg-[var(--f1-red)] transition-transform duration-150 peer-checked:scale-100" />
      </span>
      <span className="text-sm text-white/60 transition-colors group-hover:text-white/90 peer-checked:text-white">
        {value}
      </span>
    </label>
  );
}

// ── Custom checkbox indicator ──────────────────────────────────────────────

function CheckOption({
  label,
  checked,
  onChange,
}: {
  label: ChartLabel;
  checked: boolean;
  onChange: () => void;
}) {
  return (
    <label className="flex items-center gap-2 cursor-pointer group">
      <span className="relative inline-flex h-4 w-4 shrink-0">
        <input
          type="checkbox"
          checked={checked}
          onChange={onChange}
          className="peer absolute inset-0 h-full w-full cursor-pointer opacity-0"
        />
        {/* Box */}
        <span className="h-4 w-4 border-2 border-white/25 bg-transparent transition-colors peer-checked:border-[var(--f1-red)] peer-checked:bg-[var(--f1-red)]" />
        {/* Checkmark */}
        <span className="pointer-events-none absolute inset-0 flex items-center justify-center opacity-0 transition-opacity peer-checked:opacity-100">
          <svg
            width="10"
            height="8"
            viewBox="0 0 10 8"
            fill="none"
            aria-hidden="true"
          >
            <path
              d="M1 4l3 3L9 1"
              stroke="white"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </span>
      </span>
      <span className="text-sm text-white/60 transition-colors group-hover:text-white/90">
        {label}
      </span>
    </label>
  );
}

// ── SidebarClient ──────────────────────────────────────────────────────────

export function SidebarClient() {
  const { charts, toggle } = useVisibleCharts();
  const { mode, setMode } = useAIMode();
  const [raceMode, setRaceMode] = useState<string | null>(null);
  const [apiStatus, setApiStatus] = useState<"loading" | "ok" | "error">(
    "loading",
  );

  useEffect(() => {
    setRaceMode(localStorage.getItem(RACE_MODE_KEY));
  }, []);

  useEffect(() => {
    const base = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
    fetch(`${base}/healthz`)
      .then((r) => setApiStatus(r.ok ? "ok" : "error"))
      .catch(() => setApiStatus("error"));
  }, []);

  return (
    <>
      {/* Race Dashboard mode indicator */}
      {raceMode ? (
        <p className="mb-1 text-xs text-white/60">Mode: 🗂️ Historical</p>
      ) : (
        <p className="mb-1 text-xs italic text-white/40">
          Select a mode in the Race Dashboard tab to get started.
        </p>
      )}

      {/* Race Dashboard info callout */}
      <div className="border-l-2 border-[var(--f1-red)] pl-3 py-1 text-xs text-white/60">
        <strong className="text-white/80">Race Dashboard</strong> — fastest lap
        telemetry comparison, plus historical race analysis with rolling pace,
        gap charts, undercut detection, and projected finishing order.
      </div>

      {/* Visible Charts */}
      <hr className="my-4 border-white/10" />
      <details>
        <summary className="cursor-pointer select-none py-1 text-xs uppercase tracking-widest text-white/60 hover:text-white/80">
          Visible Charts
        </summary>
        <div className="mt-2 space-y-2">
          {CHART_LABELS.map((label) => (
            <CheckOption
              key={label}
              label={label}
              checked={charts[label] ?? true}
              onChange={() => toggle(label)}
            />
          ))}
        </div>
      </details>

      {/* Health Check */}
      <hr className="my-4 border-white/10" />
      <details>
        <summary className="cursor-pointer select-none py-1 text-xs uppercase tracking-widest text-white/60 hover:text-white/80">
          🩺 Health Check
        </summary>
        <div className="mt-2 text-sm">
          {apiStatus === "loading" && (
            <span className="text-white/40">Checking…</span>
          )}
          {apiStatus === "ok" && (
            <span className="text-white/70">✅ API reachable</span>
          )}
          {apiStatus === "error" && (
            <span className="text-white/70">❌ API unreachable</span>
          )}
        </div>
      </details>

      {/* AI Response Mode */}
      <hr className="my-4 border-white/10" />
      <div>
        <p className="mb-2 text-xs uppercase tracking-widest text-white/60">
          AI Response Mode
        </p>
        <div className="space-y-2">
          {AI_MODES.map((m) => (
            <RadioOption
              key={m}
              name="ai_mode"
              value={m}
              checked={mode === m}
              onChange={() => setMode(m)}
            />
          ))}
        </div>
      </div>
    </>
  );
}
