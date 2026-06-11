"use client";

import { useEffect, useRef, useState } from "react";
import { Card } from "@/components/ui/Card";
import { StatPill } from "@/components/ui/StatPill";
import { MODEL_METRICS_REFRESH_EVENT } from "@/components/SidebarClient";
import { apiClient, type EventOut, type SessionOut, type GpScheduleItem } from "@/lib/api-client";

const YEARS = [2026, 2025, 2024, 2023];

const SESSION_OPTIONS: { value: string; label: string }[] = [
  { value: "",    label: "All Sessions" },
  { value: "R",   label: "R — Race" },
  { value: "Q",   label: "Q — Qualifying" },
  { value: "S",   label: "S — Sprint" },
  { value: "SQ",  label: "SQ — Sprint Qualifying" },
  { value: "FP1", label: "FP1 — Free Practice 1" },
  { value: "FP2", label: "FP2 — Free Practice 2" },
  { value: "FP3", label: "FP3 — Free Practice 3" },
];

interface EventOption {
  id: number;
  name: string;
  round: number;
}

function FetchTelemetryPanel({ base }: { base: string }) {
  const [year, setYear] = useState(YEARS[0]);
  const [events, setEvents] = useState<EventOut[]>([]);
  const [eventId, setEventId] = useState<number | null>(null);
  const [sessions, setSessions] = useState<SessionOut[]>([]);
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [eventsLoading, setEventsLoading] = useState(false);
  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [logs, setLogs] = useState("");
  const [running, setRunning] = useState(false);

  const SELECT_CLASS =
    "border border-white/20 bg-[var(--bg-2)] px-3 py-2 text-sm text-white focus:outline-none disabled:opacity-40";
  const BTN_CLASS =
    "px-4 py-2 text-sm font-semibold bg-[var(--f1-red)] text-white hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed transition-opacity";

  // Load events (with real GP names) when year changes
  useEffect(() => {
    setEvents([]);
    setEventId(null);
    setSessions([]);
    setSessionId(null);
    setStatus(null);
    setEventsLoading(true);
    Promise.all([
      apiClient.events(year),
      apiClient.gpSchedule(year).catch((): GpScheduleItem[] => []),
    ])
      .then(([page, schedule]) => {
        const nameByRound = new Map(schedule.map((gp) => [gp.round, gp.name]));
        const enriched = page.data.map((ev) => ({
          ...ev,
          name: nameByRound.get(ev.round) ?? ev.name,
        }));
        setEvents(enriched);
      })
      .catch(() => {})
      .finally(() => setEventsLoading(false));
  }, [year]);

  // Load sessions when event changes
  useEffect(() => {
    if (!eventId) return;
    setSessions([]);
    setSessionId(null);
    setStatus(null);
    setSessionsLoading(true);
    apiClient.sessions(eventId)
      .then(setSessions)
      .catch(() => {})
      .finally(() => setSessionsLoading(false));
  }, [eventId]);

  // Auto-check cache status when session is selected
  useEffect(() => {
    if (!sessionId) { setStatus(null); return; }
    fetch(`${base}/api/v1/pipeline/telemetry-status?session_id=${sessionId}`)
      .then((r) => r.json())
      .then((d) => setStatus(
        d.fetched_at
          ? `Cached at ${d.fetched_at.slice(0, 19).replace("T", " ")}`
          : "Not cached",
      ))
      .catch(() => setStatus(null));
  }, [sessionId, base]);

  function runFetch() {
    if (!sessionId) return;
    setLogs("");
    setRunning(true);
    setStatus(null);
    const es = new EventSource(`${base}/api/v1/pipeline/fetch-telemetry?session_id=${sessionId}`);
    es.onmessage = (e) => {
      setLogs((prev) => prev + e.data + "\n");
      if (e.data.startsWith("[done]")) {
        es.close();
        setRunning(false);
        fetch(`${base}/api/v1/pipeline/telemetry-status?session_id=${sessionId}`)
          .then((r) => r.json())
          .then((d) => setStatus(
            d.fetched_at
              ? `Cached at ${d.fetched_at.slice(0, 19).replace("T", " ")}`
              : "Not cached",
          ))
          .catch(() => setStatus(null));
      }
    };
    es.onerror = () => { es.close(); setRunning(false); };
  }

  const gpDisabled = eventsLoading || events.length === 0;
  const sessionDisabled = !eventId || sessionsLoading || sessions.length === 0;

  return (
    <>
      <div className="flex flex-wrap items-end gap-3">
        {/* Year */}
        <div className="flex flex-col gap-1">
          <label className="text-xs uppercase tracking-widest text-white/50">Year</label>
          <select
            value={year}
            onChange={(e) => setYear(Number(e.target.value))}
            className={SELECT_CLASS}
          >
            {YEARS.map((y) => (
              <option key={y} value={y}>{y}</option>
            ))}
          </select>
        </div>

        {/* Grand Prix */}
        <div className="flex flex-col gap-1 min-w-[260px]">
          <label className="text-xs uppercase tracking-widest text-white/50">
            Grand Prix
            {eventsLoading && <span className="ml-2 text-white/30">Loading…</span>}
          </label>
          <select
            value={eventId ?? ""}
            onChange={(e) => setEventId(Number(e.target.value))}
            disabled={gpDisabled}
            className={SELECT_CLASS}
          >
            <option value="">{eventsLoading ? "Loading…" : "Select GP"}</option>
            {events.map((ev) => (
              <option key={ev.id} value={ev.id}>{ev.name}</option>
            ))}
          </select>
        </div>

        {/* Session */}
        <div className="flex flex-col gap-1">
          <label className="text-xs uppercase tracking-widest text-white/50">
            Session
            {sessionsLoading && <span className="ml-2 text-white/30">Loading…</span>}
          </label>
          <select
            value={sessionId ?? ""}
            onChange={(e) => setSessionId(Number(e.target.value))}
            disabled={sessionDisabled}
            className={SELECT_CLASS}
          >
            <option value="">{sessionsLoading ? "Loading…" : "Select Session"}</option>
            {sessions.map((s) => (
              <option key={s.id} value={s.id}>
                {s.type}{s.date_start ? ` (${s.date_start.slice(0, 10)})` : ""}
              </option>
            ))}
          </select>
        </div>

        {/* Fetch button */}
        <button
          onClick={runFetch}
          disabled={running || !sessionId}
          className={BTN_CLASS}
        >
          {running ? "Fetching…" : "▶ Fetch Telemetry"}
        </button>

        {/* Cache status badge */}
        {status && (
          <span className={`text-xs ${status.startsWith("Cached") ? "text-green-400" : "text-white/40"}`}>
            {status}
          </span>
        )}
      </div>
      <LogBox logs={logs} />
    </>
  );
}

interface Stats {
  dataset_rows: number;
  drivers: number;
  laps: number;
  last_updated: string | null;
}

function LogBox({ logs }: { logs: string }) {
  const ref = useRef<HTMLPreElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [logs]);
  return (
    <pre
      ref={ref}
      className="mt-3 h-48 overflow-y-auto bg-black/60 px-3 py-2 text-xs text-white/80 whitespace-pre-wrap"
    >
      {logs || <span className="text-white/30">Output will appear here…</span>}
    </pre>
  );
}

export default function PipelinePage() {
  const [stats, setStats] = useState<Stats | null>(null);

  // Hydrate controls
  const [year, setYear] = useState(2026);
  const [events, setEvents] = useState<EventOption[]>([]);
  const [eventsLoading, setEventsLoading] = useState(false);
  const [gp, setGp] = useState("");
  const [session, setSession] = useState("R");
  const [hydrateLogs, setHydrateLogs] = useState("");
  const [hydrateRunning, setHydrateRunning] = useState(false);

  // Train controls
  const [trainLogs, setTrainLogs] = useState("");
  const [trainRunning, setTrainRunning] = useState(false);

  const base = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

  useEffect(() => {
    fetch(`${base}/api/v1/pipeline/stats`)
      .then((r) => r.json())
      .then(setStats)
      .catch(() => null);
  }, [base]);

  // Load GP schedule from OpenF1 whenever year changes
  useEffect(() => {
    setEventsLoading(true);
    setGp("");
    fetch(`${base}/api/v1/pipeline/gp-schedule?year=${year}`)
      .then((r) => {
        if (!r.ok) throw new Error(`${r.status}`);
        return r.json();
      })
      .then((list: EventOption[]) => {
        setEvents(list);
        setGp(list[0]?.name ?? "");
      })
      .catch(() => setEvents([]))
      .finally(() => setEventsLoading(false));
  }, [base, year]);

  function startSSE(
    url: string,
    setLogs: (fn: (prev: string) => string) => void,
    setRunning: (v: boolean) => void,
  ) {
    setLogs(() => "");
    setRunning(true);
    const es = new EventSource(url);
    es.onmessage = (e) => {
      setLogs((prev) => prev + e.data + "\n");
      if (e.data.startsWith("[done]")) {
        es.close();
        setRunning(false);
        fetch(`${base}/api/v1/pipeline/stats`)
          .then((r) => r.json())
          .then(setStats)
          .catch(() => null);
        window.dispatchEvent(new CustomEvent(MODEL_METRICS_REFRESH_EVENT));
      }
    };
    es.onerror = () => {
      es.close();
      setRunning(false);
    };
  }

  function runHydrate() {
    if (!gp) return;
    const params = new URLSearchParams({ year: String(year), gp });
    if (session) params.set("session", session);
    startSSE(`${base}/api/v1/pipeline/hydrate?${params}`, setHydrateLogs, setHydrateRunning);
  }

  function runTrain() {
    startSSE(`${base}/api/v1/pipeline/train`, setTrainLogs, setTrainRunning);
  }

  const SELECT_CLASS =
    "border border-white/20 bg-[var(--bg-2)] px-3 py-2 text-sm text-white focus:outline-none disabled:opacity-40";
  const BTN_CLASS =
    "px-4 py-2 text-sm font-semibold bg-[var(--f1-red)] text-white hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed transition-opacity";

  return (
    <div className="flex flex-col gap-6">
      <h2 className="text-xl font-bold tracking-tight">⚙️ Pipeline &amp; Training</h2>

      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <StatPill label="Dataset rows" value={stats ? stats.dataset_rows.toLocaleString() : "—"} />
        <StatPill label="Drivers" value={stats ? String(stats.drivers) : "—"} />
        <StatPill label="Sessions" value={stats ? String(stats.laps) : "—"} />
        <StatPill
          label="Last updated"
          value={stats?.last_updated ? stats.last_updated.slice(0, 10) : "—"}
        />
      </div>

      {/* Hydrate */}
      <section>
        <h3 className="mb-3 text-base font-semibold">📥 Download Dataset</h3>
        <Card>
          <div className="flex flex-wrap items-end gap-3">
            <div className="flex flex-col gap-1">
              <label className="text-xs uppercase tracking-widest text-white/50">Year</label>
              <select
                value={year}
                onChange={(e) => setYear(Number(e.target.value))}
                className={SELECT_CLASS}
              >
                {YEARS.map((y) => (
                  <option key={y} value={y}>{y}</option>
                ))}
              </select>
            </div>

            <div className="flex flex-col gap-1 min-w-[260px]">
              <label className="text-xs uppercase tracking-widest text-white/50">
                Grand Prix
                {eventsLoading && <span className="ml-2 text-white/30">Loading…</span>}
              </label>
              <select
                value={gp}
                onChange={(e) => setGp(e.target.value)}
                disabled={eventsLoading || events.length === 0}
                className={`${SELECT_CLASS} w-full`}
              >
                {events.length === 0 && !eventsLoading && (
                  <option value="">No events found</option>
                )}
                {events.map((ev) => (
                  <option key={ev.id} value={ev.name}>
                    R{ev.round} — {ev.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex flex-col gap-1">
              <label className="text-xs uppercase tracking-widest text-white/50">Session</label>
              <select
                value={session}
                onChange={(e) => setSession(e.target.value)}
                className={SELECT_CLASS}
              >
                {SESSION_OPTIONS.map((opt) => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </select>
            </div>

            <button
              onClick={runHydrate}
              disabled={hydrateRunning || !gp || eventsLoading}
              className={BTN_CLASS}
            >
              {hydrateRunning ? "Running…" : "▶ Hydrate"}
            </button>
          </div>

          <LogBox logs={hydrateLogs} />
        </Card>
      </section>

      {/* Fetch Telemetry */}
      <section>
        <h3 className="mb-3 text-base font-semibold">📡 Fetch Telemetry</h3>
        <Card>
          <p className="mb-3 text-sm text-white/60">
            Pre-download all car telemetry for a session so compare charts load instantly
            from the database instead of calling OpenF1 on every request.
          </p>
          <FetchTelemetryPanel base={base} />
        </Card>
      </section>

      {/* Train */}
      <section>
        <h3 className="mb-3 text-base font-semibold">🧠 Train Model</h3>
        <Card>
          <button
            onClick={runTrain}
            disabled={trainRunning}
            className={BTN_CLASS}
          >
            {trainRunning ? "Training…" : "▶ Train"}
          </button>

          <LogBox logs={trainLogs} />
        </Card>
      </section>
    </div>
  );
}
