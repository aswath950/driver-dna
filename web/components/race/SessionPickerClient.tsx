"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/Button";
import { apiClient, type EventOut, type SessionOut, type GpScheduleItem } from "@/lib/api-client";

// Years for which OpenF1 has real race data (confirmed via /meetings API).
const AVAILABLE_YEARS = [2026, 2025, 2024, 2023];

const LAST_SESSION_KEY = "dna_last_session";

interface LastSession {
  year: number;
  eventId: number;
  sessionId: number;
}

function readLastSession(): LastSession | null {
  try {
    const raw = localStorage.getItem(LAST_SESSION_KEY);
    return raw ? (JSON.parse(raw) as LastSession) : null;
  } catch {
    return null;
  }
}

const SELECT_CLASS =
  "border border-white/20 bg-[var(--bg-2)] px-3 py-2 text-sm text-white focus:outline-none";
const DISABLED_CLASS = "cursor-not-allowed opacity-50";

export function SessionPickerClient() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const forcePickerMode = searchParams.get("pick") === "1";

  // Carry target eventId/sessionId to restore across async effect boundaries
  const restoreEventId   = useRef<number | null>(null);
  const restoreSessionId = useRef<number | null>(null);

  const [year, setYear] = useState(AVAILABLE_YEARS[0]);
  const [events, setEvents] = useState<EventOut[]>([]);
  const [eventId, setEventId] = useState<number | null>(null);
  const [sessions, setSessions] = useState<SessionOut[]>([]);
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [loadingEvents, setLoadingEvents] = useState(false);
  const [loadingSessions, setLoadingSessions] = useState(false);

  // On mount: auto-redirect to last session, or restore picker state when ?pick=1
  useEffect(() => {
    const stored = readLastSession();
    if (!stored) return;

    if (!forcePickerMode) {
      // Send user straight back to the session they were viewing
      router.replace(`/race/${stored.sessionId}`);
      return;
    }

    // ?pick=1 — restore dropdowns so user sees their last selection
    restoreEventId.current   = stored.eventId;
    restoreSessionId.current = stored.sessionId;
    setYear(stored.year); // triggers year effect which cascades restore
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Fetch events whenever the selected year changes (including on initial mount)
  useEffect(() => {
    let cancelled = false;
    setEvents([]);
    setEventId(null);
    setSessions([]);
    setSessionId(null);
    setLoadingEvents(true);

    Promise.all([
      apiClient.events(year),
      apiClient.gpSchedule(year).catch((): GpScheduleItem[] => []),
    ])
      .then(([page, schedule]) => {
        if (cancelled) return;
        // Map OpenF1 round number → real GP name
        const nameByRound = new Map(schedule.map((gp) => [gp.round, gp.name]));
        const enriched = page.data.map((ev) => ({
          ...ev,
          name: nameByRound.get(ev.round) ?? ev.name,
        }));
        setEvents(enriched);

        // Cascade restore: set eventId if the stored one exists in this year's events
        const toRestore = restoreEventId.current;
        if (toRestore && enriched.some((ev) => ev.id === toRestore)) {
          restoreEventId.current = null;
          setEventId(toRestore); // triggers session effect
        }
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setLoadingEvents(false);
      });

    return () => {
      cancelled = true;
    };
  }, [year]);

  // Fetch sessions whenever the selected event changes
  useEffect(() => {
    if (!eventId) return;
    let cancelled = false;
    setSessions([]);
    setSessionId(null);
    setLoadingSessions(true);
    apiClient
      .sessions(eventId)
      .then((data) => {
        if (!cancelled) {
          setSessions(data);

          // Cascade restore: set sessionId if the stored one exists in this event
          const toRestore = restoreSessionId.current;
          if (toRestore && data.some((s) => s.id === toRestore)) {
            restoreSessionId.current = null;
            setSessionId(toRestore);
          }
        }
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setLoadingSessions(false);
      });
    return () => {
      cancelled = true;
    };
  }, [eventId]);

  const gpDisabled = loadingEvents || events.length === 0;
  const sessionDisabled = loadingSessions || sessions.length === 0;

  return (
    <div className="flex flex-col gap-5">
      {/* Mode radio */}
      <div className="flex flex-col gap-1">
        <span className="text-xs uppercase tracking-widest text-white/50">Mode</span>
        <div className="flex gap-5">
          {/* Historical — always selected */}
          <label className="flex cursor-pointer items-center gap-2">
            <span className="relative inline-flex h-4 w-4 shrink-0">
              <span className="h-4 w-4 rounded-full border-2 border-[var(--f1-red)]" />
              <span className="absolute left-1/2 top-1/2 h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-[var(--f1-red)]" />
            </span>
            <span className="text-sm text-white/90">Historical</span>
          </label>
          {/* Live — always disabled */}
          <label
            className="flex cursor-not-allowed items-center gap-2"
            title="Live mode coming soon"
          >
            <span className="relative inline-flex h-4 w-4 shrink-0">
              <span className="h-4 w-4 rounded-full border-2 border-white/20" />
            </span>
            <span className="text-sm text-white/30">Live</span>
          </label>
        </div>
      </div>

      {/* Pickers + Load */}
      <div className="flex flex-wrap items-end gap-4">
        {/* Year */}
        <div className="flex flex-col gap-1">
          <label className="text-xs uppercase tracking-widest text-white/50">Year</label>
          <select
            value={year}
            onChange={(e) => setYear(Number(e.target.value))}
            className={SELECT_CLASS}
          >
            {AVAILABLE_YEARS.map((y) => (
              <option key={y} value={y}>
                {y}
              </option>
            ))}
          </select>
        </div>

        {/* Grand Prix */}
        <div className="flex flex-col gap-1">
          <label className="text-xs uppercase tracking-widest text-white/50">Grand Prix</label>
          <select
            value={eventId ?? ""}
            onChange={(e) => setEventId(Number(e.target.value))}
            disabled={gpDisabled}
            className={`${SELECT_CLASS} ${gpDisabled ? DISABLED_CLASS : ""}`}
          >
            <option value="">
              {loadingEvents ? "Loading…" : "Select GP"}
            </option>
            {events.map((ev) => (
              <option key={ev.id} value={ev.id}>
                {ev.name}
              </option>
            ))}
          </select>
        </div>

        {/* Session */}
        <div className="flex flex-col gap-1">
          <label className="text-xs uppercase tracking-widest text-white/50">Session</label>
          <select
            value={sessionId ?? ""}
            onChange={(e) => setSessionId(Number(e.target.value))}
            disabled={sessionDisabled}
            className={`${SELECT_CLASS} ${sessionDisabled ? DISABLED_CLASS : ""}`}
          >
            <option value="">
              {loadingSessions ? "Loading…" : "Select Session"}
            </option>
            {sessions.map((s) => (
              <option key={s.id} value={s.id}>
                {s.type}
                {s.date_start ? ` (${s.date_start.slice(0, 10)})` : ""}
              </option>
            ))}
          </select>
        </div>

        {/* Load button */}
        <Button
          disabled={!sessionId}
          onClick={() => {
            if (!sessionId || !eventId) return;
            try {
              localStorage.setItem(
                LAST_SESSION_KEY,
                JSON.stringify({ year, eventId, sessionId }),
              );
            } catch {}
            router.push(`/race/${sessionId}`);
          }}
          className="disabled:cursor-not-allowed disabled:opacity-50"
        >
          Load Session
        </Button>
      </div>
    </div>
  );
}
