"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { Button } from "@/components/ui/Button";
import { apiClient, type EventOut, type SessionOut, type GpScheduleItem } from "@/lib/api-client";

// Years for which OpenF1 has real race data.
const AVAILABLE_YEARS = [2026, 2025, 2024, 2023];

const LAST_SESSION_KEY = "dna_last_session";

interface LastSession {
  year: number;
  dbEventId: number;
  sessionId: number;
}

// Each entry in the GP dropdown: OpenF1-canonical name + round, and the DB
// event ID if that GP has been hydrated into the database (null = not yet).
interface EventItem {
  round: number;          // OpenF1 sequential round — used for stable ordering
  name: string;           // OpenF1 canonical name
  dbId: number | null;    // null → not hydrated; set → sessions can be loaded
  start_date: string | null;
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

  // Refs carry restore targets across async effect boundaries.
  const restoreDbEventId = useRef<number | null>(null);
  const restoreSessionId  = useRef<number | null>(null);

  const [year, setYear]                   = useState(AVAILABLE_YEARS[0]);
  const [events, setEvents]               = useState<EventItem[]>([]);
  const [selectedRound, setSelectedRound] = useState<number | null>(null);
  const [sessions, setSessions]           = useState<SessionOut[]>([]);
  const [sessionId, setSessionId]         = useState<number | null>(null);
  const [loadingEvents, setLoadingEvents]   = useState(false);
  const [loadingSessions, setLoadingSessions] = useState(false);
  const [scheduleError, setScheduleError] = useState(false);

  // Derived values from selected round.
  const selectedEvent = events.find((e) => e.round === selectedRound) ?? null;
  const dbEventId     = selectedEvent?.dbId ?? null;
  const notHydrated   = selectedRound !== null && selectedEvent !== null && dbEventId === null;

  // On mount: auto-redirect to last session, or restore picker state when ?pick=1
  useEffect(() => {
    const stored = readLastSession();
    if (!stored) return;

    if (!forcePickerMode) {
      router.replace(`/race/${stored.sessionId}`);
      return;
    }

    // ?pick=1 — restore dropdowns to the last-viewed selection
    restoreDbEventId.current = stored.dbEventId;
    restoreSessionId.current  = stored.sessionId;
    setYear(stored.year); // triggers year effect → cascades restore
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Populate the GP list whenever year changes.
  //
  // OpenF1 gp-schedule is the primary source (shows the full calendar for any
  // year). DB events are joined by name to determine which GPs are loadable.
  // Round numbers cannot be used for joining — DB stores OpenF1 meeting_key
  // as round (e.g. 1279) while gp-schedule returns sequential rounds (1, 2…).
  useEffect(() => {
    let cancelled = false;
    setEvents([]);
    setSelectedRound(null);
    setSessions([]);
    setSessionId(null);
    setScheduleError(false);
    setLoadingEvents(true);

    Promise.all([
      apiClient.gpSchedule(year).catch((): GpScheduleItem[] => []),
      apiClient.events(year).catch((): { data: EventOut[] } => ({ data: [] })),
    ])
      .then(([schedule, page]) => {
        if (cancelled) return;

        // Build name → DB event map (case-insensitive).
        const dbByName = new Map(
          page.data.map((ev) => [ev.name.toLowerCase(), ev]),
        );

        if (schedule.length === 0) {
          // OpenF1 unreachable — fall back to showing whatever is in the DB.
          setScheduleError(page.data.length === 0);
          const fallback: EventItem[] = page.data.map((ev) => ({
            round:      ev.round,
            name:       ev.name,
            dbId:       ev.id,
            start_date: ev.start_date ?? null,
          }));
          setEvents(fallback);
          return;
        }

        const enriched: EventItem[] = schedule.map((gp) => {
          const dbEv = dbByName.get(gp.name.toLowerCase());
          return {
            round:      gp.round,
            name:       gp.name,
            dbId:       dbEv?.id ?? null,
            start_date: gp.date ?? null,
          };
        });
        setEvents(enriched);

        // Cascade restore: find the item whose DB ID matches the stored one.
        const toRestore = restoreDbEventId.current;
        if (toRestore) {
          const item = enriched.find((e) => e.dbId === toRestore);
          if (item) {
            restoreDbEventId.current = null;
            setSelectedRound(item.round); // triggers session fetch
          }
        }
      })
      .finally(() => {
        if (!cancelled) setLoadingEvents(false);
      });

    return () => { cancelled = true; };
  }, [year]);

  // Fetch sessions whenever the selected DB event changes.
  useEffect(() => {
    if (!dbEventId) { setSessions([]); setSessionId(null); return; }
    let cancelled = false;
    setSessions([]);
    setSessionId(null);
    setLoadingSessions(true);
    apiClient
      .sessions(dbEventId)
      .then((data) => {
        if (!cancelled) {
          setSessions(data);
          const toRestore = restoreSessionId.current;
          if (toRestore && data.some((s) => s.id === toRestore)) {
            restoreSessionId.current = null;
            setSessionId(toRestore);
          }
        }
      })
      .catch(() => {})
      .finally(() => { if (!cancelled) setLoadingSessions(false); });
    return () => { cancelled = true; };
  }, [dbEventId]);

  const gpDisabled      = loadingEvents || events.length === 0;
  const sessionDisabled = notHydrated || loadingSessions || sessions.length === 0;

  return (
    <div className="flex flex-col gap-5">
      {/* Mode radio */}
      <div className="flex flex-col gap-1">
        <span className="text-xs uppercase tracking-widest text-white/50">Mode</span>
        <div className="flex gap-5">
          <label className="flex cursor-pointer items-center gap-2">
            <span className="relative inline-flex h-4 w-4 shrink-0">
              <span className="h-4 w-4 rounded-full border-2 border-[var(--f1-red)]" />
              <span className="absolute left-1/2 top-1/2 h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-[var(--f1-red)]" />
            </span>
            <span className="text-sm text-white/90">Historical</span>
          </label>
          <label className="flex cursor-not-allowed items-center gap-2" title="Live mode coming soon">
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
              <option key={y} value={y}>{y}</option>
            ))}
          </select>
        </div>

        {/* Grand Prix */}
        <div className="flex flex-col gap-1">
          <label className="text-xs uppercase tracking-widest text-white/50">Grand Prix</label>
          <select
            value={selectedRound ?? ""}
            onChange={(e) => setSelectedRound(Number(e.target.value) || null)}
            disabled={gpDisabled}
            className={`${SELECT_CLASS} ${gpDisabled ? DISABLED_CLASS : ""}`}
          >
            <option value="">
              {loadingEvents ? "Loading…" : "Select GP"}
            </option>
            {events.map((ev) => (
              <option key={ev.round} value={ev.round}>
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
                {s.type}{s.date_start ? ` (${s.date_start.slice(0, 10)})` : ""}
              </option>
            ))}
          </select>
        </div>

        {/* Load button */}
        <Button
          disabled={!sessionId}
          onClick={() => {
            if (!sessionId || !dbEventId) return;
            try {
              localStorage.setItem(
                LAST_SESSION_KEY,
                JSON.stringify({ year, dbEventId, sessionId }),
              );
            } catch {}
            router.push(`/race/${sessionId}`);
          }}
          className="disabled:cursor-not-allowed disabled:opacity-50"
        >
          Load Session
        </Button>
      </div>

      {/* Contextual messages */}
      {!loadingEvents && scheduleError && (
        <p className="text-sm text-white/50">
          Could not reach the schedule API — check that the backend is running.
        </p>
      )}
      {notHydrated && (
        <p className="text-sm text-white/50">
          {selectedEvent?.name} hasn&apos;t been downloaded yet.{" "}
          <Link href="/pipeline" className="text-[var(--f1-red)] underline underline-offset-2">
            Go to Pipeline
          </Link>{" "}
          and use Download Dataset to hydrate it, then come back here.
        </p>
      )}
    </div>
  );
}
