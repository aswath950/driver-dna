"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/Button";
import { apiClient, type SeasonOut, type EventOut, type SessionOut } from "@/lib/api-client";

interface Props {
  seasons: SeasonOut[];
}

const SELECT_CLASS =
  "border border-white/20 bg-[var(--bg-2)] px-3 py-2 text-sm text-white focus:outline-none";
const DISABLED_CLASS = "cursor-not-allowed opacity-50";

export function SessionPickerClient({ seasons }: Props) {
  const router = useRouter();
  const defaultYear = seasons[0]?.year ?? new Date().getFullYear();

  const [year, setYear] = useState(defaultYear);
  const [events, setEvents] = useState<EventOut[]>([]);
  const [eventId, setEventId] = useState<number | null>(null);
  const [sessions, setSessions] = useState<SessionOut[]>([]);
  const [sessionId, setSessionId] = useState<number | null>(null);
  const [loadingEvents, setLoadingEvents] = useState(false);
  const [loadingSessions, setLoadingSessions] = useState(false);

  // Fetch events whenever the selected year changes (including on initial mount)
  useEffect(() => {
    let cancelled = false;
    setEvents([]);
    setEventId(null);
    setSessions([]);
    setSessionId(null);
    setLoadingEvents(true);
    apiClient
      .events(year)
      .then((page) => {
        if (!cancelled) setEvents(page.data);
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
        if (!cancelled) setSessions(data);
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
            {seasons.map((s) => (
              <option key={s.id} value={s.year}>
                {s.year}
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
          onClick={() => sessionId && router.push(`/race/${sessionId}`)}
          className="disabled:cursor-not-allowed disabled:opacity-50"
        >
          Load Session
        </Button>
      </div>
    </div>
  );
}
