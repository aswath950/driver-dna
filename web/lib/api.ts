/**
 * Server-side typed REST client for React Server Components.
 *
 * Calls `${API_BASE_INTERNAL}/api/v1/*`, forwards the `dna_sid` cookie when
 * present so anonymous-session continuity works, and translates the RFC 7807
 * error envelope into a thrown `ApiError`.
 *
 * For client components that need to issue requests after hydration, see
 * `lib/api-client.ts` (Phase 11+).
 */

import { cookies } from "next/headers";
import { env } from "./env";
import type { components } from "./api-types";

export type SeasonOut = components["schemas"]["SeasonOut"];
export type EventOut = components["schemas"]["EventOut"];
export type SessionOut = components["schemas"]["SessionOut"];
export type DriverOut = components["schemas"]["DriverOut"];
export type RaceResultOut = components["schemas"]["RaceResultOut"];
export type StandingRowOut = components["schemas"]["StandingRowOut"];
export type LapOut = components["schemas"]["LapOut"];
export type RollingPaceRow = components["schemas"]["RollingPaceRow"];
export type GapRow = components["schemas"]["GapRow"];
export type DegradationRow = components["schemas"]["DegradationRow"];
export type ComparePayload = components["schemas"]["ComparePayload"];
export type UndercutEvent = components["schemas"]["UndercutEvent"];
export type StyleAnalystResponse = components["schemas"]["StyleAnalystResponse"];
export type DNAMatchResponse = components["schemas"]["DNAMatchResponse"];
export type ReportCardResponse = components["schemas"]["ReportCardResponse"];
export type XAIExplainResponse = components["schemas"]["XAIExplainResponse"];

// Page<T> is emitted per-instantiation (Page_SeasonOut_, etc.). Define a
// shared shape so call sites can stay generic.
export interface Page<T> {
  data: T[];
  page: { next_cursor: string | null; has_more: boolean; limit: number };
}

export interface TeamOut {
  id: number;
  name: string;
  color_hex: string | null;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    public type: string,
    public title: string,
    public detail?: string,
    public requestId?: string,
  ) {
    super(`[${status}] ${title}${detail ? ": " + detail : ""}`);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const cookieHeader = (() => {
    try {
      return cookies().toString();
    } catch {
      // Called outside a request scope (e.g. build) — no cookies available.
      return "";
    }
  })();
  const url = `${env.API_BASE_INTERNAL}${path}`;
  const res = await fetch(url, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(cookieHeader ? { Cookie: cookieHeader } : {}),
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...(init?.headers ?? {}),
    },
    // No caching by default — leaderboards/lap times are volatile.
    cache: init?.cache ?? "no-store",
  });
  if (!res.ok) {
    let body: { type?: string; title?: string; detail?: string; request_id?: string } = {};
    try {
      body = await res.json();
    } catch {
      /* envelope not JSON */
    }
    throw new ApiError(
      res.status,
      body.type ?? "unknown",
      body.title ?? res.statusText,
      body.detail,
      body.request_id,
    );
  }
  return res.json() as Promise<T>;
}

// ---------- Reads ----------

export const apiServer = {
  listSeasons: (limit = 20) =>
    request<Page<SeasonOut>>(`/api/v1/seasons?limit=${limit}`),

  listEvents: (year: number, limit = 50) =>
    request<Page<EventOut>>(`/api/v1/seasons/${year}/events?limit=${limit}`),

  listSessionsForEvent: (eventId: number | string) =>
    request<SessionOut[]>(`/api/v1/events/${eventId}/sessions`),

  getSession: (sessionId: number | string) =>
    request<SessionOut>(`/api/v1/sessions/${sessionId}`),

  getLeaderboard: (sessionId: number | string) =>
    request<RaceResultOut[]>(`/api/v1/sessions/${sessionId}/results`),

  listLaps: (sessionId: number | string, params: { driver_id?: number; from_lap?: number; to_lap?: number; limit?: number } = {}) => {
    const qs = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null) qs.set(k, String(v));
    }
    return request<Page<LapOut>>(`/api/v1/sessions/${sessionId}/laps?${qs}`);
  },

  listDrivers: (params: { season?: number; team?: string; limit?: number } = {}) => {
    const qs = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null) qs.set(k, String(v));
    }
    return request<Page<DriverOut>>(`/api/v1/drivers?${qs}`);
  },

  getStandings: (season: number) =>
    request<StandingRowOut[]>(`/api/v1/standings?season=${season}`),

  rollingPace: (sessionId: number | string, window = 5) =>
    request<RollingPaceRow[]>(
      `/api/v1/sessions/${sessionId}/analytics/rolling-pace?window=${window}`,
    ),

  gapToLeader: (sessionId: number | string) =>
    request<GapRow[]>(`/api/v1/sessions/${sessionId}/analytics/gap-to-leader`),

  tyreDegradation: (sessionId: number | string) =>
    request<DegradationRow[]>(
      `/api/v1/sessions/${sessionId}/analytics/tyre-degradation`,
    ),

  compare: (sessionId: number | string, driverA: number, driverB: number, channel = "Speed") =>
    request<ComparePayload>(
      `/api/v1/sessions/${sessionId}/compare?driver_a=${driverA}&driver_b=${driverB}&channel=${channel}`,
    ),

  undercuts: (sessionId: number | string) =>
    request<UndercutEvent[]>(
      `/api/v1/sessions/${sessionId}/analytics/undercuts`,
    ),

  listTeamsForSession: (sessionId: number | string) =>
    request<TeamOut[]>(`/api/v1/sessions/${sessionId}/teams`),
};
