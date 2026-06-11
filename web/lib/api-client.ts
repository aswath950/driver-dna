/**
 * Browser-side fetch client. Use this in `"use client"` components.
 *
 * Cookies (dna_sid) are sent automatically by the browser when the API is
 * on the same origin, OR when CORS is configured with credentials. We set
 * `credentials: "include"` so prod (Vercel → Railway, cross-origin) works.
 */

import { env } from "./env";
import type { components } from "./api-types";

// ── Shared types re-exported for client components ─────────────────────────
export type StyleAnalystResponse = components["schemas"]["StyleAnalystResponse"];
export type DNAMatchResponse = components["schemas"]["DNAMatchResponse"];
export type ReportCardResponse = components["schemas"]["ReportCardResponse"];
export type SeasonOut = components["schemas"]["SeasonOut"];
export type EventOut = components["schemas"]["EventOut"];
export type SessionOut = components["schemas"]["SessionOut"];
export type ComparePayload = components["schemas"]["ComparePayload"];
export type UndercutEvent = components["schemas"]["UndercutEvent"];

// Hand-written until openapi:gen picks up the new endpoint.
export interface SectorDriverSplits {
  driver_id: number;
  code: string;
  lap_number: number | null;
  lap_time_ms: number | null;
  sector1_ms: number | null;
  sector2_ms: number | null;
  sector3_ms: number | null;
}
export interface SectorTimesPayload {
  session_id: number;
  driver_a: SectorDriverSplits;
  driver_b: SectorDriverSplits;
  figure_json: string;
}
export interface TrackMapDriverTrace {
  driver_id: number;
  code: string;
}
export interface TrackMapPayload {
  session_id: number;
  driver_a: TrackMapDriverTrace;
  driver_b: TrackMapDriverTrace;
  circuit_x: number[];
  circuit_y: number[];
  figure_json: string;
}

interface Page<T> {
  data: T[];
  page: { next_cursor: string | null; has_more: boolean; limit: number };
}

export interface TeamOut {
  id: number;
  name: string;
  color_hex: string | null;
}

export interface CornerMetrics {
  v_min: number;
  exit_speed: number;
  throttle_dist_frac: number;
  brake_point_frac: number;
}

export interface SingleCorner {
  corner_number: number;
  corner_class: "slow" | "medium" | "high";
  apex_fraction: number;
  ref_speed_kmh: number;
  team_a: CornerMetrics;
  team_b: CornerMetrics;
}

export interface ClassSummary {
  corner_count: number;
  team_a: CornerMetrics;
  team_b: CornerMetrics;
}

export interface CornerPerformancePayload {
  session_id: number;
  team_a: { id: number; name: string; color_hex: string };
  team_b: { id: number; name: string; color_hex: string };
  corners: SingleCorner[];
  summary: Record<string, ClassSummary>;
  v_min_figure: string;
  class_summary_figure: string;
  track_map_figure: string;
}

export interface GpScheduleItem {
  name: string;
  date: string;
  round: number;
}

// Critic evaluation shape nested inside StyleAnalystResponse.critique
export type Critique = {
  confidence?: number;
  factual_errors?: string[];
  suggested_improvements?: string[];
  parse_note?: string;
};

export class ClientApiError extends Error {
  constructor(public status: number, public detail?: string) {
    super(`[${status}] ${detail ?? "request failed"}`);
  }
}

export async function clientFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${env.NEXT_PUBLIC_API_BASE}${path}`, {
    ...init,
    credentials: "include",
    headers: {
      Accept: "application/json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    let detail: string | undefined;
    try {
      const body = await res.json();
      detail = body?.detail ?? body?.title;
    } catch {
      /* not JSON */
    }
    throw new ClientApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

// ── AI endpoints ───────────────────────────────────────────────────────────

export const apiClient = {
  styleAnalyst: (driverId: number, season: number) =>
    clientFetch<StyleAnalystResponse>("/api/v1/ai/style-analyst", {
      method: "POST",
      body: JSON.stringify({ driver_id: driverId, season }),
    }),

  dnaMatch: (driverId: number, season: number) =>
    clientFetch<DNAMatchResponse>("/api/v1/ai/dna-match", {
      method: "POST",
      body: JSON.stringify({ driver_id: driverId, season }),
    }),

  reportCard: (driverId: number, season: number) =>
    clientFetch<ReportCardResponse>("/api/v1/ai/report-card", {
      method: "POST",
      body: JSON.stringify({ driver_id: driverId, season }),
    }),

  events: (year: number) =>
    clientFetch<Page<EventOut>>(`/api/v1/seasons/${year}/events?limit=50`),

  gpSchedule: (year: number) =>
    clientFetch<GpScheduleItem[]>(`/api/v1/pipeline/gp-schedule?year=${year}`),

  sessions: (eventId: number) =>
    clientFetch<SessionOut[]>(`/api/v1/events/${eventId}/sessions`),

  compare: (sessionId: number, driverA: number, driverB: number, channel: string) =>
    clientFetch<ComparePayload>(
      `/api/v1/sessions/${sessionId}/compare?driver_a=${driverA}&driver_b=${driverB}&channel=${channel}`,
    ),

  sectorTimes: (sessionId: number, driverA: number, driverB: number) =>
    clientFetch<SectorTimesPayload>(
      `/api/v1/sessions/${sessionId}/compare/sectors?driver_a=${driverA}&driver_b=${driverB}`,
    ),

  trackMap: (sessionId: number, driverA: number, driverB: number) =>
    clientFetch<TrackMapPayload>(
      `/api/v1/sessions/${sessionId}/compare/track-map?driver_a=${driverA}&driver_b=${driverB}`,
    ),

  telemetryStatus: (sessionId: number) =>
    clientFetch<{ session_id: number; fetched_at: string | null }>(
      `/api/v1/pipeline/telemetry-status?session_id=${sessionId}`,
    ),

  sessionTeams: (sessionId: number) =>
    clientFetch<TeamOut[]>(`/api/v1/sessions/${sessionId}/teams`),

  cornerPerformance: (sessionId: number, teamA: number, teamB: number) =>
    clientFetch<CornerPerformancePayload>(
      `/api/v1/sessions/${sessionId}/corner-performance?team_a=${teamA}&team_b=${teamB}`,
    ),
};
