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

interface Page<T> {
  data: T[];
  page: { next_cursor: string | null; has_more: boolean; limit: number };
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

  sessions: (eventId: number) =>
    clientFetch<SessionOut[]>(`/api/v1/events/${eventId}/sessions`),

  compare: (sessionId: number, driverA: number, driverB: number, channel: string) =>
    clientFetch<ComparePayload>(
      `/api/v1/sessions/${sessionId}/compare?driver_a=${driverA}&driver_b=${driverB}&channel=${channel}`,
    ),
};
