/**
 * Pure navigation helpers for the Race Dashboard session picker.
 *
 * Kept free of React/DOM/fetch so the redirect-vs-clear decision logic can be
 * unit-tested directly. The stored "last session" lives in localStorage and may
 * point at a session ID that no longer exists (e.g. after a DB re-hydrate that
 * reassigns IDs) — these helpers let callers validate and fall back gracefully
 * instead of redirecting into a 404.
 */

// localStorage key holding the last-viewed session selection.
export const LAST_SESSION_KEY = "dna_last_session";

// Picker route with the dropdown-restore flag — the graceful fallback target.
export const RACE_PICKER_PATH = "/race?pick=1";

export interface LastSession {
  year: number;
  dbEventId: number;
  sessionId: number;
}

/**
 * Parse and shape-guard the raw localStorage value. Returns null for empty,
 * malformed, or structurally-incomplete input.
 */
export function parseStoredSession(raw: string | null): LastSession | null {
  if (!raw) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }
  if (typeof parsed !== "object" || parsed === null) return null;
  const { year, dbEventId, sessionId } = parsed as Record<string, unknown>;
  if (
    typeof year !== "number" ||
    typeof dbEventId !== "number" ||
    typeof sessionId !== "number"
  ) {
    return null;
  }
  return { year, dbEventId, sessionId };
}

export type StoredSessionAction =
  | { kind: "none" }
  | { kind: "clearStorage" }
  | { kind: "redirect"; path: string }
  | { kind: "restore" };

/**
 * Decide what to do on picker mount given the stored session, whether the user
 * explicitly requested the picker (?pick=1), and whether the stored session
 * still exists in the backend.
 *
 * - no stored session            → do nothing
 * - stored but session gone       → clear the stale key, stay on the picker
 * - stored, valid, not picking    → auto-redirect to the session dashboard
 * - stored, valid, picking        → restore the dropdowns to that selection
 */
export function decideStoredSessionAction(args: {
  stored: LastSession | null;
  forcePicker: boolean;
  sessionExists: boolean;
}): StoredSessionAction {
  const { stored, forcePicker, sessionExists } = args;
  if (stored == null) return { kind: "none" };
  if (!sessionExists) return { kind: "clearStorage" };
  if (!forcePicker) return { kind: "redirect", path: `/race/${stored.sessionId}` };
  return { kind: "restore" };
}
