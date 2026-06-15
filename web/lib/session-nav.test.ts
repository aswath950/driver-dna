import { describe, it, expect } from "vitest";
import {
  parseStoredSession,
  decideStoredSessionAction,
  type LastSession,
} from "./session-nav";

describe("parseStoredSession", () => {
  it("parses a well-formed stored session", () => {
    const raw = JSON.stringify({ year: 2024, dbEventId: 1, sessionId: 42 });
    expect(parseStoredSession(raw)).toEqual({
      year: 2024,
      dbEventId: 1,
      sessionId: 42,
    });
  });

  it("returns null for empty / missing input", () => {
    expect(parseStoredSession(null)).toBeNull();
    expect(parseStoredSession("")).toBeNull();
  });

  it("returns null for malformed JSON", () => {
    expect(parseStoredSession("{not json")).toBeNull();
  });

  it("returns null when a required field is missing", () => {
    const raw = JSON.stringify({ year: 2024, dbEventId: 1 }); // no sessionId
    expect(parseStoredSession(raw)).toBeNull();
  });

  it("returns null when a field is the wrong type", () => {
    const raw = JSON.stringify({ year: 2024, dbEventId: 1, sessionId: "42" });
    expect(parseStoredSession(raw)).toBeNull();
  });

  it("returns null for a non-object payload", () => {
    expect(parseStoredSession(JSON.stringify(42))).toBeNull();
    expect(parseStoredSession(JSON.stringify(null))).toBeNull();
  });
});

describe("decideStoredSessionAction", () => {
  const stored: LastSession = { year: 2024, dbEventId: 1, sessionId: 42 };

  it("does nothing when there is no stored session", () => {
    expect(
      decideStoredSessionAction({ stored: null, forcePicker: false, sessionExists: false }),
    ).toEqual({ kind: "none" });
  });

  // The key not-found behavior: a stale stored session must clear the key and
  // stay on the picker, never redirect into a 404 — regardless of pick mode.
  it("clears storage when the stored session no longer exists", () => {
    expect(
      decideStoredSessionAction({ stored, forcePicker: false, sessionExists: false }),
    ).toEqual({ kind: "clearStorage" });
    expect(
      decideStoredSessionAction({ stored, forcePicker: true, sessionExists: false }),
    ).toEqual({ kind: "clearStorage" });
  });

  it("redirects to the session dashboard when valid and not in pick mode", () => {
    expect(
      decideStoredSessionAction({ stored, forcePicker: false, sessionExists: true }),
    ).toEqual({ kind: "redirect", path: "/race/42" });
  });

  it("restores the picker dropdowns when valid and in pick mode", () => {
    expect(
      decideStoredSessionAction({ stored, forcePicker: true, sessionExists: true }),
    ).toEqual({ kind: "restore" });
  });
});
