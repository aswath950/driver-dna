/**
 * Operator feature-flag registry — the single source of truth for which
 * top-level features (nav tabs + routes) are enabled.
 *
 * Toggling is a runtime operator kill-switch via the server-only DISABLED_FEATURES
 * env var, a comma-separated denylist of feature keys (e.g. "mystery-driver,pipeline").
 * Unset = every feature enabled. The value is read on the server at request time
 * (see readDisabledFeatures in lib/env.ts), so changing it takes effect on the next
 * deploy/boot with NO rebuild.
 *
 * This module is PURE: it never reads process.env. The disabled set is passed in as
 * an argument so the same helpers are safe to use from a client component (TopNav,
 * via a server-provided prop) and from edge middleware (web/middleware.ts).
 */

export type FeatureKey = "radar" | "mystery-driver" | "race" | "pipeline";

export interface FeatureDef {
  key: FeatureKey;
  label: string;
  href: string;
}

// Order = nav order. Also defines the "first enabled feature" used for redirects.
export const FEATURES: FeatureDef[] = [
  { key: "radar", label: "Driver Radar", href: "/radar" },
  { key: "mystery-driver", label: "Mystery Driver", href: "/mystery-driver" },
  { key: "race", label: "Race Dashboard", href: "/race" },
  { key: "pipeline", label: "Pipeline", href: "/pipeline" },
];

/**
 * Parse the raw DISABLED_FEATURES string (comma-separated keys) into a set.
 * Tolerates surrounding whitespace and empty entries; undefined/"" → empty set.
 * Pure — the caller supplies the raw value (read from env on the server).
 */
export function parseDisabledFeatures(raw: string | undefined): Set<string> {
  return new Set(
    (raw ?? "")
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean),
  );
}

export function isFeatureEnabled(key: FeatureKey, disabled: Set<string>): boolean {
  return !disabled.has(key);
}

export function enabledFeatures(disabled: Set<string>): FeatureDef[] {
  return FEATURES.filter((f) => isFeatureEnabled(f.key, disabled));
}

export function firstEnabledFeature(disabled: Set<string>): FeatureDef | null {
  return enabledFeatures(disabled)[0] ?? null;
}

/**
 * Map a request path to the feature that owns it.
 * Race Dashboard owns /race, /race/*, /event/*, AND the "/" landing
 * (the landing lists events that link into the race flow — web/app/page.tsx).
 * Unknown paths return null so middleware leaves them alone.
 */
export function featureForPath(pathname: string): FeatureKey | null {
  if (pathname === "/") return "race";
  if (pathname === "/radar" || pathname.startsWith("/radar/")) return "radar";
  if (pathname === "/mystery-driver" || pathname.startsWith("/mystery-driver/"))
    return "mystery-driver";
  if (pathname === "/pipeline" || pathname.startsWith("/pipeline/"))
    return "pipeline";
  if (pathname === "/race" || pathname.startsWith("/race/")) return "race";
  if (pathname === "/event" || pathname.startsWith("/event/")) return "race";
  return null;
}
