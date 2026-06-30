/**
 * Operator feature-flag registry — the single source of truth for which
 * top-level features (nav tabs + routes) are enabled.
 *
 * Toggling is a runtime operator switch via one server-only boolean env var per
 * feature (e.g. FEATURE_PIPELINE=FALSE). A feature is enabled unless its var is
 * explicitly FALSE — unset/TRUE/anything-else = enabled, so a missing var never
 * hides a section. The values are read on the server at request time (see
 * readDisabledFeatures in lib/env.ts), so changing one takes effect on the next
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
  /** Server-only env var (no NEXT_PUBLIC_ prefix) controlling this feature. */
  envVar: string;
}

// Order = nav order. Also defines the "first enabled feature" used for redirects.
export const FEATURES: FeatureDef[] = [
  { key: "radar", label: "Driver Radar", href: "/radar", envVar: "FEATURE_RADAR" },
  { key: "mystery-driver", label: "Mystery Driver", href: "/mystery-driver", envVar: "FEATURE_MYSTERY_DRIVER" },
  { key: "race", label: "Race Dashboard", href: "/race", envVar: "FEATURE_RACE" },
  { key: "pipeline", label: "Pipeline", href: "/pipeline", envVar: "FEATURE_PIPELINE" },
];

/**
 * Resolve the set of DISABLED feature keys from per-feature flag values.
 * A feature is disabled only when its value is exactly "FALSE" (case-insensitive,
 * trimmed); unset / "TRUE" / anything else leaves it enabled. Pure — the caller
 * supplies the raw values (read from env on the server).
 */
export function disabledFromFlags(
  flags: Partial<Record<FeatureKey, string | undefined>>,
): Set<string> {
  const disabled = new Set<string>();
  for (const f of FEATURES) {
    if ((flags[f.key] ?? "").trim().toUpperCase() === "FALSE") disabled.add(f.key);
  }
  return disabled;
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
