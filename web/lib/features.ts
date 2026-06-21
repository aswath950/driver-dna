/**
 * Operator feature-flag registry — the single source of truth for which
 * top-level features (nav tabs + routes) are enabled.
 *
 * Toggling is a build-time operator kill-switch via NEXT_PUBLIC_DISABLED_FEATURES,
 * a comma-separated denylist of feature keys (e.g. "mystery-driver,pipeline").
 * Unset = every feature enabled. The NEXT_PUBLIC_ value is inlined at build time,
 * so changes require a rebuild / redeploy.
 *
 * This module is pure (no Node-only deps) so it is safe to import from both a
 * client component (TopNav) and edge middleware (web/middleware.ts).
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

// Parse inside the function (not at module top-level) so vitest's vi.stubEnv
// works and the read survives NEXT_PUBLIC_ inlining.
function disabledKeys(): Set<string> {
  return new Set(
    (process.env.NEXT_PUBLIC_DISABLED_FEATURES ?? "")
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean),
  );
}

export function isFeatureEnabled(key: FeatureKey): boolean {
  return !disabledKeys().has(key);
}

export function enabledFeatures(): FeatureDef[] {
  return FEATURES.filter((f) => isFeatureEnabled(f.key));
}

export function firstEnabledFeature(): FeatureDef | null {
  return enabledFeatures()[0] ?? null;
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
