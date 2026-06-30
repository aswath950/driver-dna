/**
 * Centralised env access.
 *
 * - `API_BASE_INTERNAL` is what React Server Components use when fetching
 *   from the backend. In prod (Vercel + Railway) it's the Railway public URL.
 * - `NEXT_PUBLIC_API_BASE` is what client-side code uses (e.g. SSE).
 */

import { disabledFromFlags } from "./features";

const trim = (v: string | undefined) => (v ?? "").replace(/\/+$/, "");

export const env = {
  API_BASE_INTERNAL: trim(process.env.API_BASE_INTERNAL) || "http://localhost:8000",
  NEXT_PUBLIC_API_BASE: trim(process.env.NEXT_PUBLIC_API_BASE) || "http://localhost:8000",
  NEXT_PUBLIC_GRAPHQL_URL:
    trim(process.env.NEXT_PUBLIC_GRAPHQL_URL) || "http://localhost:8000/graphql",
};

/**
 * Server-only, runtime read of the operator feature switches.
 *
 * Each feature has its own boolean env var (FEATURE_RADAR, FEATURE_PIPELINE, …),
 * deliberately WITHOUT a `NEXT_PUBLIC_` prefix: that prefix would inline the value
 * into the client bundle at build time, defeating runtime config. Read via direct
 * static property access so the values reflect the current process env at call time
 * and stay valid in the Edge/middleware runtime. Only call this from server code
 * (middleware, Server Components); in the browser the vars are undefined → every
 * feature enabled. A feature is disabled only when its var is FALSE.
 */
export function readDisabledFeatures(): Set<string> {
  // Static property access (not process.env[var]) so Next.js exposes each value in
  // the Edge/middleware runtime; a dynamic key can read as undefined there and would
  // silently leave a disabled route reachable. Keys mirror FEATURES[].envVar.
  return disabledFromFlags({
    radar: process.env.FEATURE_RADAR,
    "mystery-driver": process.env.FEATURE_MYSTERY_DRIVER,
    race: process.env.FEATURE_RACE,
    pipeline: process.env.FEATURE_PIPELINE,
  });
}
