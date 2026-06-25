/**
 * Centralised env access.
 *
 * - `API_BASE_INTERNAL` is what React Server Components use when fetching
 *   from the backend. In prod (Vercel + Railway) it's the Railway public URL.
 * - `NEXT_PUBLIC_API_BASE` is what client-side code uses (e.g. SSE).
 */

import { parseDisabledFeatures } from "./features";

const trim = (v: string | undefined) => (v ?? "").replace(/\/+$/, "");

export const env = {
  API_BASE_INTERNAL: trim(process.env.API_BASE_INTERNAL) || "http://localhost:8000",
  NEXT_PUBLIC_API_BASE: trim(process.env.NEXT_PUBLIC_API_BASE) || "http://localhost:8000",
  NEXT_PUBLIC_GRAPHQL_URL:
    trim(process.env.NEXT_PUBLIC_GRAPHQL_URL) || "http://localhost:8000/graphql",
};

/**
 * Server-only, runtime read of the operator feature kill-switch.
 *
 * `DISABLED_FEATURES` deliberately has NO `NEXT_PUBLIC_` prefix: that prefix would
 * inline the value into the client bundle at build time, defeating runtime config.
 * Read as a function (not a frozen const) and via a direct static property access
 * so the value reflects the current process env at call time and stays valid in the
 * Edge/middleware runtime. Only call this from server code (middleware, Server
 * Components); in the browser the var is undefined → every feature enabled.
 */
export function readDisabledFeatures(): Set<string> {
  return parseDisabledFeatures(process.env.DISABLED_FEATURES);
}
