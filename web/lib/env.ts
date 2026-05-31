/**
 * Centralised env access.
 *
 * - `API_BASE_INTERNAL` is what React Server Components use when fetching
 *   from the backend. In prod (Vercel + Railway) it's the Railway public URL.
 * - `NEXT_PUBLIC_API_BASE` is what client-side code uses (e.g. SSE).
 */

const trim = (v: string | undefined) => (v ?? "").replace(/\/+$/, "");

export const env = {
  API_BASE_INTERNAL: trim(process.env.API_BASE_INTERNAL) || "http://localhost:8000",
  NEXT_PUBLIC_API_BASE: trim(process.env.NEXT_PUBLIC_API_BASE) || "http://localhost:8000",
  NEXT_PUBLIC_GRAPHQL_URL:
    trim(process.env.NEXT_PUBLIC_GRAPHQL_URL) || "http://localhost:8000/graphql",
};
