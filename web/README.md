# Driver DNA — Web (Next.js 14)

App Router, RSC-first. Communicates with the FastAPI backend at `/api/v1/*`
and GraphQL at `/graphql` (added in Phase 8).

## Setup

```bash
cd web
pnpm install                     # or `npm install`
cp .env.example .env.local
pnpm dev
```

Open <http://localhost:3000>.

The landing page is a server component that fetches the backend's `/healthz`.
If the backend isn't running, you'll see an error string — start the backend
first (see `backend/README.md`).

## Feature toggles

Each top-level feature has its own server-side boolean env var — an operator
switch. Set a feature's var to `FALSE` in `.env.local` (or your Vercel project) to
**hide it from the nav and block its routes**. A feature is shown unless its var is
`FALSE` (unset / `TRUE` / anything else = shown), so a missing var never blanks a
section.

```bash
# .env.local — hide Mystery Driver and Pipeline
FEATURE_MYSTERY_DRIVER=FALSE
FEATURE_PIPELINE=FALSE
```

| Env var | Tab |
|---|---|
| `FEATURE_RADAR` | Driver Radar |
| `FEATURE_MYSTERY_DRIVER` | Mystery Driver |
| `FEATURE_RACE` | Race Dashboard |
| `FEATURE_PIPELINE` | Pipeline |

Notes:

- The value is **server-side and read at runtime** (no `NEXT_PUBLIC_` prefix), so
  changing it takes effect on the next deploy/boot — no rebuild. Set it in
  `.env.local` (not `.env.example`, which is only a template Next.js never loads)
  and restart the dev server, since env changes are not hot-reloaded.
- Visiting a disabled feature's URL directly redirects to the first enabled
  feature instead of serving the page. Setting `FEATURE_RACE=FALSE` also redirects
  the `/` landing (it belongs to the Race Dashboard) to the first enabled feature.
- How it flows through the code:
  - [`lib/features.ts`](lib/features.ts) — pure registry + helpers (`enabledFeatures`,
    `firstEnabledFeature`, `featureForPath`); it never reads `process.env`.
  - [`lib/env.ts`](lib/env.ts) — `readDisabledFeatures()` does the runtime,
    server-only read of each `FEATURE_*` var (static access so the values are
    available in the Edge/middleware runtime).
  - [`app/layout.tsx`](app/layout.tsx) — a `force-dynamic` Server Component that
    computes the enabled list per request and passes it as a prop down to the nav,
    so the client never touches the env var.
  - [`components/TopNav.tsx`](components/TopNav.tsx) — renders the nav from that prop.
  - [`middleware.ts`](middleware.ts) — reads the same vars per request to enforce the
    route redirects.

## Real pages

Added in Phase 11:

- `/` — landing (recent races)
- `/radar/[sessionId]` — Driver Radar
- `/mystery-driver` — Mystery Driver (ML + XAI)
- `/race/[sessionId]` — Race Dashboard (charts + chat)
- `/pipeline` — link to legacy Streamlit
