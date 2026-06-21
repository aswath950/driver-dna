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

The four top-level features can be switched off per environment with a single
build-time env var — an operator kill-switch. Set `NEXT_PUBLIC_DISABLED_FEATURES`
in `.env.local` (or your Vercel project) to a comma-separated list of feature
keys to **hide them from the nav and block their routes**. Unset means every
feature is enabled.

```bash
# .env.local — hide Mystery Driver and Pipeline
NEXT_PUBLIC_DISABLED_FEATURES=mystery-driver,pipeline
```

| Key | Tab |
|---|---|
| `radar` | Driver Radar |
| `mystery-driver` | Mystery Driver |
| `race` | Race Dashboard |
| `pipeline` | Pipeline |

Notes:

- The value is read at **build time**, so restart the dev server (or redeploy)
  after changing it.
- Visiting a disabled feature's URL directly redirects to the first enabled
  feature instead of serving the page. Disabling `race` also redirects the `/`
  landing (it belongs to the Race Dashboard) to the first enabled feature.
- The registry that defines features and enforcement lives in
  [`lib/features.ts`](lib/features.ts) (nav: [`components/TopNav.tsx`](components/TopNav.tsx);
  route guard: [`middleware.ts`](middleware.ts)).

## Real pages

Added in Phase 11:

- `/` — landing (recent races)
- `/radar/[sessionId]` — Driver Radar
- `/mystery-driver` — Mystery Driver (ML + XAI)
- `/race/[sessionId]` — Race Dashboard (charts + chat)
- `/pipeline` — link to legacy Streamlit
