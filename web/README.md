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

## Real pages

Added in Phase 11:

- `/` — landing (recent races)
- `/radar/[sessionId]` — Driver Radar
- `/mystery-driver` — Mystery Driver (ML + XAI)
- `/race/[sessionId]` — Race Dashboard (charts + chat)
- `/pipeline` — link to legacy Streamlit
