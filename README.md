# Driver DNA — F1 Telemetry Platform

> A full-stack F1 analytics platform — ML driver identification, five distinct
> agentic-AI features, REST + GraphQL backend over normalised Postgres,
> Next.js 14 client, and a legacy Streamlit dashboard. Built as a
> generalist-engineer portfolio piece.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![GraphQL](https://img.shields.io/badge/Strawberry-GraphQL-FF66B3?logo=graphql&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14-000?logo=next.js&logoColor=white)
![Postgres](https://img.shields.io/badge/Postgres-16-336791?logo=postgresql&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o--mini-412991?logo=openai&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)
![CI](https://img.shields.io/badge/GitHub_Actions-3_workflows-2088FF?logo=githubactions&logoColor=white)

**What's in the box**

1. **AI / ML** — XGBoost driver classifier, SHAP explainability, five distinct
   agentic-LLM patterns (Reflexion, RAG, structured output, single-shot XAI
   narration, SSE-streamed ReAct chat).
2. **Backend** — Postgres + FastAPI REST (`/api/v1` with RFC 7807 errors,
   opaque cursor pagination, OpenAPI 3.1) co-hosted with Strawberry GraphQL
   (`/graphql` with N+1-killing DataLoaders).
3. **Frontend** — Next.js 14 App Router web client (RSC, Plotly, SSE chat)
   **plus** the legacy Streamlit dashboard — both served by the same `src/`
   Python library.

---

## At a glance

- **5** alembic migrations · **13** application tables · **17** indexes (9
  query-path + 8 unique-constraint) — with measured query plans in
  [backend/docs/query_plans.md](backend/docs/query_plans.md)
- **23** production REST endpoints under `/api/v1`
  — 9 reads · 5 analytics · 5 AI · 4 me/session
- **GraphQL** schema with 12 query fields + 9 object types + 2 enums + per-request DataLoaders
- **5 agentic-AI patterns** — Reflexion · RAG · Structured Output ·
  Single-shot XAI · SSE-streamed ReAct
- **282 tests** — 119 backend (modern stack) + 163 legacy (`src/` + Streamlit)
- **3 GitHub Actions** workflows (legacy CI · backend CI · web CI)
- **4 documented deploy paths** — local dev · Vercel+Railway · self-host VPS · Streamlit Cloud

**Live demo:** _(set to your Vercel URL after deploy)_

**Quick links:** [Architecture](#architecture) · [Feature tour](#feature-tour) ·
[Get Started](#get-started) · [Deployment](#deployment) · [Testing](#testing)

---

## Architecture

Three views: system, request lifecycle, and entity-relationship.

### System diagram

```mermaid
flowchart LR
    User([Browser]) -->|HTTPS| Vercel["Next.js 14<br/>App Router · RSC + Plotly"]
    Vercel -->|REST /api/v1<br/>GraphQL /graphql<br/>SSE chat stream| Railway["FastAPI<br/>+ Strawberry GraphQL"]
    Railway -->|asyncpg| PG[("Postgres 16<br/>13 tables · 17 indexes")]
    Railway -->|OpenAI SDK| OAI[OpenAI<br/>gpt-4o-mini]
    Railway -->|requests| OF1[OpenF1 REST]

    Streamlit["Streamlit<br/>legacy dashboard"] -.->|imports| Src[/"src/ — shared Python lib<br/>race_engine · viz · llm_layer · openf1"/]
    Railway -.->|imports| Src
    MCP["MCP server<br/>(stdio · 4 tools)"] -.->|imports| Src

    classDef new fill:#e8002d,stroke:#fff,color:#fff
    class Vercel,Railway,PG new
```

### Request lifecycle — one page load

```mermaid
sequenceDiagram
    autonumber
    participant B as Browser
    participant N as Next.js (Vercel)
    participant F as FastAPI (Railway)
    participant P as Postgres
    B->>N: GET /race/1
    N->>F: GET /api/v1/sessions/1/results<br/>(forwards dna_sid cookie)
    F->>F: UserSessionMiddleware<br/>touch last_seen_at
    F->>P: SELECT JOIN drivers JOIN teams<br/>(uses ix_race_results_session_pos)
    P-->>F: 20 rows
    F-->>N: 200 application/json<br/>+ X-Request-ID + API-Version: 1
    N-->>B: SSR HTML — leaderboard inlined
```

### Entity-relationship (high-level)

```mermaid
erDiagram
    SEASONS ||--o{ EVENTS : has
    CIRCUITS ||--o{ EVENTS : hosts
    EVENTS  ||--o{ SESSIONS : has
    SESSIONS ||--o{ SESSION_DRIVERS : roster
    SESSIONS ||--o{ RACE_RESULTS : produces
    SESSIONS ||--o{ LAP_TIMES : produces
    DRIVERS ||--o{ SESSION_DRIVERS : entered
    TEAMS   ||--o{ SESSION_DRIVERS : fielded
    DRIVERS ||--o{ DRIVER_STATS : aggregates
    SEASONS ||--o{ DRIVER_STATS : aggregates
    USER_SESSIONS ||--o{ SAVED_ANALYSES : owns
    USER_SESSIONS ||--o{ LLM_AUDIT : attributed
```

### Why this shape

**Why a shared `src/` library.** All three Python frontends — Streamlit, the
FastAPI backend, and the MCP server — consume the same race-engine /
OpenF1 / LLM modules. One bug fix benefits every frontend; no copy-paste.
The shared lib pre-dates the backend and was kept intentionally
import-stable, with `sys.path` wiring centralised in
[backend/app/__init__.py](backend/app/__init__.py).

**Why REST and GraphQL.** REST is the contract for Next.js RSC fetches,
the OpenAPI-generated TypeScript client, the MCP server, and any future
integrator. GraphQL is for graph-shaped queries (event → sessions →
results → drivers in one round-trip) where DataLoaders guarantee a
constant number of SQL statements. Both routes reuse the same Phase 6
repositories — zero duplicate SQL.

**Why cursor pagination over offset.** Stable order across pages, no
`OFFSET 1000000` slowdowns. Opaque base64url cursor encodes
`{"k": <sort_key>, "id": <pk>}` so clients never construct cursors —
they only echo what the server returned. See
[backend/app/core/pagination.py](backend/app/core/pagination.py).

---

## Tech stack

| Layer | Tools | Why |
|---|---|---|
| **Frontend** | Next.js 14 App Router · React 18 · Tailwind · Plotly.js · TypeScript strict | RSC for SEO + low-JS landing pages; Plotly for visual parity with Streamlit; types generated from `/openapi.json` |
| **Backend** | FastAPI 0.115 · Strawberry GraphQL · SQLAlchemy 2.0 async · Alembic · Pydantic v2 · structlog | Async stack throughout; OpenAPI 3.1 out-of-the-box; one factory in [app/main.py](backend/app/main.py) wires every middleware |
| **Data** | Postgres 16 · asyncpg / psycopg · pgcrypto | Normalised 3NF schema; UUID via `gen_random_uuid()`; JSONB for saved-analysis payloads |
| **AI / ML** | OpenAI (gpt-4o-mini) · XGBoost · SHAP · scikit-learn · pandas · numpy | Reused legacy ML stack; backend wraps a single sync OpenAI helper in [app/llm/openai_client.py](backend/app/llm/openai_client.py) |
| **External** | OpenF1 REST · FastF1 cache · Google Drive API v3 (OAuth 2.0) | Telemetry sources; legacy artifact persistence |
| **Ops** | Docker · docker-compose · Railway (backend + Postgres) · Vercel (web) · GitHub Actions · pytest 8 · ruff · mypy | Two free-tier cloud services + GH-hosted CI |

---

## Repository map

Top-level layout — each leaf has a one-line "what it is".

```
driver-dna/
├── backend/                       FastAPI + Strawberry + Alembic (Phases 1–12)
│   ├── app/
│   │   ├── main.py                FastAPI factory; mounts middleware + handlers + routers
│   │   ├── core/                  config, errors (RFC 7807), pagination, middleware,
│   │   │                          sessions (cookie middleware)
│   │   ├── api/v1/                REST routers + Pydantic schemas
│   │   │   ├── routers/           seasons, events, sessions, drivers, standings,
│   │   │   │                      analytics, ai, me
│   │   │   └── schemas/           One Pydantic file per resource
│   │   ├── graphql/               Strawberry schema · resolvers · DataLoaders
│   │   ├── db/                    SQLAlchemy 2.0 models · repositories (one file per aggregate)
│   │   ├── etl/                   hydrate_session.py · refresh_driver_stats.py · CLI
│   │   ├── services/              DB → DataFrame adapters for src/race_engine
│   │   └── llm/                   5 feature services · SSE race chat · audit
│   ├── alembic/versions/          5 migrations
│   ├── scripts/                   seed_demo.sql · hydrate_initial_data.sh
│   ├── tests/                     119 tests across 11 files
│   ├── docs/query_plans.md        EXPLAIN ANALYZE output before/after indexes
│   ├── Dockerfile · railway.json  Cloud deploy config
│   └── pyproject.toml
├── web/                           Next.js 14 App Router (Phase 11)
│   ├── app/
│   │   ├── page.tsx                       Landing — recent races (RSC)
│   │   ├── event/[eventId]/               Sessions per event (RSC)
│   │   ├── race/[sessionId]/              Leaderboard + tyre deg + SSE chat
│   │   ├── radar/[sessionId]/             Driver compare (Plotly client island)
│   │   ├── mystery-driver/                XAI explainer (client)
│   │   └── pipeline/                      Link to legacy Streamlit
│   ├── components/                charts/PlotlyChart · chat/RaceChatStream · ui/*
│   ├── lib/                       api (RSC + cookies) · api-client (browser) · env
│   └── package.json
├── src/                           SHARED Python library — UNCHANGED, reused by all
│   ├── race_engine.py             rolling pace · gap-to-leader · undercut · tyre deg
│   ├── llm_layer.py               Streamlit-era 5 agentic patterns (still in use)
│   ├── viz.py                     Plotly figure builders
│   ├── openf1.py                  OpenF1 sync client (used by ETL + MCP)
│   ├── model.py · pipeline.py     XGBoost classifier + telemetry features
│   └── drive_sync.py              Google Drive OAuth/upload (Streamlit only)
├── streamlit_app.py               Legacy dashboard entry point
├── mcp/server.py                  MCP server — 4 stdio tools over fastest-lap data
├── docker-compose.yml             Postgres + backend (web runs separately)
├── Makefile                       make {db,backend,web,migrate,hydrate,dev,...}
└── .github/workflows/
    ├── ci.yml                     legacy: src/ + Streamlit (ruff · mypy · pytest · eval · docker)
    ├── backend-ci.yml             backend: Postgres svc · alembic · ruff · pytest · OpenAPI snapshot
    └── web-ci.yml                 web: npm ci · tsc · next lint · next build
```

---

## Feature tour

The seven things most worth pointing at.

### 1. REST API — `/api/v1`

Every endpoint inherits four contracts:

- **Versioning** — mounted under `/api/v1`; every response carries
  `API-Version: 1`. Future v2 mounts at `/api/v2`; no breaking changes
  inside a major.
- **RFC 7807 error envelope** — `application/problem+json` with `type`,
  `title`, `status`, `detail`, `instance`, and a `request_id` extension.
- **Cursor pagination** — opaque base64url cursor; default `limit=50`,
  max 200. List endpoints return `{ data: [...], page: { next_cursor,
  has_more, limit } }`.
- **Request tracing** — `X-Request-ID` is minted (or echoed) on every
  request and bound to a structlog contextvar.

23 endpoints in production (4 sentinel `_ping*` routes excluded):

| Area | Method | Path | Returns |
|---|---|---|---|
| **Reads (9)** | GET | `/seasons` | `Page[SeasonOut]` |
| | GET | `/seasons/{year}/events` | `Page[EventOut]` |
| | GET | `/events/{event_id}/sessions` | `list[SessionOut]` |
| | GET | `/sessions/{session_id}` | `SessionOut` |
| | GET | `/sessions/{session_id}/results` | `list[RaceResultOut]` (leaderboard) |
| | GET | `/sessions/{session_id}/laps` | `Page[LapOut]` — filterable by driver/from-lap/to-lap |
| | GET | `/drivers` | `Page[DriverOut]` — filterable by season/team |
| | GET | `/drivers/{driver_id}/stats?season=` | `DriverStatsOut` |
| | GET | `/standings?season=` | `list[StandingRowOut]` |
| **Analytics (5)** | GET | `/sessions/{id}/analytics/rolling-pace?window=` | `list[RollingPaceRow]` |
| | GET | `/sessions/{id}/analytics/gap-to-leader` | `list[GapRow]` |
| | GET | `/sessions/{id}/analytics/undercuts` | `list[UndercutEvent]` |
| | GET | `/sessions/{id}/analytics/tyre-degradation` | `list[DegradationRow]` |
| | GET | `/sessions/{id}/compare?driver_a&driver_b&channel=` | `ComparePayload` (Plotly JSON) |
| **AI (5)** | POST | `/ai/style-analyst` | Reflexion narrative |
| | POST | `/ai/dna-match` | RAG historical match |
| | POST | `/ai/report-card` | Structured JSON report |
| | POST | `/ai/xai-explain` | SHAP narration |
| | POST | `/ai/race-chat/stream` | **SSE** stream (ReAct tool loop) |
| **Me (4)** | GET | `/me` | `UserSessionOut` (cookie-bound) |
| | POST | `/me/saved-analyses` | Persist analysis (cap=100) |
| | GET | `/me/saved-analyses` | `Page[SavedAnalysisOut]` |
| | DELETE | `/me/saved-analyses/{id}` | 204 |

Live OpenAPI 3.1 spec served at `GET /openapi.json`; Swagger UI at
`GET /docs`.

**Error envelope example:**

```bash
$ curl -i -H 'X-Request-ID: demo' http://localhost:8000/api/v1/sessions/999
HTTP/1.1 404 Not Found
api-version: 1
x-request-id: demo
content-type: application/problem+json

{"type":"https://driver-dna.dev/errors/not_found",
 "title":"Resource not found","status":404,
 "detail":"session 999 not found",
 "instance":"/api/v1/sessions/999","request_id":"demo"}
```

**Cursor pagination example** (walks all laps for driver 1 in session 1):

```bash
$ curl -s 'http://localhost:8000/api/v1/sessions/1/laps?driver_id=1&limit=3'
{"data":[
  {"id":1,"lap_number":1,"lap_time_ms":78011,"compound":"SOFT", ...},
  {"id":1441,"lap_number":2,"lap_time_ms":78014,"compound":"SOFT", ...},
  {"id":2881,"lap_number":3,"lap_time_ms":78017,"compound":"SOFT", ...}
 ],
 "page":{"next_cursor":"eyJrIjozLCJpZCI6Mjg4MX0","has_more":true,"limit":3}}
```

### 2. GraphQL — `/graphql`

Use GraphQL when the page needs a graph-shaped slice in one round trip
(e.g. one session with its leaderboard + each driver's current team).
DataLoaders batch every nested fetch — see the N+1 regression guard in
[backend/tests/graphql/test_schema.py](backend/tests/graphql/test_schema.py)
which asserts ≤ 10 SQL statements for a nested query over 20 rows.

**Query root** (full SDL excerpt):

```graphql
type Query {
  season(year: Int!): Season
  seasons(first: Int! = 20): [Season!]!
  event(id: ID!): Event
  events(seasonYear: Int!, first: Int! = 50): [Event!]!
  session(id: ID!): Session
  sessionsForEvent(eventId: ID!): [Session!]!
  sessionResults(sessionId: ID!): [RaceResult!]!
  sessionLaps(sessionId: ID!, driverId: ID, fromLap: Int, toLap: Int,
              first: Int! = 100): [Lap!]!
  driver(id: ID!): Driver
  drivers(season: Int, team: String, first: Int! = 50): [Driver!]!
  driverStats(driverId: ID!, season: Int!): DriverStats
  standings(season: Int!): [StandingRow!]!
}
```

**Example** — leaderboard with nested driver + team in one call:

```bash
$ curl -s -X POST http://localhost:8000/graphql \
   -H 'Content-Type: application/json' \
   -d '{"query":"{ sessionResults(sessionId:\"1\") { position driver { code currentTeam { name } } } }"}'
{"data":{"sessionResults":[
  {"position":1,"driver":{"code":"HUL","currentTeam":{"name":"Haas"}}},
  {"position":2,"driver":{"code":"VER","currentTeam":{"name":"Red Bull Racing"}}},
  ...
]}}
```

GraphiQL is served at `GET /graphql` when `ENV=local`.

### 3. Postgres schema

11 application tables in 3NF + 2 audit tables, 3 ENUM types, 17 indexes
(9 query-path + 8 unique-constraint), pgcrypto for UUID defaults.

**Migrations:**

| # | File | Adds |
|---|---|---|
| 0001 | [`0001_core_schema.py`](backend/alembic/versions/0001_core_schema.py) | seasons · circuits · events · sessions · drivers · teams · session_drivers · race_results · lap_times · driver_stats + 2 ENUMs |
| 0002 | [`0002_indexes.py`](backend/alembic/versions/0002_indexes.py) | 5 query indexes incl. **partial** `ix_lap_times_pit (session_id) WHERE is_pit_in OR is_pit_out` |
| 0003 | [`0003_circuits_unique_name.py`](backend/alembic/versions/0003_circuits_unique_name.py) | UNIQUE on `circuits.name` (ETL upsert target) |
| 0004 | [`0004_llm_audit.py`](backend/alembic/versions/0004_llm_audit.py) | `llm_audit` table + per-feature index |
| 0005 | [`0005_user_sessions.py`](backend/alembic/versions/0005_user_sessions.py) | `user_sessions` + `saved_analyses` + FK on `llm_audit` + `analysis_kind` ENUM |

**Indexed query-plan baseline** — captured before and after each
migration, see [backend/docs/query_plans.md](backend/docs/query_plans.md).
Excerpt for the leaderboard query:

```text
EXPLAIN ANALYZE SELECT ... FROM race_results WHERE session_id=$1 ORDER BY position;

BEFORE (seq scan):     Cost=37..38   actual=1.2 ms
AFTER  (index scan):   Cost= 8.4    actual=0.12 ms     ← 10× faster
       Index Scan using ix_race_results_session_pos
```

A pytest sweep in [backend/tests/test_query_plans.py](backend/tests/test_query_plans.py)
parses `EXPLAIN (FORMAT JSON)` and **asserts** every covered query plans
to an `Index Scan`, not a `Seq Scan` — so a future migration that
accidentally drops an index fails CI.

### 4. The five agentic-AI patterns

All five live in the legacy Streamlit dashboard **and** as REST endpoints
in the new backend. Every call is persisted to `llm_audit` with token
counts, latency, USD cost, status, and the `user_session_id` cookie of
the originating browser.

| # | Pattern | Streamlit tab | REST endpoint | Audit rows / call |
|---|---|---|---|---|
| 1 | **Reflexion** | Driver Style Analyst | `POST /api/v1/ai/style-analyst` | 2–3 (analyst → critic → revise) |
| 2 | **RAG** (no vector DB) | Historical DNA Matching | `POST /api/v1/ai/dna-match` | 1 |
| 3 | **Structured Output** | Driver DNA Report Card | `POST /api/v1/ai/report-card` | 1 (JSON-mode + schema validate) |
| 4 | **Single-shot XAI** | Mystery Driver explainer | `POST /api/v1/ai/xai-explain` | 1 |
| 5 | **ReAct + SSE stream** | Race Intelligence Chat | `POST /api/v1/ai/race-chat/stream` | up to `MAX_ROUNDS=3` |

**Pattern 1 — Reflexion** (analyst → critic JSON → conditional revise):

```
Analyst LLM (T=0.4)  →  driving-style narrative
                                ↓
Critic LLM  (T=0.0)  →  JSON {confidence: 1–10, issues: [...]}
                                ↓
            confidence ≥ 7 → accept narrative
            confidence < 7 → Analyst rewrites once with critique injected
```

**Pattern 5 — SSE-streamed ReAct race chat.** The model is given three
analytics tools (`get_rolling_pace_top`, `get_leader_gap_summary`,
`get_tyre_degradation_summary`) backed by the Phase 7 analytics service.
Tool decisions stream live, the final answer streams token-by-token:

```bash
$ curl -N -X POST localhost:8000/api/v1/ai/race-chat/stream \
   -H 'Content-Type: application/json' \
   -d '{"session_id":1,"message":"Who had the best pace?"}'

event: tool_call
data: {"tool":"get_rolling_pace_top","args":{"top_n":5}}

event: tool_result
data: {"tool":"get_rolling_pace_top","summary":"[{\"driver_id\":1,...}]"}

event: token
data: {"delta":"Verstappen averaged 78.04s rolling pace..."}

event: done
data: {"input_tokens":312,"output_tokens":146,"rounds":2}
```

The five event types are `token`, `tool_call`, `tool_result`, `done`,
`error` — see [backend/app/llm/sse.py](backend/app/llm/sse.py).

**Cost observability.** Every call writes one `llm_audit` row with
gpt-4o-mini pricing applied at insert time (USD per 1M tokens table in
[backend/app/llm/audit.py](backend/app/llm/audit.py)):

```sql
SELECT feature, COUNT(*) AS calls,
       ROUND(SUM(cost_usd)::numeric, 4) AS usd,
       ROUND(AVG(latency_ms))           AS avg_ms
FROM   llm_audit
WHERE  created_at > now() - interval '24h'
GROUP  BY feature
ORDER  BY calls DESC;
```

### 5. Web client — Next.js 14

App-Router, RSC-first. Client components only where needed (Plotly,
SSE chat). All RSC fetches forward the `dna_sid` cookie via the wrapper
in [web/lib/api.ts](web/lib/api.ts) so anonymous-session continuity
works seamlessly. TypeScript types are generated from `/openapi.json`
via `npm run openapi:gen` (`openapi-typescript` writes
[web/lib/api-types.ts](web/lib/api-types.ts) — 2.8 k lines, every
endpoint's request/response typed).

| Route | Mode | Server fetches | Client islands |
|---|---|---|---|
| `/` | RSC | `/seasons` + `/events` | none |
| `/event/[eventId]` | RSC | `/sessions` | none |
| `/race/[sessionId]` | RSC + island | `/results` + `/tyre-degradation` | `RaceChatStream` (SSE) |
| `/radar/[sessionId]` | RSC + island | `/results` | `CompareIsland` + `PlotlyChart` |
| `/mystery-driver` | client only | `POST /ai/xai-explain` | full page |
| `/pipeline` | static | — | link to Streamlit |

Plotly (~3 MB minified) is lazy-loaded via `next/dynamic` with
`ssr: false` — the landing page ships under 90 kB of JS.

### 6. ETL pipeline

One CLI, two subcommands, true-idempotent upserts inside a single
transaction. The OpenF1 client is reused verbatim from `src/openf1.py`
— no re-implementation.

```bash
# Hydrate one session
python -m app.etl hydrate --year 2024 --gp "Monaco Grand Prix" --session R

# Hydrate every session in a weekend
python -m app.etl hydrate --year 2024 --gp "Monaco Grand Prix"

# Dry-run (rollback before commit)
python -m app.etl hydrate --year 2024 --gp "Monaco Grand Prix" --dry-run

# Recompute driver_stats aggregates for a season
python -m app.etl refresh-stats --season 2024
```

Real output from a Monaco 2024 hydrate:

```json
{
  "season_id": 1,
  "event_id": 1,
  "session_ids": [1],
  "counts": {"session_drivers": 20, "lap_times": 1237, "race_results": 20}
}
```

Running the same command again is a true no-op — every write uses
`INSERT ... ON CONFLICT DO UPDATE` keyed on natural unique constraints,
and the entire job is wrapped in one transaction so partial failures
leave the DB untouched. Proven by
[`test_hydrate_is_idempotent`](backend/tests/etl/test_hydrate_session.py).

For initial prod seeding, use
[backend/scripts/hydrate_initial_data.sh](backend/scripts/hydrate_initial_data.sh)
— hydrates 5 representative race weekends + refreshes stats.

### 7. Anonymous-session layer

No auth, no sign-up. Every browser gets a UUID cookie on first hit;
saved analyses + LLM audit rows attribute back to it.

```bash
# 1) First request mints the cookie + creates a user_sessions row
$ curl -i -c jar localhost:8000/api/v1/me
HTTP/1.1 200 OK
set-cookie: dna_sid=76d2ca32-3c52-4789-8cf4-9b4175b4860f;
            HttpOnly; Max-Age=31536000; Path=/; SameSite=lax
{"id":"76d2ca32-...","created_at":"...","last_seen_at":"..."}

# 2) Save an analysis — capped at 100 per session, ownership-checked deletes
$ curl -b jar -X POST localhost:8000/api/v1/me/saved-analyses \
   -d '{"kind":"radar","payload":{"top":3}}'
{"id":"426e01a3-...","kind":"radar","payload":{"top":3}, ...}

# 3) List — paginated, newest first
$ curl -b jar localhost:8000/api/v1/me/saved-analyses
{"data":[{"id":"426e01a3-...","kind":"radar", ...}], "page":{...}}
```

Cookie is `HttpOnly`, `SameSite=Lax` locally, `SameSite=None; Secure`
in prod so cross-origin Vercel → Railway requests carry it. Logic lives
in [backend/app/core/sessions.py](backend/app/core/sessions.py).

---

## Get Started

Two paths: three terminals manually, or `make dev`.

### Prerequisites

| Tool | Version | Why |
|---|---|---|
| Python | 3.13 | Backend + legacy `src/` |
| Node | 20 | Next.js 14 |
| Docker + docker-compose | latest | Postgres |
| `psql` | 16 | Inspect DB / EXPLAIN |
| `npm` (or `pnpm`) | latest | Web deps |

One-shot verification:
`python --version && node --version && docker --version && psql --version`.

### Three-terminal local dev

```bash
# Terminal 1 — Postgres + pgAdmin (monitoring dashboard)
make db                                           # → postgres :5432, pgAdmin :5050

# Terminal 2 — Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env                              # add OPENAI_API_KEY if testing AI
alembic upgrade head
bash scripts/hydrate_initial_data.sh              # seed a few real race weekends
uvicorn app.main:app --reload --port 8000         # → http://localhost:8000

# Terminal 3 — Web
cd web
pnpm install
cp .env.example .env.local
pnpm dev                                          # → http://localhost:3000

# (Optional) Terminal 4 — legacy Streamlit
streamlit run streamlit_app.py                    # → http://localhost:8501
```

### One-command alternative — Makefile

The repo root [Makefile](Makefile) wraps the most common commands:

```bash
make dev             # full stack: postgres + pgAdmin + backend + web in one shot

# Or bring up pieces individually:
make db              # docker compose up -d postgres pgadmin (:5432 + :5050)
make backend-install # python -m venv backend/.venv && pip install -e backend[dev]
make backend         # uvicorn dev server on :8000
make backend-test    # full pytest suite
make migrate         # alembic upgrade head
make hydrate YEAR=2024 GP='Monaco Grand Prix' SESSION=R
make web-install     # pnpm install in web/
make web             # next dev on :3000
make web-typecheck   # tsc --noEmit
```

### Env vars reference

| File | Var | Default | Purpose |
|---|---|---|---|
| `backend/.env` | `DATABASE_URL` | `postgresql+asyncpg://dna:dna@localhost:5432/driver_dna` | App (asyncpg) |
| `backend/.env` | `DATABASE_URL_SYNC` | `postgresql+psycopg://dna:dna@localhost:5432/driver_dna` | Alembic (sync) |
| `backend/.env` | `OPENAI_API_KEY` | _empty_ | Required for `/api/v1/ai/*` |
| `backend/.env` | `OPENF1_BASE_URL` | `https://api.openf1.org/v1` | Rarely changed |
| `backend/.env` | `CORS_ORIGINS` | `http://localhost:3000` | Comma-sep allow-list (include `https://*.vercel.app` in prod) |
| `backend/.env` | `COOKIE_SECRET` | `dev-secret-change-me` | Future-proof; not currently signing |
| `backend/.env` | `RATE_LIMIT_PER_MIN` | `60` | Per-session AI rate limit |
| `backend/.env` | `ENV` | `local` | `local` / `preview` / `prod` (toggles cookie attrs) |
| `web/.env.local` | `NEXT_PUBLIC_API_BASE` | `http://localhost:8000` | Browser-visible API URL (SSE chat) |
| `web/.env.local` | `API_BASE_INTERNAL` | `http://localhost:8000` | Server-side RSC fetch URL |
| `web/.env.local` | `NEXT_PUBLIC_GRAPHQL_URL` | `http://localhost:8000/graphql` | GraphQL endpoint |

### Smoke-check checklist

After bring-up, in order:

```bash
# 1. Backend up?
curl -s localhost:8000/healthz                                # {"status":"ok",...}

# 2. v1 endpoint live + envelope working?
curl -s localhost:8000/api/v1/seasons | jq '.data | length'   # > 0

# 3. GraphQL alive?
curl -s -X POST localhost:8000/graphql \
  -d '{"query":"{ seasons(first:3) { year } }"}'              # 3 years back

# 4. Standings populated?
curl -s 'localhost:8000/api/v1/standings?season=2024' | jq '.[0].driver.code'

# 5. Cookie issued?
curl -is localhost:8000/api/v1/me | grep -i set-cookie        # dna_sid=...
```

Then in the browser:

- <http://localhost:3000> — landing should list event cards
- <http://localhost:3000/race/1> — leaderboard renders + chat input visible
- <http://localhost:8000/docs> — Swagger UI for every endpoint
- <http://localhost:8000/graphql> — GraphiQL (local only)
- <http://localhost:5050> — pgAdmin (login: `admin@example.com` / `admin`; add server with host `postgres`, port `5432`, user/pass `dna`)

---

## Deployment

Four paths. Pick by cost + setup time.

| Path | Cost | Setup | Best for |
|---|---|---|---|
| **Local dev** | $0 | 10 min | Hacking |
| **Vercel + Railway** ⭐ | $0–$5/mo | 30 min | Portfolio demo |
| **Self-host VPS** | $5–$10/mo | 60 min | Full ownership |
| **Streamlit Cloud** | $0 | 5 min | Legacy UI only |

### Path A — Vercel (web) + Railway (backend + Postgres) ⭐

The recommended portfolio setup. Free tiers cover small demo traffic.

#### Railway — backend + Postgres

1. **Create project + Postgres plugin.**
   - New Railway project → "+ New" → **Postgres**.
   - Railway auto-injects `DATABASE_URL` into other services in the project.

2. **Deploy the backend service.**
   - "+ New" → **GitHub Repo** → pick this repo.
   - Railway reads [backend/railway.json](backend/railway.json), which
     points at [backend/Dockerfile](backend/Dockerfile) and runs
     `alembic upgrade head` as the release command before starting
     uvicorn.

3. **Service env vars** (paste into the Railway dashboard):

   | Var | Value |
   |---|---|
   | `DATABASE_URL` | `${{Postgres.DATABASE_URL}}` (Railway reference, **asyncpg** scheme) — append `?sslmode=require` if needed; replace `postgresql://` with `postgresql+asyncpg://` |
   | `DATABASE_URL_SYNC` | Same DB, sync URL: `postgresql+psycopg://...` |
   | `OPENAI_API_KEY` | Your key |
   | `CORS_ORIGINS` | `https://<your-vercel-domain>,https://*.vercel.app` |
   | `COOKIE_SECRET` | Output of `openssl rand -hex 32` |
   | `ENV` | `prod` |
   | `RATE_LIMIT_PER_MIN` | `60` |

4. **Generate a public domain** (Settings → Networking → Generate
   Domain). Confirm:
   ```bash
   curl -fsS https://<railway-domain>/healthz   # {"status":"ok",...}
   ```

5. **Seed the DB** (one-off):
   ```bash
   railway run bash backend/scripts/hydrate_initial_data.sh
   ```

6. **(Optional) Schedule the stats refresh.** Add a Railway cron service:
   ```
   schedule: 0 4 * * 1            # weekly Monday 04:00 UTC
   command:  python -m app.etl refresh-stats --season 2024
   ```

#### Vercel — web

1. **Import the repo** at <https://vercel.com/new>. Set **Root Directory**
   to `web/` (Vercel auto-detects Next.js).
2. **Env vars** for both Production and Preview:

   | Var | Value |
   |---|---|
   | `NEXT_PUBLIC_API_BASE` | `https://<railway-domain>` |
   | `API_BASE_INTERNAL` | `https://<railway-domain>` (Railway has no internal-only DNS for Vercel; use the public domain) |
   | `NEXT_PUBLIC_GRAPHQL_URL` | `https://<railway-domain>/graphql` |

3. **First deploy** creates a preview URL. Promote to prod once smoke
   tests pass.

4. **Cross-fill CORS.** Copy the Vercel prod domain back into Railway's
   `CORS_ORIGINS` env var and redeploy the backend.

#### Cookie attrs in prod

`ENV=prod` switches the cookie middleware to
`SameSite=None; Secure` so cross-origin Vercel → Railway requests carry
the `dna_sid` cookie. See
[backend/app/core/sessions.py](backend/app/core/sessions.py) `_cookie_attrs`.

#### Post-deploy smoke

```bash
curl -fsS https://<railway-domain>/healthz
curl -fsS https://<railway-domain>/api/v1/seasons | jq '.data | length'
curl -fsS https://<vercel-domain>/                              # landing renders
curl -fsS https://<railway-domain>/openapi.json | jq '.openapi' # "3.1.0"
```

### Path B — Self-host on a VPS (Docker only)

For reviewers who'd rather run everything themselves on a Linux box.

1. **Provision a VPS** (1 CPU / 2 GB RAM is plenty for demo) on
   DigitalOcean / Hetzner / Linode / Fly.
2. **Install Docker + docker-compose** following the official Docker
   docs for your distro.
3. **Clone the repo + configure envs**:
   ```bash
   git clone https://github.com/<you>/driver-dna.git
   cd driver-dna
   cp backend/.env.example backend/.env       # edit OPENAI_API_KEY, ENV=prod, CORS_ORIGINS
   cp web/.env.example     web/.env.local     # set NEXT_PUBLIC_API_BASE to your domain
   ```
4. **Create a prod docker-compose overlay** (not in the repo by default
   — add this file as `docker-compose.prod.yml`):

   ```yaml
   services:
     postgres:
       restart: unless-stopped
       volumes:
         - pgdata:/var/lib/postgresql/data

     backend:
       restart: unless-stopped
       env_file: ./backend/.env
       expose: ["8000"]                # internal-only; Caddy proxies in
       depends_on:
         postgres: { condition: service_healthy }

     web:
       build:
         context: ./web
       restart: unless-stopped
       env_file: ./web/.env.local
       expose: ["3000"]
       depends_on: [backend]

     caddy:
       image: caddy:2-alpine
       restart: unless-stopped
       ports: ["80:80", "443:443"]
       volumes:
         - ./Caddyfile:/etc/caddy/Caddyfile:ro
         - caddy_data:/data
         - caddy_config:/config

   volumes:
     pgdata: {}
     caddy_data: {}
     caddy_config: {}
   ```

5. **Caddyfile** (terminates TLS via Let's Encrypt automatically):

   ```caddy
   api.example.com {
     reverse_proxy backend:8000
   }
   example.com, www.example.com {
     reverse_proxy web:3000
   }
   ```

6. **Bring everything up**:
   ```bash
   docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
   docker compose exec backend sh -c "cd /app/backend && alembic upgrade head && bash scripts/hydrate_initial_data.sh"
   ```

7. **Verify**:
   ```bash
   curl -fsS https://api.example.com/healthz
   curl -fsS https://example.com
   ```

Updates: `git pull && docker compose -f ... -f ... up -d --build`.
Logs: `docker compose logs -f backend web caddy`.

### Path C — Streamlit Cloud (legacy UI only)

The legacy Streamlit dashboard can run on the free
<https://streamlit.io/cloud> tier. It does **not** reach the Postgres
backend — it's a standalone dashboard over Google-Drive-persisted
artifacts. Use this in parallel with Path A to demo "all surfaces" on
free infra.

1. Push the repo to GitHub (must be public for the free tier).
2. <https://streamlit.io/cloud> → New app → pick repo + branch.
3. **Main file path:** `streamlit_app.py`.
4. **Secrets** (Settings → Secrets) — paste TOML, mirroring
   `.streamlit/secrets.toml`:
   ```toml
   OPENAI_API_KEY = "sk-..."
   [google_drive]
   client_id = "..."
   client_secret = "..."
   ```
5. Deploy. First boot creates the FastF1 cache (~600 MB), then restores
   model artifacts from Google Drive via OAuth on first user click.

The full legacy setup steps (Google Drive OAuth, FastF1 cache, dataset
generation, model training) live in
[the Streamlit deploy section below](#legacy-streamlit-dashboard).

---

## CI / quality bar

Three GitHub Actions workflows, each path-filtered to its area.

| Workflow | Triggers on changes to | Steps | Typical duration |
|---|---|---|---|
| [`ci.yml`](.github/workflows/ci.yml) | `src/**`, `tests/**` | ruff · mypy · pytest · LLM eval · docker build → GHCR | ~4 min |
| [`backend-ci.yml`](.github/workflows/backend-ci.yml) | `backend/**`, `src/**` | Postgres service container · ruff · `alembic upgrade head` · pytest with coverage · OpenAPI 3.1 snapshot assertion | ~3 min |
| [`web-ci.yml`](.github/workflows/web-ci.yml) | `web/**` | `npm ci` · `tsc --noEmit` · `next lint` · `next build` | ~2 min |

**Backend quality bar:**
- **ruff** subset `E F I B UP SIM` — 9 noisy rules explicitly ignored
  with documentation in [backend/pyproject.toml](backend/pyproject.toml)
  (each ignore explains why).
- **alembic upgrade head** runs against an ephemeral Postgres service
  container — every migration must apply on a fresh DB on every push.
- **pytest --cov=app** must pass; coverage report saved as artifact.
- **OpenAPI snapshot** — a tiny Python step asserts
  `openapi.json["openapi"].startswith("3.1")` so we never accidentally
  fall back to 3.0.

**Web quality bar:**
- `tsc --noEmit` with `"strict": true` in [tsconfig.json](web/tsconfig.json).
- `next lint` with [.eslintrc.json](web/.eslintrc.json) checked in (no
  interactive setup wizard in CI).
- `next build` validates SSR rendering for every dynamic route.

**Real bugs CI caught during Phase 12 lint cleanup:** two
`status.HTTP_*_*` references with no `status` import, two unused locals,
one unused import, one OpenAPI field rename
(`lap_time_sec` → `fastest_lap_time_sec`) — all surfaced by `ruff` /
`tsc` before merge.

---

## Testing

**119 backend tests** across 11 files + **163 legacy tests** for `src/` +
the Streamlit app = **282 tests total**. All required to pass before
merge.

| Suite | File | # | What it covers |
|---|---|---|---|
| Smoke | [`test_healthz.py`](backend/tests/test_healthz.py) | 3 | App boots; `/healthz`; shared-`src/` on `sys.path` |
| Contracts | [`test_contracts.py`](backend/tests/test_contracts.py) | 14 | RFC 7807 envelope shape · cursor round-trip + 100-int walk · `X-Request-ID` mint/echo · `API-Version` header |
| OpenAPI | [`test_openapi.py`](backend/tests/test_openapi.py) | 4 | 3.1 conformance · `ErrorEnvelope` referenced by every 4xx/5xx |
| Schema | [`test_schema.py`](backend/tests/test_schema.py) | 10 | Table presence · FK CASCADE/RESTRICT semantics · ENUM types · unique + check constraints |
| Query plans | [`test_query_plans.py`](backend/tests/test_query_plans.py) | 4 | `EXPLAIN (FORMAT JSON)` asserts `Index Scan` for the 3 hot queries + all indexes present |
| ETL | [`tests/etl/*.py`](backend/tests/etl/) | 11 | Mocked OpenF1 via `responses` · idempotency proven (run twice = identical rows) · dry-run rollback · compound normalisation |
| REST reads | [`test_v1_*.py`](backend/tests/api/) | 32 | Happy + 404 envelope + invalid cursor + season/team filters + per-driver lap walk |
| Analytics | [`test_v1_analytics.py`](backend/tests/api/test_v1_analytics.py) | 12 | All 4 analytics endpoints + compare endpoint with mocked OpenAPI car_data |
| GraphQL | [`tests/graphql/*.py`](backend/tests/graphql/) | 8 | Introspection · REST↔GraphQL parity for leaderboards · **N+1 regression guard** |
| LLM | [`tests/llm/*.py`](backend/tests/llm/) | 11 | OpenAI mocked via queue-fake · all 5 patterns · SSE event sequence asserted |
| /me | [`test_v1_me.py`](backend/tests/api/test_v1_me.py) | 10 | Cookie issuance · 2-client isolation · 100-row cap → 409 · ownership-check deletes |
| **Total backend** | | **119** | |

### Test infrastructure highlights

- **Per-test NullPool engine override** in
  [tests/api/conftest.py](backend/tests/api/conftest.py) — `TestClient`
  uses a fresh event-loop portal per request; the global async engine
  pools connections across loops and fails with `Event loop is closed`.
  The fixture swaps `app.db.session.{engine, AsyncSessionLocal}` and
  the middleware-internal factory for a per-test `NullPool` engine.
- **Queue-based OpenAI mock** in
  [tests/llm/conftest.py](backend/tests/llm/conftest.py) — tests enqueue
  canned responses (text, JSON, tool calls, stream chunks); the patched
  `openai.OpenAI` consumes one per call. Zero network in tests.
- **N+1 regression guard** in
  [tests/graphql/test_schema.py](backend/tests/graphql/test_schema.py)
  — attaches a SQLAlchemy `before_cursor_execute` listener, runs a
  nested query returning 20 rows, asserts ≤ 10 SELECTs total. Catches
  any future regression that drops a `joinedload` or a DataLoader.

### Local run

```bash
cd backend
pytest -v                                  # all 119
pytest tests/api -v                        # just REST
pytest tests/llm -v                        # just LLM (mocked OpenAI)
pytest --cov=app --cov-report=term-missing # with coverage
```

---

## Operations & troubleshooting

### Observability

Every request emits one structured log line (structlog → JSON to stdout)
with a bound `request_id`:

```json
{"method":"GET","path":"/api/v1/sessions/1/results","status":200,
 "latency_ms":18.4,"event":"http.request",
 "request_id":"01KSHS9M2G6FQF1CAPACTQJYW8","level":"info",
 "timestamp":"2026-05-26T09:19:17Z"}
```

Same `request_id` appears in error envelopes — copy-paste straight from
a user-visible error into the log search.

### Cost tracking — `llm_audit`

Every LLM call writes one row with token counts, latency, USD cost, and
the originating `user_session_id`. Daily per-feature spend:

```sql
SELECT feature,
       COUNT(*)                          AS calls,
       SUM(input_tokens + output_tokens) AS tokens,
       ROUND(SUM(cost_usd), 4)           AS usd,
       ROUND(AVG(latency_ms))            AS avg_ms
FROM   llm_audit
WHERE  created_at > now() - interval '1 day'
GROUP  BY feature
ORDER  BY usd DESC;
```

### Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| Web → API call returns CORS error | Missing origin in `CORS_ORIGINS` | Add `https://<vercel-domain>` to backend env, restart |
| `Event loop is closed` in tests | Reusing async engine across TestClient loops | Use the `client` fixture from `tests/api/conftest.py` (NullPool override) |
| SSE chat stalls in browser | Vercel buffers responses through its proxy | Set `NEXT_PUBLIC_API_BASE` directly to Railway domain so client hits backend, not Vercel |
| `alembic upgrade head` errors "relation already exists" | DB has hand-created tables | `docker compose down -v` to drop volume, then up + migrate |
| Empty leaderboard | Session not hydrated | `python -m app.etl hydrate --year 2024 --gp "..." --session R` |

### Migration runbook

Add migration #6:

```bash
cd backend
alembic revision -m "add_telemetry_traces_table"
# Edit the new file in alembic/versions/0006_*.py
alembic upgrade head                       # apply locally
pytest tests/test_schema.py -v             # verify table shape
git add alembic/versions/0006_*.py app/db/models.py
git commit -m "feat(db): cache telemetry traces"
```

Postgres ENUM gotcha (the trap that hit us in 0005): when a new
`saved_analyses.kind` column uses `PgEnum(create_type=True)`,
SQLAlchemy issues its own `CREATE TYPE` even if you also wrote one
manually — duplicating it. Pattern used: create the ENUM explicitly
with `kind_enum.create(op.get_bind(), checkfirst=True)`, then use
`PgEnum(..., create_type=False)` on the column.

### Re-seeding prod

```bash
railway run bash backend/scripts/hydrate_initial_data.sh
# or one race
railway run python -m app.etl hydrate --year 2024 --gp "Bahrain Grand Prix" --session R
railway run python -m app.etl refresh-stats --season 2024
```

---

## Roadmap — not yet shipped

Honest disclosure so reviewers know what's deferred, not broken.

- **GraphQL `Query.me` + `Mutation.saveAnalysis`** — REST `/me/*` covers
  this fully today; GraphQL bindings can land additively.
- **Playwright web smoke tests** — Phase 12 deferred them; current web
  CI relies on `next build` succeeding as the smoke.
- **Rate limiter is in-memory** — adequate for one Railway dyno; a Redis
  backend would unblock multi-instance scaling.
- **Compare endpoint channels** — supports Speed / Throttle / Brake;
  Time-Delta + Track-Map builders exist in
  [src/viz.py](src/viz.py) but aren't wired into REST yet.
- **Telemetry trace cache** — `/compare` hits OpenF1 live every call.
  A `telemetry_traces` cache table keyed on
  `(session_key, driver_number, channel)` would cut compare latency
  ~5×.
- **Streaming GraphQL subscriptions** — race chat uses SSE on REST;
  could move to GraphQL subscriptions when we have a second
  streaming use case.

---

## Credits & license

**Data sources**
- [OpenF1](https://openf1.org) — live + historical race telemetry REST API
- [FastF1](https://docs.fastf1.dev/) — Python F1 telemetry SDK + cache

**Libraries**
- FastAPI · Strawberry · SQLAlchemy · Alembic · Pydantic
- Next.js · React · Plotly · Tailwind
- XGBoost · SHAP · scikit-learn · pandas · numpy
- OpenAI (gpt-4o-mini)

**Design**
- Typography: [Space Grotesk](https://fonts.google.com/specimen/Space+Grotesk)
- Brand red: `#E8002D` (F1)

**License**
- MIT — see [LICENSE](LICENSE) if present; otherwise treat the repo as
  MIT-licensed source available for portfolio review.

---

## Legacy Streamlit dashboard

The full original README — Streamlit visual tour, MCP server tools,
Google Drive OAuth setup, ML pipeline details — is preserved in git
history before the Phase 12 rewrite. If you need:

- **Streamlit tab-by-tab feature reference** (Driver Radar, Mystery
  Driver, Race Dashboard, Pipeline) → `git log --diff-filter=D -- README.md`
  to locate the pre-rewrite revision, or run `streamlit run streamlit_app.py`
  and explore the UI directly.
- **MCP server details** — see [mcp/server.py](mcp/server.py) and the
  inline docstrings. Four tools: `list_sessions`, `list_drivers`,
  `get_fastest_lap_data`, `get_channel_comparison`. Run with
  `python mcp/server.py` over stdio or via the MCP Inspector:
  `npx @modelcontextprotocol/inspector python mcp/server.py`.
- **Google Drive artifact persistence** — see
  [src/drive_sync.py](src/drive_sync.py). OAuth 2.0 with `drive.file`
  scope (least-privilege). Tokens at `data/.google_token.json`. Files
  synced: dataset.parquet, model joblibs, metrics.json, confusion
  matrix HTML, SHAP PNG, llm_audit.jsonl.
- **ML training pipeline** — `python src/pipeline.py` to extract
  features from a FastF1 session; `python src/model.py` to train the
  XGBoost classifier with 5-fold stratified CV.
