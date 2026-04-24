# Driver DNA + Race Intelligence Dashboard

**Can a machine learning model identify an F1 driver purely from their telemetry — without ever seeing their name?**

Driver DNA is a full-stack AI/ML platform built on Formula 1 telemetry data. It combines an XGBoost driver-identification model, a six-chart driving style analysis suite, an Explainable AI layer, a real-time race analytics engine, and **five distinct Agentic AI features** powered by GPT-4o-mini — all served through an interactive Streamlit dashboard across four tabs. A dedicated **Pipeline & Training tab** lets you download F1 telemetry datasets and retrain the classifier directly from the UI, with live progress logs and background threading. The UI uses a neo-brutalist design system with F1's red/black/white palette and Space Grotesk typography.

---

## Tech Stack

| Category | Technologies |
|---|---|
| **Machine Learning** | XGBoost (multi-class classification), scikit-learn, SHAP (Explainable AI), 5-fold stratified cross-validation |
| **Agentic AI / LLM** | OpenAI API (`gpt-4o-mini`), tool calling (ReAct), Reflexion self-critique loop, RAG via cosine similarity, JSON-mode structured output, prompt engineering, multi-step agent orchestration |
| **Data Engineering** | FastF1, pandas, NumPy, Apache Parquet, time-series resampling, 200-point distance-normalised feature extraction |
| **Visualisation** | Streamlit, Plotly (radar/spider charts, heatmaps, bar charts, line overlays, scatter track maps, horizontal similarity bars) |
| **APIs** | OpenF1 REST API (live + historical), OpenAI Chat Completions API |
| **MLOps** | joblib model serialisation, SHAP feature importance, confusion matrix evaluation, cross-validation scoring, per-driver precision/recall/F1 metrics, JSONL audit logging, LLM cost dashboard |
| **Testing & Quality** | pytest (89 tests, 4 test modules), pytest-cov (coverage XML), ruff (linting), mypy (type checking), GitHub Actions CI |
| **Deployment** | Docker (multi-stage build, python:3.13-slim), GitHub Container Registry (GHCR), Streamlit Community Cloud |
| **Security** | Prompt injection prevention, input sanitisation, API key management via `st.secrets`, tool name allowlisting, argument clamping, rate limiting, LLM response sanitisation, JSON schema validation |

---

## Architecture

The codebase is split into focused, independently testable modules with a one-directional dependency chain — no circular imports.

```
app.py  (2 200+ lines — UI, tabs, session state, background threads)
  ├── from features import ...     feature engineering + Streamlit caching
  ├── from viz import ...          Plotly figure builders — no st. calls
  ├── from llm_layer import ...    LLM orchestration, prompts, tool functions
  ├── from model import ...        XGBoost training + evaluation
  ├── from pipeline import ...     dataset download + feature extraction
  ├── from openf1 import ...       OpenF1 REST API client
  └── from race_engine import ...  pace analytics + strategy detection

llm_layer.py  (1 470 lines)
  └── from features import ...     classify_archetype, EXTENDED_RADAR_FEATURES

viz.py  (577 lines)
  └── from features import ...     N_POINTS, OPENF1_TRACE_COLS
```

`llm_layer.py` and `viz.py` have zero Streamlit dependencies — they can be imported, tested, and benchmarked without a running Streamlit server. All session-state logic stays in `app.py`.

---

## Project Structure

```
driver-dna/
├── src/
│   ├── app.py               # Dashboard — 4 tabs, 5 AI features, 13+ chart types
│   ├── features.py          # Feature constants, engineering helpers, classify_archetype
│   ├── viz.py               # Plotly figure builders (no Streamlit dependency)
│   ├── llm_layer.py         # LLM orchestration: Reflexion, RAG, ReAct, structured output
│   ├── eval_llm.py          # Offline LLM evaluation framework + token cost dashboard
│   ├── race_engine.py       # Pace analytics, gap tracking, undercut detection, projections
│   ├── openf1.py            # OpenF1 REST API client (live + historical modes)
│   ├── model.py             # XGBoost training, CV evaluation, SHAP, metrics.json
│   ├── pipeline.py          # Data extraction, telemetry resampling, feature engineering
│   └── generate_circuits.py # One-off: generates data/circuits.json
├── tests/
│   ├── test_app_helpers.py   # 29 tests — LLM layer: parsing, validation, sanitisation, RAG
│   ├── test_race_engine.py   # 30 tests — pace, gaps, undercuts, projections, degradation
│   ├── test_openf1.py        # 18 tests — OpenF1 client response parsing, edge cases
│   └── test_pipeline_tab.py  # 12 tests — pipeline download + model training session logic
├── .github/workflows/
│   ├── ci.yml               # Lint (ruff) → type check (mypy) → pytest with coverage
│   ├── docker.yml           # Build & push to GHCR on every push to main
│   └── eval.yml             # LLM evaluation + PR comment with RAG sanity metrics
├── data/                    # Git-ignored — generated locally
│   ├── dataset.parquet      # Driver DNA training data (pipeline.py output)
│   ├── dataset_meta.json    # Session metadata — GP name, year, session type
│   └── circuits.json        # Circuit XY outlines for Track Map
├── models/                  # Git-ignored — generated locally
│   ├── driver_dna_clf.joblib
│   ├── label_encoder.joblib
│   ├── metrics.json         # CV scores, train accuracy, per-driver F1
│   ├── confusion_matrix.html
│   ├── shap_importance.png
│   └── llm_audit.jsonl      # Per-call LLM audit log (feature, tokens, latency, cost)
├── Dockerfile               # Multi-stage build — builder + lean runtime (python:3.13-slim)
├── requirements.txt
├── requirements-dev.txt     # pytest, pytest-cov, ruff, mypy, types-requests
└── streamlit_app.py         # Streamlit Community Cloud entry point
```

---

## Dashboard Overview

Four tabs navigated via a boxy rectangular tab bar (neo-brutalist F1 design — red active state, dark inactive). Each tab is independently functional.

| Tab | Name | What it does |
|---|---|---|
| 1 | **Driver Radar** | Driving style fingerprint visualisation + three agentic AI analysis features |
| 2 | **Mystery Driver** | ML-based driver identification from a single lap + SHAP explainer + XAI narration + model accuracy panel |
| 3 | **Race Dashboard** | Live and historical race analytics + conversational AI race analyst |
| 4 | **⚙️ Pipeline** | Download F1 telemetry datasets and retrain the classifier — all from the UI, with live progress logs |

### Visual Design

Neo-brutalist design system with F1's red/black/white palette:

| Token | Value | Used for |
|---|---|---|
| Page background | `#111111` | Base canvas |
| Card background | `#1a1a1a` | Panels, report cards, alert boxes |
| F1 Red | `#E8002D` | Active tabs, borders, shadows, buttons |
| Offset shadow | `3–4px solid #E8002D` | All card and button shadows |
| Border radius | `0px` | All containers (hard edges throughout) |
| Font | Space Grotesk (Google Fonts) | All headings and body text |

Active tab: red fill (`#E8002D`) with black text and `3px 3px 0 #000` shadow. Inactive tabs: dark `#1a1a1a` with `#333` border, red border on hover. Streamlit theme (`config.toml`) sets `primaryColor`, `backgroundColor`, `secondaryBackgroundColor`, and `textColor` to enforce F1 palette at the component level.

### Sidebar Controls

- **Model accuracy** — CV accuracy (±std dev), train accuracy, driver count, and lap count
- **AI Response Mode toggle** — switch between **Concise** (3–6 sentence focused answers) and **Detailed** (multi-paragraph rich analysis, ~50% more tokens per call) across all five AI features simultaneously
- **API key status** — live check for OpenAI key and required library imports

---

## Tab 1 — Driver Radar

A full driving style analysis suite. Select 2–4 drivers to compare across six visualisation layers and three AI features.

### Visualisation 1 — Driver Style Radar

An interactive spider chart showing normalised driving style across up to **12 dimensions**. A toggle switches between Standard (6) and Extended (12) mode.

**Standard dimensions (6):** Avg Speed, Throttle Input, Throttle Variation, Brake Frequency, Steering Precision, Gear Shifts

**Extended dimensions (12, adds):** Top Speed, Corner Speed, Brake Pressure, Full Throttle %, Trail Braking, Coasting %

The three trace-derived dimensions are computed at runtime from the 200-point telemetry arrays:

| Derived feature | Computation |
|---|---|
| `throttle_pickup_pct` | Fraction of lap distance where throttle > 80% |
| `coasting_pct` | Fraction where both throttle < 2% and brake < 1% |
| `trail_brake_score` | Fraction where throttle > 0 and brake > 0 simultaneously |

### Visualisation 2 — Driver Archetype Cards

A rule-based classifier assigns each selected driver a driving style archetype based on their normalised radar profile.

| Archetype | Trigger conditions |
|---|---|
| 💥 Aggressive Braker | High brake frequency and high brake pressure |
| 🎯 Trail Braking Specialist | High trail braking score and high steering variance |
| 🚀 High Entry Speed | High average speed and low coasting fraction |
| 🧊 Smooth Operator | High full-throttle % and low throttle variation |
| ⚙️ Technical Driver | High gear-shift rate and high steering variance |
| ⚖️ Balanced Driver | No dominant dimension |

### Visualisation 3 — Speed Profile Comparison

A multi-driver line chart showing **mean speed at every one of 200 normalised distance points** along the lap (`hovermode="x unified"`). Directly reveals where each driver goes faster — in corners, under braking, and on straights.

### Visualisation 4 — Lap-by-Lap Consistency Heatmap

A drivers × features heatmap coloured by cross-lap standard deviation. Red = high variance = inconsistent. Answers a question the mean-based radar cannot: *who is erratic under race pressure?*

### Visualisation 5 — Lap Zone Performance

The 200-point lap is split into three equal distance zones (0–33%, 33–67%, 67–100%). A grouped bar chart shows per-zone performance across Speed, Throttle, and Braking channels.

### Visualisation 6 — Throttle Application Map

The circuit outline (reconstructed from FastF1 XY coordinates) is coloured by average throttle — green for full throttle, red for coasting or braking. One panel per selected driver. Turns an abstract percentage into a spatial fingerprint.

---

## Tab 1 — Agentic AI Features

Three distinct agentic patterns, each architecturally different from the others and from the features in Tabs 2 and 3.

### AI Feature 1 — Reflexion Loop: Driver Style Analyst

**Pattern: Reflexion (Shinn et al. 2023) — iterative self-improvement via LLM self-critique**

```
Analyst LLM  →  generates driving style narrative from radar data
                        ↓
Critic LLM   →  evaluates narrative against raw data
                returns JSON: {confidence: 1–10, factual_errors: [...], suggested_improvements: [...]}
                        ↓
  confidence ≥ 7  →  accept narrative
  confidence < 7  →  Analyst receives critique and rewrites
                        ↓
Critic LLM (round 2)  →  re-evaluates the revised narrative
                        ↓
                  Final narrative accepted
```

The dashboard renders each round transparently in collapsible expanders — confidence score, flagged factual errors, and the revision side by side. The self-improvement process is visible to the user.

The Critic runs at `temperature=0.0` (deterministic JSON); the Analyst at `temperature=0.4`. Parse failure on the Critic response gracefully accepts the existing narrative rather than crashing.

**Token budget:** Concise — 400 (analyst) / 300 (critic). Detailed — 650 / 300.

---

### AI Feature 2 — RAG: Historical Driver DNA Matching

**Pattern: Retrieval-Augmented Generation — without a vector database**

```
Driver's 12-dimensional radar vector
           ↓
Cosine similarity vs. 10 curated historical F1 driver profiles
           ↓
Top-2 most similar profiles retrieved
           ↓
Retrieved profiles injected into LLM prompt as grounding context
           ↓
GPT-4o-mini explains WHY the current driver's style aligns with the historical matches,
referencing specific dimension values from both profiles
```

The knowledge base encodes 10 F1 legends (Schumacher, Senna, Prost, Alonso, Vettel, Hamilton, Verstappen, Räikkönen, Sainz, Rosberg) as 12-dimensional normalised vectors. Retrieval uses cosine similarity directly on the feature vectors — no embedding API, no vector database.

A horizontal bar chart shows all 10 similarity scores (x-axis zoomed to 0.80–1.0), with the top-2 matches highlighted. The LLM narration is anchored exclusively to retrieved data — no speculation.

**Token budget:** Concise — 350. Detailed — 600.

---

### AI Feature 3 — Structured Output: Driver DNA Report Card

**Pattern: JSON-mode structured generation + schema validation + retry-on-failure**

```
GPT-4o-mini called with response_format={"type": "json_object"}
           ↓
Python validation layer checks every key, type, length constraint, and value range
  → integers coerced from floats (e.g. 7.0 → 7)
  → lists trimmed silently if too long
  → HTML stripped from all string values
           ↓
  valid    →  render report card
  invalid  →  retry once with specific validation error injected back into the prompt
           ↓
Rendered as a structured UI card with rating bars and pill badges
```

The schema enforces: `headline` (≥2 data references), `strengths` (exactly 3, each citing ≥2 metrics), `weaknesses` (exactly 2 with root-cause diagnosis), three integer ratings (1–10), `tactical_tips` (exactly 2 engineer-grade recommendations), `style_tags` (exactly 4 labels).

**Token budget:** Concise — 500. Detailed — 700.

---

## Tab 2 — Mystery Driver

### ML Prediction

Select any lap from the dataset. Click **Identify Driver**. The XGBoost classifier returns a predicted driver and a full probability distribution across all drivers, visualised as a horizontal bar chart.

### XAI Explainer — AI Feature 4

**Pattern: Single-shot LLM narration of SHAP values**

1. **SHAP inference** — `shap.TreeExplainer` computes per-feature SHAP values for the predicted sample, identifying each feature's contribution to the predicted class
2. **Percentile ranking** — each feature value is ranked against the full training dataset distribution
3. **Validated prompt** — SHAP magnitudes, feature values, and percentile ranks are assembled into a structured prompt. No raw user input ever enters the LLM context.
4. **LLM narration** — GPT-4o-mini produces a plain-English explanation
5. **Session cache** — the explanation is stored in `st.session_state` and cleared when the user picks a different lap

**Token budget:** Concise — 300. Detailed — 500.

### Model Accuracy Panel

Full evaluation results loaded from `models/metrics.json`:
- CV Accuracy (±std dev), Train Accuracy, driver count, lap count
- Per-fold CV scores with delta vs. mean
- Per-driver precision, recall, F1, and support — colour-coded green/amber/red by F1

---

## Tab 3 — Race Dashboard

Real-time and historical race analysis powered by the OpenF1 API. No API key required for the data.

### Race Intelligence Chat Agent — AI Feature 5

**Pattern: ReAct tool-calling agent (Reason + Act loop)**

```
User question
      ↓
GPT-4o-mini decides which tools to call (up to 3 per round)
      ↓
App executes tools against live RaceAnalyser data
      ↓
Tool results returned to GPT-4o-mini
      ↓
GPT-4o-mini synthesises or calls more tools (up to 3 rounds)
      ↓
If loop exhausted without a text answer → forced synthesis pass
      ↓
Plain-prose answer rendered in chat UI
```

**6 tools exposed to the agent:**

| Tool | Underlying call | Returns |
|---|---|---|
| `get_rolling_pace` | `analyser.rolling_pace(window)` | Per-driver rolling-average lap times |
| `get_gap_to_leader` | `analyser.gap_to_leader()` | Cumulative gap per driver to the leader |
| `detect_strategy_events` | `analyser.detect_undercuts()` | Undercut / overcut detections |
| `project_finishing_order` | `analyser.project_finishing_order(laps_remaining)` | Projected final classification |
| `get_tyre_degradation` | `analyser.tyre_degradation()` | Degradation rate per driver/compound |
| `get_pace_summary` | `analyser.pace_summary()` | Mean, median, fastest lap per driver |

**Security guardrails — 11 layers:**

| # | Threat | Mitigation |
|---|---|---|
| 0 | Prompt injection via long/malicious input | `_ca_sanitize_input()` — strips control chars, 500-char limit |
| 1 | Out-of-range LLM-supplied tool arguments | `_ca_validate_tool_args()` — typed schema validation + value clamping |
| 2 | HTML/link injection via LLM response | `st.text()` renders plain text only |
| 3 | Unknown tool names hallucinated by LLM | `_CA_ALLOWED_TOOLS` explicit allowlist |
| 4 | Tool call flood within one round | `_CA_MAX_TOOLS_PER_ROUND = 3` cap |
| 5 | History poisoning via long LLM responses | Per-message 2000-char truncation |
| 6 | Hardcoded API key | `_get_openai_api_key()` reads `st.secrets` / env var only |
| 7 | Infinite agent loop | `MAX_TOOL_ROUNDS = 3` hard cap + forced synthesis fallback |
| 8 | Context window / cost blowout | 20-message history cap + `max_tokens` controlled by response mode |
| 9 | API spam / cost abuse | 5-second per-session rate limiter |
| 10 | Corrupt state on failed turn | User message popped from history on API error |

**Example questions the agent can answer:**
- *"Who has the best pace over the last 10 laps?"*
- *"Is Verstappen likely to undercut Norris? What does the gap look like?"*
- *"Which driver has the worst tyre degradation on softs?"*
- *"Who is projected to win with 20 laps remaining?"*

### Fastest Lap Telemetry Comparison

Five channels comparing two drivers over their fastest recorded lap, fetched live from OpenF1:

| Channel | What it shows |
|---|---|
| **Time Delta** | Cumulative time gap through the lap, with shaded gain/loss regions |
| **Track Map** | Circuit outline with ~999 microsectors coloured by faster driver, braking zone overlays, full-throttle overlays, auto-detected corner labels (T1–T12), start/finish marker, lap time subtitle, and rich per-microsector hover |
| **Speed** | Overlaid speed traces (km/h) vs. distance |
| **Throttle** | Overlaid throttle application (%) vs. distance |
| **Brake** | Overlaid brake pressure (%) vs. distance |

### 7 Race Analysis Charts

| Chart | What it shows |
|---|---|
| **Rolling Race Pace** | 5-lap rolling-average lap time. Pit-out and safety car laps filtered automatically. |
| **Gap to Leader** | Cumulative time gap to the race leader per lap, shaded per driver. |
| **Undercut / Overcut Alerts** | Triangle markers on the gap chart at the exact lap a pit strategy caused a position swap. |
| **Projected Finishing Order** | Projected final gaps accounting for current pace, tyre degradation, and DNF flags. |
| **Average Race Pace** | Bars per driver coloured by consistency (Viridis scale). Hover shows mean, median, std dev, fastest lap, lap count. |
| **Race Pace Ranking** | Bars sorted fastest to slowest with ±std dev error bars. |
| **Tyre Degradation by Compound** | Scatter of mean stint pace vs. degradation rate (s/lap) from per-stint linear regression. |

**Live mode** — connects to the OpenF1 live feed during an active F1 race weekend with auto-refresh at 10s / 30s / 60s intervals.

**Historical mode** — any race from 2022 onwards via the OpenF1 archive.

---

## Tab 4 — Pipeline & Training

A self-contained data and model management tab. Everything previously requiring a terminal is now accessible from the dashboard, including live progress feedback for both long-running operations.

### Download Dataset

A form mirroring the inputs of `src/pipeline.py`:

| Field | Description |
|---|---|
| **Year** | Season year (1950–2100) |
| **Grand Prix name** | e.g. `Italian Grand Prix` |
| **Session type** | `Q`, `R`, `FP1`, `FP2`, `FP3`, `S`, `SS` |
| **Driver filter** | Optional — comma-separated 3-letter codes (e.g. `VER,HAM`). Leave blank for all drivers. |

Clicking **Download Dataset** launches a background thread (via `threading.Thread`) that calls `extract_session_telemetry` → `save_dataset` → `save_meta`. The log panel updates every second while the thread is running. On success, the `get_data` and `get_dataset_meta` Streamlit caches are cleared automatically so the rest of the dashboard reflects the new data immediately.

### Train Model

Displays the current model's CV accuracy, train accuracy, driver count, and lap count loaded from `models/metrics.json`. Clicking **Train Model** launches a background thread that runs `load_and_prepare` → `train_model` (5-fold CV) → `evaluate_model` → `save_model`. The `get_model` Streamlit resource cache is cleared on completion, reloading the classifier for the Mystery Driver tab without a page refresh.

A guard raises a clear error if the dataset contains fewer than 2 unique drivers — preventing a cryptic StratifiedKFold failure.

### Implementation details

Both operations use `threading.Thread(daemon=True)` — the same polling approach as the Live Race mode — with a 1-second auto-rerun loop (`time.sleep(1)` + `st.rerun()`) while any thread is active. Button guards (`disabled=True`) prevent double-launches. Errors are caught in a `finally` block, ensuring the `*_running` flag is always cleared regardless of outcome.

---

## Agentic AI Patterns — Summary

Five AI features across tabs 1–3, each implementing a distinct real-world production pattern:

| Feature | Tab | Pattern | Key technique |
|---|---|---|---|
| Driver Style Analyst | Radar | **Reflexion** | Analyst → Critic (JSON) → conditional revision loop |
| Historical DNA Matching | Radar | **RAG** | Cosine similarity retrieval → context-augmented prompt |
| Driver DNA Report Card | Radar | **Structured Output** | JSON mode + schema validation + retry-on-failure |
| XAI Prediction Explainer | Mystery Driver | **Single-shot narration** | SHAP values + percentile ranks → validated prompt → prose |
| Race Intelligence Chat | Race Dashboard | **ReAct** | Tool-calling loop, multi-round, forced synthesis fallback |

---

## Machine Learning Pipeline

### 1. Data extraction — `src/pipeline.py`

FastF1 pulls high-frequency telemetry from the F1 live timing feed. Invalid laps (in-laps, out-laps, deleted times) are excluded. Each valid lap is resampled to **200 evenly spaced distance-based points** via linear interpolation — normalising every lap to the same resolution regardless of circuit length or car speed.

### 2. Feature engineering — `src/features.py` + `src/pipeline.py`

Each lap is compressed into **10 scalar driving-style features** plus three 200-point trace arrays:

| Feature | What it captures |
|---|---|
| `lap_time_seconds` | Overall pace |
| `mean_speed`, `max_speed`, `min_speed` | Speed profile across the lap |
| `throttle_mean`, `throttle_std` | Throttle aggression and smoothness |
| `brake_mean`, `brake_events` | Braking intensity and frequency |
| `gear_changes` | Shift frequency — mechanical aggression |
| `steer_std` | Steering smoothness |
| `speed_trace`, `throttle_trace`, `brake_trace` | Full 200-point traces (radar + zone charts) |
| `x_trace`, `y_trace` | Circuit XY coordinates (throttle map) |

Three additional features derived at runtime from traces for the Extended Radar:

| Derived feature | Computation |
|---|---|
| `coasting_pct` | `mean(throttle < 2% and brake < 1%)` |
| `throttle_pickup_pct` | `mean(throttle > 80%)` |
| `trail_brake_score` | `mean(throttle > 0 and brake > 0)` |

### 3. XGBoost classifier — `src/model.py`

A gradient-boosted tree multi-class classifier trained with **5-fold stratified cross-validation**. SHAP `TreeExplainer` generates both global feature importance (saved as `shap_importance.png` at training time) and local per-prediction explanations at inference time.

`evaluate_model` saves `models/metrics.json` with CV mean accuracy, std dev, per-fold scores, full-dataset train accuracy, and per-driver precision / recall / F1 / support.

### 4. Circuit geometry store — `src/generate_circuits.py`

Extracts XY circuit outlines for all 24 Grand Prix from FastF1 at **500 resampled distance points** per circuit and writes them to `data/circuits.json`. Resume-safe — skips circuits already written.

---

## Testing & Quality

### Test suite — 89 tests across 4 modules

| Module | Tests | What it covers |
|---|---|---|
| `test_app_helpers.py` | 29 | LLM layer: JSON parsing, schema validation, input sanitisation, tool arg clamping, cosine similarity |
| `test_race_engine.py` | 30 | Rolling pace, gap-to-leader, undercut detection, finishing order projection, tyre degradation, stint analysis |
| `test_openf1.py` | 18 | OpenF1 API client: response parsing, error handling, live/historical edge cases |
| `test_pipeline_tab.py` | 12 | Pipeline download session-state logic, model training session-state logic, error capture, finally-block cleanup, argument propagation, low-sample guard |

```bash
pytest tests/ -v                                      # run all 89 tests
pytest tests/ --cov=src --cov-report=term-missing     # with line coverage
```

All business logic in `llm_layer.py`, `features.py`, `race_engine.py`, and the pipeline tab wrappers is imported directly — no inline copies, no Streamlit runtime required.

### CI pipeline — `.github/workflows/ci.yml`

Three jobs run on every push to `main` and every pull request:

```
lint (ruff)  ──┐
               ├──→  test (pytest + coverage upload)
type-check  ──┘
(mypy)
```

### LLM evaluation — `src/eval_llm.py`

An offline evaluation framework for all five AI features:

| Evaluation | Mode | What it measures |
|---|---|---|
| RAG retrieval sanity | Offline | Self-match rate and retrieval margin across all 10 historical profiles |
| Report Card compliance | Live (opt-in) | First-attempt pass rate, retry rate, failure rate against JSON schema |
| Reflexion critique quality | Live (opt-in) | Mean confidence score, revision rate, mean factual error count |
| Token cost dashboard | Offline | Per-feature call count, mean latency, total tokens, estimated USD cost |

Results are saved to `models/eval_*.json`. The CI eval workflow (`eval.yml`) runs the offline evaluations on every push and posts a RAG sanity summary as a PR comment.

```bash
python src/eval_llm.py               # offline evaluations (RAG + token cost)
python src/eval_llm.py --audit       # parse llm_audit.jsonl for cost report
python src/eval_llm.py --live        # full suite with real OpenAI calls
```

---

## Deployment

### Docker

A multi-stage Dockerfile produces a lean production image:

- **Stage 1 (builder):** `python:3.13-slim` + gcc/g++ to compile C-extension packages (numpy, shap, xgboost). Dependencies installed to `/install`.
- **Stage 2 (runtime):** `python:3.13-slim` — copies only the compiled packages and application code. No build toolchain in the final image.

```bash
docker build -t driver-dna .
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=sk-proj-... \
  driver-dna
```

A health check polls `/_stcore/health` every 30 seconds (`--interval=30s --timeout=10s --retries=3`).

### GitHub Container Registry

The `docker.yml` workflow builds and pushes to GHCR on every push to `main`, tagged as both `latest` and `sha-<commit>`.

### Streamlit Community Cloud

Deploy directly from this repository. Set the OpenAI API key via the **Secrets** panel — no `.toml` file needed in production.

---

## AI Response Mode Toggle

A sidebar radio button switches all five AI features between two response depths simultaneously:

| Mode | Behaviour | Token budget |
|---|---|---|
| **Concise** (default) | 3–6 sentence focused answers | Lower cost |
| **Detailed** | Multi-paragraph rich analysis with bold numbers, data citations, forward-looking observations | ~50% more tokens per call |

**Token budget by feature:**

| Feature | Concise | Detailed |
|---|---|---|
| XAI Explainer | 300 | 500 |
| Reflexion Analyst | 400 | 650 |
| Reflexion Critic | 300 | 300 (unchanged) |
| RAG DNA Match | 350 | 600 |
| Report Card | 500 | 700 |
| Race Chat | 500 | 700 |

---

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd driver-dna
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Always activate the project venv first.** Running with a system Python causes `ImportError: numpy.core.multiarray failed to import` due to version mismatches. Verify with `which python` — it should point to `.venv/bin/python`.

### 2. Configure the OpenAI API key

All five AI features require a paid OpenAI account. Create `.streamlit/secrets.toml` (gitignored):

```toml
[openai]
api_key = "sk-proj-..."
```

Or export as an environment variable:

```bash
export OPENAI_API_KEY="sk-proj-..."
```

### 3. Generate circuit outlines *(required for Track Map and Throttle Map)*

Downloads XY geometry for all 24 Grand Prix circuits. Run once — takes 10–30 minutes. Resume-safe.

```bash
python src/generate_circuits.py
```

### 4. Build the Driver DNA dataset

Downloads session data, engineers features, and writes `data/dataset.parquet` and `data/dataset_meta.json`.

**Option A — terminal:**
```bash
python src/pipeline.py
```

**Option B — dashboard:** Launch the app (step 6), open the **⚙️ Pipeline** tab, fill in the Year / Grand Prix / Session form, and click **Download Dataset**.

### 5. Train the classifier

Runs 5-fold stratified CV and saves the model, label encoder, confusion matrix, SHAP importance chart, and `models/metrics.json`.

**Option A — terminal:**
```bash
python src/model.py
```

**Option B — dashboard:** In the **⚙️ Pipeline** tab, click **Train Model**. Progress is shown live and the model cache is refreshed automatically on completion.

### 6. Launch the dashboard

```bash
streamlit run src/app.py
```

> The Race Dashboard and Race Intelligence Chat work independently of steps 4–5 — they query the OpenF1 API directly. Live mode only works during active F1 race weekends; historical mode works any time from 2022 onwards.

---

## Debugging

A VS Code debug configuration is included at [`.vscode/launch.json`](.vscode/launch.json).

| Configuration | What it runs |
|---|---|
| **Streamlit: app.py** | Full dashboard with breakpoints on `localhost:8501` |
| **Python: pipeline.py** | Data extraction with interactive breakpoints |
| **Python: model.py** | Full train → evaluate → save pipeline |
| **Python: race_engine.py** | Race engine in isolation |
| **Python: current file** | Debugs the currently open file |

### Common Issues

**`ImportError: numpy.core.multiarray failed to import`**
Wrong Python interpreter. In VS Code: `⇧⌘P` → **Python: Select Interpreter** → choose the `.venv` entry.

**Track Map or Throttle Map shows no data**
`data/circuits.json` not yet generated. Run `python src/generate_circuits.py`.

**Model accuracy panel shows nothing in Mystery Driver tab**
`models/metrics.json` not yet generated. Re-run `python src/model.py`.

**AI features: "OpenAI rate limit reached"**
The OpenAI API requires a paid account. Free-tier accounts have no API access — enable billing at `platform.openai.com/settings/billing`.

**AI features: "OpenAI API key not configured"**
`.streamlit/secrets.toml` is missing or the placeholder key was not replaced. See Setup step 2.

**Mystery Driver tab shows "—" for Grand Prix / Season / Session**
`data/dataset_meta.json` was not generated. Re-run `python src/pipeline.py` or use the **⚙️ Pipeline** tab to download a dataset.

**Pipeline tab: "Download Dataset" button is greyed out**
A download or training operation is already running in the background. Wait for the active operation to complete — the button re-enables automatically.

**Pipeline tab: "Train Model" button is greyed out**
Either a download or training thread is still active, or `data/dataset.parquet` does not yet exist. Download a dataset first.

**Pipeline tab: "Need at least 2 drivers to train a classifier"**
The downloaded dataset contains laps from only one driver. Re-download with a different session or remove the driver filter to include all drivers.

**Driver Radar AI features: no buttons visible**
The three agentic AI features only appear once 2–4 drivers are selected in the multiselect and an OpenAI API key is configured.
