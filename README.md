# Driver DNA + Race Intelligence Dashboard

**Can a machine learning model identify an F1 driver purely from their telemetry — without ever seeing their name?**

Driver DNA is a full-stack AI/ML platform built on Formula 1 telemetry data. It combines an XGBoost driver-identification model, a six-chart driving style analysis suite, an Explainable AI layer, a real-time race analytics engine, and **five distinct Agentic AI features** powered by GPT-4o-mini — all served through an interactive Streamlit dashboard across three tabs.

---

## Tech Stack

| Category | Technologies |
|---|---|
| **Machine Learning** | XGBoost (multi-class classification), scikit-learn, SHAP (Explainable AI), 5-fold stratified cross-validation |
| **Agentic AI / LLM** | OpenAI API (`gpt-4o-mini`), tool calling (ReAct), Reflexion self-critique loop, RAG via cosine similarity, JSON-mode structured output, prompt engineering, multi-step agent orchestration |
| **Data Engineering** | FastF1, pandas, NumPy, Apache Parquet, time-series resampling, 200-point distance-normalised feature extraction |
| **Visualisation** | Streamlit, Plotly (radar/spider charts, heatmaps, bar charts, line overlays, scatter track maps, horizontal similarity bars) |
| **APIs** | OpenF1 REST API (live + historical), OpenAI Chat Completions API |
| **MLOps** | joblib model serialisation, SHAP feature importance, confusion matrix evaluation, cross-validation scoring, per-driver precision/recall/F1 metrics |
| **Security** | Prompt injection prevention, input sanitisation, API key management (`st.secrets`), tool name allowlisting, argument clamping, rate limiting, LLM response sanitisation, JSON schema validation |

---

## Dashboard Overview

The dashboard has three tabs navigated via a custom pill-style navigation bar. Each tab is independently functional.

| Tab | Name | What it does |
|---|---|---|
| 1 | **Driver Radar** | Driving style fingerprint visualisation + three agentic AI analysis features |
| 2 | **Mystery Driver** | ML-based driver identification from a single lap + SHAP explainer + XAI narration + model accuracy panel |
| 3 | **Race Dashboard** | Live and historical race analytics + conversational AI race analyst |

### Sidebar Controls

The sidebar provides global controls available across all tabs:

- **Model accuracy** — CV accuracy (±std dev), train accuracy, driver count, and lap count, loaded from `models/metrics.json`
- **AI Response Mode toggle** — switch between **Concise** (3–6 sentence focused answers, lower token cost) and **Detailed** (multi-paragraph rich analysis, ~50% more tokens per call) across all five AI features simultaneously
- **API key status** — live check for OpenAI key and required library imports

---

## Tab 1 — Driver Radar

A full driving style analysis suite. Select 2–4 drivers to compare across six visualisation layers and three AI features. The AI features are displayed at the top of the tab for immediate access.

### Visualisation 1 — Driver Style Radar

An interactive spider chart showing normalised driving style across up to **12 dimensions**. A toggle switches between Standard (6) and Extended (12) mode.

**Standard dimensions (6):** Avg Speed, Throttle Input, Throttle Variation, Brake Frequency, Steering Precision, Gear Shifts

**Extended dimensions (12, adds):** Top Speed, Corner Speed, Brake Pressure, Full Throttle %, Trail Braking, Coasting %

The three new trace-derived dimensions are computed from the 200-point telemetry arrays stored in the dataset — not from pre-computed scalars:
- **Full Throttle %** — fraction of lap distance where throttle > 80%
- **Coasting %** — fraction where both throttle < 2% and brake < 1%
- **Trail Braking** — fraction where throttle > 0 and brake > 0 simultaneously

### Visualisation 2 — Driver Archetype Cards

A rule-based classifier assigns each selected driver a driving style archetype based on their normalised radar profile. Rendered as styled cards below the radar.

| Archetype | Trigger conditions |
|---|---|
| 💥 Aggressive Braker | High brake frequency and high brake pressure |
| 🎯 Trail Braking Specialist | High trail braking score and high steering variance |
| 🚀 High Entry Speed | High average speed and low coasting fraction |
| 🧊 Smooth Operator | High full-throttle % and low throttle variation |
| ⚙️ Technical Driver | High gear-shift rate and high steering variance |
| ⚖️ Balanced Driver | No dominant dimension |

### Visualisation 3 — Speed Profile Comparison

A multi-driver line chart showing **mean speed at every one of 200 normalised distance points** along the lap. Hovering shows all driver speeds at the same point simultaneously (`hovermode="x unified"`). Directly reveals where each driver goes faster — in corners, under braking, and on straights.

### Visualisation 4 — Lap-by-Lap Consistency Heatmap

A drivers × features heatmap coloured by cross-lap standard deviation. Red = high variance = inconsistent. Each cell shows the raw std dev value. Answers a question the mean-based radar cannot: *who is erratic under race pressure?*

### Visualisation 5 — Lap Zone Performance

The 200-point lap is split into three equal distance zones (0–33%, 33–67%, 67–100%). A grouped bar chart shows per-zone performance. A channel switcher toggles between Speed (km/h), Throttle (%), and Braking Intensity — revealing which driver dominates which phase of the lap.

### Visualisation 6 — Throttle Application Map

The circuit outline (reconstructed from FastF1 XY coordinates stored per lap) is coloured by average throttle value — green for full throttle, red for coasting or braking. One panel per selected driver. Shows *where on track* each driver's throttle behaviour differs, turning an abstract percentage into a spatial fingerprint.

---

## Tab 1 — Agentic AI Features

Three distinct Agentic AI patterns, each architecturally different from the others and from the features in Tabs 2 and 3.

### AI Feature 1 — Reflexion Loop: Driver Style Analyst

**Pattern: Reflexion (Shinn et al. 2023) — iterative self-improvement via LLM self-critique**

```
Analyst LLM
  → generates a driving style narrative from radar data
      ↓
Critic LLM
  → evaluates the narrative against the raw data
  → returns JSON: {confidence: 1-10, factual_errors: [...], suggested_improvements: [...]}
      ↓
  if confidence ≥ 7  →  accept narrative
  if confidence < 7  →  Analyst receives critique and rewrites
      ↓
Critic LLM (round 2)
  → re-evaluates the revised narrative
      ↓
Final narrative accepted
```

The dashboard renders each round transparently in collapsible expanders — showing the critic's confidence score, any flagged factual errors, and the revision side by side. The user can see the self-improvement process live.

The Critic is called with `temperature=0.0` (deterministic JSON) while the Analyst uses `temperature=0.4`. Parse failure on the Critic response gracefully accepts the existing narrative rather than failing.

**Token budget:** Concise mode — 400 tokens (analyst). Detailed mode — 650 tokens (analyst); Critic unchanged at 300 tokens in both modes.

**This pattern is used in production by:** Reflexion-based code-generation agents, automated essay revision systems, LLM-based test-case refinement pipelines.

---

### AI Feature 2 — RAG: Historical Driver DNA Matching

**Pattern: Retrieval-Augmented Generation — without a vector database**

```
Driver's 12-dimensional radar vector (already computed)
     ↓
Cosine similarity vs. 10 curated historical F1 driver profiles
     ↓
Top-2 most similar profiles retrieved (the "knowledge base lookup")
     ↓
Retrieved profiles injected into LLM prompt as context
     ↓
GPT-4o-mini explains WHY the current driver's style aligns with the historical matches,
referencing specific dimension values from both profiles
```

The knowledge base is a curated dictionary of 10 F1 legends — Schumacher, Senna, Prost, Alonso, Vettel, Hamilton, Verstappen, Räikkönen, Sainz, and Rosberg — each encoded as a 12-dimensional normalised vector representing their well-documented driving characteristics. Retrieval uses cosine similarity directly on the feature vectors, eliminating the need for an embedding API or vector database.

A horizontal bar chart shows all 10 similarity scores (x-axis zoomed to 0.80–1.0 to surface fine-grained ranking differences), with the top-2 matches highlighted in gold. The LLM narration is anchored exclusively to the retrieved data — no speculation.

**Token budget:** Concise — 350 tokens. Detailed — 600 tokens.

**This pattern is used in production by:** Enterprise knowledge base Q&A systems, document retrieval pipelines, AI assistants that ground responses in proprietary data.

---

### AI Feature 3 — Structured Output: Driver DNA Report Card

**Pattern: JSON-mode structured generation + schema validation + retry-on-failure**

```
GPT-4o-mini called with response_format={"type": "json_object"}
  → enforced to return a JSON object matching a strict schema
      ↓
Python validation layer checks every key, type, length constraint, and value range
  → integers coerced from floats (e.g. 7.0 → 7)
  → lists trimmed silently if too long
  → HTML stripped from all string values
      ↓
  if valid  →  render report card
  if invalid  →  retry once with the specific validation error injected back into the prompt
      ↓
Rendered as a structured UI card
```

The report card schema enforces: `headline` (1–2 sentences with ≥2 data references), `strengths` (exactly 3, each citing ≥2 metrics), `weaknesses` (exactly 2 with root-cause diagnosis), `race_craft_rating` / `raw_pace_rating` / `consistency_rating` (integers 1–10), `tactical_tips` (exactly 2 engineer-grade recommendations), `style_tags` (exactly 4 short labels).

The UI renders this as: a headline, pill-style tag badges, rating bars with `st.progress`, a strengths/weaknesses two-column layout, and `st.info` tip cards.

This is the only place in the codebase that uses `response_format={"type": "json_object"}` — a distinct call pattern from every other LLM call in the project.

**Token budget:** Concise — 500 tokens. Detailed — 700 tokens.

**This pattern is used in production by:** Structured data extraction pipelines, AI-generated reports, form-filling agents, API response generation.

---

## Tab 2 — Mystery Driver

### AI Explainer (top of tab)

The XAI explainer is displayed at the top of the Mystery Driver tab. Once a driver has been predicted, the **✨ Ask AI to Explain** button is immediately available without scrolling.

### ML Prediction

Select any lap from the dataset. Click **Identify Driver**. The XGBoost classifier returns a predicted driver and a full probability distribution across all drivers, visualised as a horizontal bar chart.

### XAI Explainer — AI Feature 4

**Pattern: Single-shot LLM narration of SHAP values**

After the model predicts, clicking **✨ Ask AI to Explain** triggers the explainability pipeline:

1. **SHAP inference** — `shap.TreeExplainer` computes per-feature SHAP values for the predicted sample at runtime, identifying each feature's contribution to the predicted driver class
2. **Percentile ranking** — each feature value is ranked against the full training dataset distribution
3. **Structured prompt** — SHAP magnitudes, feature values, and percentile ranks are assembled into a validated prompt. No raw user text ever enters the prompt.
4. **LLM narration** — `gpt-4o-mini` produces a plain-English explanation
5. **Session cache** — the explanation is stored in `st.session_state` and cleared when the user picks a different lap

**Token budget:** Concise — 300 tokens. Detailed — 500 tokens (with richer 2–3 paragraph system prompt covering feature-by-feature SHAP breakdown and driver technique fingerprinting).

**Example output (Concise):**
> *"The model identified this lap as Verstappen with 73% confidence. The key signals were unusually high brake_events at the 91st percentile and low steer_std at the 8th percentile — indicating aggressive, precisely timed braking with surgical steering control. max_speed at the 88th percentile further reinforced the high-speed commitment characteristic of his telemetry fingerprint."*

### Model Accuracy Panel

A dedicated section at the bottom of the Mystery Driver tab shows full model evaluation results loaded from `models/metrics.json`:

- **Top-line metrics** — CV Accuracy (±std dev), Train Accuracy, driver count, lap count
- **Per-fold CV scores** — collapsible expander showing each fold's accuracy with delta vs. the mean
- **Per-driver metrics table** — precision, recall, F1 score, and support per driver, colour-coded by F1 (green ≥ 90%, amber ≥ 70%, red < 70%)

These metrics are also summarised in the sidebar for quick reference.

---

## Tab 3 — Race Dashboard

Real-time and historical race analysis powered by the OpenF1 API. No API key required for the data.

### Race Intelligence Chat Agent — AI Feature 5 *(top of tab)*

The chat panel appears at the top of the Race Dashboard tab so it's immediately accessible. The input is a tab-scoped inline form rather than a floating page-level bar.

**Pattern: ReAct tool-calling agent (Reason + Act loop)**

A conversational AI analyst embedded in the Race Dashboard. Users ask natural language questions about the loaded race; the agent autonomously selects tools, executes them against real race data, and synthesises a data-grounded answer.

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
If loop exhausted without text answer → forced synthesis pass
     ↓
Plain-prose answer rendered in chat UI
```

**6 tools exposed to the agent:**

| Tool | Calls | Returns |
|---|---|---|
| `get_rolling_pace` | `analyser.rolling_pace(window)` | Per-driver rolling-average lap times |
| `get_gap_to_leader` | `analyser.gap_to_leader()` | Cumulative gap per driver to the leader |
| `detect_strategy_events` | `analyser.detect_undercuts()` | Undercut / overcut detections |
| `project_finishing_order` | `analyser.project_finishing_order(laps_remaining)` | Projected final classification |
| `get_tyre_degradation` | `analyser.tyre_degradation()` | Degradation rate per driver/compound |
| `get_pace_summary` | `analyser.pace_summary()` | Mean, median, fastest lap per driver |

**Token budget:** Concise — 500 tokens. Detailed — 700 tokens (6–10 sentences with citations, forward-looking close, bold key numbers).

**Example questions:**
> *"Who has the best pace over the last 10 laps?"*
> *"Is Verstappen likely to undercut Norris? What does the gap look like?"*
> *"Which driver has the worst tyre degradation on softs?"*
> *"Who is projected to win with 20 laps remaining?"*

**Security guardrails (11 layers):**

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

### Fastest Lap Telemetry Comparison

Five channels comparing two drivers over their fastest recorded lap, fetched live from OpenF1:

| Channel | What it shows |
|---|---|
| **Time Delta** | Cumulative time gap through the lap. Shaded regions show where each driver is gaining. |
| **Track Map** | Circuit outline with circle microsectors (~999 segments) coloured by faster driver. Includes braking zone overlays (hollow red rings), full-throttle overlays (hollow green rings), auto-detected corner labels (T1–T12), a start/finish marker, a lap time subtitle showing both drivers' times and the gap, and rich per-microsector hover showing speed, throttle, and brake for both drivers. |
| **Speed** | Overlaid speed traces (km/h) vs. distance |
| **Throttle** | Overlaid throttle application (%) vs. distance |
| **Brake** | Overlaid brake pressure (%) vs. distance |

### 7 Race Analysis Charts

| Chart | What it shows |
|---|---|
| **Rolling Race Pace** | 5-lap rolling-average lap time per driver. Pit-out and safety car laps filtered automatically. |
| **Gap to Leader** | Cumulative time gap to the race leader per lap, shaded per driver. |
| **Undercut / Overcut Alerts** | Triangle markers on the gap chart at the exact lap a pit strategy caused a position swap. |
| **Projected Finishing Order** | Projected final gaps accounting for current pace, tyre degradation, and DNF flags. |
| **Average Race Pace** | Bars per driver coloured by consistency (Viridis scale). Hover shows mean, median, std dev, fastest lap, lap count. |
| **Race Pace Ranking** | Bars sorted fastest to slowest with ±std dev error bars. |
| **Tyre Degradation by Compound** | Scatter of mean stint pace vs. degradation rate (s/lap) from per-stint linear regression. |

**Live mode** — connects to the OpenF1 live feed during an active F1 race weekend. Auto-refreshes at 10s / 30s / 60s intervals.

**Historical mode** — analyse any race from 2022 onwards from the OpenF1 archive.

---

## AI Response Mode Toggle

A sidebar radio button lets users switch all five AI features between two response depths simultaneously:

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

Switching modes does not change the data passed to the model — only the system prompt and token budget change. All existing session state and caching behaviour is preserved.

---

## Agentic AI Patterns Summary

Five AI features across the three tabs — each implementing a distinct real-world pattern:

| Feature | Tab | Pattern | Key technique |
|---|---|---|---|
| Driver Style Analyst | Radar | **Reflexion** | Analyst → Critic (JSON) → conditional revision loop |
| Historical DNA Matching | Radar | **RAG** | Cosine similarity retrieval → context-augmented LLM prompt |
| Driver DNA Report Card | Radar | **Structured Output** | JSON mode + Python schema validation + retry-on-failure |
| XAI Prediction Explainer | Mystery Driver | **Single-shot narration** | SHAP values + percentile ranks → validated prompt → prose |
| Race Intelligence Chat | Race Dashboard | **ReAct** | Tool-calling loop, multi-round, forced synthesis fallback |

---

## Machine Learning Pipeline

### 1. Data extraction (`src/pipeline.py`)

FastF1 pulls high-frequency telemetry from the F1 live timing feed. Invalid laps (in-laps, out-laps, deleted times) are excluded. Each valid lap is resampled to **200 evenly spaced distance-based points** via linear interpolation — normalising every lap to the same resolution regardless of circuit length or car speed.

### 2. Feature engineering (`src/pipeline.py`)

Each lap is compressed into **10 scalar driving-style features** plus three 200-point trace arrays:

| Feature | What it captures |
|---|---|
| `lap_time_seconds` | Overall pace |
| `mean_speed`, `max_speed`, `min_speed` | Speed profile across the lap |
| `throttle_mean`, `throttle_std` | Throttle aggression and smoothness |
| `brake_mean`, `brake_events` | Braking intensity and frequency |
| `gear_changes` | Shift frequency — mechanical aggression |
| `steer_std` | Steering smoothness |
| `speed_trace`, `throttle_trace`, `brake_trace` | Full 200-point traces (used by radar visualisations) |
| `x_trace`, `y_trace` | Circuit XY coordinates at each trace point (used by throttle map) |

Three additional features are derived at runtime from the 200-point traces for the Extended Radar:

| Derived feature | Computation |
|---|---|
| `coasting_pct` | Fraction of points where throttle < 2% and brake < 1% |
| `throttle_pickup_pct` | Fraction of points where throttle > 80% |
| `trail_brake_score` | Fraction of points where throttle > 0 and brake > 0 simultaneously |

### 3. XGBoost classifier (`src/model.py`)

A gradient-boosted tree multi-class classifier trained with **5-fold stratified cross-validation**. Each F1 driver is one class. SHAP `TreeExplainer` generates both global feature importance (saved as `models/shap_importance.png` at training time) and local per-prediction explanations at inference time.

After training, `evaluate_model` saves `models/metrics.json` with:
- CV mean accuracy and standard deviation
- Per-fold accuracy scores
- Full-dataset train accuracy
- Per-driver precision, recall, F1 score, and support (lap count)
- Total sample and driver counts

### 4. Circuit geometry store (`src/generate_circuits.py`)

A one-off script extracts XY circuit outlines for all 24 Grand Prix from FastF1 and writes them to `data/circuits.json` at **500 resampled distance points** per circuit. Resume-safe — skips circuits already written.

---

## Data Sources

| Source | What it provides |
|---|---|
| **FastF1** | High-frequency car telemetry (speed, throttle, brake, steering, gear, RPM) per lap — used to build the Driver DNA training dataset and circuit geometry store |
| **OpenF1 API** | Live and historical race data — lap times, tyre stints, track positions, car telemetry, driver/team metadata. No API key required. |
| **OpenAI API** | `gpt-4o-mini` via Chat Completions — powers all five AI features across the dashboard |

---

## Project Structure

```
driver-dna/
├── src/
│   ├── app.py                   # Streamlit dashboard — 3 tabs, 5 AI features, 13+ chart types
│   ├── pipeline.py              # Data extraction, telemetry resampling, feature engineering
│   ├── model.py                 # XGBoost training, CV evaluation, SHAP analysis, metrics.json
│   ├── race_engine.py           # Pace analytics, gap tracking, undercut detection, projections
│   ├── openf1.py                # OpenF1 REST API client (live + historical modes)
│   └── generate_circuits.py     # One-off: generates data/circuits.json
├── data/                        # Git-ignored — generated locally
│   ├── dataset.parquet          # Driver DNA training data (pipeline.py output)
│   ├── dataset_meta.json        # Session metadata — GP name, year, session type
│   └── circuits.json            # Circuit XY outlines for Track Map
├── models/                      # Git-ignored — generated locally
│   ├── driver_dna_clf.joblib    # Trained XGBoost classifier
│   ├── label_encoder.joblib     # sklearn LabelEncoder
│   ├── accuracy.txt             # Cross-validation accuracy (plain float)
│   ├── metrics.json             # Detailed metrics: CV scores, train accuracy, per-driver F1
│   ├── confusion_matrix.html    # Per-driver confusion matrix heatmap
│   └── shap_importance.png      # Global SHAP feature importance chart
├── .streamlit/
│   ├── config.toml              # Dark theme, F1 red (#E8002D)
│   └── secrets.toml             # OpenAI API key — gitignored, never commit
├── streamlit_app.py             # Streamlit Community Cloud entry point
├── requirements.txt
└── README.md
```

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

> **Always activate the project venv first.** Running with a system Python (e.g. Anaconda) causes `ImportError: numpy.core.multiarray failed to import` due to numpy/pandas version mismatches. Verify with `which python` — it should point to `.venv/bin/python`.

### 2. Configure the OpenAI API key

All five AI features require an OpenAI paid API account. Create `.streamlit/secrets.toml` (this file is gitignored):

```toml
[openai]
api_key = "sk-proj-..."
```

Or export as an environment variable:

```bash
export OPENAI_API_KEY="sk-proj-..."
```

For Streamlit Community Cloud deployment, add the key via the **Secrets** panel in the app settings instead of a local file.

### 3. Generate circuit outlines *(required for Track Map and Throttle Map)*

Downloads XY geometry for all 24 Grand Prix circuits and writes `data/circuits.json`. Run once — takes 10–30 minutes. Resume-safe.

```bash
python src/generate_circuits.py
```

### 4. Build the Driver DNA dataset

Downloads session data from the F1 timing feed, engineers features, and writes `data/dataset.parquet` and `data/dataset_meta.json`. Required for the Driver Radar and Mystery Driver tabs.

```bash
python src/pipeline.py
```

### 5. Train the classifier

Runs 5-fold stratified CV, saves the model, label encoder, accuracy, confusion matrix, SHAP importance chart, and detailed `models/metrics.json` to `models/`:

```bash
python src/model.py
```

The terminal will print per-fold accuracy, mean CV accuracy ±std dev, full-dataset train accuracy, and a complete per-driver classification report.

### 6. Launch the dashboard

```bash
streamlit run src/app.py
```

> The Race Dashboard and Race Intelligence Chat Agent work independently of steps 4–5 — they query the OpenF1 API directly. Live mode only works during active F1 race weekends; historical mode works any time from 2022 onwards.

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
`models/metrics.json` not yet generated. Re-run `python src/model.py`. If the panel still does not appear, `models/accuracy.txt` is used as a fallback in the sidebar.

**AI features: "OpenAI rate limit reached"**
The OpenAI API requires a paid account. Free-tier accounts have no API access — enable billing at `platform.openai.com/settings/billing`.

**AI features: "OpenAI API key not configured"**
`.streamlit/secrets.toml` is missing or the placeholder key was not replaced. See Setup step 2.

**Mystery Driver tab shows "—" for Grand Prix / Season / Session**
`data/dataset_meta.json` was not generated. Re-run `python src/pipeline.py`.

**Driver Radar AI features: no buttons visible**
The three agentic AI features (Reflexion Analyst, DNA Matching, Report Card) only appear once 2–4 drivers are selected in the multiselect and an OpenAI API key is configured.

---

## Key Findings

> Run the full pipeline and fill these in with your results.

- **Finding 1** — which feature had the highest mean SHAP value across all drivers, and what driving behaviour it reflects
- **Finding 2** — which pair of drivers the model most often confused, and a data-backed explanation
- **Finding 3** — overall CV accuracy vs. random-chance baseline (1 / number of drivers)
