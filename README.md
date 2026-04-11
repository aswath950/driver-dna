# Driver DNA + Race Intelligence Dashboard

**Can a machine learning model identify an F1 driver purely from their telemetry — without ever seeing their name?**

Driver DNA is a full-stack AI/ML platform built on F1 telemetry data. It combines an XGBoost driver-identification model with an Explainable AI layer, a real-time race analytics engine, and two Agentic AI features powered by GPT-4o-mini — all served through an interactive Streamlit dashboard.

---

## Tech Stack

| Category | Technologies |
|---|---|
| **Machine Learning** | XGBoost (multi-class classification), scikit-learn, SHAP (Explainable AI), 5-fold stratified cross-validation |
| **Agentic AI / LLM** | OpenAI API (`gpt-4o-mini`), OpenAI tool calling (function calling), multi-step agent orchestration, prompt engineering |
| **Data Engineering** | FastF1, pandas, NumPy, Apache Parquet, time-series resampling, distance-normalised feature extraction |
| **Visualisation** | Streamlit, Plotly (spider/radar charts, time-series, scatter, bar, track maps) |
| **APIs** | OpenF1 REST API (live + historical), OpenAI Chat Completions API |
| **MLOps** | joblib model serialisation, SHAP feature importance, confusion matrix evaluation |
| **Security** | Prompt injection prevention, input sanitisation, API key management (`st.secrets`), tool name allowlisting, rate limiting, LLM response sanitisation |

---

## Agentic AI Features

### 🤖 Feature 1 — Race Intelligence Chat Agent

A conversational AI analyst embedded directly in the Race Dashboard. Users ask natural language questions about any loaded race and the agent autonomously decides which analytical tools to invoke, executes them against live race data, and synthesises a data-grounded answer.

**Architecture — OpenAI Tool Calling (Function Calling):**

```
User question
     ↓
GPT-4o-mini decides which tools to call
     ↓
App executes tools against RaceAnalyser (real data)
     ↓
Results returned to GPT-4o-mini
     ↓
GPT-4o-mini synthesises a natural language answer
     ↓
Answer displayed in chat UI
```

This is the same pattern used in enterprise AI platforms (Salesforce Einstein, GitHub Copilot Workspace, Palantir AIP).

**6 tools exposed to the agent:**

| Tool | Calls | Returns |
|---|---|---|
| `get_rolling_pace` | `analyser.rolling_pace(window)` | Per-driver rolling-average lap times |
| `get_gap_to_leader` | `analyser.gap_to_leader()` | Cumulative gap per driver to the leader |
| `detect_strategy_events` | `analyser.detect_undercuts()` | Undercut / overcut detections |
| `project_finishing_order` | `analyser.project_finishing_order(laps_remaining)` | Projected final classification |
| `get_tyre_degradation` | `analyser.tyre_degradation()` | Degradation rate per driver/compound |
| `get_pace_summary` | `analyser.pace_summary()` | Mean, median, fastest lap per driver |

**Example interactions:**
> *"Who has the best pace over the last 10 laps?"*
> *"Is Verstappen likely to undercut Norris? What does the gap look like?"*
> *"Which driver has the worst tyre degradation on softs?"*
> *"Who is projected to win with 20 laps remaining?"*

**Security guardrails (11 layers):**

| # | Threat | Mitigation |
|---|---|---|
| 0 | Prompt injection via long/malicious input | `_ca_sanitize_input()` — strips control chars, 500-char limit |
| 1 | Out-of-range LLM-supplied tool args | `_ca_validate_tool_args()` — schema validation + clamping |
| 2 | HTML/link injection via LLM response | `st.text()` renders plain text only |
| 3 | Unknown tool names from LLM | `_CA_ALLOWED_TOOLS` explicit allowlist |
| 4 | Tool call flood attack | `_CA_MAX_TOOLS_PER_ROUND = 3` cap |
| 5 | History poisoning via long responses | Per-message 2000-char truncation |
| 6 | Hardcoded API key | `_get_openai_api_key()` reads `st.secrets` / env var only |
| 7 | Infinite agent loop | `MAX_TOOL_ROUNDS = 3` hard cap |
| 8 | Context window / cost blowout | 20-message history cap + `max_tokens=500` |
| 9 | API spam / cost abuse | 5-second per-session rate limiter |
| 10 | Corrupt state on failed turn | User message popped from history on API error |

---

### ✨ Feature 2 — XAI Prediction Explainer

After the Mystery Driver ML prediction, an AI explanation pipeline runs automatically when the user clicks **✨ Ask AI to Explain**. It bridges classical ML explainability (SHAP) with generative AI narration.

**Pipeline:**
1. **SHAP inference** — `shap.TreeExplainer` computes per-feature SHAP values for the single predicted sample at runtime, identifying each feature's contribution to the predicted driver class
2. **Percentile ranking** — each feature value is ranked against the full training dataset distribution
3. **Structured prompt** — SHAP values, feature magnitudes, and percentile ranks are assembled into a validated prompt (no raw user text enters the prompt)
4. **LLM narration** — `gpt-4o-mini` produces a plain-English explanation via the Chat Completions API
5. **Session cache** — explanation is stored in `st.session_state` to avoid repeat API calls

**Example output:**
> *"The model identified this as Verstappen with 73% confidence. The key signals were unusually high brake_events at the 91st percentile and low steer_std at the 8th percentile — indicating aggressive, precisely timed braking with surgical steering control. max_speed at the 88th percentile further reinforced the high-speed commitment characteristic of his telemetry fingerprint."*

---

## Machine Learning Pipeline

### 1. Data extraction (`src/pipeline.py`)
FastF1 pulls high-frequency telemetry from the F1 live timing feed. Invalid laps (in-laps, out-laps, deleted times) are excluded. Each valid lap is resampled to **200 evenly spaced distance-based points** via linear interpolation — normalising every lap to the same resolution regardless of circuit length or car speed.

### 2. Feature engineering (`src/pipeline.py`)
Each lap is compressed into **10 scalar driving-style features:**

| Feature | What it captures |
|---|---|
| `lap_time_seconds` | Overall pace |
| `mean_speed`, `max_speed`, `min_speed` | Speed profile across the lap |
| `throttle_mean`, `throttle_std` | Throttle aggression and smoothness |
| `brake_mean`, `brake_events` | Braking intensity and frequency |
| `gear_changes` | Shift frequency — mechanical aggression |
| `steer_std` | Steering smoothness |

### 3. XGBoost classifier (`src/model.py`)
A gradient-boosted tree multi-class classifier trained with **5-fold stratified cross-validation**. Each F1 driver is one class. SHAP `TreeExplainer` generates both global feature importance (saved as a chart at training time) and local per-prediction explanations at inference time.

### 4. Circuit geometry store (`src/generate_circuits.py`)
A one-off script extracts XY circuit outlines for all 24 Grand Prix from FastF1 and writes them to `data/circuits.json` at **500 resampled distance points** per circuit. Resume-safe — skips circuits already written.

---

## Race Dashboard

Real-time and historical race analysis powered by the OpenF1 API. No API key required.

### Fastest Lap Telemetry Comparison

Five channels comparing two drivers over their fastest recorded lap, fetched live from OpenF1:

| Channel | What it shows |
|---|---|
| **Time Delta** | Cumulative time gap through the lap. Shaded regions show where each driver is gaining. |
| **Track Map** | Circuit outline coloured by faster driver per microsector (~999 segments). |
| **Speed** | Overlaid speed traces (km/h) vs. distance |
| **Throttle** | Overlaid throttle application (%) vs. distance |
| **Brake** | Overlaid brake pressure (%) vs. distance |

### 7 Race Analysis Charts

| Chart | What it shows |
|---|---|
| **Rolling Race Pace** | 5-lap rolling-average lap time per driver. Pit-out and safety car laps filtered automatically. |
| **Gap to Leader** | Cumulative time gap to the race leader per lap, shaded per driver. |
| **Undercut / Overcut Alerts** | Triangle markers on the gap chart at the exact lap a pit strategy caused a position swap. |
| **Projected Finishing Order** | Projected final gaps accounting for current pace, tyre degradation (+0.07 s/lap after 20 laps), and DNF flags. |
| **Average Race Pace** | Bars per driver coloured by consistency (Viridis scale). Hover shows mean, median, std dev, fastest lap, lap count. |
| **Race Pace Ranking** | Bars sorted fastest to slowest with ±std dev error bars. |
| **Tyre Degradation by Compound** | Scatter of mean stint pace vs. degradation rate (s/lap) from per-stint linear regression. |

**Live mode** — connects to the OpenF1 live feed during an active F1 race weekend. Auto-refreshes at 10s / 30s / 60s intervals.

**Historical mode** — analyse any race from 2022 onwards from the OpenF1 archive.

---

## Data Sources

| Source | What it provides |
|---|---|
| **FastF1** | High-frequency car telemetry (speed, throttle, brake, steering, gear, RPM) per lap — used to build the Driver DNA training dataset and circuit geometry store |
| **OpenF1 API** | Live and historical race data — lap times, tyre stints, track positions, car telemetry, driver/team metadata. No API key required. |
| **OpenAI API** | `gpt-4o-mini` via Chat Completions — powers both the XAI explainer and the Race Intelligence chat agent |

---

## Project Structure

```
driver-dna/
├── src/
│   ├── app.py                   # Streamlit dashboard — 3 tabs, XAI explainer, chat agent
│   ├── pipeline.py              # Data extraction, telemetry resampling, feature engineering
│   ├── model.py                 # XGBoost training, CV evaluation, SHAP analysis
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
│   ├── accuracy.txt             # Cross-validation accuracy
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

Both AI features (XAI Explainer and Race Intelligence Chat) require an OpenAI paid API account. Create `.streamlit/secrets.toml` (this file is gitignored):

```toml
[openai]
api_key = "sk-proj-..."
```

Or export as an environment variable:

```bash
export OPENAI_API_KEY="sk-proj-..."
```

For Streamlit Community Cloud deployment, add the key via the **Secrets** panel in the app settings instead of a local file.

### 3. Generate circuit outlines *(required for Track Map)*

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

Runs 5-fold stratified CV, saves the model, label encoder, accuracy, confusion matrix, and SHAP importance chart to `models/`:

```bash
python src/model.py
```

### 6. Launch the dashboard

```bash
streamlit run src/app.py
```

> The Race Dashboard and Race Intelligence chat agent work independently of steps 4–5 — they query the OpenF1 API directly. Live mode only works during active F1 race weekends; historical mode works any time from 2022 onwards.

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

**Track Map shows "No circuit data"**
`data/circuits.json` not yet generated. Run `python src/generate_circuits.py`.

**Race Intelligence / XAI Explainer: "OpenAI rate limit reached"**
The OpenAI API requires a paid account. Free-tier accounts have no API access — enable billing at `platform.openai.com/settings/billing`.

**Race Intelligence / XAI Explainer: "OpenAI API key not configured"**
`.streamlit/secrets.toml` is missing or the placeholder key was not replaced. See Setup step 2.

**Mystery Driver tab shows "—" for Grand Prix / Season / Session**
`data/dataset_meta.json` was not generated (it is written by `pipeline.py` alongside `dataset.parquet`). Re-run `python src/pipeline.py` to generate it.

---

## Key Findings

> Run the full pipeline and fill these in with your results.

- **Finding 1** — which feature had the highest mean SHAP value across all drivers, and what driving behaviour it reflects
- **Finding 2** — which pair of drivers the model most often confused, and a data-backed explanation
- **Finding 3** — overall CV accuracy vs. random-chance baseline (1 / number of drivers)
