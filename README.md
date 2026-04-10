# Driver DNA + Race Dashboard

**Can a machine learning model tell two F1 drivers apart from their telemetry alone — without ever seeing their name?**

Driver DNA is an end-to-end ML system that extracts lap telemetry from FastF1, engineers driving-style features, and trains an XGBoost classifier to identify F1 drivers purely from how they brake, accelerate, steer, and shift. It is paired with a real-time race analytics dashboard powered by the OpenF1 API, and an **Agentic AI layer** that uses GPT-4o-mini to generate natural language explanations of model predictions — combining Explainable AI (XAI) with Generative AI.

---

## Tech Stack

| Category | Technologies |
|---|---|
| **Machine Learning** | XGBoost, scikit-learn, SHAP (Explainable AI) |
| **Generative AI / LLM** | OpenAI API (`gpt-4o-mini`), prompt engineering, structured tool output |
| **Data Engineering** | FastF1, pandas, NumPy, Parquet, feature engineering, time-series resampling |
| **Visualisation** | Streamlit, Plotly (interactive charts, radar/spider charts, scatter, bar) |
| **APIs** | OpenF1 REST API (live + historical), OpenAI Chat Completions API |
| **ML Ops** | joblib model serialisation, 5-fold stratified cross-validation, SHAP feature importance |
| **Security** | Input validation, prompt injection prevention, API key management (`st.secrets`), rate limiting |

---

## Features

### 🧬 Driver DNA Fingerprinter

Extracts telemetry from qualifying/race sessions, engineers 10 scalar driving-style features per lap, and trains an XGBoost multi-class classifier to identify drivers. The Streamlit dashboard includes:

- **Driver Radar** — normalised spider chart of driving-style features for 2–4 drivers side by side
- **Mystery Driver** — pick any lap, let the model predict the driver, then reveal whether it was correct

### 🤖 AI-Powered Prediction Explainer (Agentic AI)

After any Mystery Driver prediction, an **"Explain this Prediction"** button triggers an Agentic AI pipeline:

1. **SHAP inference** — computes per-feature SHAP values at runtime using `shap.TreeExplainer`, isolating the contribution of each driving-style feature to the predicted class
2. **Percentile context** — ranks the lap's feature values against the full dataset distribution
3. **Prompt construction** — builds a structured prompt embedding the SHAP values, feature magnitudes, and percentile ranks — all pre-validated inputs, no raw user text
4. **LLM generation** — calls OpenAI `gpt-4o-mini` via the Chat Completions API to produce a concise natural language explanation

Example output:
> *"The model identified this as Verstappen with 73% confidence. The key signals were unusually high brake_events (91st percentile) and low steer_std (8th percentile), indicating aggressive, precise braking with surgical steering control. max_speed at the 88th percentile added further evidence of the high-speed commitment typical of his style."*

**Security guardrails implemented:**
- API key loaded exclusively from `st.secrets` / environment variable — never hardcoded
- Driver names validated against the model's class list before prompt interpolation (prompt injection prevention)
- Feature values checked for finite floats within domain-specific bounds before the API call
- All OpenAI exception types caught and mapped to user-friendly messages (raw errors never surfaced)
- 10-second session-level rate limiter with a disabled button + countdown label
- `max_tokens=300` hard cap to prevent runaway API costs

### 🏁 Race Dashboard

Real-time and historical race analysis powered by the OpenF1 API (no API key required).

**Fastest Lap Telemetry Comparison** — five channels comparing two drivers over their single fastest recorded lap:

| Channel | What it shows |
|---|---|
| **Time Delta** | Cumulative time gap through the lap. Shaded regions show where each driver is gaining. |
| **Track Map** | Circuit coloured by faster driver per microsector (~999 segments). Geometry from pre-generated `data/circuits.json`. |
| **Speed** | Overlaid speed traces (km/h) vs. distance |
| **Throttle** | Overlaid throttle application (%) vs. distance |
| **Brake** | Overlaid brake pressure (%) vs. distance |

**Live mode** — connects to the OpenF1 live feed during an active F1 session. Auto-refreshes at a configurable interval (10s / 30s / 60s). Only active during race weekends.

**Historical mode** — analyse any race from 2022 onwards from the OpenF1 archive.

**7 race analysis charts:**

| Chart | What it shows |
|---|---|
| **Rolling Race Pace** | 5-lap rolling-average lap time. Pit-out laps and safety car laps filtered automatically. |
| **Gap to Leader** | Cumulative time gap to the race leader per lap, shaded per driver. |
| **Undercut / Overcut Alerts** | Triangle markers on the gap chart where pit strategy caused a position swap. |
| **Projected Finishing Order** | Projected final gaps accounting for current pace, tyre degradation (+0.07 s/lap after 20 laps), and DNF flags. |
| **Average Race Pace** | Horizontal bars per driver, coloured by consistency (Viridis scale). Hover shows mean, median, std dev, fastest lap, lap count. |
| **Race Pace Ranking** | Vertical bars sorted fastest to slowest with ± std dev error bars. |
| **Tyre Degradation by Compound** | Scatter of mean stint pace vs. degradation rate (s/lap) from linear regression, coloured by compound. |

---

## How It Works

### 1. Data extraction (`src/pipeline.py`)
FastF1 pulls high-frequency telemetry from the F1 timing feed. Invalid laps (in-laps, out-laps, deleted times) are dropped. Each lap is resampled to **200 evenly spaced distance-based points** via linear interpolation — normalising every lap to the same length regardless of circuit or speed.

### 2. Feature engineering (`src/pipeline.py`)
Each lap is compressed into **10 scalar features** capturing driving style:

| Feature | What it captures |
|---|---|
| `lap_time_seconds` | Overall pace |
| `mean_speed`, `max_speed`, `min_speed` | Speed profile across the lap |
| `throttle_mean`, `throttle_std` | Throttle aggression and smoothness |
| `brake_mean`, `brake_events` | Braking intensity and frequency |
| `gear_changes` | Shift frequency (mechanical aggression) |
| `steer_std` | Steering smoothness |

### 3. XGBoost classifier + SHAP (`src/model.py`)
A gradient-boosted tree classifier is trained with **5-fold stratified cross-validation**. Each driver is one class. SHAP `TreeExplainer` identifies which features drive each prediction — both globally (training-time importance chart) and locally (per-prediction at inference time for the LLM explainer).

### 4. Agentic AI explainer (`src/app.py`)
At prediction time, `shap.TreeExplainer` computes SHAP values for the single input sample. These values, along with percentile ranks against the full dataset, are assembled into a structured prompt and sent to `gpt-4o-mini` via the OpenAI Chat Completions API. The response is cached in Streamlit session state to avoid repeat API calls.

### 5. Circuit geometry store (`src/generate_circuits.py`)
A one-off script that uses FastF1 to extract XY circuit outlines for all 24 Grand Prix and writes them to `data/circuits.json` at **500 evenly resampled distance points** per circuit. Resume-safe — skips circuits already written.

### 6. Streamlit dashboard (`src/app.py`)
Three interactive tabs:

| Tab | Contents |
|---|---|
| **Driver Radar** | Normalised driving-style spider chart for 2–4 drivers |
| **Mystery Driver** | ML prediction with probability breakdown + GPT-4o-mini XAI explanation |
| **Race Dashboard** | Fastest Lap Telemetry (5 channels, live OpenF1) + 7 race analysis charts |

---

## Data Sources

| Source | What it provides |
|---|---|
| **FastF1** | High-frequency car telemetry (speed, throttle, brake, steering, gear, RPM) per lap — used to build the Driver DNA training dataset and circuit geometry |
| **OpenF1 API** | Live and historical race data — lap times, tyre stints, track positions, car telemetry, driver/team metadata. No API key required. |
| **OpenAI API** | `gpt-4o-mini` Chat Completions for natural language prediction explanations |

> `data/circuits.json` and `data/dataset.parquet` are git-ignored and must be generated locally. See Setup below.

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd driver-dna
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Always use the project venv.** Running with a system Python (e.g. Anaconda) causes `ImportError: numpy.core.multiarray failed to import` due to numpy/pandas version conflicts. Verify with `which python` — it should point to `.venv/bin/python`.

### 2. Configure the OpenAI API key

The LLM explainer requires an OpenAI API key with billing enabled (paid tier). Create `.streamlit/secrets.toml`:

```toml
[openai]
api_key = "sk-proj-..."
```

This file is gitignored. For Streamlit Community Cloud deployment, use the **Secrets** panel in the app settings instead of a local file.

Alternatively, export as an environment variable:

```bash
export OPENAI_API_KEY="sk-proj-..."
```

### 3. Generate circuit outlines (required for Track Map)

Downloads XY geometry for all 24 Grand Prix circuits and writes `data/circuits.json`. Run once — takes 10–30 minutes due to FastF1 downloads. Resume-safe.

```bash
python src/generate_circuits.py
```

### 4. Build the Driver DNA dataset

Downloads session data from the F1 timing feed and writes `data/dataset.parquet`. Required for Driver Radar and Mystery Driver tabs.

```bash
python src/pipeline.py
```

### 5. Train the classifier

Runs 5-fold stratified CV, saves the model, label encoder, accuracy, confusion matrix, and SHAP importance plot to `models/`:

```bash
python src/model.py
```

### 6. Launch the dashboard

```bash
streamlit run src/app.py
```

> The Race Dashboard live mode only works during active F1 race weekends. Historical mode works any time from 2022 onwards. The Race Dashboard charts query the OpenF1 API directly and do not require steps 4–5.

---

## Project Structure

```
driver-dna/
├── data/                        # FastF1 cache + generated files (git-ignored)
│   ├── dataset.parquet          # Driver DNA training data (pipeline.py output)
│   └── circuits.json            # Circuit XY outlines for Track Map
├── models/                      # Saved model artefacts (git-ignored)
│   ├── driver_dna_clf.joblib    # Trained XGBoost classifier
│   ├── label_encoder.joblib     # sklearn LabelEncoder
│   ├── accuracy.txt             # Cross-validation accuracy
│   ├── confusion_matrix.html    # Per-driver confusion matrix heatmap
│   └── shap_importance.png      # Global SHAP feature importance chart
├── src/
│   ├── pipeline.py              # Data extraction & feature engineering (FastF1)
│   ├── model.py                 # XGBoost training, CV evaluation, SHAP analysis
│   ├── generate_circuits.py     # One-off: generate data/circuits.json
│   ├── openf1.py                # OpenF1 REST API client (live + historical)
│   ├── race_engine.py           # Pace analytics, tyre degradation, undercut detection
│   └── app.py                   # Streamlit dashboard (3 tabs + LLM explainer)
├── .streamlit/
│   ├── config.toml              # Dark theme, F1 red (#E8002D)
│   └── secrets.toml             # OpenAI API key — gitignored, never commit
├── streamlit_app.py             # Streamlit Community Cloud entry point
├── requirements.txt
└── README.md
```

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
Running with the wrong Python interpreter. In VS Code: `⇧⌘P` → **Python: Select Interpreter** → choose the `.venv` entry.

**`ModuleNotFoundError` when debugging**
`PYTHONPATH` must be set to `src/`. The `launch.json` sets this automatically. From the terminal: `source .venv/bin/activate` first.

**Track Map shows "No circuit data"**
`data/circuits.json` not yet generated. Run `python src/generate_circuits.py`.

**LLM explainer: "OpenAI rate limit reached"**
The OpenAI API requires a paid account. Free-tier accounts have no API access — billing must be enabled at `platform.openai.com/settings/billing`.

**LLM explainer: "OpenAI API key not configured"**
`.streamlit/secrets.toml` is missing or the key placeholder was not replaced. See Setup step 2.

---

## Key Findings

> Run the full pipeline and fill these in with your results.

- **Finding 1** — e.g. which feature had the highest mean SHAP value across all drivers, and what driving behaviour it reflects
- **Finding 2** — e.g. which pair of drivers the model most often confused, and a data-backed explanation
- **Finding 3** — e.g. overall CV accuracy vs. random-chance baseline (1 / number of drivers)
