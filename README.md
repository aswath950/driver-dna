# Driver DNA + Race Pace

**Can a machine learning model tell two F1 drivers apart from their telemetry alone — without ever seeing their name?**

Driver DNA Fingerprinter extracts lap telemetry from FastF1, engineers driving-style features, and trains an XGBoost classifier to identify drivers purely from how they brake, accelerate, steer, and shift — no car number, no team, no context.

---

## Features

### 🧬 Driver DNA Fingerprinter

Extracts telemetry from qualifying/race sessions, engineers 10 scalar driving-style features per lap, and trains an XGBoost classifier to identify drivers. The Streamlit dashboard includes:

- **Telemetry Overlay** — compare two drivers' average Speed, Throttle, or Brake trace across all their laps
- **Driver Radar** — spider chart of normalised style features for 2–4 drivers
- **Mystery Driver** — pick any lap, let the model guess the driver, then reveal whether it was right

### 🏁 Race Pace Dashboard

Real-time and historical race analysis powered by the OpenF1 API (no API key required).

**Live mode** — connects to the OpenF1 live feed during an active F1 session. Auto-refreshes at a configurable interval (10s / 30s / 60s) with a pulsing LIVE badge. Only works during active F1 race weekends.

**Historical mode** — analyse any race from 2022 onwards. Select a year and Grand Prix, and all charts load from the OpenF1 archive.

**7 analysis charts:**

| Chart | What it shows |
|---|---|
| **Rolling Race Pace** | Rolling-average lap time per driver over the last 5 laps. Filters out pit-out laps and safety car laps. A dashed line marks the session median. |
| **Gap to Leader** | Cumulative time gap to the race leader at each lap. Each driver's area is shaded at low opacity so gaps are easy to read at a glance. |
| **Undercut / Overcut Alerts** | Triangle markers overlaid on the gap chart at the exact lap where a position swap was caused by a pit strategy move. Hover for details. |
| **Projected Finishing Order** | Horizontal bars showing projected gap to the winner. Accounts for current pace, tyre degradation (+0.07s/lap beyond 20 laps on a compound), and flags DNF drivers. Bars are green (gaining), red (losing), or grey (holding). |
| **Average Race Pace** | Horizontal bars per driver sorted fastest to slowest, coloured by pace consistency (Viridis scale — yellow = consistent, dark = variable). Hover shows mean, median, std dev, fastest lap, and lap count. |
| **Race Pace Ranking** | Vertical bars sorted fastest to slowest with ± std dev error bars. Each driver has a distinct colour. Fastest lap time is annotated above each bar. |
| **Tyre Degradation by Compound** | Scatter plot of mean stint pace vs. degradation rate (s/lap) from linear regression. Coloured by compound: SOFT (red), MEDIUM (yellow), HARD (grey), INTERMEDIATE (green), WET (blue). A zero-line marks the no-degradation baseline. |

All driver labels display 3-letter acronyms (e.g. HAM, VER) rather than raw driver numbers.

---

## Data sources

| Source | What it provides | Link |
|---|---|---|
| **FastF1** | High-frequency car telemetry (speed, throttle, brake, steering, gear, RPM) per lap | https://docs.fastf1.dev |
| **OpenF1** | Live and historical race data — lap times, tyre stints, track positions | https://openf1.org |

---

## How it works

### 1. Data extraction (`src/pipeline.py`)
FastF1 pulls raw telemetry for each lap from the F1 timing feed. Only clean laps are kept — in-laps, out-laps, and laps with deleted times are dropped. Each valid lap's telemetry is resampled to **200 evenly spaced distance-based points** via linear interpolation, so every lap is the same length regardless of circuit or car speed.

### 2. Feature engineering (`src/pipeline.py`)
Each lap is compressed into **10 scalar features** that capture driving style:

| Feature | What it captures |
|---|---|
| `lap_time_seconds` | Overall pace |
| `mean_speed`, `max_speed`, `min_speed` | Speed profile |
| `throttle_mean`, `throttle_std` | Throttle aggression and smoothness |
| `brake_mean`, `brake_events` | Braking frequency and intensity |
| `gear_changes` | Shift frequency (mechanical aggression) |
| `steer_std` | Steering smoothness |

### 3. XGBoost classifier (`src/model.py`)
A gradient-boosted tree classifier is trained with **5-fold stratified cross-validation**. Each driver is one class. The model learns the combination of scalar features that makes each driver's style distinct. SHAP values reveal which features matter most.

### 4. Streamlit dashboard (`src/app.py`)
Four interactive tabs covering both the Driver DNA analysis and the Race Pace Dashboard described in the Features section above.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Important — always use the project venv.**
> If you have Anaconda or another Python installation, running scripts with the wrong interpreter causes `ImportError: numpy.core.multiarray failed to import` due to numpy/pandas version mismatches.
> Always run `source .venv/bin/activate` (or `.venv\Scripts\activate` on Windows) before any `python` command. You can verify you're in the right environment with `which python` — it should point to `.venv/bin/python`.

---

## Usage

### 1. Extract telemetry and build the dataset

Downloads the session from the F1 timing feed and writes `data/dataset.parquet`:

```bash
python src/pipeline.py
```

> First run will download and cache session data — this takes a few minutes.

### 2. Train the classifier

Runs 5-fold CV, saves the model, label encoder, accuracy, confusion matrix, and SHAP plot to `models/`:

```bash
python src/model.py
```

### 3. Launch the dashboard

```bash
streamlit run src/app.py
```

> **Note:** The Race Pace Dashboard's live mode only works during active F1 race weekends (practice, qualifying, or race sessions). Historical mode works any time for races from 2022 onwards.

---

## Debugging

A VS Code debug configuration is included at [`.vscode/launch.json`](.vscode/launch.json). It automatically uses the `.venv` interpreter and sets `PYTHONPATH` to `src/` so cross-module imports resolve correctly.

### Available debug configurations

| Configuration | What it debugs |
|---|---|
| **Streamlit: app.py** | Full dashboard with breakpoints. Opens on `localhost:8501`. |
| **Python: pipeline.py** | Interactive data extraction — breakpoints work inside `extract_session_telemetry`. |
| **Python: model.py** | Full train → evaluate → save pipeline with breakpoints. |
| **Python: race_engine.py** | Race engine module in isolation. |
| **Python: current file** | Debugs whichever file is currently open in the editor. |

### How to use

1. Open the project folder in VS Code.
2. Press `F5` (or open **Run → Start Debugging**) and select a configuration from the dropdown.
3. Set breakpoints by clicking the gutter to the left of any line number.

### Common issues

**`ImportError: numpy.core.multiarray failed to import`**
This means the script is running with a system Python (e.g. Anaconda) instead of the project venv. In VS Code, open the Command Palette (`⇧⌘P`), run **Python: Select Interpreter**, and choose the `.venv` entry. The `.vscode/settings.json` sets this automatically, but it can be overridden by a workspace-level selection. From the terminal always run `source .venv/bin/activate` first.

**`ModuleNotFoundError` when debugging**
The `PYTHONPATH` in `launch.json` is set to `src/`, which is required because `app.py` imports from `model`, `openf1`, and `race_engine` by short name. If you see import errors, confirm `.vscode/settings.json` points to `.venv/bin/python`.

**Streamlit reloads wipe breakpoints mid-session**
Streamlit's file-watcher reruns the script on save, which re-executes past any paused breakpoint. To prevent this, add `--server.fileWatcherType=none` to the `args` list in `launch.json` for the Streamlit configuration while debugging.

**pipeline.py waits for input when debugging**
`pipeline.py` uses `input()` prompts. These appear in the VS Code integrated terminal — click the terminal panel and type your responses there.

**FastF1 cache**
On first run, FastF1 downloads session data to `./data/`. This can take several minutes. Subsequent runs read from cache and are fast.

---

## Project structure

```
driver-dna/
├── data/               # FastF1 cache + dataset.parquet (git-ignored)
├── models/             # Saved model files (git-ignored)
│   ├── driver_dna_clf.joblib
│   ├── label_encoder.joblib
│   ├── accuracy.txt
│   ├── confusion_matrix.html
│   └── shap_importance.png
├── src/
│   ├── pipeline.py     # Data extraction & feature engineering
│   ├── model.py        # Train, evaluate, save classifier
│   ├── openf1.py       # OpenF1 API client (live + historical)
│   ├── race_engine.py  # Pace analysis, tyre degradation, undercut detection, projections
│   └── app.py          # Streamlit dashboard (4 tabs)
├── .streamlit/
│   └── config.toml     # Dark theme with F1 red (#E8002D)
├── streamlit_app.py    # Streamlit Community Cloud entry point
├── requirements.txt
└── README.md
```

---

## Key findings

> Run the full pipeline and fill these in from your results.

- **Finding 1** — e.g. which feature had the highest mean SHAP value, and what that says about the most distinctive driver behaviour
- **Finding 2** — e.g. which pair of drivers the model most often confused, and a possible explanation
- **Finding 3** — e.g. overall CV accuracy and how it compares to random-chance baseline (1 / number of drivers)
