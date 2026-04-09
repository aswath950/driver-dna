# Driver DNA + Race Dashboard

**Can a machine learning model tell two F1 drivers apart from their telemetry alone — without ever seeing their name?**

Driver DNA Fingerprinter extracts lap telemetry from FastF1, engineers driving-style features, and trains an XGBoost classifier to identify drivers purely from how they brake, accelerate, steer, and shift — no car number, no team, no context.

---

## Features

### 🧬 Driver DNA Fingerprinter

Extracts telemetry from qualifying/race sessions, engineers 10 scalar driving-style features per lap, and trains an XGBoost classifier to identify drivers. The Streamlit dashboard includes:

- **Driver Radar** — spider chart of normalised style features for 2–4 drivers
- **Mystery Driver** — pick any lap, let the model guess the driver, then reveal whether it was right

### 🏁 Race Dashboard

Real-time and historical race analysis powered by the OpenF1 API (no API key required).

**Fastest Lap Telemetry Comparison** — compare the single fastest recorded lap between two drivers, live from the OpenF1 API for the selected session. Five channels are available:

| Channel | What it shows |
|---|---|
| **Time Delta** | Cumulative time gap through the lap. X axis = elapsed lap time, Y axis = Δ gap (positive = Driver A ahead). Shaded regions show which driver is gaining at each point. |
| **Track Map** | Circuit outline coloured by the faster driver per microsector (~999 segments). Each segment is coloured by whichever driver had higher speed at that distance point. Circuit geometry comes from a pre-generated static store (`data/circuits.json`) keyed by Grand Prix name — always shows the correct track for the selected race. |
| **Speed** | Overlaid speed trace (km/h) vs. distance through the lap |
| **Throttle** | Overlaid throttle application (%) vs. distance |
| **Brake** | Overlaid brake pressure (%) vs. distance |

Driver colours are assigned from team colours (sourced live from the OpenF1 `/drivers` endpoint). When two drivers from the **same team** are compared, one receives an alternative colour from a distinct palette to ensure they are always visually distinguishable. Legend labels include both the driver acronym and the lap number used.

**Live mode** — connects to the OpenF1 live feed during an active F1 session. Auto-refreshes at a configurable interval (10s / 30s / 60s) with a pulsing LIVE badge. Only works during active F1 race weekends.

**Historical mode** — analyse any race from 2022 onwards. Select a year and Grand Prix, and all charts load from the OpenF1 archive.

**7 race analysis charts:**

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
| **FastF1** | High-frequency car telemetry (speed, throttle, brake, steering, gear, RPM) per lap — used by `pipeline.py` for the Driver DNA dataset and by `generate_circuits.py` for circuit XY geometry | https://docs.fastf1.dev |
| **OpenF1** | Live and historical race data — lap times, tyre stints, track positions, car telemetry, and driver/team metadata | https://openf1.org |

> `data/circuits.json` (circuit XY outlines) and `data/dataset.parquet` (Driver DNA training data) are both git-ignored and must be generated locally before the full dashboard is available.

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

### 4. Circuit geometry store (`src/generate_circuits.py`)
A one-off script that uses FastF1 to extract the XY circuit outline for every Grand Prix in the dashboard dropdown and writes them to `data/circuits.json`. Each entry contains **500 evenly resampled distance points** for a smooth rendering resolution. The script is resume-safe — it skips circuits already written and saves after each one. Run once before launching the dashboard.

### 5. Streamlit dashboard (`src/app.py`)
Three interactive tabs covering both the Driver DNA analysis and the Race Dashboard:

| Tab | Contents |
|---|---|
| **Driver Radar** | Normalised driving-style spider chart for 2–4 drivers |
| **Mystery Driver** | ML model prediction with probability breakdown |
| **Race Dashboard** | Fastest Lap Telemetry Comparison (5 channels, live OpenF1 data) + full race analysis (7 charts) |

The Race Dashboard's Fastest Lap Telemetry Comparison fetches car telemetry directly from the OpenF1 `car_data` API for the user-selected session and drivers — it is not reading from `dataset.parquet`. The Track Map channel draws circuit geometry from `data/circuits.json` (keyed by Grand Prix name) and colours it by live speed data, so the correct track is always shown regardless of which session was last run through `pipeline.py`.

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

### 1. Generate circuit outlines (required for Track Map)

Downloads XY geometry for all 24 Grand Prix circuits and writes `data/circuits.json`. Run once; takes 10–30 minutes due to FastF1 downloads. Progress is saved after each circuit so the script can be interrupted and resumed.

```bash
python src/generate_circuits.py
```

### 2. Extract telemetry and build the Driver DNA dataset

Downloads the session from the F1 timing feed and writes `data/dataset.parquet`. Required for the Driver Radar and Mystery Driver tabs.

```bash
python src/pipeline.py
```

> First run will download and cache session data — this takes a few minutes.

### 3. Train the classifier

Runs 5-fold CV, saves the model, label encoder, accuracy, confusion matrix, and SHAP plot to `models/`:

```bash
python src/model.py
```

### 4. Launch the dashboard

```bash
streamlit run src/app.py
```

> **Note:** The Race Dashboard's live mode only works during active F1 race weekends (practice, qualifying, or race sessions). Historical mode works any time for races from 2022 onwards.
>
> The Race Dashboard's Fastest Lap Telemetry and race analysis charts work independently of `dataset.parquet` — they query the OpenF1 API directly. Only the Driver Radar and Mystery Driver tabs require steps 2 and 3.

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

**Track Map shows "No circuit data"**
`data/circuits.json` has not been generated yet. Run `python src/generate_circuits.py` once (with the venv active). The script is resume-safe — re-run it to add any missing circuits.

**FastF1 cache**
On first run, FastF1 downloads session data to `./data/`. This can take several minutes. Subsequent runs read from cache and are fast.

---

## Project structure

```
driver-dna/
├── data/                        # FastF1 cache + generated data files (git-ignored)
│   ├── dataset.parquet          # Driver DNA training data (pipeline.py output)
│   └── circuits.json            # Circuit XY outlines for Track Map (generate_circuits.py output)
├── models/                      # Saved model files (git-ignored)
│   ├── driver_dna_clf.joblib
│   ├── label_encoder.joblib
│   ├── accuracy.txt
│   ├── confusion_matrix.html
│   └── shap_importance.png
├── src/
│   ├── pipeline.py              # Data extraction & feature engineering (FastF1)
│   ├── model.py                 # Train, evaluate, save XGBoost classifier
│   ├── generate_circuits.py     # One-off script: generate data/circuits.json
│   ├── openf1.py                # OpenF1 API client (live + historical modes)
│   ├── race_engine.py           # Pace analysis, tyre degradation, undercut detection, projections
│   └── app.py                   # Streamlit dashboard (3 tabs)
├── .streamlit/
│   └── config.toml              # Dark theme with F1 red (#E8002D)
├── streamlit_app.py             # Streamlit Community Cloud entry point
├── requirements.txt
└── README.md
```

---

## Key findings

> Run the full pipeline and fill these in from your results.

- **Finding 1** — e.g. which feature had the highest mean SHAP value, and what that says about the most distinctive driver behaviour
- **Finding 2** — e.g. which pair of drivers the model most often confused, and a possible explanation
- **Finding 3** — e.g. overall CV accuracy and how it compares to random-chance baseline (1 / number of drivers)
