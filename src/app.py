"""
app.py — Streamlit dashboard for Driver DNA Fingerprinter.

Run with:
    streamlit run src/app.py

Expected columns from dataset.parquet:
  driver, lap_time_seconds, mean_speed, max_speed, min_speed,
  throttle_mean, throttle_std, brake_mean, brake_events,
  gear_changes, steer_std, speed_trace, throttle_trace, brake_trace
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent))

from model import load_model, FEATURE_COLS, DATA_DIR, MODELS_DIR
from openf1 import OpenF1Client
from race_engine import RaceAnalyser

DATASET_PATH = DATA_DIR / "dataset.parquet"
CIRCUITS_PATH = DATA_DIR / "circuits.json"
ACCURACY_PATH = MODELS_DIR / "accuracy.txt"
DATASET_META_PATH = DATA_DIR / "dataset_meta.json"
N_POINTS = 200
TRACE_COLS = {"Speed": "speed_trace", "Throttle": "throttle_trace", "Brake": "brake_trace"}
# OpenF1 car_data column names for the same channels
OPENF1_TRACE_COLS = {"Speed": "speed", "Throttle": "throttle", "Brake": "brake"}
# Channels that require special rendering (not a simple 1D line overlay)
SPECIAL_CHANNELS = ["Track Map", "Time Delta"]
RADAR_FEATURES = ["mean_speed", "throttle_mean", "throttle_std", "brake_events", "steer_std", "gear_changes"]
EXTENDED_RADAR_FEATURES = [
    "mean_speed", "max_speed", "min_speed",
    "throttle_mean", "throttle_std", "throttle_pickup_pct",
    "brake_mean", "brake_events", "trail_brake_score",
    "steer_std", "gear_changes", "coasting_pct",
]
FEATURE_LABELS = {
    "mean_speed":           "Avg Speed",
    "max_speed":            "Top Speed",
    "min_speed":            "Corner Speed",
    "throttle_mean":        "Throttle Input",
    "throttle_std":         "Throttle Variation",
    "brake_mean":           "Brake Pressure",
    "brake_events":         "Brake Frequency",
    "steer_std":            "Steering Precision",
    "gear_changes":         "Gear Shifts",
    "coasting_pct":         "Coasting %",
    "throttle_pickup_pct":  "Full Throttle %",
    "trail_brake_score":    "Trail Braking",
}
# Sidebar visibility toggle keys — order determines display order in the expander
CHART_KEYS = {
    "Fastest Lap Telemetry": ("show_telemetry",      True),
    "Rolling Race Pace":     ("show_rolling_pace",    False),
    "Gap to Leader":         ("show_gap_to_leader",   False),
    "Projected Finish":      ("show_projected_order", False),
    "Average Race Pace":     ("show_avg_pace",        False),
    "Race Pace Ranking":     ("show_pace_ranking",    True),
    "Tyre Degradation":      ("show_tyre_deg",        True),
}

# --- Session state defaults for non-widget keys only ---------------
# Widget keys (ta_a, ta_b, radar_drivers, mystery_lap, rp_mode, etc.)
# must NOT be pre-set here — Streamlit manages widget state internally
# and raises a conflict error if a widget key already exists in
# session_state when the widget renders with its own default value.
STATE_DEFAULTS: dict = {
    # Tab 4 — non-widget state
    "rp_last_refresh": None,
    "rp_live_client": None,
    # Tab 4 historical — persists loaded race data across reruns
    "rp_hist_analyser": None,
    "rp_hist_driver_map": {},
    "rp_hist_team_colour_map": {},   # {int(driver_number): "#RRGGBB"}
    "rp_hist_session_key": None,     # int session_key of the currently loaded session
    # Tab 4 historical — available sessions fetched from OpenF1
    "rp_available_sessions": None,   # pd.DataFrame or None
    "rp_fetched_for": None,          # (year, gp) tuple — invalidates cache on change
    # Tab 4 live — driver map for current session
    "rp_live_driver_map": {},
    "rp_live_team_colour_map": {},   # {int(driver_number): "#RRGGBB"}
    "rp_live_meeting_name": "",      # GP name for live Track Map circuit lookup
    # Tab 2 — Mystery Driver / LLM Explainer
    "md_explanation": None,          # str | None — cached LLM explanation
    "md_prediction_ready": False,    # bool — True after Identify Driver runs
    "md_last_explain_ts": 0.0,       # float — unix timestamp of last explain call
    "md_last_lap_idx": None,         # int | None — lap index of last prediction
    # Tab 3 — Race Intelligence Chat Agent (ca_ prefix)
    "ca_messages": [],               # list[dict] — {role, content} history, capped at 20
    "ca_last_send_ts": 0.0,          # float — unix timestamp of last user message (rate limiting)
    "ca_session_cache_key": None,    # int|str|None — clears history when loaded session changes
}

st.set_page_config(page_title="Driver DNA Fingerprinter", layout="wide")

for _key, _val in STATE_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val

# --- Sidebar ---
with st.sidebar:
    st.title("🧬 Driver DNA + 🏁 Race Pace")

    # -- Driver DNA section --
    st.subheader("Driver DNA")
    st.markdown("Identify F1 drivers from telemetry alone using XGBoost.")
    if ACCURACY_PATH.exists():
        acc = float(ACCURACY_PATH.read_text().strip())
        st.metric("Model CV Accuracy", f"{acc:.1%}")
    else:
        st.caption("Run `model.py` to generate accuracy.txt")
    st.info("**Mystery Driver** — pick any lap and see if the ML model can identify the driver from driving style alone.")

    st.divider()

    # -- Race Dashboard section --
    st.subheader("Race Dashboard")
    rp_mode_val = st.session_state.get("rp_mode", "")
    if rp_mode_val.startswith("📡"):
        st.markdown("**Mode:** 📡 Live")
        last_ts = st.session_state.get("rp_last_refresh")
        if last_ts:
            st.caption(f"Last refreshed: {last_ts}")
        else:
            st.caption("Waiting for first refresh...")
    elif rp_mode_val.startswith("🗂️"):
        st.markdown("**Mode:** 🗂️ Historical")
        loaded_gp = st.session_state.get("rp_gp", "—")
        loaded_year = st.session_state.get("rp_year", "—")
        st.caption(f"{loaded_year} {loaded_gp}")
    else:
        st.caption("Select a mode in the Race Dashboard tab to get started.")
    st.info("**Race Dashboard** — fastest lap telemetry comparison, plus live or historical race analysis with rolling pace, gap charts, undercut detection, and projected finishing order.")

    # -- Visible Charts expander --
    st.divider()
    with st.expander("Visible Charts", expanded=False):
        st.caption("Toggle sections on or off in the Race Dashboard.")
        for _chart_label, (_chart_key, _chart_default) in CHART_KEYS.items():
            st.checkbox(_chart_label, value=_chart_default, key=_chart_key)
        st.divider()
        st.checkbox("Live Mode", value=False, key="show_live_mode")
        st.caption("Enable to access live race weekend data.")

    # -- Health Check expander --
    st.divider()
    with st.expander("🩺 Health Check"):
        _ok = "✅"
        _fail = "❌"

        # Dataset
        st.markdown(f"{_ok if DATASET_PATH.exists() else _fail} `dataset.parquet` exists")

        # Model
        _model_exists = (MODELS_DIR / "driver_dna_clf.joblib").exists()
        st.markdown(f"{_ok if _model_exists else _fail} `driver_dna_clf.joblib` exists")

        # OpenF1 API reachable (cached for 60s so it doesn't block every rerun)
        @st.cache_data(ttl=60, show_spinner=False)
        def _check_openf1() -> bool:
            try:
                import requests as _req
                _resp = _req.get(
                    "https://api.openf1.org/v1/sessions", timeout=5, params={"limit": 1}
                )
                return _resp.status_code == 200
            except Exception:
                return False

        _api_ok = _check_openf1()
        st.markdown(f"{_ok if _api_ok else _fail} OpenF1 API reachable")

        # Required libraries
        _libs = ["fastf1", "pandas", "numpy", "xgboost", "sklearn", "shap",
                 "streamlit", "plotly", "joblib", "matplotlib", "requests"]
        _missing_libs = []
        for _lib in _libs:
            try:
                __import__(_lib)
            except ImportError:
                _missing_libs.append(_lib)
        if _missing_libs:
            st.markdown(f"{_fail} Missing libraries: {', '.join(_missing_libs)}")
        else:
            st.markdown(f"{_ok} All required libraries importable")

# --- Load resources ---
@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_parquet(DATASET_PATH)


@st.cache_data
def get_circuits() -> dict:
    """Load pre-generated circuit XY outlines keyed by Grand Prix name."""
    if not CIRCUITS_PATH.exists():
        return {}
    with open(CIRCUITS_PATH) as f:
        return json.load(f)


@st.cache_data
def get_dataset_meta() -> dict:
    """Load session metadata written by pipeline.py alongside dataset.parquet."""
    if not DATASET_META_PATH.exists():
        return {}
    with open(DATASET_META_PATH) as f:
        return json.load(f)


# ── Radar derived-feature computation ───────────────────────────────────────

@st.cache_data
def compute_extended_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute coasting_pct, throttle_pickup_pct, trail_brake_score per lap from traces."""
    records = []
    for _, row in df.iterrows():
        try:
            thr = np.array(row["throttle_trace"], dtype=float)
            brk = np.array(row["brake_trace"], dtype=float)
            records.append({
                "driver": row["driver"],
                "lap_number": row.get("lap_number", 0),
                "coasting_pct": float(np.nanmean((thr < 2) & (brk < 1))),
                "throttle_pickup_pct": float(np.nanmean(thr > 80)),
                "trail_brake_score": float(np.nanmean((thr > 0) & (brk > 0))),
            })
        except Exception:
            continue
    if not records:
        df["coasting_pct"] = np.nan
        df["throttle_pickup_pct"] = np.nan
        df["trail_brake_score"] = np.nan
        return df
    ext = pd.DataFrame(records)
    return df.merge(ext, on=["driver", "lap_number"], how="left")


def compute_mean_trace(df: pd.DataFrame, driver: str, col: str) -> np.ndarray:
    """Return mean of a 200-point trace column across all laps for a driver."""
    rows = df[df["driver"] == driver][col]
    stack = np.stack([np.array(r, dtype=float) for r in rows])
    return np.nanmean(stack, axis=0)


def compute_sector_means(df: pd.DataFrame, driver: str, col: str) -> list[float]:
    """Split 200-point trace into three equal zones and return mean per zone."""
    stack = np.stack([np.array(r, dtype=float) for r in df[df["driver"] == driver][col]])
    return [float(np.nanmean(stack[:, s])) for s in [slice(0, 67), slice(67, 134), slice(134, 200)]]


# ── LLM Explainer helpers ───────────────────────────────────────────────────

@st.cache_resource
def get_shap_explainer(_clf):
    """Cache a shap.TreeExplainer. Underscore prefix prevents Streamlit hashing clf."""
    import shap
    return shap.TreeExplainer(_clf)


def compute_shap_for_prediction(
    explainer, X_input: np.ndarray, pred_class_idx: int
) -> np.ndarray:
    """Return 1-D SHAP values (n_features,) for the predicted class only.

    X_input always has shape (1, n_features), so the first axis of the raw
    SHAP output is a reliable discriminator:
      - New SHAP (>=0.41): (n_samples=1, n_features, n_classes) → shape[0] == 1
      - Old SHAP:          (n_classes, n_samples=1, n_features) → shape[0] == n_classes
    """
    raw = explainer.shap_values(X_input)
    arr = np.array(raw)
    if arr.ndim == 2:           # binary: (1, n_features)
        return arr[0]
    elif arr.ndim == 3:
        if arr.shape[0] == 1:   # new SHAP: (1, n_features, n_classes)
            return arr[0, :, pred_class_idx]
        else:                   # old SHAP: (n_classes, 1, n_features)
            return arr[pred_class_idx, 0, :]
    return np.abs(arr).mean(axis=tuple(range(arr.ndim - 1)))      # fallback


_FEATURE_BOUNDS: dict = {
    "lap_time_seconds":  (60.0,  200.0),
    "mean_speed":        (50.0,  380.0),
    "max_speed":         (80.0,  420.0),
    "min_speed":         (0.0,   250.0),
    "throttle_mean":     (0.0,   100.0),
    "throttle_std":      (0.0,    60.0),
    "brake_mean":        (0.0,   100.0),
    "brake_events":      (0.0,   200.0),
    "gear_changes":      (0.0,  1500.0),
    "steer_std":         (0.0,   100.0),
}


def _validate_feature_values(feature_dict: dict) -> tuple:
    """Check all values are finite floats within F1-reasonable bounds.
    Returns (True, '') on success or (False, reason) on failure."""
    import math
    for feat, val in feature_dict.items():
        try:
            fval = float(val)
        except (TypeError, ValueError):
            return False, f"Feature '{feat}' is not numeric."
        if not math.isfinite(fval):
            return False, f"Feature '{feat}' is not finite (got {val})."
        lo, hi = _FEATURE_BOUNDS.get(feat, (-1e9, 1e9))
        if not (lo <= fval <= hi):
            return False, f"Feature '{feat}' value {fval:.2f} outside expected range [{lo}, {hi}]."
    return True, ""


def _sanitize_driver_name(name: str, known_drivers: list) -> str:
    """Validate driver name is in the known list before prompt interpolation.
    Raises ValueError if not found — prevents prompt injection."""
    if name not in known_drivers:
        raise ValueError(f"Unknown driver '{name}' — not in model's class list.")
    return name


def _get_openai_api_key():
    """Read API key from st.secrets with os.environ fallback. Returns None if absent."""
    try:
        return st.secrets["openai"]["api_key"]
    except (KeyError, FileNotFoundError):
        pass
    return os.environ.get("OPENAI_API_KEY")


def _build_explain_prompt(
    pred_driver: str,
    confidence: float,
    feature_dict: dict,
    shap_dict: dict,
    percentile_dict: dict,
) -> tuple:
    """Build (system_msg, user_msg) for GPT-4o-mini. All interpolated values are
    pre-validated floats or a sanitized driver name — no raw user text enters the prompt.
    Total prompt stays under ~500 tokens."""

    ranked = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    lines = [
        f"  {f}: {feature_dict[f]:.2f}  "
        f"(p{percentile_dict[f]:.0f}, SHAP: {'+'if shap_dict[f] >= 0 else ''}{shap_dict[f]:.4f})"
        for f, _ in ranked
    ]
    system_msg = (
        "You are an F1 telemetry analyst assistant embedded in a machine learning dashboard. "
        "Explain ML model predictions in clear, engaging prose for a motorsport audience. "
        "Focus on the top 2-3 features by SHAP magnitude. Be concise (3-5 sentences). "
        "Do not speculate beyond the data. No markdown, bullet points, or headers."
    )
    user_msg = (
        f"An XGBoost classifier predicted this lap was driven by {pred_driver} "
        f"with {confidence:.1%} confidence.\n\n"
        "Feature values (percentile rank vs all drivers, SHAP contribution toward this prediction):\n"
        + "\n".join(lines)
        + "\n\nExplain in 3-5 plain prose sentences WHY the model made this prediction, "
        "citing the 2-3 features with the largest |SHAP| values and their actual numbers."
    )
    return system_msg, user_msg


def _call_openai_explainer(api_key: str, system_msg: str, user_msg: str) -> tuple:
    """Call GPT-4o-mini. Returns (text, None) on success or (None, error_msg) on failure.
    Raw exceptions are never surfaced to the UI."""
    try:
        from openai import OpenAI, AuthenticationError, RateLimitError, APIError
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=300,
            temperature=0.4,
        )
        text = resp.choices[0].message.content
        if not text:
            return None, "OpenAI returned an empty response. Try again."
        return text.strip(), None
    except AuthenticationError:
        return None, "OpenAI API key is invalid. Check `.streamlit/secrets.toml`."
    except RateLimitError:
        return None, "OpenAI rate limit reached. Wait a moment and try again."
    except APIError as e:
        return None, f"OpenAI service error ({type(e).__name__}). Try again shortly."
    except Exception:
        return None, "Unexpected error contacting OpenAI. Try again."


def _call_openai_chat(api_key: str, messages: list, tools: list) -> tuple:
    """Call GPT-4o-mini with optional tool definitions for the chat agent.

    Returns one of three shapes:
      ("text",       final_text,      None)  — LLM produced a final answer
      ("tool_calls", tool_calls_list, None)  — LLM wants to invoke tools
      (None,         None,            error) — API failure
    """
    try:
        from openai import OpenAI, AuthenticationError, RateLimitError, APIError
        client = OpenAI(api_key=api_key)
        if tools:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=500,
                temperature=0.4,
            )
        else:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=500,
                temperature=0.4,
            )
        choice = resp.choices[0]
        msg = choice.message
        if choice.finish_reason == "tool_calls" and msg.tool_calls:
            return "tool_calls", msg.tool_calls, None
        text = msg.content
        if not text:
            return None, None, "OpenAI returned an empty response. Try again."
        return "text", text.strip(), None
    except AuthenticationError:
        return None, None, "OpenAI API key is invalid. Check `.streamlit/secrets.toml`."
    except RateLimitError:
        return None, None, "OpenAI rate limit reached. Wait a moment and try again."
    except APIError as e:
        return None, None, f"OpenAI service error ({type(e).__name__}). Try again shortly."
    except Exception:
        return None, None, "Unexpected error contacting OpenAI. Try again."

# ── End LLM Explainer helpers ───────────────────────────────────────────────

# ============================================================================
# RACE INTELLIGENCE CHAT AGENT
# ============================================================================

# -- Security constants ------------------------------------------------------

_CA_MAX_INPUT_LEN = 500        # max user message length (chars)
_CA_MAX_TOOLS_PER_ROUND = 3    # max tool calls processed per LLM response

_CA_ALLOWED_TOOLS: set[str] = {
    "get_rolling_pace", "get_gap_to_leader", "detect_strategy_events",
    "project_finishing_order", "get_tyre_degradation", "get_pace_summary",
}

_CA_TOOL_ARG_SCHEMAS: dict[str, dict] = {
    "get_rolling_pace":        {"window": (int, 2, 15), "top_n": (int, 1, 20)},
    "detect_strategy_events":  {"lookahead": (int, 1, 10)},
    "project_finishing_order": {"laps_remaining": (int, 1, 80)},
}

# -- System prompt -----------------------------------------------------------

_CA_SYSTEM_PROMPT: str = (
    "You are Race Intelligence, an expert F1 race analyst embedded in a live telemetry "
    "dashboard. You have access to real-time race data for the currently loaded session.\n\n"
    "Your tools:\n"
    "  get_rolling_pace        — recent lap pace trend per driver\n"
    "  get_gap_to_leader       — cumulative time gap from each driver to the leader\n"
    "  detect_strategy_events  — undercut and overcut detections\n"
    "  project_finishing_order — projected final race order\n"
    "  get_tyre_degradation    — degradation rate per driver/compound/stint\n"
    "  get_pace_summary        — aggregate mean, median, fastest lap per driver\n\n"
    "Rules:\n"
    "- Always call at least one tool before answering pace, strategy, or position questions.\n"
    "- Synthesise tool data into a concise, confident answer (3-6 sentences).\n"
    "- Cite specific numbers from the data (lap times, gaps, degradation rates).\n"
    "- Use F1 driver acronyms (e.g. VER, NOR, HAM) as provided by the tools.\n"
    "- If data is insufficient, say so clearly.\n"
    "- No markdown, bullet points, or headers — plain prose only.\n"
    "- Do not speculate beyond the available data."
)

# -- Tool definitions (OpenAI function-calling schema) -----------------------

_CA_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_rolling_pace",
            "description": (
                "Returns rolling-average lap pace (seconds) per driver over a sliding window. "
                "Use to identify the driver with the strongest recent race pace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "window": {"type": "integer", "description": "Laps in rolling window. Default 5, range 2-15.", "default": 5},
                    "top_n":  {"type": "integer", "description": "Drivers to include. Default 5, max 20.", "default": 5},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_gap_to_leader",
            "description": "Returns current cumulative time gap (seconds) from each driver to the race leader.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_strategy_events",
            "description": (
                "Detects undercut and overcut events — position changes caused by pit-stop timing. "
                "Use for strategy questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lookahead": {"type": "integer", "description": "Laps after pit to check for position swap. Default 3, range 1-10.", "default": 3},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "project_finishing_order",
            "description": "Projects final race finishing order based on current pace and tyre degradation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "laps_remaining": {"type": "integer", "description": "Estimated laps remaining in the race. Range 1-80."},
                },
                "required": ["laps_remaining"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_tyre_degradation",
            "description": "Returns tyre degradation rates (seconds per lap lost) per driver by stint and compound.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pace_summary",
            "description": "Returns aggregate race pace statistics per driver: mean, median, std dev, fastest lap, lap count.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# -- Guardrail helpers -------------------------------------------------------

def _ca_sanitize_input(text: str) -> tuple:
    """Sanitize user chat input before it reaches the LLM.
    Returns (cleaned_text, None) on success or ('', error_reason) if rejected."""
    text = text.strip()
    if not text:
        return "", "Please enter a question."
    if len(text) > _CA_MAX_INPUT_LEN:
        return "", f"Message too long ({len(text)} chars). Please keep it under {_CA_MAX_INPUT_LEN} characters."
    # Strip null bytes and ASCII control chars (0x00-0x1F) except newline/tab
    cleaned = "".join(ch for ch in text if ch in ("\n", "\t") or ord(ch) >= 0x20)
    if not cleaned:
        return "", "Message contained only invalid characters."
    return cleaned, None


def _ca_validate_tool_args(tool_name: str, args: dict) -> tuple:
    """Validate and clamp LLM-supplied tool arguments against known schemas.
    Returns (cleaned_args, None) on success or ({}, error_string) on failure."""
    schema = _CA_TOOL_ARG_SCHEMAS.get(tool_name, {})
    cleaned: dict = {}
    for key, (typ, lo, hi) in schema.items():
        raw = args.get(key)
        if raw is None:
            cleaned[key] = lo   # use lower bound as safe default
            continue
        try:
            val = typ(raw)
        except (ValueError, TypeError):
            return {}, f"Tool '{tool_name}': argument '{key}' must be {typ.__name__}, got {raw!r}"
        cleaned[key] = max(lo, min(val, hi))
    return cleaned, None

# -- Tool executors ----------------------------------------------------------

def _ca_tool_get_rolling_pace(analyser, driver_map: dict, window: int = 5, top_n: int = 5) -> str:
    window = max(2, min(int(window), 15))
    top_n  = max(1, min(int(top_n), 20))
    df = analyser.rolling_pace(window=window)
    if df.empty:
        return "No rolling pace data available yet."
    latest = df.iloc[-1].dropna().sort_values()
    valid = [(k, v) for k, v in latest.items() if int(k) in driver_map][:top_n]
    if not valid:
        return "No valid driver pace data available."
    lines = [f"  P{i+1}: {driver_map[int(drv)]}  {pace:.3f}s/lap" for i, (drv, pace) in enumerate(valid)]
    return f"Rolling pace (window={window} laps), latest lap:\n" + "\n".join(lines)


def _ca_tool_get_gap_to_leader(analyser, driver_map: dict) -> str:
    df = analyser.gap_to_leader()
    if df.empty:
        return "No gap data available yet."
    latest = df.iloc[-1].dropna().sort_values()
    valid = [(k, v) for k, v in latest.items() if int(k) in driver_map]
    if not valid:
        return "No gap data for known drivers."
    lines = [f"  {driver_map[int(drv)]}: +{gap:.3f}s" for drv, gap in valid]
    return f"Gap to leader at lap {df.index[-1]}:\n" + "\n".join(lines)


def _ca_tool_detect_strategy_events(analyser, driver_map: dict, lookahead: int = 3) -> str:
    lookahead = max(1, min(int(lookahead), 10))
    events = analyser.detect_undercuts(lookahead=lookahead)
    if not events:
        return "No undercut or overcut events detected in this race."
    lines = [
        f"  Lap {e['lap']}: "
        f"{driver_map.get(int(e['attacking_driver']), str(e['attacking_driver']))} "
        f"{e['type']}d "
        f"{driver_map.get(int(e['defending_driver']), str(e['defending_driver']))}"
        for e in events
    ]
    return f"Strategy events detected ({len(events)} total):\n" + "\n".join(lines)


def _ca_tool_project_finishing_order(analyser, driver_map: dict, laps_remaining: int = 10) -> str:
    laps_remaining = max(1, min(int(laps_remaining), 80))
    projections = analyser.project_finishing_order(laps_remaining=laps_remaining)
    if not projections:
        return "Insufficient data to project finishing order."
    lines = []
    for p in projections:
        label = driver_map.get(int(p["driver"]), str(p["driver"]))
        if p.get("status") == "DNF":
            lines.append(f"  P{p['projected_position']}: {label} (DNF)")
        else:
            change = p.get("position_change")
            if change is not None:
                arrow = f" [{'↑' if change > 0 else '↓' if change < 0 else '='}{abs(change)}]"
            else:
                arrow = ""
            lines.append(f"  P{p['projected_position']}: {label}{arrow}")
    return f"Projected finishing order ({laps_remaining} laps remaining):\n" + "\n".join(lines)


def _ca_tool_get_tyre_degradation(analyser, driver_map: dict) -> str:
    df = analyser.tyre_degradation()
    if df.empty:
        return "No tyre degradation data available."
    if "driver_number" in df.columns:
        df = df[df["driver_number"].apply(lambda x: int(x) in driver_map)]
    if df.empty:
        return "No degradation data for known drivers."
    lines = [
        f"  {driver_map.get(int(r['driver_number']), str(r['driver_number']))} "
        f"(S{int(r.get('stint_number', 0))}, {str(r.get('compound', '?')).upper()}, "
        f"{int(r.get('laps_in_stint', 0))} laps): "
        f"{r.get('deg_per_lap', float('nan')):+.3f}s/lap, "
        f"{r.get('mean_pace', float('nan')):.3f}s mean"
        for _, r in df.iterrows()
    ]
    return "Tyre degradation per driver/stint:\n" + "\n".join(lines)


def _ca_tool_get_pace_summary(analyser, driver_map: dict) -> str:
    df = analyser.pace_summary()
    if df.empty:
        return "No pace summary data available."
    lines = [
        f"  {driver_map[int(drv_num)]}: mean={row['mean_pace']:.3f}s, "
        f"median={row['median_pace']:.3f}s, fastest={row['fastest_lap']:.3f}s, "
        f"laps={int(row['lap_count'])}"
        for drv_num, row in df.iterrows()
        if int(drv_num) in driver_map
    ]
    if not lines:
        return "No pace summary data for known drivers."
    return "Pace summary:\n" + "\n".join(lines)

# -- Tool dispatcher ---------------------------------------------------------

def _ca_dispatch_tool(tool_name: str, tool_args: dict, analyser, driver_map: dict) -> str:
    """Route an LLM tool_call to the correct executor with full validation."""
    if tool_name not in _CA_ALLOWED_TOOLS:
        return f"Tool '{tool_name}' is not a recognised tool — skipped."

    cleaned_args, arg_err = _ca_validate_tool_args(tool_name, tool_args)
    if arg_err:
        return arg_err
    tool_args = cleaned_args

    try:
        if tool_name == "get_rolling_pace":
            return _ca_tool_get_rolling_pace(
                analyser, driver_map,
                window=int(tool_args.get("window", 5)),
                top_n=int(tool_args.get("top_n", 5)),
            )
        elif tool_name == "get_gap_to_leader":
            return _ca_tool_get_gap_to_leader(analyser, driver_map)
        elif tool_name == "detect_strategy_events":
            return _ca_tool_detect_strategy_events(
                analyser, driver_map,
                lookahead=int(tool_args.get("lookahead", 3)),
            )
        elif tool_name == "project_finishing_order":
            try:
                lr = int(tool_args["laps_remaining"])
            except (KeyError, ValueError, TypeError):
                lr = 10
            return _ca_tool_project_finishing_order(analyser, driver_map, laps_remaining=lr)
        elif tool_name == "get_tyre_degradation":
            return _ca_tool_get_tyre_degradation(analyser, driver_map)
        elif tool_name == "get_pace_summary":
            return _ca_tool_get_pace_summary(analyser, driver_map)
    except (ValueError, TypeError) as e:
        return f"Tool '{tool_name}' argument error: {e}"
    except Exception:
        return f"Tool '{tool_name}' execution failed (unexpected error)."
    return f"Tool '{tool_name}' produced no output."

# -- Message history helpers -------------------------------------------------

def _ca_append_message(role: str, content: str) -> None:
    """Append to chat history. Truncates long messages and caps history at 20 entries."""
    if len(content) > 2000:
        content = content[:2000] + " [truncated]"
    st.session_state["ca_messages"].append({"role": role, "content": content})
    if len(st.session_state["ca_messages"]) > 20:
        st.session_state["ca_messages"] = st.session_state["ca_messages"][-20:]


def _ca_maybe_clear_history(session_cache_key) -> None:
    """Clear chat history when the loaded session changes."""
    if st.session_state["ca_session_cache_key"] != session_cache_key:
        st.session_state["ca_messages"] = []
        st.session_state["ca_session_cache_key"] = session_cache_key

# -- Orchestration -----------------------------------------------------------

def _ca_run_chat_turn(user_message: str, analyser, driver_map: dict, api_key: str) -> tuple:
    """Run one chat turn with the tool-calling loop.
    Returns (answer_text, None) on success or (None, error_string) on failure."""
    MAX_TOOL_ROUNDS = 3

    messages: list[dict] = [{"role": "system", "content": _CA_SYSTEM_PROMPT}]
    for m in st.session_state["ca_messages"]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_message})

    for _ in range(MAX_TOOL_ROUNDS):
        kind, payload, err = _call_openai_chat(api_key, messages, _CA_TOOLS)
        if err:
            return None, err
        if kind == "text":
            return payload, None
        if kind == "tool_calls":
            # Append the assistant's tool-call request in the format OpenAI requires
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in payload
                ],
            })
            # Execute each tool (capped per round) and append results
            for tc in payload[:_CA_MAX_TOOLS_PER_ROUND]:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}
                result = _ca_dispatch_tool(tc.function.name, args, analyser, driver_map)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

    # Fell through all rounds — force a synthesis pass with no tools
    messages.append({
        "role": "user",
        "content": "Summarise what you found in the tool results above.",
    })
    kind, payload, err = _call_openai_chat(api_key, messages, [])
    if err:
        return None, err
    if kind == "text":
        return payload, None
    return None, "Could not generate a response after tool execution."

# -- Chat UI -----------------------------------------------------------------

def _render_chat_panel(analyser, driver_map: dict, session_cache_key) -> None:
    """Render the Race Intelligence chat panel below the race analysis charts."""
    _ca_maybe_clear_history(session_cache_key)
    api_key = _get_openai_api_key()

    st.divider()
    st.markdown("#### 🤖 Race Intelligence — Ask me anything about this race")

    # Render existing message history
    for msg in st.session_state["ca_messages"]:
        with st.chat_message(msg["role"]):
            st.text(msg["content"])

    user_input = st.chat_input("Ask a question about the race...", key="ca_input")

    if user_input:
        # Guardrail 0 — sanitize input
        user_input, input_err = _ca_sanitize_input(user_input)
        if input_err:
            st.warning(input_err)
            return

        # Rate limit check
        elapsed = time.time() - st.session_state["ca_last_send_ts"]
        if elapsed < 5.0:
            st.warning(f"Please wait {5.0 - elapsed:.1f}s before sending another message.")
            return

        # API key check
        if not api_key:
            st.error(
                "No OpenAI API key configured. "
                "Add `[openai]\\napi_key = 'sk-...'` to `.streamlit/secrets.toml`."
            )
            return

        with st.chat_message("user"):
            st.text(user_input)
        _ca_append_message("user", user_input)
        st.session_state["ca_last_send_ts"] = time.time()

        with st.chat_message("assistant"):
            with st.spinner("Analysing race data..."):
                answer, error = _ca_run_chat_turn(user_input, analyser, driver_map, api_key)
            if error:
                st.error(error)
                st.session_state["ca_messages"].pop()   # remove failed user message
            else:
                st.text(answer)
                _ca_append_message("assistant", answer)

# ── End Race Intelligence Chat Agent ────────────────────────────────────────


if not DATASET_PATH.exists():
    st.error("Run `python src/pipeline.py` first to fetch telemetry and build the dataset.")
    st.stop()

model_path = MODELS_DIR / "driver_dna_clf.joblib"
if not model_path.exists():
    st.error("Run `python src/model.py` first to train the model.")
    st.stop()

try:
    clf, le = get_model()
    shap_explainer = get_shap_explainer(clf)
    df = get_data()
except Exception as e:
    st.error(f"Failed to load resources: {e}")
    st.stop()

circuits = get_circuits()

drivers = sorted(df["driver"].unique())
palette = px.colors.qualitative.Plotly
color_map = {d: palette[i % len(palette)] for i, d in enumerate(drivers)}

def classify_archetype(norm_means: dict) -> tuple[str, str]:
    """Assign a driving style archetype from normalised (0-1) feature means."""
    if norm_means.get("brake_events", 0) > 0.65 and norm_means.get("brake_mean", 0) > 0.60:
        return "Aggressive Braker", "💥"
    if norm_means.get("trail_brake_score", 0) > 0.55 and norm_means.get("steer_std", 0) > 0.60:
        return "Trail Braking Specialist", "🎯"
    if norm_means.get("mean_speed", 0) > 0.70 and norm_means.get("coasting_pct", 0) < 0.35:
        return "High Entry Speed", "🚀"
    if norm_means.get("throttle_pickup_pct", 0) > 0.60 and norm_means.get("throttle_std", 0) < 0.40:
        return "Smooth Operator", "🧊"
    if norm_means.get("gear_changes", 0) > 0.65 and norm_means.get("steer_std", 0) > 0.55:
        return "Technical Driver", "⚙️"
    return "Balanced Driver", "⚖️"


def _build_throttle_map_fig(df: pd.DataFrame, selected: list[str], color_map: dict) -> go.Figure:
    """Build a multi-panel track map coloured by mean throttle per driver."""
    from plotly.subplots import make_subplots
    n = len(selected)
    fig = make_subplots(
        rows=1, cols=n,
        subplot_titles=selected,
        horizontal_spacing=0.04,
    )
    for i, drv in enumerate(selected, start=1):
        df_drv = df[df["driver"] == drv]
        try:
            x = np.nanmean(np.stack([np.array(r, dtype=float) for r in df_drv["x_trace"]]), axis=0)
            y = np.nanmean(np.stack([np.array(r, dtype=float) for r in df_drv["y_trace"]]), axis=0)
            thr = np.nanmean(np.stack([np.array(r, dtype=float) for r in df_drv["throttle_trace"]]), axis=0)
        except Exception:
            continue
        # Background outline
        fig.add_trace(
            go.Scatter(x=x, y=y, mode="lines",
                       line=dict(color="#333333", width=6),
                       showlegend=False, hoverinfo="skip"),
            row=1, col=i,
        )
        # Throttle-coloured layer
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="markers",
                marker=dict(
                    color=thr,
                    colorscale="RdYlGn",
                    cmin=0, cmax=100,
                    size=4,
                    colorbar=dict(title="Throttle %", thickness=12, len=0.6) if i == n else None,
                    showscale=(i == n),
                ),
                name=drv,
                showlegend=False,
                hovertemplate=f"<b>{drv}</b><br>Throttle: %{{marker.color:.0f}}%<extra></extra>",
            ),
            row=1, col=i,
        )
        axis_cfg = dict(visible=False, scaleanchor=f"y{i if i > 1 else ''}", scaleratio=1)
        fig.update_xaxes(axis_cfg, row=1, col=i)
        fig.update_yaxes(dict(visible=False), row=1, col=i)
    fig.update_layout(
        height=340,
        margin=dict(l=10, r=30, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


tab1, tab2, tab3 = st.tabs(["Driver Radar", "Mystery Driver", "Race Dashboard"])

# ================================================================
# TAB 1 — Driver Radar
# ================================================================
with tab1:
    st.subheader("Driver Style Fingerprint")

    selected = st.multiselect(
        "Select 2–4 drivers",
        options=drivers,
        default=drivers[:min(3, len(drivers))],
        key="radar_drivers",
    )

    if len(selected) < 2:
        st.warning("Select at least 2 drivers.")
    elif len(selected) > 4:
        st.warning("Select at most 4 drivers.")
    else:
        # ── Compute extended trace-derived features ──────────────────────
        df_ext = compute_extended_features(df)

        radar_mode = st.radio(
            "Radar dimensions",
            ["Standard (6)", "Extended (12)"],
            horizontal=True,
            key="radar_mode",
        )
        use_extended = "Extended" in radar_mode
        features = EXTENDED_RADAR_FEATURES if use_extended else RADAR_FEATURES

        # Build normalised driver means for the chosen feature set
        feat_df = df_ext[features].copy()
        feat_min = feat_df.min()
        feat_max = feat_df.max()
        norm_df = (feat_df - feat_min) / (feat_max - feat_min).replace(0, 1)
        norm_df["driver"] = df_ext["driver"].values
        driver_means = norm_df.groupby("driver")[features].mean()

        # Human-readable axis labels
        theta_labels = [FEATURE_LABELS.get(f, f) for f in features]
        categories = theta_labels + [theta_labels[0]]

        fig = go.Figure()
        for drv in selected:
            vals = driver_means.loc[drv].tolist()
            vals += [vals[0]]
            fig.add_trace(go.Scatterpolar(
                r=vals,
                theta=categories,
                fill="toself",
                name=drv,
                line=dict(color=color_map[drv]),
                opacity=0.7,
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Driver Style Radar (normalised 0–1)",
            legend_title="Driver",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch", key="tab1_driver_radar")

        # ── Archetype cards ──────────────────────────────────────────────
        # Use extended normalised means for archetype classification
        ext_feat_df = df_ext[EXTENDED_RADAR_FEATURES].copy()
        ext_norm_df = (ext_feat_df - ext_feat_df.min()) / (ext_feat_df.max() - ext_feat_df.min()).replace(0, 1)
        ext_norm_df["driver"] = df_ext["driver"].values
        ext_driver_means = ext_norm_df.groupby("driver")[EXTENDED_RADAR_FEATURES].mean()

        arch_cols = st.columns(len(selected))
        for col_idx, drv in enumerate(selected):
            norm_vals = ext_driver_means.loc[drv].to_dict()
            archetype, emoji = classify_archetype(norm_vals)
            drv_color = color_map[drv]
            arch_cols[col_idx].markdown(
                f"""<div style="background:#1e1e2e; border-left:4px solid {drv_color};
                border-radius:6px; padding:12px; margin:4px 0;">
                <div style="font-size:1.6em">{emoji}</div>
                <div style="font-weight:700; color:{drv_color}; font-size:1.1em">{drv}</div>
                <div style="color:#aaa; font-size:0.9em; margin-top:4px">{archetype}</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.divider()

        # ── Speed Trace Overlay ──────────────────────────────────────────
        st.markdown("### Speed Profile Comparison")
        st.caption("Average speed at each point along the lap — reveals where each driver goes faster.")
        fig_speed = go.Figure()
        for drv in selected:
            try:
                trace = compute_mean_trace(df_ext, drv, "speed_trace")
                fig_speed.add_trace(go.Scatter(
                    x=list(range(200)),
                    y=trace,
                    mode="lines",
                    name=drv,
                    line=dict(color=color_map[drv], width=2),
                ))
            except Exception:
                continue
        fig_speed.update_layout(
            xaxis_title="Lap Distance (normalised)",
            yaxis_title="Speed (km/h)",
            hovermode="x unified",
            legend_title="Driver",
            height=360,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_speed, width="stretch", key="tab1_speed_overlay")

        # ── Lap Consistency Heatmap ──────────────────────────────────────
        st.markdown("### Lap-by-Lap Consistency")
        st.caption("How variable each driver is across laps — red means less consistent.")
        cons_records = {
            drv: {feat: float(df_ext[df_ext["driver"] == drv][feat].std(ddof=0))
                  for feat in RADAR_FEATURES}
            for drv in selected
        }
        cons_df = pd.DataFrame(cons_records).T
        norm_cons = (cons_df - cons_df.min()) / (cons_df.max() - cons_df.min() + 1e-9)
        heatmap_labels = [FEATURE_LABELS.get(f, f) for f in RADAR_FEATURES]
        fig_heat = go.Figure(go.Heatmap(
            z=norm_cons.values,
            x=heatmap_labels,
            y=list(norm_cons.index),
            colorscale="Reds",
            text=cons_df.round(3).values,
            texttemplate="%{text}",
            hovertemplate="<b>%{y}</b> — %{x}<br>Std dev: %{text}<extra></extra>",
            showscale=True,
            colorbar=dict(title="Inconsistency", thickness=14),
        ))
        fig_heat.update_layout(
            height=300,
            margin=dict(l=10, r=80, t=20, b=60),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_heat, width="stretch", key="tab1_consistency_heatmap")

        st.divider()

        # ── Distance-Third Breakdown ─────────────────────────────────────
        st.markdown("### Lap Zone Performance")
        st.caption("Lap split into three equal distance zones — shows where each driver excels.")
        channel_options = {
            "Speed (km/h)": ("speed_trace", "Speed (km/h)"),
            "Throttle (%)": ("throttle_trace", "Throttle (%)"),
            "Braking (0–1)": ("brake_trace", "Braking Intensity"),
        }
        sector_channel = st.selectbox(
            "Channel",
            list(channel_options.keys()),
            key="sector_channel",
        )
        trace_col, y_label = channel_options[sector_channel]
        zone_labels = ["Zone 1 (0–33%)", "Zone 2 (33–67%)", "Zone 3 (67–100%)"]
        fig_sect = go.Figure()
        for drv in selected:
            try:
                zone_vals = compute_sector_means(df_ext, drv, trace_col)
                fig_sect.add_trace(go.Bar(
                    name=drv,
                    x=zone_labels,
                    y=zone_vals,
                    marker_color=color_map[drv],
                ))
            except Exception:
                continue
        fig_sect.update_layout(
            barmode="group",
            xaxis_title="Lap Zone",
            yaxis_title=y_label,
            legend_title="Driver",
            height=360,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_sect, width="stretch", key="tab1_sector_breakdown")

        # ── Throttle Map on Track ────────────────────────────────────────
        st.markdown("### Throttle Application Map")
        st.caption("Track outline coloured by average throttle — green: full throttle, red: braking / coasting.")
        try:
            fig_tmap = _build_throttle_map_fig(df_ext, selected, color_map)
            st.plotly_chart(fig_tmap, width="stretch", key="tab1_throttle_map")
        except Exception as e:
            st.info(f"Track map unavailable: {e}")

# ================================================================
# TAB 2 — Mystery Driver
# ================================================================
with tab2:
    st.subheader("Can the model identify the driver?")

    _meta = get_dataset_meta()
    if _meta:
        _col1, _col2, _col3 = st.columns(3)
        _col1.metric("Grand Prix", _meta.get("grand_prix", "—"))
        _col2.metric("Season", str(_meta.get("year", "—")))
        _col3.metric("Session", _meta.get("session_label", _meta.get("session_type", "—")))
    else:
        st.caption("Training session info unavailable — re-run `python src/pipeline.py` to generate it.")

    def _fmt_lap(i: int) -> str:
        row = df.iloc[i]
        driver = row.get("driver", "???")
        lap_num = row.get("lap_number", i + 1)
        return f"{driver} (Lap {int(lap_num)})"

    lap_idx = st.selectbox(
        "Pick a lap",
        range(len(df)),
        format_func=_fmt_lap,
        key="mystery_lap",
    )

    # Clear stale explanation whenever the user picks a different lap
    if st.session_state.get("md_last_lap_idx") != lap_idx:
        st.session_state["md_prediction_ready"] = False
        st.session_state["md_explanation"] = None

    if st.button("Identify Driver", key="identify_btn"):
        lap_row = df.iloc[lap_idx]
        actual_driver = lap_row["driver"]

        X_input = lap_row[FEATURE_COLS].to_numpy().reshape(1, -1)
        proba = clf.predict_proba(X_input)[0]
        pred_driver = le.classes_[proba.argmax()]
        pred_class_idx = int(proba.argmax())

        prob_df = pd.DataFrame({
            "Driver": le.classes_,
            "Probability": proba,
        }).sort_values("Probability", ascending=True)

        fig = go.Figure(go.Bar(
            x=prob_df["Probability"],
            y=prob_df["Driver"],
            orientation="h",
            marker_color=[color_map.get(d, "#888") for d in prob_df["Driver"]],
        ))
        fig.update_layout(
            title="Model Prediction Probabilities",
            xaxis_title="Probability",
            yaxis_title="Driver",
            xaxis=dict(range=[0, 1]),
        )
        st.plotly_chart(fig, width="stretch", key="tab3_mystery_proba")

        correct = pred_driver == actual_driver
        icon = "✅" if correct else "❌"
        st.markdown(f"### Actual driver: **{actual_driver}** {icon}")
        if correct:
            st.success(f"Correct! The model identified **{pred_driver}** with {proba.max():.1%} confidence.")
        else:
            st.error(f"Wrong! The model predicted **{pred_driver}** but the actual driver was **{actual_driver}**.")

        # Persist state so the explainer button can read it on the next rerun
        st.session_state["md_prediction_ready"] = True
        st.session_state["md_last_lap_idx"] = lap_idx
        st.session_state["md_explanation"] = None           # clear on new prediction
        st.session_state["_md_pred_driver"] = pred_driver
        st.session_state["_md_confidence"] = float(proba.max())
        st.session_state["_md_X_input"] = X_input
        st.session_state["_md_pred_class_idx"] = pred_class_idx

    # ── Explain button — only visible after a prediction has been made ──────
    _RATE_LIMIT_SECS = 10

    if st.session_state.get("md_prediction_ready"):
        api_key = _get_openai_api_key()

        if api_key is None:
            st.warning(
                "OpenAI API key not configured. To enable AI explanations:\n\n"
                "1. Open `.streamlit/secrets.toml`\n"
                "2. Replace the placeholder with your real key:\n"
                "```toml\n[openai]\napi_key = \"sk-proj-...\"\n```\n"
                "3. Restart the app."
            )
        else:
            elapsed = time.time() - st.session_state.get("md_last_explain_ts", 0.0)
            cooldown = max(0.0, _RATE_LIMIT_SECS - elapsed)
            btn_label = (
                f"✨ Ask AI to Explain (wait {cooldown:.0f}s)"
                if cooldown > 0
                else "✨ Ask AI to Explain"
            )

            if st.button(btn_label, key="explain_btn", disabled=cooldown > 0):
                _pred_driver    = st.session_state["_md_pred_driver"]
                _confidence     = st.session_state["_md_confidence"]
                _X_input        = st.session_state["_md_X_input"]
                _pred_class_idx = st.session_state["_md_pred_class_idx"]

                lap_features = {
                    feat: float(_X_input[0, i])
                    for i, feat in enumerate(FEATURE_COLS)
                }

                # Guardrail 1 — validate feature values are finite and in-bounds
                valid, reason = _validate_feature_values(lap_features)
                if not valid:
                    st.error(f"Input validation failed: {reason}")
                else:
                    # Guardrail 2 — sanitize driver name before prompt interpolation
                    try:
                        safe_driver = _sanitize_driver_name(_pred_driver, list(le.classes_))
                    except ValueError as exc:
                        st.error(str(exc))
                        safe_driver = None

                    if safe_driver is not None:
                        with st.spinner("Asking the AI to explain this prediction..."):
                            # Compute SHAP values for this single sample
                            shap_vals = compute_shap_for_prediction(
                                shap_explainer, _X_input, _pred_class_idx
                            )
                            shap_dict = {
                                feat: float(shap_vals[i])
                                for i, feat in enumerate(FEATURE_COLS)
                            }

                            # Percentile rank of this lap vs the full dataset
                            percentile_dict = {
                                feat: float(
                                    (df[feat].dropna() <= lap_features[feat]).mean() * 100
                                )
                                for feat in FEATURE_COLS
                            }

                            system_msg, user_msg = _build_explain_prompt(
                                safe_driver, _confidence,
                                lap_features, shap_dict, percentile_dict,
                            )
                            explanation, err = _call_openai_explainer(
                                api_key, system_msg, user_msg
                            )

                            if err:
                                st.error(err)
                            else:
                                st.session_state["md_explanation"] = explanation

                        st.session_state["md_last_explain_ts"] = time.time()

    # Display cached explanation — persists across reruns without re-calling the API
    if st.session_state.get("md_explanation"):
        st.info(st.session_state["md_explanation"])

# ================================================================
# TAB 4 — Race Pace Dashboard
# ================================================================

# ---- helpers shared by both modes --------------------------------

RACE_PALETTE = px.colors.qualitative.Dark24



def _fetch_fastest_lap_openf1(
    session_key: int | str,
    driver_number: int,
    laps_df: pd.DataFrame,
    channel: str,
) -> tuple[np.ndarray | None, float | None, int | None]:
    """
    Fetch the fastest-lap telemetry trace for a driver from the OpenF1 car_data API.

    Parameters
    ----------
    session_key : int | str
        The OpenF1 session key for the currently loaded session, or ``"latest"``
        for the live session.
    driver_number : int
        The OpenF1 driver number (e.g. 44 for Hamilton).
    laps_df : pd.DataFrame
        The laps DataFrame already fetched from OpenF1 for this session.
        Used to identify the fastest lap and its time window — avoids a second API call.
    channel : str
        One of ``"Speed"``, ``"Throttle"``, ``"Brake"``.

    Returns
    -------
    (trace, lap_time, lap_number) where trace is a float64 ndarray of length N_POINTS,
    or (None, None, None) on any failure.

    How it works
    ------------
    1. Filter laps to this driver, drop pit-out laps, find the row with the
       minimum ``lap_duration``.
    2. Use ``date_start`` and ``date_start + lap_duration`` as the car_data
       time window (ISO-8601 strings passed to ``date>`` / ``date<``).
    3. Fetch car_data from OpenF1 for that window.
    4. Sort by date, compute a cumulative-distance proxy from speed × dt, then
       resample the requested channel to N_POINTS evenly spaced distance points
       using np.interp — matching the normalisation pipeline.py applies.
    """
    OPENF1_CHANNEL_COL = {"Speed": "speed", "Throttle": "throttle", "Brake": "brake"}
    col = OPENF1_CHANNEL_COL.get(channel)
    if col is None:
        return None, None, None

    # --- 1. Find the fastest lap for this driver ---
    drv_laps = laps_df[laps_df["driver_number"] == driver_number].copy()
    if "is_pit_out_lap" in drv_laps.columns:
        drv_laps = drv_laps[drv_laps["is_pit_out_lap"] != True]  # noqa: E712
    drv_laps = drv_laps.dropna(subset=["lap_duration"])
    if drv_laps.empty:
        return None, None, None

    fastest = drv_laps.sort_values("lap_duration").iloc[0]
    lap_time = float(fastest["lap_duration"])
    lap_number: int | None = int(fastest["lap_number"]) if pd.notna(fastest.get("lap_number")) else None

    # --- 2. Build the time window ---
    date_start = fastest.get("date_start")
    if date_start is None or pd.isna(date_start):
        return None, None, None
    ts_start = pd.Timestamp(date_start)
    ts_end = ts_start + pd.Timedelta(seconds=lap_time)
    # Add a small buffer so we don't clip the final sample
    date_gte = ts_start.isoformat()
    date_lte = (ts_end + pd.Timedelta(seconds=0.5)).isoformat()

    # --- 3. Fetch car_data ---
    client = OpenF1Client(mode="historical")
    car = client.get_car_data(
        session_key=session_key,
        driver_number=driver_number,
        date_gte=date_gte,
        date_lte=date_lte,
    )
    if car.empty or col not in car.columns or "date" not in car.columns:
        return None, None, None

    car = car.sort_values("date").reset_index(drop=True)
    values = car[col].to_numpy(dtype=float)
    if len(values) < 2:
        return None, None, None

    # --- 4. Resample to N_POINTS evenly spaced distance points ---
    # Compute cumulative distance proxy: distance ≈ speed × dt (speed in km/h → m/s)
    dt = car["date"].diff().dt.total_seconds().fillna(0.0).to_numpy(dtype=float)[1:]
    speeds_ms = car["speed"].to_numpy(dtype=float)[:-1] / 3.6            # km/h → m/s
    dist_increments = np.where(np.isfinite(speeds_ms) & np.isfinite(dt), speeds_ms * dt, 0.0)
    dist = np.concatenate([[0.0], np.cumsum(dist_increments)])

    dist_grid = np.linspace(dist[0], dist[-1], N_POINTS)
    trace = np.interp(dist_grid, dist, values)
    return trace, lap_time, lap_number


def _fetch_fastest_lap_all_openf1(
    session_key: int | str,
    driver_number: int,
    laps_df: pd.DataFrame,
) -> dict | None:
    """
    Fetch all telemetry channels for a driver's fastest lap from the OpenF1 API.

    Returns a dict with keys: ``'speed'``, ``'throttle'``, ``'brake'``,
    ``'lap_time'``, ``'lap_number'``, ``'x'``, ``'y'``.

    ``'x'`` and ``'y'`` are always ``None`` — OpenF1 car_data does not include
    circuit XY coordinates. Track Map falls back to dataset.parquet for these.
    """
    drv_laps = laps_df[laps_df["driver_number"] == driver_number].copy()
    if "is_pit_out_lap" in drv_laps.columns:
        drv_laps = drv_laps[drv_laps["is_pit_out_lap"] != True]  # noqa: E712
    drv_laps = drv_laps.dropna(subset=["lap_duration"])
    if drv_laps.empty:
        return None

    fastest = drv_laps.sort_values("lap_duration").iloc[0]
    lap_time = float(fastest["lap_duration"])
    lap_number: int | None = int(fastest["lap_number"]) if pd.notna(fastest.get("lap_number")) else None

    date_start = fastest.get("date_start")
    if date_start is None or pd.isna(date_start):
        return None
    ts_start = pd.Timestamp(date_start)
    ts_end = ts_start + pd.Timedelta(seconds=lap_time)
    date_gte = ts_start.isoformat()
    date_lte = (ts_end + pd.Timedelta(seconds=0.5)).isoformat()

    client = OpenF1Client(mode="historical")
    car = client.get_car_data(
        session_key=session_key,
        driver_number=driver_number,
        date_gte=date_gte,
        date_lte=date_lte,
    )
    if car.empty or "date" not in car.columns or "speed" not in car.columns:
        return None

    car = car.sort_values("date").reset_index(drop=True)
    speeds = car["speed"].to_numpy(dtype=float)
    if len(speeds) < 2:
        return None

    # Cumulative elapsed time from real timestamps — immune to missing samples.
    cumtime_raw = (car["date"] - car["date"].iloc[0]).dt.total_seconds().to_numpy(dtype=float)

    dt = car["date"].diff().dt.total_seconds().fillna(0.0).to_numpy(dtype=float)[1:]
    speeds_ms = speeds[:-1] / 3.6
    dist_increments = np.where(np.isfinite(speeds_ms) & np.isfinite(dt), speeds_ms * dt, 0.0)
    dist = np.concatenate([[0.0], np.cumsum(dist_increments)])
    dist_grid = np.linspace(dist[0], dist[-1], N_POINTS)

    def _resample(col: str) -> np.ndarray:
        if col not in car.columns:
            return np.full(N_POINTS, np.nan)
        return np.interp(dist_grid, dist, car[col].to_numpy(dtype=float))

    # Resample cumtime using raw sample index as the independent variable, not dist.
    # dist can be flat (non-increasing) in slow/stopped sections which makes it an
    # invalid xp for np.interp and produces incorrect interpolated cumtime values.
    # The sample index is always strictly increasing, so interpolation is well-defined.
    raw_idx = np.linspace(0.0, 1.0, len(cumtime_raw))
    grid_idx = np.linspace(0.0, 1.0, N_POINTS)

    return {
        "speed": _resample("speed"),
        "throttle": _resample("throttle"),
        "brake": _resample("brake"),
        "cumtime": np.interp(grid_idx, raw_idx, cumtime_raw),  # elapsed seconds at each distance point
        "x": None,   # not available from OpenF1 car_data; Track Map uses dataset.parquet
        "y": None,
        "lap_time": lap_time,
        "lap_number": lap_number,
    }


def _build_track_map_fig(
    data_a: dict, acronym_a: str, color_a: str,
    data_b: dict, acronym_b: str, color_b: str,
) -> "go.Figure | None":
    """
    Circuit coloured by the faster driver per microsector.

    Each of the N-1 distance gaps is coloured by whichever driver had the higher
    speed at that point. Consecutive same-winner segments are merged into one
    Scatter trace to keep the figure lightweight.
    Returns None if X/Y coordinates are missing from the dataset.
    """
    x_a, y_a, spd_a = data_a["x"], data_a["y"], data_a["speed"]
    spd_b = data_b["speed"]

    if x_a is None or y_a is None:
        return None

    # Upsample to 1000 display points for a smoother circuit outline and finer
    # microsector boundaries. XY (from circuits.json, 500 pts) and speed traces
    # (from OpenF1, N_POINTS=200) may have different source lengths, so each is
    # interpolated on its own unit-interval grid before being mapped to t_fine.
    N_DISPLAY = 1000
    t_fine = np.linspace(0.0, 1.0, N_DISPLAY)

    t_xy = np.linspace(0.0, 1.0, len(x_a))
    x_fine = np.interp(t_fine, t_xy, x_a)
    y_fine = np.interp(t_fine, t_xy, y_a)

    t_spd = np.linspace(0.0, 1.0, len(spd_a))
    spd_a_fine = np.interp(t_fine, t_spd, spd_a)
    spd_b_fine = np.interp(t_fine, t_spd, spd_b)

    # winner[i] = "a" or "b" for the microsector between point i and i+1
    winner = np.where(spd_a_fine[:-1] >= spd_b_fine[:-1], "a", "b")

    fig = go.Figure()

    # Grey background track (full circuit, behind the coloured segments)
    fig.add_trace(go.Scatter(
        x=x_fine, y=y_fine,
        mode="lines",
        line=dict(color="rgba(120,120,120,0.25)", width=10),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Group consecutive same-winner segments and draw each as a single trace
    i = 0
    n = len(winner)
    legend_added: dict[str, bool] = {"a": False, "b": False}
    while i < n:
        w = winner[i]
        j = i
        while j < n and winner[j] == w:
            j += 1
        seg_x = x_fine[i: j + 1]
        seg_y = y_fine[i: j + 1]
        col = color_a if w == "a" else color_b
        label = acronym_a if w == "a" else acronym_b
        fig.add_trace(go.Scatter(
            x=seg_x, y=seg_y,
            mode="lines",
            line=dict(color=col, width=5),
            name=label,
            showlegend=not legend_added[w],
            legendgroup=w,
            hoverinfo="skip",
        ))
        legend_added[w] = True
        i = j

    fig.update_layout(
        title=f"Track Map — {acronym_a} vs {acronym_b} (faster driver by microsector)",
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        legend_title="Faster driver",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=40, b=0),
        height=520,
    )
    return fig


def _build_time_delta_fig(
    data_a: dict, acronym_a: str, color_a: str,
    data_b: dict, acronym_b: str, color_b: str,
) -> go.Figure:
    """
    Redesigned time-delta chart.

    X axis: Lap Distance (0–100%) — symmetric, matches F1 broadcast convention.
    Y axis: cumulative gap in seconds (positive = A ahead, negative = B ahead).

    Delta is computed from speed traces reconstructed to cumulative time, then
    differenced. The line is coloured by whichever driver is ahead at each point.
    Crossover markers, peak advantage markers, and a final-gap annotation are added
    so the chart is readable at a glance.
    """
    def _to_rgba(color: str, alpha: float = 0.30) -> str:
        """Convert any hex or rgb() colour to an rgba() string Plotly accepts."""
        if color.startswith("rgba("):
            return color
        if color.startswith("rgb("):
            return color.replace("rgb(", "rgba(").replace(")", f",{alpha})")
        h = color.lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    # Use real timestamp-derived cumulative time — avoids speed-reconstruction artifacts
    # that cause mid-lap delta spikes when car_data samples are unevenly distributed.
    cumtime_a = data_a["cumtime"]
    cumtime_b = data_b["cumtime"]
    delta = cumtime_b - cumtime_a   # positive = A gaining

    n = len(delta)
    x_pct = np.linspace(0.0, 100.0, n)

    # Per-point plain-language hover text
    hover_text = [
        f"{acronym_a} +{abs(d):.3f}s ahead" if d >= 0 else f"{acronym_b} +{abs(d):.3f}s ahead"
        for d in delta
    ]

    fig = go.Figure()

    # --- Shaded fill regions (opacity 0.30) ---
    pos = np.where(delta >= 0, delta, 0.0)
    neg = np.where(delta < 0, delta, 0.0)

    fig.add_trace(go.Scatter(
        x=np.concatenate([[x_pct[0]], x_pct, [x_pct[-1]]]),
        y=np.concatenate([[0], pos, [0]]),
        mode="lines",
        fill="tozeroy",
        fillcolor=_to_rgba(color_a, 0.30),
        line=dict(width=0),
        name=f"{acronym_a} faster",
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([[x_pct[0]], x_pct, [x_pct[-1]]]),
        y=np.concatenate([[0], neg, [0]]),
        mode="lines",
        fill="tozeroy",
        fillcolor=_to_rgba(color_b, 0.30),
        line=dict(width=0),
        name=f"{acronym_b} faster",
        showlegend=False,
        hoverinfo="skip",
    ))

    # --- Coloured delta line: A's colour above zero, B's colour below ---
    delta_a = np.where(delta >= 0, delta, np.nan)
    delta_b = np.where(delta < 0,  delta, np.nan)

    fig.add_trace(go.Scatter(
        x=x_pct, y=delta_a,
        mode="lines",
        connectgaps=False,
        line=dict(color=color_a, width=2.5),
        name=f"{acronym_a} faster",
        customdata=hover_text,
        hovertemplate="%{x:.0f}%  —  %{customdata}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x_pct, y=delta_b,
        mode="lines",
        connectgaps=False,
        line=dict(color=color_b, width=2.5),
        name=f"{acronym_b} faster",
        customdata=hover_text,
        hovertemplate="%{x:.0f}%  —  %{customdata}<extra></extra>",
    ))

    # --- Lead-change markers ---
    crossings = np.where(np.diff(np.sign(delta)) != 0)[0]
    if len(crossings) > 0:
        cx = x_pct[crossings]
        fig.add_trace(go.Scatter(
            x=cx, y=np.zeros(len(cx)),
            mode="markers",
            marker=dict(color="white", size=8, symbol="circle",
                        line=dict(color="rgba(0,0,0,0.6)", width=1)),
            name="Lead change",
            hovertemplate="Lead change at %{x:.0f}%<extra></extra>",
        ))

    # --- Peak advantage markers (only if meaningful gap > 0.05s) ---
    max_delta = float(np.nanmax(delta))
    min_delta = float(np.nanmin(delta))

    if max_delta > 0.05:
        idx_max = int(np.argmax(delta))
        fig.add_trace(go.Scatter(
            x=[x_pct[idx_max]], y=[max_delta],
            mode="markers",
            marker=dict(color=color_a, size=10, symbol="triangle-up",
                        line=dict(color="white", width=1)),
            name=f"Peak {acronym_a}",
            hovertemplate=f"Max {acronym_a} +{max_delta:.3f}s at %{{x:.0f}}%<extra></extra>",
        ))

    if min_delta < -0.05:
        idx_min = int(np.argmin(delta))
        fig.add_trace(go.Scatter(
            x=[x_pct[idx_min]], y=[min_delta],
            mode="markers",
            marker=dict(color=color_b, size=10, symbol="triangle-down",
                        line=dict(color="white", width=1)),
            name=f"Peak {acronym_b}",
            hovertemplate=f"Max {acronym_b} +{abs(min_delta):.3f}s at %{{x:.0f}}%<extra></extra>",
        ))

    # --- Final gap annotation at right edge ---
    final_gap = float(delta[-1])
    if abs(final_gap) < 0.001:
        gap_text = "Dead heat"
        gap_color = "white"
    elif final_gap > 0:
        gap_text = f"+{final_gap:.3f}s\n{acronym_a}"
        gap_color = color_a
    else:
        gap_text = f"{final_gap:.3f}s\n{acronym_b}"
        gap_color = color_b

    fig.add_annotation(
        x=x_pct[-1], y=final_gap,
        text=gap_text,
        showarrow=True,
        arrowhead=2,
        arrowcolor=gap_color,
        arrowwidth=1.5,
        ax=40, ay=0,
        font=dict(color=gap_color, size=12),
        xanchor="left",
    )

    # --- Title with subtitle ---
    lap_a_num = data_a.get("lap_number")
    lap_b_num = data_b.get("lap_number")
    lap_a_str = f"Lap {lap_a_num}, {data_a['lap_time']:.3f}s" if lap_a_num else f"{data_a['lap_time']:.3f}s"
    lap_b_str = f"Lap {lap_b_num}, {data_b['lap_time']:.3f}s" if lap_b_num else f"{data_b['lap_time']:.3f}s"
    title_text = (
        f"Lap Time Delta — {acronym_a} vs {acronym_b}"
        f"<br><sup>{acronym_a}: {lap_a_str}  |  {acronym_b}: {lap_b_str}</sup>"
    )

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14)),
        xaxis=dict(
            title="Lap Distance (%)",
            range=[0, 100],
            zeroline=False,
        ),
        yaxis=dict(
            title=f"Gap (s)   ↑ {acronym_a} faster   ·   {acronym_b} faster ↓",
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.6)",
            zerolinewidth=1.5,
        ),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=460,
        margin=dict(t=80, b=50, l=60, r=120),
    )
    return fig


def _driver_color_map(
    driver_numbers: list,
    team_colour_map: dict | None = None,
) -> dict:
    """
    Assign a unique colour to each driver number.

    Prefers the team colour from *team_colour_map* when available.
    If two drivers share the same team colour (teammates), the second one
    falls back to the next unused Dark24 colour so every driver is distinct.
    """
    result: dict = {}
    used_colours: set[str] = set()
    palette_idx = 0
    for d in sorted(driver_numbers):
        team_col = team_colour_map.get(d) if team_colour_map else None
        if team_col and team_col not in used_colours:
            result[d] = team_col
            used_colours.add(team_col)
        else:
            # No team colour, or teammate collision — pick next unused Dark24 colour
            while RACE_PALETTE[palette_idx % len(RACE_PALETTE)] in used_colours:
                palette_idx += 1
            col = RACE_PALETTE[palette_idx % len(RACE_PALETTE)]
            result[d] = col
            used_colours.add(col)
            palette_idx += 1
    return result


def _resolve_pair_colours(
    drv_a: int,
    drv_b: int,
    team_colour_map: dict | None,
) -> tuple[str, str]:
    """
    Return a pair of distinct colours for a two-driver telemetry comparison.

    Rules:
    - Driver A always gets their team colour (falls back to Dark24[0] if unknown).
    - Driver B gets their team colour when it differs from Driver A's.
    - If both drivers share the same team colour (teammates), Driver B gets
      the next Dark24 colour that is not already used by Driver A.
    """
    tc = team_colour_map or {}
    col_a = tc.get(drv_a) or RACE_PALETTE[0]
    col_b_candidate = tc.get(drv_b)

    if col_b_candidate and col_b_candidate != col_a:
        # Different teams — use both team colours directly
        return col_a, col_b_candidate

    # Same team (or Driver B has no team colour) — find an unused Dark24 colour for B
    for candidate in RACE_PALETTE:
        if candidate != col_a:
            return col_a, candidate

    # Unreachable in practice (Dark24 has 24 colours), but safe fallback
    return col_a, "#888888"


@st.cache_data(ttl=30, show_spinner=False)
def _cached_analysis(laps_hash: str, _analyser: RaceAnalyser, laps_remaining: int) -> dict:
    """Cache the 4 analysis results so reruns don't recompute.

    *laps_hash* is a hashable key derived from the lap data so the cache
    invalidates when new laps arrive.  *_analyser* is prefixed with ``_``
    so Streamlit skips hashing it (it's not picklable).
    """
    return {
        "rolling_pace": _analyser.rolling_pace(window=5),
        "gap_to_leader": _analyser.gap_to_leader(),
        "undercuts": _analyser.detect_undercuts(),
        "projections": _analyser.project_finishing_order(laps_remaining),
        "driver_numbers": sorted(_analyser._clean["driver_number"].unique()),
        "tyre_deg": _analyser.tyre_degradation(),
        "pace_summary": _analyser.pace_summary(),
    }


def _render_charts(
    analyser: RaceAnalyser,
    laps_remaining: int,
    container,
    driver_map: dict | None = None,
    selected_drivers: list | None = None,
    team_colour_map: dict | None = None,
) -> None:
    """Render all Race-Pace charts inside *container*.

    Parameters
    ----------
    driver_map : dict, optional
        Mapping of ``{driver_number: acronym}`` e.g. ``{44: 'HAM'}``.
        Falls back to the raw driver number string when not provided or
        when a number has no entry in the map.
    selected_drivers : list, optional
        Subset of driver numbers to display. Defaults to all drivers.
    team_colour_map : dict, optional
        Mapping of ``{driver_number: "#RRGGBB"}`` for team colours.
        Drivers with a unique team colour use it; teammates fall back to Dark24.
    """
    if driver_map is None:
        driver_map = {}

    def _label(num) -> str:
        """Return acronym if known, otherwise the number as a string."""
        return driver_map.get(int(num), str(num)) if driver_map else str(num)

    # Build a hash key from the laps data so the cache invalidates on new data
    laps_hash = str(hash(tuple(analyser._laps["lap_number"].tolist()[:50])))
    results = _cached_analysis(laps_hash, analyser, laps_remaining)

    # Resolve the active driver set (filter or all)
    active_drivers = set(selected_drivers) if selected_drivers else set(results["driver_numbers"])
    cmap = _driver_color_map(list(active_drivers), team_colour_map=team_colour_map)

    # ---- Chart 1: Rolling Race Pace --------------------------------
    if st.session_state.get("show_rolling_pace", True):
        container.markdown("### Rolling Race Pace")
        rp = results["rolling_pace"]
        fig1 = go.Figure()
        for drv in [d for d in rp.columns if d in active_drivers]:
            fig1.add_trace(go.Scatter(
                x=rp.index, y=rp[drv],
                mode="lines", name=_label(drv),
                line=dict(color=cmap.get(drv, "#888"), width=2),
            ))
        # Session median pace (dashed reference)
        all_vals = rp.values[~np.isnan(rp.values)]
        if len(all_vals):
            median_pace = float(np.median(all_vals))
            fig1.add_hline(
                y=median_pace, line_dash="dash", line_color="white",
                opacity=0.5,
                annotation_text=f"Median {median_pace:.2f}s",
                annotation_position="top left",
            )
        fig1.update_layout(
            xaxis_title="Lap Number", yaxis_title="Rolling Avg Lap Time (s)",
            hovermode="x unified", legend_title="Driver",
        )
        container.plotly_chart(fig1, width="stretch", key="tab4_rolling_pace")

    # ---- Chart 2 & 3: Gap to Leader + Undercut markers -------------
    if st.session_state.get("show_gap_to_leader", True):
        container.markdown("### Gap to Leader")
        gap = results["gap_to_leader"]
        undercuts = results["undercuts"]

        fig2 = go.Figure()
        for drv in [d for d in gap.columns if d in active_drivers]:
            fig2.add_trace(go.Scatter(
                x=gap.index, y=gap[drv],
                mode="lines", name=_label(drv),
                line=dict(color=cmap.get(drv, "#888"), width=2),
                fill="tozeroy", opacity=0.3,
            ))
        # Overlay undercut/overcut markers (only when both drivers are selected)
        visible_undercuts = [
            e for e in undercuts
            if e["attacking_driver"] in active_drivers and e["defending_driver"] in active_drivers
        ]
        for evt in visible_undercuts:
            lap = evt["lap"]
            atk = evt["attacking_driver"]
            kind = evt["type"]
            marker_symbol = "triangle-up" if kind == "undercut" else "triangle-down"
            try:
                y_val = gap.loc[lap, atk].item()  # type: ignore[union-attr]
            except (KeyError, AttributeError):
                y_val = 0.0
            label = f"Lap {lap}: {_label(atk)} {kind} {_label(evt['defending_driver'])}"
            fig2.add_trace(go.Scatter(
                x=[lap], y=[y_val],
                mode="markers+text",
                marker=dict(
                    symbol=marker_symbol, size=14,
                    color="lime" if kind == "undercut" else "orange",
                    line=dict(width=1, color="white"),
                ),
                text=[f"{'▲' if kind == 'undercut' else '▽'}"],
                textposition="top center",
                hovertext=[label],
                hoverinfo="text",
                showlegend=False,
            ))
        fig2.update_layout(
            xaxis_title="Lap Number", yaxis_title="Gap to Leader (s)",
            hovermode="x unified", legend_title="Driver",
        )
        container.plotly_chart(fig2, width="stretch", key="tab4_gap_to_leader")

    # ---- Chart 4: Projected Finishing Order -------------------------
    if st.session_state.get("show_projected_order", True):
        container.markdown("### Projected Finishing Order")
        projections = [p for p in results["projections"] if p["driver"] in active_drivers]
        if projections:
            # Winner time comes from the first active (non-DNF) driver
            active = [p for p in projections if p.get("status") != "DNF"]
            winner_time = active[0]["projected_total_time"] if active else 0
            proj_data = []
            for p in projections:
                is_dnf = p.get("status") == "DNF"
                gap_to_win = 0.0 if is_dnf else p["projected_total_time"] - winner_time
                change = p.get("position_change")
                if is_dnf:
                    bar_color = "#555555"
                elif change is not None and change > 0:
                    bar_color = "#2ecc71"
                elif change is not None and change < 0:
                    bar_color = "#e74c3c"
                else:
                    bar_color = "#95a5a6"
                cur = p.get("current_position", "?")
                proj_pos = p["projected_position"]
                label = "DNF" if is_dnf else f"P{cur} → P{proj_pos}"
                proj_data.append({
                    "driver": _label(p["driver"]),
                    "gap": round(gap_to_win, 2),
                    "color": bar_color,
                    "label": label,
                })
            # Reverse so P1 is at the top of the horizontal bar chart
            proj_data = proj_data[::-1]
            fig4 = go.Figure(go.Bar(
                x=[d["gap"] for d in proj_data],
                y=[d["driver"] for d in proj_data],
                orientation="h",
                marker_color=[d["color"] for d in proj_data],
                text=[d["label"] for d in proj_data],
                textposition="outside",
                hovertemplate="Driver %{y}<br>Gap: +%{x:.2f}s<extra></extra>",
            ))
            fig4.update_layout(
                xaxis_title="Projected Gap to Winner (s)",
                yaxis_title="Driver",
                showlegend=False,
            )
            container.plotly_chart(fig4, width="stretch", key="tab4_projected_order")
        else:
            container.info("Not enough data to project finishing order.")

    # ---- Chart 5: Average Race Pace --------------------------------
    if st.session_state.get("show_avg_pace", True):
        container.markdown("### Average Race Pace")
        pace_sum = results["pace_summary"]
        if pace_sum.empty:
            container.info("Not enough clean lap data for pace summary.")
        else:
            pace_sum = pace_sum.reset_index()
            pace_sum = pace_sum[pace_sum["driver_number"].isin(active_drivers)]
            pace_sum["label"] = pace_sum["driver_number"].apply(_label)
            pace_sum = pace_sum.sort_values("median_pace", ascending=False)  # fastest at top
            fig5 = go.Figure(go.Bar(
                x=pace_sum["median_pace"],
                y=pace_sum["label"],
                orientation="h",
                marker=dict(
                    color=pace_sum["std_pace"],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title="Pace Std Dev (s)"),
                ),
                customdata=pace_sum[["mean_pace", "std_pace", "fastest_lap", "lap_count"]].values,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Median: %{x:.3f}s<br>"
                    "Mean: %{customdata[0]:.3f}s<br>"
                    "Std Dev: %{customdata[1]:.3f}s<br>"
                    "Fastest lap: %{customdata[2]:.3f}s<br>"
                    "Laps counted: %{customdata[3]}<extra></extra>"
                ),
            ))
            fig5.update_layout(
                xaxis_title="Median Lap Time (s)",
                yaxis_title="Driver",
                showlegend=False,
            )
            container.plotly_chart(fig5, width="stretch", key="tab4_avg_pace")

    # ---- Chart 6: Race Pace Ranking --------------------------------
    if st.session_state.get("show_pace_ranking", True):
        container.markdown("### Race Pace Ranking")
        pace_sum = results["pace_summary"]
        if pace_sum.empty:
            container.info("Not enough clean lap data for pace ranking.")
        else:
            pace_sum = pace_sum.reset_index()
            pace_sum = pace_sum[pace_sum["driver_number"].isin(active_drivers)]
            pace_sum["label"] = pace_sum["driver_number"].apply(_label)
            ranked = pace_sum.sort_values("mean_pace")
            bar_colors = [cmap.get(int(row["driver_number"]), "#888") for _, row in ranked.iterrows()]
            fig6 = go.Figure(go.Bar(
                x=ranked["label"],
                y=ranked["mean_pace"],
                error_y=dict(type="data", array=ranked["std_pace"].tolist(), visible=True),
                marker_color=bar_colors,
                text=ranked["fastest_lap"].round(3).astype(str) + "s",
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Mean pace: %{y:.3f}s<br>"
                    "Fastest lap: %{text}<extra></extra>"
                ),
            ))
            fig6.update_layout(
                xaxis_title="Driver",
                yaxis_title="Mean Lap Time (s)",
                showlegend=False,
            )
            container.plotly_chart(fig6, width="stretch", key="tab4_pace_ranking")

    # ---- Chart 7: Tyre Degradation by Compound ---------------------
    if st.session_state.get("show_tyre_deg", True):
        container.markdown("### Tyre Degradation by Compound")
        tyre_deg = results["tyre_deg"]
        tyre_deg = tyre_deg[tyre_deg["driver_number"].isin(active_drivers)]
        if tyre_deg.empty:
            container.info("Stint data unavailable — tyre degradation chart requires stint information.")
        else:
            COMPOUND_COLORS = {
                "SOFT": "#e8002d",
                "MEDIUM": "#ffd700",
                "HARD": "#cccccc",
                "INTERMEDIATE": "#39b54a",
                "WET": "#0067ff",
            }
            fig7 = go.Figure()
            for compound, grp in tyre_deg.groupby("compound"):
                color = COMPOUND_COLORS.get(str(compound).upper(), "#888888")
                fig7.add_trace(go.Scatter(
                    x=grp["mean_pace"],
                    y=grp["deg_per_lap"],
                    mode="markers",
                    name=str(compound),
                    marker=dict(color=color, size=10, line=dict(width=1, color="white")),
                    customdata=grp[["driver_number", "laps_in_stint"]].assign(
                        label=grp["driver_number"].apply(_label)
                    )[["label", "laps_in_stint"]].values,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b> — " + str(compound) + "<br>"
                        "Mean pace: %{x:.3f}s<br>"
                        "Degradation: %{y:+.4f}s/lap<br>"
                        "Laps in stint: %{customdata[1]}<extra></extra>"
                    ),
                ))
            fig7.add_hline(
                y=0, line_dash="dash", line_color="white",
                opacity=0.4, annotation_text="No degradation",
                annotation_position="top left",
            )
            fig7.update_layout(
                xaxis_title="Mean Stint Pace (s)",
                yaxis_title="Degradation (s / lap)",
                legend_title="Compound",
            )
            container.plotly_chart(fig7, width="stretch", key="tab4_tyre_deg")


# ---- Tab 3 content -----------------------------------------------

with tab3:
    st.subheader("Race Dashboard")

    # ---- Race Analysis ---------------------------------------------
    _mode_options = ["🗂️ Historical (select a race)"]
    if st.session_state.get("show_live_mode", False):
        _mode_options.append("📡 Live (current race weekend)")
    mode = st.radio(
        "Mode",
        _mode_options,
        horizontal=True,
        key="rp_mode",
    )

    if mode.startswith("🗂️"):
        # ---- HISTORICAL MODE ----
        h1, h2 = st.columns(2)
        hist_year = h1.selectbox("Year", [2026, 2025, 2024, 2023, 2022], key="rp_year")
        _gp_options = [
            "Australian Grand Prix",
            "Bahrain Grand Prix",
            "Saudi Arabian Grand Prix",
            "Japanese Grand Prix",
            "Chinese Grand Prix",
            "Miami Grand Prix",
            "Emilia Romagna Grand Prix",
            "Monaco Grand Prix",
            "Canadian Grand Prix",
            "Spanish Grand Prix",
            "Austrian Grand Prix",
            "British Grand Prix",
            "Hungarian Grand Prix",
            "Belgian Grand Prix",
            "Dutch Grand Prix",
            "Italian Grand Prix",
            "Azerbaijan Grand Prix",
            "Singapore Grand Prix",
            "United States Grand Prix",
            "Mexico City Grand Prix",
            "São Paulo Grand Prix",
            "Las Vegas Grand Prix",
            "Qatar Grand Prix",
            "Abu Dhabi Grand Prix",
        ]
        hist_gp = h2.selectbox(
            "Grand Prix",
            options=_gp_options,
            index=_gp_options.index("Italian Grand Prix"),
            key="rp_gp",
        )
        # Invalidate cached sessions when year or GP changes
        if st.session_state["rp_fetched_for"] != (hist_year, hist_gp):
            st.session_state["rp_available_sessions"] = None
            st.session_state["rp_hist_analyser"] = None

        # Step 1 — Fetch available sessions for the selected GP
        if st.button("Fetch Sessions", key="rp_fetch_sessions"):
            with st.spinner("Fetching sessions from OpenF1..."):
                _client = OpenF1Client(mode="historical")
                _sessions = _client.get_sessions(hist_year, hist_gp)
            if _sessions.empty:
                st.error(
                    "No sessions found. Either the year/Grand Prix name is wrong, "
                    "or the OpenF1 API is unreachable (check your internet connection)."
                )
            else:
                st.session_state["rp_available_sessions"] = _sessions
                st.session_state["rp_fetched_for"] = (hist_year, hist_gp)

        # Step 2 — Session selector + Load button (shown once sessions are fetched)
        if st.session_state["rp_available_sessions"] is not None:
            _sessions_df = st.session_state["rp_available_sessions"]
            _session_names = _sessions_df["session_name"].tolist() if "session_name" in _sessions_df.columns else []
            # Default to Race if available
            _default_idx = next(
                (i for i, n in enumerate(_session_names) if "race" in str(n).lower() and "sprint" not in str(n).lower()),
                0,
            )
            hist_session_name = st.selectbox(
                "Session",
                options=_session_names,
                index=_default_idx,
                key="rp_session_name",
            )

            if st.button("Load Session Data", key="rp_load"):
                _row = _sessions_df[_sessions_df["session_name"] == hist_session_name].iloc[0]
                session_key = int(_row["session_key"])
                with st.spinner(f"Loading {hist_session_name} data from OpenF1..."):
                    client = OpenF1Client(mode="historical")
                    laps = client.get_laps(session_key)
                    stints = client.get_stints(session_key)
                    position = client.get_position(session_key)
                    drivers_df = client.get_drivers(session_key)

                if laps.empty:
                    st.error("No lap data returned for this session.")
                else:
                    driver_map: dict = {}
                    team_colour_map: dict = {}
                    if not drivers_df.empty and "driver_number" in drivers_df.columns and "name_acronym" in drivers_df.columns:
                        driver_map = {
                            int(row["driver_number"]): str(row["name_acronym"])
                            for _, row in drivers_df.iterrows()
                            if pd.notna(row["driver_number"]) and pd.notna(row["name_acronym"])
                        }
                        if "team_colour" in drivers_df.columns:
                            team_colour_map = {
                                int(row["driver_number"]): f"#{str(row['team_colour']).strip().upper()}"
                                for _, row in drivers_df.iterrows()
                                if pd.notna(row["driver_number"]) and pd.notna(row.get("team_colour"))
                            }
                    st.session_state["rp_hist_analyser"] = RaceAnalyser(laps, stints, position)
                    st.session_state["rp_hist_driver_map"] = driver_map
                    st.session_state["rp_hist_team_colour_map"] = team_colour_map
                    st.session_state["rp_hist_session_key"] = session_key
        else:
            st.info("Select a year and Grand Prix, then click **Fetch Sessions** to see available sessions.")

        # Render charts from session state if data has been loaded
        if st.session_state["rp_hist_analyser"] is not None:
            _analyser = st.session_state["rp_hist_analyser"]
            _dmap = st.session_state["rp_hist_driver_map"]
            _all_drvs = sorted(_analyser._clean["driver_number"].dropna().unique().tolist())
            # ---- Fastest Lap Telemetry Comparison ------------------
            if st.session_state.get("show_telemetry", True):
                st.markdown("### Fastest Lap Telemetry Comparison")
            _tc1, _tc2, _tc3 = st.columns(3)
            _tel_drv_a = _tc1.selectbox(
                "Driver A",
                options=_all_drvs,
                format_func=lambda n: _dmap.get(int(n), str(n)),
                key="rp_tel_drv_a",
            )
            _tel_drv_b = _tc2.selectbox(
                "Driver B",
                options=_all_drvs,
                index=min(1, len(_all_drvs) - 1),
                format_func=lambda n: _dmap.get(int(n), str(n)),
                key="rp_tel_drv_b",
            )
            _tel_channel = _tc3.selectbox(
                "Channel",
                options=SPECIAL_CHANNELS + list(TRACE_COLS.keys()),
                key="rp_tel_channel",
            )

            def _dlabel(n: int) -> str:
                return _dmap.get(n, str(n))

            if not st.session_state.get("show_telemetry", True):
                pass  # section hidden by user
            elif _tel_drv_a is None or _tel_drv_b is None:
                st.info("Select two drivers to compare telemetry.")
            else:
                _drv_a_int, _drv_b_int = int(_tel_drv_a), int(_tel_drv_b)
                _col_a, _col_b = _resolve_pair_colours(
                    _drv_a_int, _drv_b_int,
                    st.session_state.get("rp_hist_team_colour_map"),
                )
                _tel_cmap = _driver_color_map(
                    list(_all_drvs),
                    team_colour_map=st.session_state.get("rp_hist_team_colour_map"),
                )

                _hist_sk: int | str | None = st.session_state.get("rp_hist_session_key")
                _hist_laps = _analyser._laps

                if _hist_sk is None:
                    st.info("Session not loaded yet. Load a race first.")
                elif _tel_channel in SPECIAL_CHANNELS:
                    with st.spinner("Fetching fastest-lap telemetry from OpenF1..."):
                        _data_a = _fetch_fastest_lap_all_openf1(_hist_sk, _drv_a_int, _hist_laps)
                        _data_b = _fetch_fastest_lap_all_openf1(_hist_sk, _drv_b_int, _hist_laps)
                    if _data_a is None:
                        st.caption(f"Could not fetch telemetry for {_dlabel(_drv_a_int)} from OpenF1.")
                    elif _data_b is None:
                        st.caption(f"Could not fetch telemetry for {_dlabel(_drv_b_int)} from OpenF1.")
                    elif _tel_channel == "Track Map":
                        _gp_name = st.session_state.get("rp_gp", "")
                        _circuit = circuits.get(_gp_name)
                        if _circuit is None:
                            st.info(f"No circuit data for '{_gp_name}'. Run `python src/generate_circuits.py` to generate it.")
                        else:
                            _cx = np.asarray(_circuit["x"], dtype=float)
                            _cy = np.asarray(_circuit["y"], dtype=float)
                            _acronym_a, _acronym_b = _dlabel(_drv_a_int), _dlabel(_drv_b_int)
                            _map_a = {"x": _cx, "y": _cy, "speed": _data_a["speed"]}
                            _map_b = {"x": _cx, "y": _cy, "speed": _data_b["speed"]}
                            _special_fig = _build_track_map_fig(
                                _map_a, _acronym_a, _col_a,
                                _map_b, _acronym_b, _col_b,
                            )
                            if _special_fig is not None:
                                st.plotly_chart(_special_fig, width="stretch", key="tab3_track_map_hist")
                    elif _tel_channel == "Time Delta":
                        _special_fig = _build_time_delta_fig(
                            _data_a, _dlabel(_drv_a_int), _col_a,
                            _data_b, _dlabel(_drv_b_int), _col_b,
                        )
                        st.plotly_chart(_special_fig, width="stretch", key="tab3_time_delta_hist")
                else:
                    _tel_fig = go.Figure()
                    _any_trace = False
                    for _drv_int in (_drv_a_int, _drv_b_int):
                        with st.spinner(f"Fetching {_tel_channel} telemetry for {_dlabel(_drv_int)}..."):
                            _trace, _lap_t, _lap_num = _fetch_fastest_lap_openf1(
                                _hist_sk, _drv_int, _hist_laps, _tel_channel
                            )
                        if _trace is None:
                            st.caption(f"Could not fetch {_tel_channel} telemetry for {_dlabel(_drv_int)} from OpenF1.")
                            continue
                        _any_trace = True
                        _lap_label = f"Lap {_lap_num}, {_lap_t:.3f}s" if _lap_num is not None else f"{_lap_t:.3f}s"
                        _tel_fig.add_trace(go.Scatter(
                            x=np.arange(N_POINTS), y=_trace,
                            mode="lines",
                            name=f"{_dlabel(_drv_int)} ({_lap_label})",
                            line=dict(color=_tel_cmap.get(_drv_int, "#888"), width=2.5),
                        ))
                    if _any_trace:
                        _tel_fig.update_layout(
                            title=(
                                f"Fastest Lap {_tel_channel} — "
                                f"{_dlabel(_drv_a_int)} vs {_dlabel(_drv_b_int)}"
                            ),
                            xaxis_title="Normalised Distance (0–200 points)",
                            yaxis_title=_tel_channel,
                            legend_title="Driver",
                            hovermode="x unified",
                        )
                        st.plotly_chart(_tel_fig, width="stretch", key="tab3_tel_hist")

            st.divider()

            _selected = st.multiselect(
                "Drivers",
                options=_all_drvs,
                default=_all_drvs,
                format_func=lambda n: _dmap.get(int(n), str(n)),
                key="rp_driver_filter",
            )
            _render_charts(
                _analyser,
                0,
                st,
                driver_map=_dmap,
                selected_drivers=_selected or _all_drvs,
                team_colour_map=st.session_state.get("rp_hist_team_colour_map"),
            )

            # ── Race Intelligence Chat Agent ─────────────────────────
            _hist_cache_key = st.session_state.get("rp_hist_session_key")
            if _hist_cache_key is not None:
                _render_chat_panel(_analyser, _dmap, session_cache_key=_hist_cache_key)

    else:
        # ---- LIVE MODE ----
        if not st.session_state.get("show_live_mode", False):
            st.info("Enable **Live Mode** in the sidebar (Visible Charts) to access live race weekend data.")
        else:
            st.markdown(
                """
                <style>
                @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
                .live-badge {
                    display: inline-block;
                    font-size: 1.4rem;
                    font-weight: 700;
                    color: #ff4444;
                    animation: pulse 1.5s ease-in-out infinite;
                }
                </style>
                <span class="live-badge">🔴 LIVE</span>
                """,
                unsafe_allow_html=True,
            )

            refresh_interval = st.selectbox(
                "Auto-refresh interval",
                options=[10, 30, 60],
                format_func=lambda s: f"{s}s",
                key="rp_refresh",
            )
            live_laps_remaining = st.number_input(
                "Estimated laps remaining", min_value=0, value=20, key="rp_live_laps",
            )
    
            chart_container = st.empty()
    
            try:
                # Persist live client across reruns so watermarks survive
                if st.session_state["rp_live_client"] is None:
                    st.session_state["rp_live_client"] = OpenF1Client(mode="live")
                client = st.session_state["rp_live_client"]
    
                with st.spinner("Fetching live data from OpenF1..."):
                    laps = client.get_live_laps()
                    stints = client.get_live_stints()
                    position = client.get_live_position()
                    live_drivers_df = client.get_live_drivers()
                    st.session_state["rp_last_refresh"] = datetime.now().strftime("%H:%M:%S")
    
                # Build / refresh driver map and team colour map
                if not live_drivers_df.empty and "driver_number" in live_drivers_df.columns and "name_acronym" in live_drivers_df.columns:
                    st.session_state["rp_live_driver_map"] = {
                        int(row["driver_number"]): str(row["name_acronym"])
                        for _, row in live_drivers_df.iterrows()
                        if pd.notna(row["driver_number"]) and pd.notna(row["name_acronym"])
                    }
                    if "team_colour" in live_drivers_df.columns:
                        st.session_state["rp_live_team_colour_map"] = {
                            int(row["driver_number"]): f"#{str(row['team_colour']).strip().upper()}"
                            for _, row in live_drivers_df.iterrows()
                            if pd.notna(row["driver_number"]) and pd.notna(row.get("team_colour"))
                        }
    
                if laps.empty:
                    st.warning("No active session found. Try during a race weekend.")
                else:
                    analyser = RaceAnalyser(laps, stints, position)
                    _live_dmap = st.session_state["rp_live_driver_map"]
                    _all_drvs = sorted(analyser._clean["driver_number"].dropna().unique().tolist())
    
                    # ---- Fastest Lap Telemetry Comparison (live) -------
                    if st.session_state.get("show_telemetry", True):
                        st.markdown("### Fastest Lap Telemetry Comparison")
                    _lc1, _lc2, _lc3 = st.columns(3)
                    _ltel_drv_a = _lc1.selectbox(
                        "Driver A",
                        options=_all_drvs,
                        format_func=lambda n: _live_dmap.get(int(n), str(n)),
                        key="rp_live_tel_drv_a",
                    )
                    _ltel_drv_b = _lc2.selectbox(
                        "Driver B",
                        options=_all_drvs,
                        index=min(1, len(_all_drvs) - 1),
                        format_func=lambda n: _live_dmap.get(int(n), str(n)),
                        key="rp_live_tel_drv_b",
                    )
                    _ltel_channel = _lc3.selectbox(
                        "Channel",
                        options=SPECIAL_CHANNELS + list(TRACE_COLS.keys()),
                        key="rp_live_tel_channel",
                    )
    
                    def _live_dlabel(n: int) -> str:
                        return _live_dmap.get(n, str(n))
    
                    if not st.session_state.get("show_telemetry", True):
                        pass  # section hidden by user
                    elif _ltel_drv_a is None or _ltel_drv_b is None:
                        st.info("Select two drivers to compare telemetry.")
                    else:
                        _ldrv_a_int, _ldrv_b_int = int(_ltel_drv_a), int(_ltel_drv_b)
                        _lcol_a, _lcol_b = _resolve_pair_colours(
                            _ldrv_a_int, _ldrv_b_int,
                            st.session_state.get("rp_live_team_colour_map"),
                        )
                        _ltel_cmap = _driver_color_map(
                            list(_all_drvs),
                            team_colour_map=st.session_state.get("rp_live_team_colour_map"),
                        )
    
                        # Live mode: use "latest" as the session key for car_data calls
                        _live_laps = analyser._laps
    
                        if _ltel_channel in SPECIAL_CHANNELS:
                            with st.spinner("Fetching fastest-lap telemetry from OpenF1..."):
                                _ldata_a = _fetch_fastest_lap_all_openf1("latest", _ldrv_a_int, _live_laps)
                                _ldata_b = _fetch_fastest_lap_all_openf1("latest", _ldrv_b_int, _live_laps)
                            if _ldata_a is None:
                                st.caption(f"Could not fetch telemetry for {_live_dlabel(_ldrv_a_int)} from OpenF1.")
                            elif _ldata_b is None:
                                st.caption(f"Could not fetch telemetry for {_live_dlabel(_ldrv_b_int)} from OpenF1.")
                            elif _ltel_channel == "Track Map":
                                _live_gp = st.session_state.get("rp_live_meeting_name", "")
                                _lcircuit = circuits.get(_live_gp) if _live_gp else None
                                if _lcircuit is None:
                                    st.info("Circuit outline unavailable for the current live session.")
                                else:
                                    _lcx = np.asarray(_lcircuit["x"], dtype=float)
                                    _lcy = np.asarray(_lcircuit["y"], dtype=float)
                                    _lacronym_a, _lacronym_b = _live_dlabel(_ldrv_a_int), _live_dlabel(_ldrv_b_int)
                                    _lmap_a = {"x": _lcx, "y": _lcy, "speed": _ldata_a["speed"]}
                                    _lmap_b = {"x": _lcx, "y": _lcy, "speed": _ldata_b["speed"]}
                                    _lspecial_fig = _build_track_map_fig(
                                        _lmap_a, _lacronym_a, _lcol_a,
                                        _lmap_b, _lacronym_b, _lcol_b,
                                    )
                                    if _lspecial_fig is not None:
                                        st.plotly_chart(_lspecial_fig, width="stretch", key="tab3_track_map_live")
                            elif _ltel_channel == "Time Delta":
                                _lspecial_fig = _build_time_delta_fig(
                                    _ldata_a, _live_dlabel(_ldrv_a_int), _lcol_a,
                                    _ldata_b, _live_dlabel(_ldrv_b_int), _lcol_b,
                                )
                                st.plotly_chart(_lspecial_fig, width="stretch", key="tab3_time_delta_live")
                        else:
                            _ltel_fig = go.Figure()
                            _lany_trace = False
                            for _ldrv_int in (_ldrv_a_int, _ldrv_b_int):
                                with st.spinner(f"Fetching {_ltel_channel} telemetry for {_live_dlabel(_ldrv_int)}..."):
                                    _ltrace, _llap_t, _llap_num = _fetch_fastest_lap_openf1(
                                        "latest", _ldrv_int, _live_laps, _ltel_channel
                                    )
                                if _ltrace is None:
                                    st.caption(f"Could not fetch {_ltel_channel} telemetry for {_live_dlabel(_ldrv_int)} from OpenF1.")
                                    continue
                                _lany_trace = True
                                _llap_label = f"Lap {_llap_num}, {_llap_t:.3f}s" if _llap_num is not None else f"{_llap_t:.3f}s"
                                _ltel_fig.add_trace(go.Scatter(
                                    x=np.arange(N_POINTS), y=_ltrace,
                                    mode="lines",
                                    name=f"{_live_dlabel(_ldrv_int)} ({_llap_label})",
                                    line=dict(color=_ltel_cmap.get(_ldrv_int, "#888"), width=2.5),
                                ))
                            if _lany_trace:
                                _ltel_fig.update_layout(
                                    title=(
                                        f"Fastest Lap {_ltel_channel} — "
                                        f"{_live_dlabel(_ldrv_a_int)} vs {_live_dlabel(_ldrv_b_int)}"
                                    ),
                                    xaxis_title="Normalised Distance (0–200 points)",
                                    yaxis_title=_ltel_channel,
                                    legend_title="Driver",
                                    hovermode="x unified",
                                )
                                st.plotly_chart(_ltel_fig, width="stretch", key="tab3_tel_live")
    
                    st.divider()
    
                    _selected = st.multiselect(
                        "Drivers",
                        options=_all_drvs,
                        default=_all_drvs,
                        format_func=lambda n: _live_dmap.get(int(n), str(n)),
                        key="rp_driver_filter",
                    )
                    with chart_container.container():
                        _render_charts(
                            analyser,
                            live_laps_remaining,
                            st,
                            driver_map=_live_dmap,
                            selected_drivers=_selected or _all_drvs,
                            team_colour_map=st.session_state.get("rp_live_team_colour_map"),
                        )
                    # ── Race Intelligence Chat Agent ──────────────────
                    _render_chat_panel(analyser, _live_dmap, session_cache_key="live")
            except Exception as exc:
                st.warning(f"Live refresh failed: {exc}")
    
            # Only auto-rerun if still on live mode — prevents resetting other tabs
            if st.session_state.get("rp_mode", "").startswith("📡"):
                time.sleep(refresh_interval)
                st.rerun()
