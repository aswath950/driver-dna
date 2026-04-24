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
import logging
import os
import sys
import threading
import time
from datetime import datetime
from typing import cast
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent))

from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from model import load_model, load_and_prepare, train_model, evaluate_model, save_model, FEATURE_COLS, DATA_DIR, MODELS_DIR
from pipeline import extract_session_telemetry, save_dataset, save_meta, VALID_SESSION_TYPES
from openf1 import OpenF1Client
from race_engine import RaceAnalyser
from features import (
    N_POINTS, TRACE_COLS, SPECIAL_CHANNELS,
    RADAR_FEATURES, EXTENDED_RADAR_FEATURES, FEATURE_LABELS,
    compute_extended_features, compute_mean_trace, compute_sector_means,
    get_shap_explainer, compute_shap_for_prediction,
    _validate_feature_values, _sanitize_driver_name, classify_archetype,
)
from viz import (
    _build_throttle_map_fig, _fetch_fastest_lap_openf1,
    _fetch_fastest_lap_all_openf1, _build_track_map_fig,
    _build_time_delta_fig, _driver_color_map, _resolve_pair_colours,
)
from llm_layer import (
    _build_explain_prompt,
    _call_openai_llm, _call_openai_chat,
    _ra_build_driver_data_block, _ra_run_reflexion,
    _rag_retrieve_top_k, _RAG_HISTORICAL_PROFILES,
    _rag_run_dna_match,
    _rc_run_report,
    _ca_sanitize_input, _ca_dispatch_tool,
    _CA_TOOLS, _CA_MAX_TOOLS_PER_ROUND,
    _CA_SYSTEM_PROMPT, _CA_SYSTEM_VERBOSE,
    _RA_RATE_LIMIT_SECS, _RAG_RATE_LIMIT_SECS, _RC_RATE_LIMIT_SECS,
)

DATASET_PATH = DATA_DIR / "dataset.parquet"
CIRCUITS_PATH = DATA_DIR / "circuits.json"
ACCURACY_PATH = MODELS_DIR / "accuracy.txt"
METRICS_PATH  = MODELS_DIR / "metrics.json"
DATASET_META_PATH = DATA_DIR / "dataset_meta.json"

# ── Structured logging ───────────────────────────────────────────────────────
logging.basicConfig(format="%(message)s", level=logging.INFO)


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
    # Tab 1 — Reflexion Analyst (ra_ prefix)
    "ra_last_analyse_ts":  0.0,      # float — rate limit timestamp
    "ra_result":           None,     # dict | None — {iterations, final_narrative}
    "ra_analysed_driver":  None,     # str | None
    # Tab 1 — RAG DNA Matching (rag_ prefix)
    "rag_last_match_ts":   0.0,      # float — rate limit timestamp
    "rag_result":          None,     # dict | None — {driver, matches, narrative}
    "rag_analysed_driver": None,     # str | None
    # Tab 1 — DNA Report Card (rc_ prefix)
    "rc_last_report_ts":   0.0,      # float — rate limit timestamp
    "rc_result":           None,     # dict | None — validated report card
    "rc_analysed_driver":  None,     # str | None
    # Global — AI response verbosity toggle
    "ai_response_mode":    "Concise",  # "Concise" | "Detailed"
    # Global — custom tab navigation
    "active_tab":          "Driver Radar",  # one of the three tab names
    # Tab 1 — unified AI panel
    "radar_ai_feature":    "Driver Style Analyst",
    "radar_ai_driver":     None,
    # Tab 4 — Pipeline: dataset download
    "pipe_running":  False,
    "pipe_done":     False,
    "pipe_error":    None,
    "pipe_log":      [],
    "pipe_progress": 0.0,
    "pipe_status":   "",
    # Tab 4 — Pipeline: model training
    "train_running":  False,
    "train_done":     False,
    "train_error":    None,
    "train_log":      [],
    "train_progress": 0.0,
    "train_status":   "",
}

st.set_page_config(page_title="Driver DNA Fingerprinter", layout="wide")

# ── Neo-brutalist F1 global CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&display=swap');

html, body, .stApp {
    font-family: 'Space Grotesk', sans-serif !important;
}
/* Target text-bearing Streamlit elements — no span/SVG selectors to avoid breaking Material Icons ligatures */
p, label, div[data-testid="stMarkdownContainer"],
div[data-testid="stText"], div[data-testid="stCaption"],
div[data-testid="stMetricLabel"], div[data-testid="stMetricValue"],
div[data-testid="stMetricDelta"],
div[data-testid="stSelectbox"] label, div[data-testid="stMultiSelect"] label,
div[data-testid="stRadio"] label, div[data-testid="stCheckbox"] label,
button, input, textarea {
    font-family: 'Space Grotesk', sans-serif !important;
}
h1, h2, h3, h4 {
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
/* Buttons — secondary (inactive tabs, general actions) */
[data-testid="baseButton-secondary"] {
    border-radius: 0 !important;
    border: 2px solid #333 !important;
    background: #1a1a1a !important;
    color: #ccc !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    box-shadow: none !important;
    transition: border-color 0.15s, color 0.15s !important;
}
[data-testid="baseButton-secondary"]:hover {
    border-color: #E8002D !important;
    color: #fff !important;
}
/* Buttons — primary (active tab, submit actions) */
[data-testid="baseButton-primary"] {
    border-radius: 0 !important;
    border: 2px solid #E8002D !important;
    background: #E8002D !important;
    color: #000 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    box-shadow: 3px 3px 0 #000 !important;
    transition: background 0.15s, color 0.15s !important;
}
[data-testid="baseButton-primary"]:hover {
    background: #ff1a3c !important;
    box-shadow: none !important;
}
/* Metric cards */
[data-testid="stMetric"] {
    border: 2px solid #E8002D !important;
    border-radius: 0 !important;
    padding: 12px !important;
    background: #1a1a1a !important;
    box-shadow: 2px 2px 0 #E8002D !important;
}
/* Sidebar */
[data-testid="stSidebar"] {
    border-right: 2px solid #E8002D !important;
}
/* Expanders — target inner details element to avoid clipping the overflow:hidden wrapper */
[data-testid="stExpander"] details {
    border: 1px solid #333;
    border-radius: 0;
}
[data-testid="stExpander"] details summary,
[data-testid="stExpander"] details > div:first-of-type {
    border-radius: 0;
}
/* Selectbox / radio / multiselect */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    border-radius: 0 !important;
    border-color: #E8002D !important;
}
</style>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────────────────────

for _key, _val in STATE_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val


@st.cache_data
def get_model_metrics() -> dict:
    """Load the detailed metrics.json written by model.py. Returns {} if not yet generated."""
    if not METRICS_PATH.exists():
        return {}
    with open(METRICS_PATH) as f:
        return json.load(f)


# --- Sidebar ---
with st.sidebar:
    st.title("🧬 Driver DNA + 🏁 Race Pace")

    # -- Driver DNA section --
    st.subheader("Driver DNA")
    st.markdown("Identify F1 drivers from telemetry alone using XGBoost.")
    _metrics = get_model_metrics()
    if _metrics:
        _mcol1, _mcol2 = st.columns(2)
        _mcol1.metric("CV Accuracy", f"{_metrics['cv_mean']:.1%}", f"±{_metrics['cv_std']:.1%}")
        _mcol2.metric("Train Accuracy", f"{_metrics['train_accuracy']:.1%}")
        st.caption(f"{_metrics.get('n_drivers', '?')} drivers · {_metrics.get('n_samples', '?')} laps")
    elif ACCURACY_PATH.exists():
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

    # -- AI Response Mode toggle --
    st.divider()
    st.markdown("#### AI Response Mode")
    st.session_state["ai_response_mode"] = st.sidebar.radio(
        "Response depth",
        options=["Concise", "Detailed"],
        index=0 if st.session_state.get("ai_response_mode", "Concise") == "Concise" else 1,
        key="ai_mode_radio",
        help=(
            "Concise: 3-6 sentence focused answers, lower token cost. "
            "Detailed: multi-paragraph rich analysis, ~50% more tokens per call."
        ),
    )

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


def _get_openai_api_key():
    """Read API key from st.secrets with os.environ fallback. Returns None if absent."""
    try:
        return st.secrets["openai"]["api_key"]
    except (KeyError, FileNotFoundError):
        pass
    return os.environ.get("OPENAI_API_KEY")


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

def _ca_run_chat_turn(user_message: str, analyser, driver_map: dict, api_key: str, verbose: bool = False) -> tuple:
    """Run one chat turn with the tool-calling loop.
    Returns (answer_text, None) on success or (None, error_string) on failure.
    Pass verbose=True for the Detailed-mode system prompt and higher token budget.
    """
    MAX_TOOL_ROUNDS = 3
    _ca_sys = _CA_SYSTEM_VERBOSE if verbose else _CA_SYSTEM_PROMPT
    _ca_tokens = 700 if verbose else 500

    messages: list[dict] = [{"role": "system", "content": _ca_sys}]
    for m in st.session_state["ca_messages"]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_message})

    for _ in range(MAX_TOOL_ROUNDS):
        kind, payload, err = _call_openai_chat(api_key, messages, _CA_TOOLS, max_tokens=_ca_tokens)
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
    kind, payload, err = _call_openai_chat(api_key, messages, [], max_tokens=_ca_tokens)
    if err:
        return None, err
    if kind == "text":
        return payload, None
    return None, "Could not generate a response after tool execution."

# -- Chat UI -----------------------------------------------------------------

def _run_pipeline_download(year: int, gp: str, session_type: str, drivers: list[str] | None) -> None:
    """Background thread: download F1 telemetry and save to dataset.parquet."""
    ss = st.session_state

    def _cb(frac: float, msg: str) -> None:
        ss["pipe_progress"] = frac
        ss["pipe_status"] = msg
        ss["pipe_log"].append(msg)

    try:
        ss["pipe_log"] = []
        ss["pipe_progress"] = 0.0
        _cb(0.0, f"Connecting to FastF1 for {year} {gp} ({session_type})…")
        df = extract_session_telemetry(year, gp, session_type, drivers, progress_cb=_cb)
        _cb(0.97, "Saving dataset…")
        save_dataset(df, DATA_DIR / "dataset.parquet")
        save_meta(year, gp, session_type)
        _cb(1.0, f"Done — {len(df)} laps saved to data/dataset.parquet")
        ss["pipe_done"] = True
    except Exception as exc:
        ss["pipe_error"] = str(exc)
    finally:
        ss["pipe_running"] = False


def _run_model_training() -> None:
    """Background thread: train the XGBoost classifier on the current dataset."""
    ss = st.session_state

    def _cb(frac: float, msg: str) -> None:
        ss["train_progress"] = frac
        ss["train_status"] = msg
        ss["train_log"].append(msg)

    try:
        ss["train_log"] = []
        ss["train_progress"] = 0.0
        _cb(0.0, "Loading dataset…")
        X, y, le = load_and_prepare(DATA_DIR / "dataset.parquet")
        n_drivers = len(set(y))
        if n_drivers < 2:
            raise ValueError(f"Need at least 2 drivers to train a classifier; found {n_drivers}.")
        _cb(0.05, f"Loaded {len(X)} samples, {n_drivers} drivers. Starting 5-fold CV…")
        clf, acc = train_model(X, y, progress_cb=lambda f, m: _cb(0.05 + f * 0.67, m))
        _cb(0.75, "Evaluating — computing metrics and SHAP values…")
        evaluate_model(clf, X, y, le)
        _cb(0.97, "Saving model artifacts…")
        save_model(clf, le)
        _cb(1.0, "Done — model saved to models/")
        ss["train_done"] = True
    except Exception as exc:
        ss["train_error"] = str(exc)
    finally:
        ss["train_running"] = False


def _render_chat_panel(analyser, driver_map: dict, session_cache_key) -> None:
    """Render the Race Intelligence chat panel below the race analysis charts."""
    _ca_maybe_clear_history(session_cache_key)
    api_key = _get_openai_api_key()

    _chat_verbose = st.session_state.get("ai_response_mode", "Concise") == "Detailed"

    st.divider()
    st.markdown("#### 🤖 Race Intelligence — Ask me anything about this race")
    st.caption(f"Response mode: **{st.session_state.get('ai_response_mode', 'Concise')}** · Change in the sidebar.")

    # Render existing message history
    for msg in st.session_state["ca_messages"]:
        with st.chat_message(msg["role"]):
            st.text(msg["content"])

    # Use an inline form so the input stays inside the tab (st.chat_input floats
    # to the bottom of the entire page and bleeds into other tabs)
    with st.form("ca_chat_form", clear_on_submit=True):
        _col_input, _col_btn = st.columns([8, 1])
        with _col_input:
            user_input = st.text_input(
                "Ask a question about the race…",
                label_visibility="collapsed",
                placeholder="Ask a question about the race…",
            )
        with _col_btn:
            _submitted = st.form_submit_button("Send")

    if _submitted and user_input:
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
                answer, error = _ca_run_chat_turn(user_input, analyser, driver_map, api_key, verbose=_chat_verbose)
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

_TABS = [
    ("🎯  Driver Radar",   "Driver Radar"),
    ("🕵️  Mystery Driver", "Mystery Driver"),
    ("🏁  Race Dashboard", "Race Dashboard"),
    ("⚙️  Pipeline",       "Pipeline"),
]

_tab_cols = st.columns(len(_TABS))
for _col, (_label, _key) in zip(_tab_cols, _TABS):
    _is_active = st.session_state.get("active_tab", "Driver Radar") == _key
    with _col:
        if st.button(
            _label,
            key=f"tab_nav_{_key}",
            use_container_width=True,
            type="primary" if _is_active else "secondary",
        ):
            st.session_state["active_tab"] = _key
            st.rerun()

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
st.divider()

_active_tab = st.session_state.get("active_tab", "Driver Radar")

# ================================================================
# TAB 1 — Driver Radar
# ================================================================
if _active_tab == "Driver Radar":
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

        # ============================================================
        # AGENTIC AI FEATURES
        # ============================================================
        _api_key = _get_openai_api_key()
        _verbose = st.session_state.get("ai_response_mode", "Concise") == "Detailed"

        # ── Unified AI Feature Panel ─────────────────────────────────
        st.divider()
        st.markdown("## AI Analysis")

        _ai_feature = st.selectbox(
            "Select AI feature",
            options=["Driver Style Analyst", "Historical DNA Matching", "Driver DNA Report Card"],
            key="radar_ai_feature",
        )

        _ai_driver = st.selectbox(
            "Select driver to analyse",
            options=selected,
            key="radar_ai_driver",
        )

        if _ai_feature == "Driver Style Analyst":
            st.caption(
                "**Reflexion pattern** (Shinn et al. 2023): an Analyst LLM generates a driving "
                "style narrative → a Critic LLM evaluates it and returns a confidence score + "
                "critique in JSON → if confidence < 7/10 the Analyst revises using the critique "
                "as context. Max 2 rounds."
            )
            if _api_key is None:
                st.info("Add an OpenAI API key to `.streamlit/secrets.toml` to enable this feature.")
            else:
                _elapsed_ra = time.time() - st.session_state["ra_last_analyse_ts"]
                _cooldown_ra = max(0.0, _RA_RATE_LIMIT_SECS - _elapsed_ra)
                _btn_label_ra = (
                    f"Analyse Driving Style (wait {_cooldown_ra:.0f}s)"
                    if _cooldown_ra > 0
                    else "Analyse Driving Style"
                )
                if st.button(_btn_label_ra, key="ra_analyse_btn", disabled=_cooldown_ra > 0):
                    st.session_state["ra_last_analyse_ts"] = time.time()
                    st.session_state["ra_result"] = None
                    st.session_state["ra_analysed_driver"] = _ai_driver
                    if _ai_driver is not None:
                        try:
                            _data_block = _ra_build_driver_data_block(_ai_driver, df_ext, ext_driver_means)
                            with st.spinner("Running Reflexion loop — Analyst → Critic → (optional) Revision…"):
                                _ra_result, _ra_err = _ra_run_reflexion(_ai_driver, _data_block, _api_key, verbose=_verbose)
                            if _ra_err:
                                st.error(f"Reflexion error: {_ra_err}")
                            else:
                                st.session_state["ra_result"] = _ra_result
                        except Exception as _ra_exc:
                            st.error(f"Unexpected error: {_ra_exc}")

                if (
                    st.session_state["ra_result"] is not None
                    and st.session_state["ra_analysed_driver"] == _ai_driver
                ):
                    _ra_res = st.session_state["ra_result"]
                    for _iter in _ra_res["iterations"]:
                        _rn = _iter["round_num"]
                        _label = (
                            f"Round {_rn} — Initial Draft"
                            if _rn == 1
                            else f"Round {_rn} — Revised Narrative"
                        )
                        _is_last = _rn == _ra_res["iterations"][-1]["round_num"]
                        with st.expander(_label, expanded=_is_last):
                            st.write(_iter["narrative"])
                            if _iter.get("critique"):
                                _crit = _iter["critique"]
                                _conf = _crit.get("confidence", 10)
                                _c1, _c2 = st.columns([1, 3])
                                _c1.metric("Critic Confidence", f"{_conf}/10")
                                if _conf < 7:
                                    _c2.warning("Confidence below 7/10 — Analyst will revise.")
                                else:
                                    _c2.success("Confidence meets threshold — narrative accepted.")
                                if _crit.get("factual_errors"):
                                    st.markdown("**Factual errors flagged:**")
                                    for _fe in _crit["factual_errors"]:
                                        st.markdown(f"- {_fe}")
                                if _crit.get("suggested_improvements"):
                                    st.markdown("**Suggested improvements:**")
                                    for _si in _crit["suggested_improvements"]:
                                        st.markdown(f"- {_si}")
                                if _crit.get("parse_note"):
                                    st.caption(_crit["parse_note"])
                    st.success(_ra_res["final_narrative"])

        elif _ai_feature == "Historical DNA Matching":
            st.caption(
                "**RAG pattern**: the driver's 12-dimensional normalised radar vector is compared to "
                "10 legendary F1 drivers by cosine similarity. The top-2 most stylistically similar "
                "historical profiles are retrieved and used to augment the LLM explanation."
            )
            if _api_key is None:
                st.info("Add an OpenAI API key to `.streamlit/secrets.toml` to enable this feature.")
            else:
                _elapsed_rag = time.time() - st.session_state["rag_last_match_ts"]
                _cooldown_rag = max(0.0, _RAG_RATE_LIMIT_SECS - _elapsed_rag)
                _btn_label_rag = (
                    f"Find DNA Match (wait {_cooldown_rag:.0f}s)"
                    if _cooldown_rag > 0
                    else "Find DNA Match"
                )
                if st.button(_btn_label_rag, key="rag_match_btn", disabled=_cooldown_rag > 0):
                    st.session_state["rag_last_match_ts"] = time.time()
                    st.session_state["rag_result"] = None
                    st.session_state["rag_analysed_driver"] = _ai_driver
                    if _ai_driver is not None:
                        try:
                            _drv_series = cast(pd.Series, ext_driver_means.loc[_ai_driver])
                            _drv_vec = np.array(_drv_series, dtype=float)
                            with st.spinner("Computing cosine similarity and generating narrative…"):
                                _rag_result, _rag_err = _rag_run_dna_match(_ai_driver, _drv_vec, _api_key, verbose=_verbose)
                            if _rag_err:
                                st.error(f"DNA match error: {_rag_err}")
                            else:
                                st.session_state["rag_result"] = _rag_result
                        except KeyError:
                            st.warning(f"Extended features unavailable for {_ai_driver}.")
                        except Exception as _rag_exc:
                            st.error(f"Unexpected error: {_rag_exc}")

                if (
                    st.session_state["rag_result"] is not None
                    and st.session_state["rag_analysed_driver"] == _ai_driver
                ):
                    _rag_res = st.session_state["rag_result"]
                    try:
                        _drv_vec_render = cast(pd.Series, ext_driver_means.loc[_ai_driver]).to_numpy(dtype=float)
                        _all_scores = _rag_retrieve_top_k(_drv_vec_render, _RAG_HISTORICAL_PROFILES, k=10)
                        _top2_names = {m["name"] for m in _rag_res["matches"]}
                        _bar_colors = [
                            "#f1c40f" if s["name"] in _top2_names else "#555555"
                            for s in reversed(_all_scores)
                        ]
                        _fig_sim = go.Figure(go.Bar(
                            x=[s["similarity"] for s in reversed(_all_scores)],
                            y=[s["name"] for s in reversed(_all_scores)],
                            orientation="h",
                            marker_color=_bar_colors,
                            hovertemplate="%{y}<br>Similarity: %{x:.4f}<extra></extra>",
                        ))
                        _fig_sim.update_layout(
                            title=f"Cosine Similarity to Historical F1 Drivers — {_ai_driver}",
                            xaxis=dict(range=[0.80, 1.0], title="Cosine Similarity"),
                            yaxis_title="",
                            height=380,
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(_fig_sim, width="stretch", key="tab1_rag_similarity_bar")
                    except Exception:
                        pass

                    _match_cols = st.columns(2)
                    for _ci, _match in enumerate(_rag_res["matches"]):
                        with _match_cols[_ci]:
                            st.metric(
                                label=f"#{_ci + 1} DNA Match",
                                value=_match["name"],
                                delta=f"Similarity: {_match['similarity']:.1%}",
                            )
                            st.caption(f"Era: {_match['era']}")
                            st.write(_match["description"])

                    st.markdown("#### Why the DNA aligns")
                    st.info(_rag_res["narrative"])

        elif _ai_feature == "Driver DNA Report Card":
            st.caption(
                "**Structured output pattern**: GPT-4o-mini is called in JSON mode "
                "(`response_format={\"type\": \"json_object\"}`). The schema is enforced via "
                "prompt engineering and validated in Python. If validation fails, the prompt "
                "is retried once with the specific error injected."
            )
            if _api_key is None:
                st.info("Add an OpenAI API key to `.streamlit/secrets.toml` to enable this feature.")
            else:
                _elapsed_rc = time.time() - st.session_state["rc_last_report_ts"]
                _cooldown_rc = max(0.0, _RC_RATE_LIMIT_SECS - _elapsed_rc)
                _btn_label_rc = (
                    f"Generate DNA Report (wait {_cooldown_rc:.0f}s)"
                    if _cooldown_rc > 0
                    else "Generate DNA Report"
                )
                if st.button(_btn_label_rc, key="rc_report_btn", disabled=_cooldown_rc > 0):
                    st.session_state["rc_last_report_ts"] = time.time()
                    st.session_state["rc_result"] = None
                    st.session_state["rc_analysed_driver"] = _ai_driver
                    if _ai_driver is not None:
                        with st.spinner("Generating structured report card (JSON mode)…"):
                            _rc_result, _rc_err = _rc_run_report(_ai_driver, df_ext, ext_driver_means, _api_key, verbose=_verbose)
                        if _rc_err:
                            st.error(f"Report card error: {_rc_err}")
                        else:
                            st.session_state["rc_result"] = _rc_result

                if (
                    st.session_state["rc_result"] is not None
                    and st.session_state["rc_analysed_driver"] == _ai_driver
                ):
                    _report = st.session_state["rc_result"]

                    # ── Headline ──────────────────────────────────────
                    st.markdown(f"### {_report['headline']}")

                    # ── Style tags ────────────────────────────────────
                    _tag_html = " ".join(
                        f'<span style="background:#111;border:2px solid #E8002D;border-radius:0;'
                        f'padding:3px 10px;font-size:0.85em;color:#fff;margin:2px;font-weight:600">{tag}</span>'
                        for tag in _report.get("style_tags", [])
                    )
                    st.markdown(_tag_html, unsafe_allow_html=True)
                    st.markdown("")

                    # ── Overall score ─────────────────────────────────
                    _rc_val  = _report.get("race_craft_rating", 0)
                    _rp_val  = _report.get("raw_pace_rating", 0)
                    _con_val = _report.get("consistency_rating", 0)
                    _overall = round((_rc_val + _rp_val + _con_val) / 3, 1)

                    def _rating_tier(v: int) -> tuple[str, str]:
                        """Return (tier label, hex colour) for a 1-10 rating."""
                        if v >= 9:
                            return "Elite", "#00e676"
                        if v >= 7:
                            return "Strong", "#69f0ae"
                        if v >= 5:
                            return "Good", "#ffd740"
                        if v >= 3:
                            return "Average", "#ff9100"
                        return "Below Average", "#ff5252"

                    def _rating_bar_html(label: str, value: int, tooltip: str) -> str:
                        tier, color = _rating_tier(value)
                        pct = value * 10
                        return (
                            f'<div style="margin-bottom:14px">'
                            f'  <div style="display:flex;justify-content:space-between;margin-bottom:4px">'
                            f'    <span style="font-weight:600;font-size:0.95em">{label}</span>'
                            f'    <span style="color:{color};font-weight:700;font-size:1.05em">'
                            f'      {value}/10 &nbsp;<span style="font-size:0.8em;color:#aaa">({tier})</span>'
                            f'    </span>'
                            f'  </div>'
                            f'  <div style="background:#1a1a1a;border:1px solid #333;border-radius:0;height:10px;width:100%">'
                            f'    <div style="background:{color};width:{pct}%;height:10px;border-radius:0;'
                            f'               transition:width 0.4s ease"></div>'
                            f'  </div>'
                            f'  <div style="font-size:0.78em;color:#888;margin-top:3px">{tooltip}</div>'
                            f'</div>'
                        )

                    _overall_tier, _overall_color = _rating_tier(round(_overall))
                    st.markdown(
                        f'<div style="background:#1a1a1a;border:2px solid #E8002D;border-radius:0;'
                        f'box-shadow:4px 4px 0 #E8002D;padding:16px 20px;margin-bottom:18px">'
                        f'  <div style="font-size:0.8em;color:#888;margin-bottom:4px;text-transform:uppercase;'
                        f'             letter-spacing:0.08em">Overall Driver Rating</div>'
                        f'  <div style="font-size:2.2em;font-weight:800;color:{_overall_color}">'
                        f'    {_overall}<span style="font-size:0.5em;color:#aaa">/10</span>'
                        f'    &nbsp;<span style="font-size:0.45em;color:{_overall_color}">{_overall_tier}</span>'
                        f'  </div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # ── Rating bars ───────────────────────────────────
                    st.markdown(
                        _rating_bar_html(
                            "Race Craft", _rc_val,
                            "Based on trail braking precision, brake timing, and steering control under load",
                        )
                        + _rating_bar_html(
                            "Raw Pace", _rp_val,
                            "Based on mean speed, max speed, and full-throttle fraction across all laps",
                        )
                        + _rating_bar_html(
                            "Consistency", _con_val,
                            "Based on lap-to-lap variability in speed, throttle, braking, and steering",
                        ),
                        unsafe_allow_html=True,
                    )

                    st.divider()

                    # ── Strengths ─────────────────────────────────────
                    st.markdown("#### Strengths")
                    for _s in _report.get("strengths", []):
                        st.markdown(
                            f'<div style="background:#1a1a1a;border:1px solid #00c864;border-left:3px solid #00c864;'
                            f'border-radius:0;box-shadow:2px 2px 0 #00c864;padding:10px 14px;margin-bottom:8px;font-size:0.93em">'
                            f'✅ &nbsp;{_s}</div>',
                            unsafe_allow_html=True,
                        )

                    # ── Weaknesses ────────────────────────────────────
                    st.markdown("#### Weaknesses")
                    for _w in _report.get("weaknesses", []):
                        st.markdown(
                            f'<div style="background:#1a1a1a;border:1px solid #ff9100;border-left:3px solid #ff9100;'
                            f'border-radius:0;box-shadow:2px 2px 0 #ff9100;padding:10px 14px;margin-bottom:8px;font-size:0.93em">'
                            f'⚠️ &nbsp;{_w}</div>',
                            unsafe_allow_html=True,
                        )

                    st.divider()

                    # ── Tactical Tips ─────────────────────────────────
                    st.markdown("#### Tactical Tips")
                    for _ti, _tip in enumerate(_report.get("tactical_tips", []), 1):
                        st.markdown(
                            f'<div style="background:#1a1a1a;border:1px solid #1e78ff;border-left:3px solid #1e78ff;'
                            f'border-radius:0;box-shadow:2px 2px 0 #1e78ff;padding:10px 14px;margin-bottom:8px;font-size:0.93em">'
                            f'<span style="font-weight:700;color:#1e78ff">Tip {_ti}:</span> &nbsp;{_tip}</div>',
                            unsafe_allow_html=True,
                        )

        st.divider()

        arch_cols = st.columns(len(selected))
        for col_idx, drv in enumerate(selected):
            norm_vals = ext_driver_means.loc[drv].to_dict()
            archetype, emoji = classify_archetype(norm_vals)
            drv_color = color_map[drv]
            arch_cols[col_idx].markdown(
                f"""<div style="background:#1a1a1a; border:2px solid #E8002D;
                border-radius:0; box-shadow:3px 3px 0 #E8002D; padding:12px; margin:4px 0;">
                <div style="font-size:1.6em">{emoji}</div>
                <div style="font-weight:700; color:{drv_color}; font-size:1.1em; font-family:'Space Grotesk',sans-serif; letter-spacing:0.04em">{drv}</div>
                <div style="color:#ccc; font-size:0.9em; margin-top:4px; text-transform:uppercase; letter-spacing:0.06em">{archetype}</div>
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
if _active_tab == "Mystery Driver":
    st.subheader("Can the model identify the driver?")

    # ── AI Explanation (top of tab — shown once a prediction exists) ─
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

                valid, reason = _validate_feature_values(lap_features)
                if not valid:
                    st.error(f"Input validation failed: {reason}")
                else:
                    try:
                        safe_driver: str | None = _sanitize_driver_name(_pred_driver, list(le.classes_))
                    except ValueError as exc:
                        st.error(str(exc))
                        safe_driver = None

                    if safe_driver is not None:
                        with st.spinner("Asking the AI to explain this prediction..."):
                            shap_vals = compute_shap_for_prediction(
                                shap_explainer, _X_input, _pred_class_idx
                            )
                            shap_dict = {
                                feat: float(shap_vals[i])
                                for i, feat in enumerate(FEATURE_COLS)
                            }
                            percentile_dict = {
                                feat: float(
                                    (df[feat].dropna() <= lap_features[feat]).mean() * 100
                                )
                                for feat in FEATURE_COLS
                            }
                            _md_verbose = st.session_state.get("ai_response_mode", "Concise") == "Detailed"
                            _md_max_tokens = 500 if _md_verbose else 300
                            system_msg, user_msg = _build_explain_prompt(
                                safe_driver, _confidence,
                                lap_features, shap_dict, percentile_dict,
                                verbose=_md_verbose,
                            )
                            explanation, err = _call_openai_llm(
                                api_key, system_msg, user_msg, max_tokens=_md_max_tokens,
                                feature_name="xai_explainer",
                            )
                            if err:
                                st.error(err)
                            else:
                                st.session_state["md_explanation"] = explanation
                        st.session_state["md_last_explain_ts"] = time.time()

    if st.session_state.get("md_explanation"):
        st.info(st.session_state["md_explanation"])

    st.divider()

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

    # ── Model Accuracy Panel ──────────────────────────────────────────────────
    _mets = get_model_metrics()
    if _mets:
        st.divider()
        st.markdown("### Model Accuracy")

        # Top-line metrics
        _a1, _a2, _a3, _a4 = st.columns(4)
        _a1.metric("CV Accuracy", f"{_mets['cv_mean']:.1%}", f"±{_mets['cv_std']:.1%}")
        _a2.metric("Train Accuracy", f"{_mets['train_accuracy']:.1%}")
        _a3.metric("Drivers", str(_mets.get("n_drivers", "—")))
        _a4.metric("Laps", str(_mets.get("n_samples", "—")))

        # Per-fold CV breakdown
        _fold_scores = _mets.get("fold_scores", [])
        if _fold_scores:
            with st.expander("Per-fold CV scores", expanded=False):
                _fcols = st.columns(len(_fold_scores))
                for _fi, (_fc, _fs) in enumerate(zip(_fcols, _fold_scores), 1):
                    _delta_color = "normal" if _fs >= _mets["cv_mean"] else "inverse"
                    _fc.metric(f"Fold {_fi}", f"{_fs:.1%}",
                               f"{(_fs - _mets['cv_mean'])*100:+.1f}pp")

        # Per-driver accuracy table
        _per = _mets.get("per_driver", {})
        if _per:
            st.markdown("#### Per-Driver Metrics")
            st.caption("Evaluated on the full training set. Precision, recall, and F1 per driver.")
            _rows = []
            for _drv, _dm in sorted(_per.items()):
                _rows.append({
                    "Driver":     _drv,
                    "Precision":  f"{_dm['precision']:.1%}",
                    "Recall":     f"{_dm['recall']:.1%}",
                    "F1 Score":   f"{_dm['f1_score']:.1%}",
                    "Laps (support)": _dm["support"],
                })
            _met_df = pd.DataFrame(_rows).set_index("Driver")

            # Colour-code rows: green ≥ 0.90, amber ≥ 0.70, red < 0.70
            def _colour_f1(val: str) -> str:
                v = float(val.strip("%")) / 100
                if v >= 0.90:
                    return "background-color: rgba(50,200,80,0.15); color:#7fffaa"
                if v >= 0.70:
                    return "background-color: rgba(255,180,0,0.15); color:#ffd966"
                return "background-color: rgba(220,50,50,0.15); color:#ff9999"

            st.write(_met_df.style.map(_colour_f1, subset=["F1 Score"]))  # type: ignore[call-overload]


# ================================================================
# TAB 4 — Race Pace Dashboard
# ================================================================

# ---- helpers shared by both modes --------------------------------

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

if _active_tab == "Race Dashboard":
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

            # ── Race Intelligence Chat Agent (top of dashboard) ──────
            _hist_cache_key = st.session_state.get("rp_hist_session_key")
            if _hist_cache_key is not None:
                _render_chat_panel(_analyser, _dmap, session_cache_key=_hist_cache_key)

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
                    color: #E8002D;
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

                    # ── Race Intelligence Chat Agent (top of dashboard) ─
                    _render_chat_panel(analyser, _live_dmap, session_cache_key="live")

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
            except Exception as exc:
                st.warning(f"Live refresh failed: {exc}")
    
            # Only auto-rerun if still on live mode — prevents resetting other tabs
            if st.session_state.get("rp_mode", "").startswith("📡"):
                time.sleep(refresh_interval)
                st.rerun()

# ================================================================
# TAB 4 — Pipeline & Training
# ================================================================
elif _active_tab == "Pipeline":
    ss = st.session_state

    st.subheader("⚙️ Pipeline & Training")

    # ── Current dataset info card ────────────────────────────────────────────
    _meta = get_dataset_meta()
    if _meta:
        _mc1, _mc2, _mc3, _mc4 = st.columns(4)
        _mc1.metric("Grand Prix", _meta.get("grand_prix", "—"))
        _mc2.metric("Year", str(_meta.get("year", "—")))
        _mc3.metric("Session", _meta.get("session_label", _meta.get("session_type", "—")))
        if DATASET_PATH.exists():
            _lap_count = len(pd.read_parquet(DATASET_PATH, columns=["driver"]))
            _mc4.metric("Laps", str(_lap_count))
    else:
        st.info("No dataset found. Download one below to get started.")

    st.divider()

    # ── Section 1: Download Dataset ──────────────────────────────────────────
    st.markdown("### 📥 Download Dataset")

    _dl_col1, _dl_col2, _dl_col3 = st.columns(3)
    with _dl_col1:
        _pipe_year = st.number_input(
            "Year", min_value=1950, max_value=2100,
            value=2025, step=1, key="pipe_year",
        )
    with _dl_col2:
        _pipe_gp = st.text_input("Grand Prix name", placeholder="e.g. Italian Grand Prix", key="pipe_gp")
    with _dl_col3:
        _pipe_session = st.selectbox(
            "Session type", options=VALID_SESSION_TYPES,
            index=VALID_SESSION_TYPES.index("R"), key="pipe_session",
        )

    _pipe_drivers_raw = st.text_input(
        "Driver filter (optional — comma-separated codes, e.g. VER,HAM,NOR)",
        placeholder="Leave blank for all drivers",
        key="pipe_drivers",
    )

    _pipe_busy = ss.get("pipe_running", False) or ss.get("train_running", False)
    if st.button("Download Dataset", disabled=_pipe_busy, type="primary", key="pipe_dl_btn"):
        if not _pipe_gp.strip():
            st.error("Grand Prix name cannot be empty.")
        else:
            _driver_list: list[str] | None = (
                [d.strip().upper() for d in _pipe_drivers_raw.split(",") if d.strip()]
                if _pipe_drivers_raw.strip() else None
            )
            ss["pipe_running"] = True
            ss["pipe_done"] = False
            ss["pipe_error"] = None
            ss["pipe_log"] = []
            ss["pipe_progress"] = 0.0
            ss["pipe_status"] = ""
            _dl_thread = threading.Thread(
                target=_run_pipeline_download,
                args=(int(_pipe_year), _pipe_gp.strip(), _pipe_session, _driver_list),
                daemon=True,
            )
            add_script_run_ctx(_dl_thread, get_script_run_ctx())
            _dl_thread.start()
            st.rerun()

    if ss.get("pipe_running"):
        st.progress(ss.get("pipe_progress", 0.0))
        st.caption(ss.get("pipe_status", "Starting…"))
    if ss.get("pipe_log"):
        with st.expander("Download log", expanded=bool(ss.get("pipe_running"))):
            st.code("\n".join(ss["pipe_log"]), language=None)
    if ss.get("pipe_done"):
        st.success("Dataset downloaded successfully. Cache cleared — refresh the page to use the new data.")
        get_data.clear()
        get_dataset_meta.clear()
        ss["pipe_done"] = False
    if ss.get("pipe_error"):
        st.error(f"Download failed: {ss['pipe_error']}")
        ss["pipe_error"] = None

    st.divider()

    # ── Section 2: Train Model ───────────────────────────────────────────────
    st.markdown("### 🧠 Train Model")

    _mets = get_model_metrics()
    if _mets:
        _tm1, _tm2, _tm3 = st.columns(3)
        _tm1.metric("CV Accuracy", f"{_mets['cv_mean']:.1%}", f"±{_mets['cv_std']:.1%}")
        _tm2.metric("Train Accuracy", f"{_mets['train_accuracy']:.1%}")
        _tm3.metric("Drivers / Laps", f"{_mets.get('n_drivers','?')} / {_mets.get('n_samples','?')}")
        st.caption("Current model metrics — press Train to retrain on the active dataset.")
    else:
        st.caption("No trained model found. Download a dataset first, then train.")

    _train_busy = ss.get("train_running", False) or ss.get("pipe_running", False)
    if st.button("Train Model", disabled=_train_busy or not DATASET_PATH.exists(), type="primary", key="train_btn"):
        ss["train_running"] = True
        ss["train_done"] = False
        ss["train_error"] = None
        ss["train_log"] = []
        ss["train_progress"] = 0.0
        ss["train_status"] = ""
        _train_thread = threading.Thread(target=_run_model_training, daemon=True)
        add_script_run_ctx(_train_thread, get_script_run_ctx())
        _train_thread.start()
        st.rerun()

    if ss.get("train_running"):
        st.progress(ss.get("train_progress", 0.0))
        st.caption(ss.get("train_status", "Starting…"))
    if ss.get("train_log"):
        with st.expander("Training log", expanded=bool(ss.get("train_running"))):
            st.code("\n".join(ss["train_log"]), language=None)
    if ss.get("train_done"):
        st.success("Model trained and saved. Reloading model cache…")
        get_model.clear()
        ss["train_done"] = False
        st.rerun()
    if ss.get("train_error"):
        st.error(f"Training failed: {ss['train_error']}")
        ss["train_error"] = None

    # Auto-rerun while any background operation is running
    if ss.get("pipe_running") or ss.get("train_running"):
        time.sleep(1)
        st.rerun()
