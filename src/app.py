"""
app.py — Streamlit dashboard for Driver DNA Fingerprinter.

Run with:
    streamlit run src/app.py

Expected columns from dataset.parquet:
  driver, lap_time_seconds, mean_speed, max_speed, min_speed,
  throttle_mean, throttle_std, brake_mean, brake_events,
  gear_changes, steer_std, speed_trace, throttle_trace, brake_trace
"""

import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import cast
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
METRICS_PATH  = MODELS_DIR / "metrics.json"
DATASET_META_PATH = DATA_DIR / "dataset_meta.json"
LLM_AUDIT_PATH = MODELS_DIR / "llm_audit.jsonl"

# ── Structured logging ───────────────────────────────────────────────────────
logging.basicConfig(format="%(message)s", level=logging.INFO)
_logger = logging.getLogger("driver_dna")


def _log_llm_call(
    feature: str,
    model: str,
    latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    success: bool,
    error_type: str | None = None,
) -> None:
    """Append one JSON line to the LLM audit log (models/llm_audit.jsonl)."""
    entry: dict = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "feature": feature,
        "model": model,
        "latency_ms": round(latency_ms),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "success": success,
    }
    if error_type:
        entry["error_type"] = error_type
    line = json.dumps(entry)
    _logger.info(line)
    try:
        with open(LLM_AUDIT_PATH, "a") as fh:
            fh.write(line + "\n")
    except OSError:
        pass  # never crash the dashboard due to logging
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
# ── Agentic AI rate limits ───────────────────────────────────────────────────
_RA_RATE_LIMIT_SECS  = 10   # Reflexion Analyst
_RAG_RATE_LIMIT_SECS = 10   # RAG DNA Matching
_RC_RATE_LIMIT_SECS  = 10   # Structured Report Card

# ── RAG: Historical F1 Driver Knowledge Base ─────────────────────────────────
# 12-element vectors map to EXTENDED_RADAR_FEATURES order:
# mean_speed, max_speed, min_speed, throttle_mean, throttle_std, throttle_pickup_pct,
# brake_mean, brake_events, trail_brake_score, steer_std, gear_changes, coasting_pct
_RAG_HISTORICAL_PROFILES: dict = {
    "Michael Schumacher": {
        "era": "1991–2012",
        "description": (
            "Seven-time champion defined by clinical trail braking, ultra-high brake pressure, "
            "and extreme lap-to-lap consistency. Notoriously late on the brakes with massive "
            "deceleration into corners."
        ),
        "vector": [0.85, 0.80, 0.55, 0.80, 0.35, 0.78, 0.90, 0.88, 0.85, 0.0, 0.78, 0.05],
    },
    "Ayrton Senna": {
        "era": "1984–1994",
        "description": (
            "Three-time champion known for aggressive braking, exceptional car control under "
            "trail-braking, and extremely high corner entry speed — he carried massive speed "
            "through the apex."
        ),
        "vector": [0.80, 0.75, 0.90, 0.75, 0.65, 0.70, 0.85, 0.80, 0.80, 0.0, 0.75, 0.04],
    },
    "Alain Prost": {
        "era": "1980–1993",
        "description": (
            "The Professor: maximised mechanical grip with smooth, low-variance inputs, minimal "
            "tyre wear, very high throttle efficiency, and low coasting — economical and precise "
            "rather than spectacular."
        ),
        "vector": [0.75, 0.70, 0.65, 0.85, 0.25, 0.88, 0.40, 0.45, 0.20, 0.0, 0.60, 0.08],
    },
    "Fernando Alonso": {
        "era": "2001–present",
        "description": (
            "Two-time champion exceptionally smooth on throttle with very low variation, a high "
            "coasting percentage in conservation phases, precise gear selection, and moderate "
            "braking — the master of tyre management."
        ),
        "vector": [0.70, 0.65, 0.72, 0.72, 0.28, 0.68, 0.38, 0.42, 0.18, 0.0, 0.72, 0.55],
    },
    "Sebastian Vettel": {
        "era": "2007–2022",
        "description": (
            "Four-time champion characterised by very high brake-event frequency with intense, "
            "precise braking zones, clean throttle application, and an early-apex technique "
            "that generated strong mechanical grip."
        ),
        "vector": [0.82, 0.78, 0.68, 0.76, 0.40, 0.74, 0.60, 0.85, 0.55, 0.0, 0.76, 0.07],
    },
    "Lewis Hamilton": {
        "era": "2007–present",
        "description": (
            "Seven-time champion defined by smooth, low-variance throttle application, minimal "
            "braking relative to the grid, very high corner speed, and exceptional lap-to-lap "
            "consistency across race stints."
        ),
        "vector": [0.85, 0.80, 0.74, 0.75, 0.30, 0.72, 0.25, 0.38, 0.28, 0.0, 0.68, 0.06],
    },
    "Max Verstappen": {
        "era": "2015–present",
        "description": (
            "Three-time champion who distinguishes himself with very late braking, high throttle "
            "variation (aggressive on/off application), significant trail-braking into corners, "
            "and high steering inputs at low-speed apexes."
        ),
        "vector": [0.88, 0.85, 0.72, 0.72, 0.78, 0.68, 0.45, 0.60, 0.55, 0.0, 0.70, 0.12],
    },
    "Kimi Räikkönen": {
        "era": "2001–2021",
        "description": (
            "The Iceman: extremely low coasting fraction, smooth and consistent throttle delivery, "
            "one of the highest full-throttle percentages on the grid, and an exceptionally clean "
            "steering trace — relentless mechanical efficiency."
        ),
        "vector": [0.78, 0.72, 0.70, 0.80, 0.22, 0.85, 0.35, 0.40, 0.15, 0.0, 0.65, 0.04],
    },
    "Carlos Sainz": {
        "era": "2015–present",
        "description": (
            "Known for aggressive throttle pickup and trail braking on entry, above-average brake "
            "pressure relative to the grid, and a high gear-change rate reflecting his attacking, "
            "high-commitment driving style."
        ),
        "vector": [0.76, 0.70, 0.75, 0.74, 0.55, 0.72, 0.65, 0.68, 0.62, 0.0, 0.72, 0.10],
    },
    "Nico Rosberg": {
        "era": "2006–2016",
        "description": (
            "One-time champion characterised by clean, systematic driving with consistent lap "
            "times, one of the highest full-throttle percentages on the grid, moderate coasting "
            "in fuel-saving phases, and low brake pressure."
        ),
        "vector": [0.80, 0.75, 0.68, 0.82, 0.30, 0.82, 0.30, 0.42, 0.22, 0.0, 0.66, 0.14],
    },
}

# ── Real-world driver context for Report Card calibration ────────────────────
# Keys must match 3-letter driver codes used in the dataset (df["driver"].unique()).
# When a driver is found here, _rc_build_user_prompt appends this block so the LLM
# can reconcile telemetry statistics with the driver's actual career standing.
_RC_DRIVER_CONTEXT: dict[str, dict] = {
    "LEC": {
        "full_name": "Charles Leclerc",
        "team": "Ferrari",
        "championships": 0,
        "wins": 8,
        "poles": 25,
        "career_seasons": "2018–present",
        "known_strengths": [
            "Elite one-lap qualifying pace — consistently extracts maximum from the car",
            "Exceptional trail braking and late-apex entry — one of the best on sector-1 type circuits",
            "High-risk, high-reward racecraft — renowned for aggressive wheel-to-wheel overtaking",
        ],
        "known_weaknesses": [
            "Race management under tyre pressure occasionally inconsistent",
            "Pushes beyond grip limits, which can lead to errors when chasing performance",
        ],
        "context_note": (
            "Leclerc is regarded as one of the fastest drivers on the current grid. "
            "His telemetry often shows high variance because he exploits the limit aggressively; "
            "this should be interpreted as attacking style, not inconsistency."
        ),
    },
    "VER": {
        "full_name": "Max Verstappen",
        "team": "Red Bull Racing",
        "championships": 4,
        "wins": 63,
        "poles": 40,
        "career_seasons": "2015–present",
        "known_strengths": [
            "Dominant race pace and tyre management over long stints",
            "Fearless overtaking and exceptional wet-weather performance",
            "Exceptional car-setup intuition and racecraft under pressure",
        ],
        "known_weaknesses": [
            "Historically aggressive wheel-to-wheel, leading to early-career penalties",
        ],
        "context_note": (
            "Verstappen is the reigning four-time world champion and widely regarded as the best "
            "driver of his generation. High telemetry variability reflects offensive driving, not poor consistency."
        ),
    },
    "HAM": {
        "full_name": "Lewis Hamilton",
        "team": "Ferrari (from 2025)",
        "championships": 7,
        "wins": 104,
        "poles": 104,
        "career_seasons": "2007–present",
        "known_strengths": [
            "Supreme tyre conservation — can manage degradation better than almost any driver",
            "Exceptional racecraft and strategic awareness over grand prix distance",
            "Consistent qualifying and race pace across all circuit types",
        ],
        "known_weaknesses": [
            "Recent seasons showed reduced peak pace relative to his dominant years",
        ],
        "context_note": (
            "Hamilton holds the all-time records for wins and poles. "
            "His smooth telemetry profile reflects deliberate tyre management, not lack of aggression."
        ),
    },
    "NOR": {
        "full_name": "Lando Norris",
        "team": "McLaren",
        "championships": 0,
        "wins": 4,
        "poles": 8,
        "career_seasons": "2019–present",
        "known_strengths": [
            "Strong one-lap qualifying pace — regularly beats teammates and top-four rivals",
            "Excellent tyre management and racecraft over grand prix distance",
            "Aggressive but controlled braking style with strong trail-braking technique",
        ],
        "known_weaknesses": [
            "Historically found it difficult to convert pace into wins under pressure, though significantly improved in 2024",
        ],
        "context_note": (
            "Norris is considered among the top three drivers on the current grid. "
            "His 2024 season showed consistent podium-level pace across the full calendar."
        ),
    },
    "SAI": {
        "full_name": "Carlos Sainz",
        "team": "Williams (from 2025)",
        "championships": 0,
        "wins": 4,
        "poles": 6,
        "career_seasons": "2015–present",
        "known_strengths": [
            "Extremely consistent race-day performer — rarely makes unforced errors",
            "Strong tyre management and strategic racecraft over long stints",
            "Smooth, calculated driving style with exceptional repeatability",
        ],
        "known_weaknesses": [
            "One-lap qualifying pace occasionally a fraction below the absolute elite",
        ],
        "context_note": (
            "Sainz is regarded as one of the most complete and consistent drivers on the grid. "
            "Low telemetry variability is a genuine strength for him, reflecting precision driving."
        ),
    },
    "RUS": {
        "full_name": "George Russell",
        "team": "Mercedes",
        "championships": 0,
        "wins": 3,
        "poles": 4,
        "career_seasons": "2019–present",
        "known_strengths": [
            "Exceptional qualifying pace and one-lap speed — frequently outqualifies faster cars",
            "Strong tyre management, particularly in difficult conditions",
            "Clinical racecraft and composed race management",
        ],
        "known_weaknesses": [
            "Race pace occasionally slightly below absolute elite when the car is not at peak",
        ],
        "context_note": (
            "Russell is widely regarded as a top-tier talent and Mercedes' long-term leader. "
            "His structured, precise driving style means low telemetry variability is a positive trait."
        ),
    },
    "PIA": {
        "full_name": "Oscar Piastri",
        "team": "McLaren",
        "championships": 0,
        "wins": 3,
        "poles": 2,
        "career_seasons": "2023–present",
        "known_strengths": [
            "Exceptionally smooth and precise driving style — outstanding natural tyre management",
            "Strong race pace that frequently exceeds qualifying position",
            "Composed racecraft under pressure despite limited F1 experience",
        ],
        "known_weaknesses": [
            "Still accumulating experience in wheel-to-wheel battles at the front of the grid",
        ],
        "context_note": (
            "Piastri is considered one of the most naturally talented drivers of his generation. "
            "His smooth telemetry profile is a hallmark of his style, not a weakness."
        ),
    },
    "ALO": {
        "full_name": "Fernando Alonso",
        "team": "Aston Martin",
        "championships": 2,
        "wins": 32,
        "poles": 22,
        "career_seasons": "2001–present",
        "known_strengths": [
            "Legendary racecraft and strategic intelligence — widely regarded as the smartest race driver ever",
            "Consistent top-10 delivery from sub-par machinery across multiple decades",
            "Exceptional adaptability and tyre conservation",
        ],
        "known_weaknesses": [
            "One-lap pace has diminished slightly with age compared to his dominant years",
        ],
        "context_note": (
            "Alonso is a two-time world champion widely considered among the greatest drivers of all time. "
            "His telemetry variability often reflects calculated risk-taking, not inconsistency."
        ),
    },
    "PER": {
        "full_name": "Sergio Perez",
        "team": "Red Bull Racing",
        "championships": 0,
        "wins": 13,
        "poles": 3,
        "career_seasons": "2011–present",
        "known_strengths": [
            "Elite tyre management — one of the best in the field at extending stints",
            "Excellent race pace over long runs with minimal degradation",
            "Strong undercut and overcut execution from the pitwall perspective",
        ],
        "known_weaknesses": [
            "One-lap qualifying pace significantly below elite teammates",
            "Inconsistent in changeable or wet conditions",
        ],
        "context_note": (
            "Perez is a race-pace specialist whose value lies in tyre management and strategy, not raw speed. "
            "His relatively smooth telemetry reflects deliberate tyre preservation, a genuine strength."
        ),
    },
    "STR": {
        "full_name": "Lance Stroll",
        "team": "Aston Martin",
        "championships": 0,
        "wins": 0,
        "poles": 1,
        "career_seasons": "2017–present",
        "known_strengths": [
            "Strong wet-weather pace — performed well above expectations in rain",
            "Good racecraft over long stints with tyre management",
        ],
        "known_weaknesses": [
            "Qualifying pace and one-lap speed consistently below top-10 benchmark",
            "Error-prone in high-pressure situations",
        ],
        "context_note": (
            "Stroll is a midfield driver. His ratings should reflect solid but not elite standing."
        ),
    },
    "GAS": {
        "full_name": "Pierre Gasly",
        "team": "Alpine",
        "championships": 0,
        "wins": 1,
        "poles": 0,
        "career_seasons": "2017–present",
        "known_strengths": [
            "Strong race pace in clean air — comfortable top-10 performer",
            "Aggressive overtaking and good tactical awareness",
        ],
        "known_weaknesses": [
            "One-lap pace inconsistent at the sharp end of the grid",
        ],
        "context_note": (
            "Gasly is a reliable midfield points scorer with occasional top-5 pace."
        ),
    },
    "OCO": {
        "full_name": "Esteban Ocon",
        "team": "Haas (from 2025)",
        "championships": 0,
        "wins": 1,
        "poles": 0,
        "career_seasons": "2016–present",
        "known_strengths": [
            "Consistent midfield scoring — reliable in clean conditions",
            "Good tyre management and racecraft over long stints",
        ],
        "known_weaknesses": [
            "Qualifying pace below top-10 benchmark in competitive seasons",
            "Occasional clashes with teammates reducing overall race performance",
        ],
        "context_note": (
            "Ocon is a solid midfield driver. Ratings should reflect mid-grid calibre."
        ),
    },
    "ALB": {
        "full_name": "Alexander Albon",
        "team": "Williams",
        "championships": 0,
        "wins": 0,
        "poles": 0,
        "career_seasons": "2019–present",
        "known_strengths": [
            "Excellent car development feedback — consistently outperforms machinery",
            "Strong racecraft and tyre management relative to the Williams package",
        ],
        "known_weaknesses": [
            "One-lap pace typically below top-10 benchmark",
        ],
        "context_note": (
            "Albon is highly regarded for his feedback and racecraft relative to his car's performance level."
        ),
    },
    "TSU": {
        "full_name": "Yuki Tsunoda",
        "team": "Red Bull Racing (from 2025)",
        "championships": 0,
        "wins": 0,
        "poles": 0,
        "career_seasons": "2021–present",
        "known_strengths": [
            "Aggressive one-lap pace — capable of strong qualifying results",
            "Fast learner who has improved significantly season over season",
        ],
        "known_weaknesses": [
            "Racecraft and consistency under pressure still developing",
            "Prone to errors in high-pressure wheel-to-wheel situations",
        ],
        "context_note": (
            "Tsunoda is a developing talent who has shown flashes of strong pace. "
            "Mid-grid ratings are appropriate."
        ),
    },
    "HUL": {
        "full_name": "Nico Hülkenberg",
        "team": "Sauber",
        "championships": 0,
        "wins": 0,
        "poles": 1,
        "career_seasons": "2010–present",
        "known_strengths": [
            "Consistent qualifying pace — often qualifies well above car's race potential",
            "Strong racecraft and technical feedback to engineers",
        ],
        "known_weaknesses": [
            "Never won an F1 race despite 200+ starts",
            "Race pace sometimes below qualifying form",
        ],
        "context_note": (
            "Hülkenberg is a reliable, experienced midfield competitor. Mid-grid ratings are appropriate."
        ),
    },
}

# ── Structured Report Card Schema ────────────────────────────────────────────
# (type, min_constraint, max_constraint) — None means unconstrained
_RC_SCHEMA: dict = {
    "headline":           (str,  None, None),
    "strengths":          (list, 3,    3),
    "weaknesses":         (list, 2,    2),
    "race_craft_rating":  (int,  1,    10),
    "raw_pace_rating":    (int,  1,    10),
    "consistency_rating": (int,  1,    10),
    "tactical_tips":      (list, 2,    2),
    "style_tags":         (list, 3,    5),
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
}

st.set_page_config(page_title="Driver DNA Fingerprinter", layout="wide")

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
    verbose: bool = False,
) -> tuple:
    """Build (system_msg, user_msg) for GPT-4o-mini. All interpolated values are
    pre-validated floats or a sanitized driver name — no raw user text enters the prompt.
    Pass verbose=True to use the Detailed-mode system prompt."""

    ranked = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    lines = [
        f"  {f}: {feature_dict[f]:.2f}  "
        f"(p{percentile_dict[f]:.0f}, SHAP: {'+'if shap_dict[f] >= 0 else ''}{shap_dict[f]:.4f})"
        for f, _ in ranked
    ]
    system_msg = _EXPLAIN_SYSTEM_VERBOSE if verbose else (
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
        + "\n\nExplain WHY the model made this prediction, "
        "citing the features with the largest |SHAP| values and their actual numbers."
    )
    return system_msg, user_msg


def _call_openai_llm(
    api_key: str,
    system_msg: str,
    user_msg: str,
    max_tokens: int = 300,
    temperature: float = 0.4,
    json_mode: bool = False,
    feature_name: str = "llm",
) -> tuple:
    """Unified LLM helper for all agentic AI features.

    Returns (text, None) on success or (None, error_msg) on failure.
    Set json_mode=True to request response_format={"type": "json_object"}.
    Every call is timed and appended to models/llm_audit.jsonl.
    """
    from openai import OpenAI, AuthenticationError, RateLimitError, APIError
    _MODEL = "gpt-4o-mini"
    t0 = time.perf_counter()
    try:
        client = OpenAI(api_key=api_key)
        kwargs: dict = dict(
            model=_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content
        latency_ms = (time.perf_counter() - t0) * 1000
        usage = resp.usage
        _log_llm_call(
            feature=feature_name,
            model=_MODEL,
            latency_ms=latency_ms,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            success=True,
        )
        if not text:
            return None, "OpenAI returned an empty response. Try again."
        return text.strip(), None
    except AuthenticationError:
        _log_llm_call(feature_name, _MODEL, (time.perf_counter()-t0)*1000, 0, 0, False, "AuthenticationError")
        return None, "OpenAI API key is invalid. Check `.streamlit/secrets.toml`."
    except RateLimitError:
        _log_llm_call(feature_name, _MODEL, (time.perf_counter()-t0)*1000, 0, 0, False, "RateLimitError")
        return None, "OpenAI rate limit reached. Wait a moment and try again."
    except APIError as e:
        _log_llm_call(feature_name, _MODEL, (time.perf_counter()-t0)*1000, 0, 0, False, type(e).__name__)
        return None, f"OpenAI service error ({type(e).__name__}). Try again shortly."
    except Exception:
        _log_llm_call(feature_name, _MODEL, (time.perf_counter()-t0)*1000, 0, 0, False, "UnexpectedError")
        return None, "Unexpected error contacting OpenAI. Try again."


# ── Reflexion Analyst helpers (ra_) ─────────────────────────────────────────

def _ra_build_driver_data_block(
    driver: str,
    df_ext: pd.DataFrame,
    ext_norm_means: pd.DataFrame,
) -> str:
    """Build a structured plain-text data block for LLM prompts."""
    drv_df = df_ext[df_ext["driver"] == driver]
    raw: pd.Series = drv_df[
        ["mean_speed", "max_speed", "min_speed", "throttle_mean", "throttle_std",
         "throttle_pickup_pct", "brake_mean", "brake_events", "trail_brake_score",
         "gear_changes", "coasting_pct"]
    ].mean()
    try:
        norm = ext_norm_means.loc[driver]
        archetype, emoji = classify_archetype(norm.to_dict())
    except KeyError:
        archetype, emoji = "Unknown", "?"

    lines = [
        f"DRIVER: {driver}",
        "--- Normalised radar profile (0 = lowest on grid, 1 = highest) ---",
    ]
    for feat in EXTENDED_RADAR_FEATURES:
        label = FEATURE_LABELS.get(feat, feat)
        try:
            val = float(cast(float, ext_norm_means.loc[driver, feat]))
        except Exception:
            val = float("nan")
        lines.append(f"  {label:<22} {val:.3f}")
    lines += [
        "--- Raw averages (real engineering units) ---",
        f"  Mean Speed:         {raw.get('mean_speed', float('nan')):.1f} km/h",
        f"  Max Speed:          {raw.get('max_speed', float('nan')):.1f} km/h",
        f"  Min Speed:          {raw.get('min_speed', float('nan')):.1f} km/h",
        f"  Throttle Input:     {raw.get('throttle_mean', float('nan')):.1f} %",
        f"  Throttle Variation: {raw.get('throttle_std', float('nan')):.2f}",
        f"  Full Throttle %:    {raw.get('throttle_pickup_pct', float('nan')):.4f} (fraction of lap)",
        f"  Brake Pressure:     {raw.get('brake_mean', float('nan')):.3f}",
        f"  Brake Events/lap:   {raw.get('brake_events', float('nan')):.1f}",
        f"  Trail Brake Score:  {raw.get('trail_brake_score', float('nan')):.4f}",
        f"  Gear Changes/lap:   {raw.get('gear_changes', float('nan')):.1f}",
        f"  Coasting Fraction:  {raw.get('coasting_pct', float('nan')):.4f}",
        f"--- Rule-engine archetype ---",
        f"  {emoji} {archetype}",
    ]
    return "\n".join(lines)


def _ra_parse_critique(raw_text: str) -> tuple:
    """Parse critic JSON response. Returns (critique_dict, None) or (None, error)."""
    try:
        data = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        return None, "Critic returned non-JSON response."
    if not isinstance(data, dict):
        return None, "Critic response was not a JSON object."
    try:
        confidence = max(1, min(10, int(data.get("confidence", 10))))
    except (TypeError, ValueError):
        return None, "Critic 'confidence' field missing or invalid."
    factual_errors = data.get("factual_errors", [])
    improvements = data.get("suggested_improvements", [])
    return {
        "confidence": confidence,
        "factual_errors": [str(e) for e in (factual_errors if isinstance(factual_errors, list) else [])[:5]],
        "suggested_improvements": [str(s) for s in (improvements if isinstance(improvements, list) else [])[:3]],
    }, None


_RA_ANALYST_SYSTEM = (
    "You are a professional F1 telemetry analyst. You will receive normalised radar data "
    "(0 = lowest on the grid, 1 = highest) and real-unit averages for a single driver. "
    "Write a driving style narrative in 4-6 sentences of fluent, confident prose suitable "
    "for a motorsport audience.\n\n"
    "Rules:\n"
    "- Cite at least 3 specific numbers from the data (use the real-unit averages for readability).\n"
    "- Anchor every claim to the provided numbers — do not speculate beyond the data.\n"
    "- Structure: open with the defining characteristic, develop with 2-3 supporting traits, "
    "close with a tactical implication.\n"
    "- No bullet points, headers, or markdown. Plain prose only.\n"
    "- Do not repeat the driver's name more than once."
)

_RA_CRITIC_SYSTEM = (
    "You are a strict F1 data auditor. Evaluate a driving style narrative against the telemetry "
    "data it was based on. Respond ONLY with valid JSON — no prose, no markdown, no code fences.\n\n"
    "JSON schema (return exactly these keys):\n"
    '{"confidence": <integer 1-10>, "factual_errors": [<string>, ...], '
    '"suggested_improvements": [<string>, ...]}\n\n'
    "Scoring guide for 'confidence': 10=all claims accurate; 7-9=minor omissions; "
    "4-6=at least one claim contradicts data; 1-3=multiple factual errors.\n"
    "'factual_errors': list each inaccurate claim with the correct number. Empty list if none.\n"
    "'suggested_improvements': 1-3 concrete ways to strengthen or correct the narrative."
)


def _ra_run_reflexion(
    driver: str,
    data_block: str,
    api_key: str,
    verbose: bool = False,
) -> tuple:
    """Orchestrate the Reflexion self-critique loop.

    Returns ({iterations, final_narrative}, None) or (None, error_msg).
    Each iteration: {round_num, narrative, critique (dict | None)}.
    Pass verbose=True for the Detailed-mode analyst prompt and higher token budget.
    """
    _CONFIDENCE_THRESHOLD = 7
    _analyst_system = _RA_ANALYST_SYSTEM_VERBOSE if verbose else _RA_ANALYST_SYSTEM
    _analyst_tokens = 650 if verbose else 400

    def analyst_call(user_msg: str) -> tuple:
        return _call_openai_llm(api_key, _analyst_system, user_msg,
                                max_tokens=_analyst_tokens, temperature=0.4,
                                feature_name="reflexion_analyst")

    def critic_call(narrative: str) -> tuple:
        user_msg = (
            f"Evaluate this narrative against the telemetry data below.\n\n"
            f"NARRATIVE:\n{narrative}\n\n"
            f"TELEMETRY DATA:\n{data_block}\n\n"
            "Respond with a JSON object only."
        )
        return _call_openai_llm(api_key, _RA_CRITIC_SYSTEM, user_msg,
                                max_tokens=300, temperature=0.0,
                                feature_name="reflexion_critic")

    iterations: list[dict] = []

    # Round 1 — Analyst
    user_r1 = (
        f"Analyse the driving style of {driver} based on the following telemetry data:\n\n"
        f"{data_block}\n\nWrite a 4-6 sentence driving style narrative."
    )
    narrative_1, err = analyst_call(user_r1)
    if err:
        return None, err
    iter1: dict = {"round_num": 1, "narrative": narrative_1, "critique": None}

    # Critique 1
    raw_critique_1, _ = critic_call(narrative_1)
    critique_1, parse_err = _ra_parse_critique(raw_critique_1 or "")
    if parse_err or critique_1 is None:
        # Critic failed — accept narrative as-is
        iter1["critique"] = {"confidence": 10, "factual_errors": [], "suggested_improvements": [],
                             "parse_note": "Critic response could not be parsed — narrative accepted."}
        return {"iterations": [iter1], "final_narrative": narrative_1}, None
    iter1["critique"] = critique_1
    iterations.append(iter1)

    if critique_1["confidence"] >= _CONFIDENCE_THRESHOLD:
        return {"iterations": iterations, "final_narrative": narrative_1}, None

    # Round 2 — Revision
    errors_str = "\n".join(f"  - {e}" for e in critique_1["factual_errors"]) or "  (none)"
    improve_str = "\n".join(f"  - {s}" for s in critique_1["suggested_improvements"]) or "  (none)"
    user_r2 = (
        f"Your previous narrative has been reviewed by a critic. Revise it to address the issues below.\n\n"
        f"Original narrative:\n{narrative_1}\n\n"
        f"Critic feedback:\n"
        f"  Confidence score: {critique_1['confidence']}/10\n"
        f"  Factual errors:\n{errors_str}\n"
        f"  Suggested improvements:\n{improve_str}\n\n"
        f"Current telemetry data:\n{data_block}\n\n"
        "Write an improved 4-6 sentence narrative. Do not mention the critique process."
    )
    narrative_2, err2 = analyst_call(user_r2)
    if err2:
        # Return what we have with round 1 only
        return {"iterations": iterations, "final_narrative": narrative_1}, None
    iter2: dict = {"round_num": 2, "narrative": narrative_2, "critique": None}

    # Critique 2
    raw_critique_2, _ = critic_call(narrative_2)
    critique_2, _ = _ra_parse_critique(raw_critique_2 or "")
    iter2["critique"] = critique_2 or {"confidence": 10, "factual_errors": [], "suggested_improvements": []}
    iterations.append(iter2)

    return {"iterations": iterations, "final_narrative": narrative_2}, None


# ── RAG DNA Matching helpers (rag_) ─────────────────────────────────────────

def _rag_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity in [0, 1]. Returns 0.0 if either vector is near-zero."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def _rag_retrieve_top_k(
    driver_vector: np.ndarray,
    profiles: dict,
    k: int = 2,
) -> list[dict]:
    """Return top-k historical profiles sorted by cosine similarity descending."""
    scores = []
    for name, profile in profiles.items():
        ref_vec = np.array(profile["vector"], dtype=float)
        sim = _rag_cosine_similarity(driver_vector, ref_vec)
        scores.append({
            "name": name,
            "similarity": sim,
            "description": profile["description"],
            "era": profile["era"],
            "vector": ref_vec,
        })
    scores.sort(key=lambda x: x["similarity"], reverse=True)
    return scores[:k]


_RAG_SYSTEM = (
    "You are an F1 driving style analyst. You will be given a current driver's 12-dimensional "
    "radar profile and the two most stylistically similar historical F1 drivers, retrieved by "
    "cosine similarity over normalised telemetry dimensions.\n\n"
    "Write 4-6 sentences explaining WHY this driver's style resembles these two legends. "
    "Be specific: reference at least 2 dimension values from the data.\n\n"
    "Rules: No bullet points, headers, or markdown. Plain prose only. "
    "Do not speculate beyond the provided numbers."
)


def _rag_run_dna_match(
    driver: str,
    driver_vector: np.ndarray,
    api_key: str,
    verbose: bool = False,
) -> tuple:
    """Retrieve top-2 historical matches and call LLM to narrate the comparison.

    Returns ({driver, matches, narrative}, None) or (None, error_msg).
    Pass verbose=True for the Detailed-mode system prompt and higher token budget.
    """
    matches = _rag_retrieve_top_k(driver_vector, _RAG_HISTORICAL_PROFILES, k=2)

    def _fmt_vector(vec: np.ndarray) -> str:
        return "  ".join(
            f"{FEATURE_LABELS.get(f, f)}: {vec[i]:.3f}"
            for i, f in enumerate(EXTENDED_RADAR_FEATURES)
        )

    user_msg = (
        f"CURRENT DRIVER: {driver}\n"
        f"Radar vector (12 normalised dimensions, 0=lowest on grid, 1=highest):\n"
        f"  {_fmt_vector(driver_vector)}\n\n"
    )
    for idx, match in enumerate(matches, 1):
        user_msg += (
            f"MOST SIMILAR HISTORICAL DRIVER #{idx}: {match['name']} ({match['era']})\n"
            f"  Cosine Similarity: {match['similarity']:.4f}\n"
            f"  Known style: {match['description']}\n"
            f"  Reference radar: {_fmt_vector(match['vector'])}\n\n"
        )
    user_msg += (
        f"Explain in 4-6 sentences why {driver}'s driving style is most similar to "
        f"{matches[0]['name']} and {matches[1]['name']}. Reference the specific dimensions "
        "that drove the similarity."
    )

    _rag_sys = _RAG_SYSTEM_VERBOSE if verbose else _RAG_SYSTEM
    _rag_tokens = 600 if verbose else 350
    narrative, err = _call_openai_llm(api_key, _rag_sys, user_msg,
                                      max_tokens=_rag_tokens, temperature=0.4,
                                      feature_name="rag_dna_match")
    if err:
        return None, err
    return {"driver": driver, "matches": matches, "narrative": narrative}, None


# ── Structured Report Card helpers (rc_) ────────────────────────────────────

def _rc_validate_report(data: dict) -> tuple:
    """Validate a parsed report card dict against _RC_SCHEMA.

    Returns (True, "") on success or (False, reason) on failure.
    Mutates data in-place to normalise types (int coercion, list trimming, HTML strip).
    """
    for key, (expected_type, lo, hi) in _RC_SCHEMA.items():
        if key not in data:
            return False, f"Missing required key: '{key}'"
        val = data[key]

        if expected_type == int:
            try:
                int_val = int(val)
            except (TypeError, ValueError):
                return False, f"Key '{key}' must be integer, got {type(val).__name__}"
            if lo is not None and not (lo <= int_val <= hi):
                return False, f"Key '{key}' value {int_val} outside range [{lo}, {hi}]"
            data[key] = int_val

        elif expected_type == list:
            if not isinstance(val, list):
                return False, f"Key '{key}' must be a list, got {type(val).__name__}"
            if lo is not None and len(val) < lo:
                return False, f"Key '{key}' has {len(val)} items, minimum is {lo}"
            if hi is not None and len(val) > hi:
                data[key] = val[:hi]
            for i, item in enumerate(data[key]):
                if not isinstance(item, str):
                    return False, f"Key '{key}[{i}]' must be a string"
                data[key][i] = item.replace("<", "").replace(">", "")

        elif expected_type == str:
            if not isinstance(val, str):
                return False, f"Key '{key}' must be a string, got {type(val).__name__}"
            data[key] = val.replace("<", "").replace(">", "")

    return True, ""


_RC_SYSTEM = (
    "You are a senior F1 telemetry engineer producing a driver DNA report card. "
    "You will receive normalised radar dimensions (0=lowest on grid, 1=highest) and "
    "raw engineering averages for a single driver.\n\n"
    "You MUST respond with a single valid JSON object that matches EXACTLY this schema "
    "— no prose before or after, no markdown code fences:\n"
    '{"headline": "<One sentence with a specific number>", '
    '"strengths": ["<str with data ref>", "<str>", "<str>"], '
    '"weaknesses": ["<str with data ref>", "<str>"], '
    '"race_craft_rating": <int 1-10>, '
    '"raw_pace_rating": <int 1-10>, '
    '"consistency_rating": <int 1-10>, '
    '"tactical_tips": ["<actionable tip>", "<actionable tip>"], '
    '"style_tags": ["<tag>", "<tag>", "<tag>"]}\n\n'
    "Constraints:\n"
    "- headline: exactly 1 sentence, must include at least one number from the data.\n"
    "- strengths: exactly 3 strings, each referencing a specific metric value.\n"
    "- weaknesses: exactly 2 strings, each referencing a specific metric value.\n"
    "- race_craft_rating: base on trail braking, brake precision, and steering consistency.\n"
    "- raw_pace_rating: base on mean speed, max speed, and full throttle percentage.\n"
    "- consistency_rating: base on lap-to-lap variability metrics provided.\n"
    "- tactical_tips: exactly 2 actionable, data-grounded strings.\n"
    "- style_tags: 3-5 short tag strings (e.g. 'Late Braker', 'Smooth Operator', 'High Entry Speed').\n"
    "- All string values must be safe for direct display — no HTML or markdown."
)


def _rc_build_user_prompt(driver: str, df_ext: pd.DataFrame, ext_norm_means: pd.DataFrame) -> str:
    """Build the user prompt for the report card LLM call."""
    drv_df = df_ext[df_ext["driver"] == driver]
    try:
        norm = cast(pd.Series, ext_norm_means.loc[driver])
    except KeyError:
        norm = pd.Series({f: float("nan") for f in EXTENDED_RADAR_FEATURES})
    raw: pd.Series = drv_df[EXTENDED_RADAR_FEATURES].mean()
    variability: pd.Series = drv_df[RADAR_FEATURES].std(ddof=0)

    def nv(feat: str) -> str:
        try:
            return f"{float(norm.get(feat, float('nan'))):.3f}"
        except Exception:
            return "n/a"

    def rv(feat: str) -> str:
        try:
            return f"{float(raw.get(feat, float('nan'))):.3f}"
        except Exception:
            return "n/a"

    try:
        archetype, emoji = classify_archetype(norm.to_dict())
    except Exception:
        archetype, emoji = "Unknown", "?"

    lines = [
        f"Generate a DNA Report Card for driver: {driver}\n",
        "NORMALISED RADAR (0 = lowest on grid, 1 = highest):",
    ]
    for feat in EXTENDED_RADAR_FEATURES:
        lines.append(f"  {FEATURE_LABELS.get(feat, feat):<22} {nv(feat)}")
    lines += [
        "\nRAW AVERAGES:",
        f"  Mean Speed: {rv('mean_speed')} km/h  |  Max Speed: {rv('max_speed')} km/h",
        f"  Throttle Input: {rv('throttle_mean')} %  |  Throttle Variation: {rv('throttle_std')}",
        f"  Full Throttle (fraction): {rv('throttle_pickup_pct')}",
        f"  Brake Pressure: {rv('brake_mean')}  |  Brake Frequency: {rv('brake_events')} /lap",
        f"  Trail Brake Score: {rv('trail_brake_score')}",
        f"  Gear Changes/lap: {rv('gear_changes')}",
        f"  Coasting Fraction: {rv('coasting_pct')}",
        "\nLAP-TO-LAP VARIABILITY (std dev across laps — higher = less consistent):",
    ]
    for feat in RADAR_FEATURES:
        try:
            std_val = f"{float(variability.get(feat, float('nan'))):.3f}"
        except Exception:
            std_val = "n/a"
        lines.append(f"  {FEATURE_LABELS.get(feat, feat):<22} {std_val}")
    lines.append(f"\nRule-engine archetype: {emoji} {archetype}")

    # Real-world context injection — calibrate ratings against known ground truth
    _ctx = _RC_DRIVER_CONTEXT.get(driver)
    if _ctx:
        lines += [
            "\nREAL-WORLD DRIVER CONTEXT (use this to calibrate ratings against known ground truth):",
            f"  Full name:        {_ctx['full_name']}",
            f"  Team:             {_ctx['team']}",
            f"  Championships:    {_ctx['championships']}",
            f"  Career wins:      {_ctx['wins']}",
            f"  Career poles:     {_ctx['poles']}",
            f"  Active seasons:   {_ctx['career_seasons']}",
            "  Known strengths:",
        ]
        for _s in _ctx["known_strengths"]:
            lines.append(f"    - {_s}")
        lines.append("  Known weaknesses:")
        for _w in _ctx["known_weaknesses"]:
            lines.append(f"    - {_w}")
        lines.append(f"  Analyst note: {_ctx['context_note']}")
        lines.append(
            "\nIMPORTANT: The telemetry above is a snapshot from a limited number of captured "
            "sessions, normalised only relative to drivers in this dataset — it is not a complete "
            "career picture. Use the real-world context above to calibrate race_craft_rating, "
            "raw_pace_rating, and consistency_rating so they reflect the driver's true standing on "
            "the current grid. A driver with 25 poles cannot score 3/10 for Raw Pace."
        )

    lines.append("\nRespond with the JSON object only.")
    return "\n".join(lines)


def _rc_run_report(
    driver: str,
    df_ext: pd.DataFrame,
    ext_norm_means: pd.DataFrame,
    api_key: str,
    verbose: bool = False,
) -> tuple:
    """Call JSON-mode LLM, validate, retry once on failure.

    Returns (validated_report_dict, None) or (None, error_msg).
    Pass verbose=True for the Detailed-mode system prompt and higher token budget.
    """
    user_msg = _rc_build_user_prompt(driver, df_ext, ext_norm_means)
    _rc_sys = _RC_SYSTEM_VERBOSE if verbose else _RC_SYSTEM
    _rc_tokens = 700 if verbose else 500

    # Attempt 1
    raw_json, err = _call_openai_llm(api_key, _rc_sys, user_msg,
                                     max_tokens=_rc_tokens, temperature=0.4, json_mode=True,
                                     feature_name="report_card")
    if err:
        return None, err
    try:
        data = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        data = {}
    valid, reason = _rc_validate_report(data)
    if valid:
        return data, None

    # Attempt 2 — inject correction
    correction_msg = (
        user_msg
        + f"\n\nYour previous response failed schema validation: {reason}\n"
        "Fix the issue and respond with the corrected JSON object only."
    )
    raw_json2, err2 = _call_openai_llm(api_key, _rc_sys, correction_msg,
                                       max_tokens=_rc_tokens, temperature=0.4, json_mode=True,
                                       feature_name="report_card_retry")
    if err2:
        return None, err2
    try:
        data2 = json.loads(raw_json2)
    except (json.JSONDecodeError, TypeError):
        return None, f"JSON parsing failed on retry. Original error: {reason}"
    valid2, reason2 = _rc_validate_report(data2)
    if valid2:
        return data2, None
    return None, f"Report validation failed after retry: {reason2}"


def _call_openai_chat(api_key: str, messages: list, tools: list, max_tokens: int = 500) -> tuple:
    """Call GPT-4o-mini with optional tool definitions for the chat agent.

    Returns one of three shapes:
      ("text",       final_text,      None)  — LLM produced a final answer
      ("tool_calls", tool_calls_list, None)  — LLM wants to invoke tools
      (None,         None,            error) — API failure
    Every call is timed and appended to models/llm_audit.jsonl.
    """
    from openai import OpenAI, AuthenticationError, RateLimitError, APIError
    _MODEL = "gpt-4o-mini"
    t0 = time.perf_counter()
    try:
        client = OpenAI(api_key=api_key)
        kwargs: dict = dict(
            model=_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.4,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message
        usage = resp.usage
        _log_llm_call(
            feature="chat_agent",
            model=_MODEL,
            latency_ms=(time.perf_counter() - t0) * 1000,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            success=True,
        )
        if choice.finish_reason == "tool_calls" and msg.tool_calls:
            return "tool_calls", msg.tool_calls, None
        text = msg.content
        if not text:
            return None, None, "OpenAI returned an empty response. Try again."
        return "text", text.strip(), None
    except AuthenticationError:
        _log_llm_call("chat_agent", _MODEL, (time.perf_counter()-t0)*1000, 0, 0, False, "AuthenticationError")
        return None, None, "OpenAI API key is invalid. Check `.streamlit/secrets.toml`."
    except RateLimitError:
        _log_llm_call("chat_agent", _MODEL, (time.perf_counter()-t0)*1000, 0, 0, False, "RateLimitError")
        return None, None, "OpenAI rate limit reached. Wait a moment and try again."
    except APIError as e:
        _log_llm_call("chat_agent", _MODEL, (time.perf_counter()-t0)*1000, 0, 0, False, type(e).__name__)
        return None, None, f"OpenAI service error ({type(e).__name__}). Try again shortly."
    except Exception:
        _log_llm_call("chat_agent", _MODEL, (time.perf_counter()-t0)*1000, 0, 0, False, "UnexpectedError")
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

# ── Verbose (Detailed mode) prompt constants ─────────────────────────────────

_EXPLAIN_SYSTEM_VERBOSE: str = (
    "You are an expert F1 telemetry analyst and machine-learning interpreter. "
    "Explain this model prediction in 2-3 paragraphs:\n"
    "1. The headline: what the prediction and confidence score mean in practice.\n"
    "2. Feature-by-feature breakdown of the top 3-4 SHAP contributors — cite each value, "
    "its percentile rank, and what it reveals about this driver's technique.\n"
    "3. Closing observation: what this fingerprints about the driver's signature style "
    "and whether any other driver was a close call.\n"
    "Use vivid motorsport language (trail braking, throttle pickup, traction management). "
    "Bold key numbers with **number**. Write in paragraphs, no bullet points or headers."
)

_RA_ANALYST_SYSTEM_VERBOSE: str = (
    "You are a senior F1 race engineer presenting a driver style dossier to team management. "
    "Write a multi-paragraph analysis (8-12 sentences across 2-3 paragraphs):\n"
    "1. Opening: the driver's defining philosophy — what kind of racing driver are they at their core?\n"
    "2. Technical deep-dive: cite at least 5 specific numbers; explain what each reveals and draw "
    "connections between related dimensions (e.g. how coasting % relates to throttle pickup, "
    "how trail braking score relates to corner speed).\n"
    "3. Tactical: preferred track types, where this driver gains or loses time versus the grid, "
    "and what a race engineer would prioritise improving.\n"
    "Write in the style of a real F1 engineer's notebook — precise, observational, no bullet points."
)

_RAG_SYSTEM_VERBOSE: str = (
    "You are an F1 historian and data analyst comparing a modern driver's telemetry fingerprint "
    "to legendary F1 drivers. Write a rich 3-paragraph comparison:\n"
    "1. Primary match: which specific dimensions align most strongly with the top legend — cite "
    "exact values from both drivers and explain what they mean in concrete racing terms.\n"
    "2. Secondary match and nuance: what the second legend adds to the picture; note any dimensions "
    "where the modern driver diverges from both legends and what that implies about their unique style.\n"
    "3. Synthesis: what track types and conditions suit this driver, what their inherent limitations "
    "are, and how their style would have fared racing against their matched legends in period cars.\n"
    "Be specific with numbers. Write as a genuine motorsport historical analysis — no bullet points."
)

_RC_SYSTEM_VERBOSE: str = (
    "You are a senior F1 data engineer producing a detailed driver DNA report card for team "
    "management. Your report must be technically precise AND narratively rich.\n\n"
    "Respond ONLY with a JSON object (no prose, no code fences) matching exactly this schema:\n"
    '{"headline": "<1-2 sentences with 2+ specific numbers capturing this driver\'s essence>", '
    '"strengths": ["<strength citing 2+ metrics with engineering context>", '
    '"<strength citing 2+ metrics>", "<strength citing 2+ metrics>"], '
    '"weaknesses": ["<root-cause analysis referencing specific values>", '
    '"<root-cause analysis referencing specific values>"], '
    '"race_craft_rating": <int 1-10>, "raw_pace_rating": <int 1-10>, '
    '"consistency_rating": <int 1-10>, '
    '"tactical_tips": ["<engineer-grade actionable recommendation with data reference>", '
    '"<engineer-grade actionable recommendation>"], '
    '"style_tags": ["<tag>", "<tag>", "<tag>", "<tag>"]}\n\n'
    "Constraints: headline 1-2 sentences with ≥2 numbers; strengths exactly 3, each citing ≥2 "
    "metric values; weaknesses exactly 2 with root-cause reasoning; tactical_tips exactly 2 "
    "detailed engineer-grade recommendations; style_tags exactly 4 short descriptors. "
    "No HTML or markdown in string values."
)

_CA_SYSTEM_VERBOSE: str = (
    "You are Race Intelligence, an expert F1 race strategist embedded in a live telemetry "
    "dashboard. Think and communicate like a race engineer who has seen everything.\n\n"
    "Your tools:\n"
    "  get_rolling_pace        — recent lap pace trend per driver\n"
    "  get_gap_to_leader       — cumulative time gap from each driver to the leader\n"
    "  detect_strategy_events  — undercut and overcut detections\n"
    "  project_finishing_order — projected final race order\n"
    "  get_tyre_degradation    — degradation rate per driver/compound/stint\n"
    "  get_pace_summary        — aggregate mean, median, fastest lap per driver\n\n"
    "Rules:\n"
    "- Always call at least one tool before answering pace, strategy, or position questions.\n"
    "- Provide a thorough analysis (6-10 sentences): lead with the direct answer, explain the "
    "data evidence in detail, draw inferences about strategy or performance trends, and close "
    "with a forward-looking observation — what to watch for on the next 5 laps.\n"
    "- Cite specific numbers: lap times in full (e.g. 1:23.456), gaps in seconds, degradation "
    "rates, stint lengths, and position changes.\n"
    "- Explain the *implication* of the data — not just what it shows but what it means for "
    "strategy, tyre management, or the race outcome.\n"
    "- Bold key numbers and driver acronyms using **text**.\n"
    "- If data is insufficient, explain exactly what you would need to give a fuller answer.\n"
    "- Draw reasoned inferences from patterns in the data, but never fabricate numbers."
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


# ── Custom pill-style tab navigation ────────────────────────────────────────
_TABS = [
    ("🎯  Driver Radar",   "Driver Radar"),
    ("🕵️  Mystery Driver", "Mystery Driver"),
    ("🏁  Race Dashboard", "Race Dashboard"),
]

st.markdown("""
<style>
div[data-testid="stHorizontalBlock"] > div:has(> div > button.tab-pill) {
    gap: 0 !important;
}
button.tab-pill {
    all: unset;
    cursor: pointer;
    padding: 10px 26px;
    border-radius: 30px;
    font-size: 0.93rem;
    font-weight: 500;
    transition: background 0.2s, color 0.2s, box-shadow 0.2s;
    white-space: nowrap;
    letter-spacing: 0.01em;
}
</style>
""", unsafe_allow_html=True)

_tab_cols = st.columns(len(_TABS))
for _col, (_label, _key) in zip(_tab_cols, _TABS):
    _is_active = st.session_state.get("active_tab", "Driver Radar") == _key
    with _col:
        _btn_style = (
            "background:#5865f2; color:#fff; box-shadow:0 2px 10px rgba(88,101,242,0.45);"
            if _is_active
            else "background:rgba(255,255,255,0.06); color:#aaa;"
        )
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
                        f'<span style="background:#2d2d50;border:1px solid #555;border-radius:12px;'
                        f'padding:3px 10px;font-size:0.85em;color:#ccc;margin:2px">{tag}</span>'
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
                        if v >= 9:  return "Elite",         "#00e676"
                        if v >= 7:  return "Strong",        "#69f0ae"
                        if v >= 5:  return "Good",          "#ffd740"
                        if v >= 3:  return "Average",       "#ff9100"
                        return              "Below Average", "#ff5252"

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
                            f'  <div style="background:#2a2a3a;border-radius:6px;height:10px;width:100%">'
                            f'    <div style="background:{color};width:{pct}%;height:10px;border-radius:6px;'
                            f'               transition:width 0.4s ease"></div>'
                            f'  </div>'
                            f'  <div style="font-size:0.78em;color:#888;margin-top:3px">{tooltip}</div>'
                            f'</div>'
                        )

                    _overall_tier, _overall_color = _rating_tier(round(_overall))
                    st.markdown(
                        f'<div style="background:#1a1a2e;border:1px solid #333;border-radius:10px;'
                        f'padding:16px 20px;margin-bottom:18px">'
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
                            f'<div style="background:rgba(0,200,100,0.07);border-left:3px solid #00c864;'
                            f'border-radius:6px;padding:10px 14px;margin-bottom:8px;font-size:0.93em">'
                            f'✅ &nbsp;{_s}</div>',
                            unsafe_allow_html=True,
                        )

                    # ── Weaknesses ────────────────────────────────────
                    st.markdown("#### Weaknesses")
                    for _w in _report.get("weaknesses", []):
                        st.markdown(
                            f'<div style="background:rgba(255,150,0,0.07);border-left:3px solid #ff9100;'
                            f'border-radius:6px;padding:10px 14px;margin-bottom:8px;font-size:0.93em">'
                            f'⚠️ &nbsp;{_w}</div>',
                            unsafe_allow_html=True,
                        )

                    st.divider()

                    # ── Tactical Tips ─────────────────────────────────
                    st.markdown("#### Tactical Tips")
                    for _ti, _tip in enumerate(_report.get("tactical_tips", []), 1):
                        st.markdown(
                            f'<div style="background:rgba(30,120,255,0.07);border-left:3px solid #1e78ff;'
                            f'border-radius:6px;padding:10px 14px;margin-bottom:8px;font-size:0.93em">'
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
                        safe_driver = _sanitize_driver_name(_pred_driver, list(le.classes_))
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
        mode="markers",
        marker=dict(color="rgba(120,120,120,0.25)", size=8, symbol="circle"),
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
            mode="markers",
            marker=dict(color=col, size=7, symbol="circle"),
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
