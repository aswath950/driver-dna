"""llm_layer.py — LLM orchestration, system prompts, and agentic AI feature helpers.

No Streamlit UI code — all functions take plain Python arguments and return plain
Python values (strings, dicts, tuples). The four functions that read/write
st.session_state or st.secrets remain in app.py:
  _get_openai_api_key, _ca_append_message, _ca_maybe_clear_history, _ca_run_chat_turn
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import cast

import numpy as np
import pandas as pd

from model import MODELS_DIR
from features import (
    EXTENDED_RADAR_FEATURES,
    FEATURE_LABELS,
    RADAR_FEATURES,
    classify_archetype,
)

LLM_AUDIT_PATH = MODELS_DIR / "llm_audit.jsonl"
_logger = logging.getLogger("driver_dna")

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

# ── Chat agent security constants ────────────────────────────────────────────
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

# ── System prompts ───────────────────────────────────────────────────────────

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

_RAG_SYSTEM = (
    "You are an F1 driving style analyst. You will be given a current driver's 12-dimensional "
    "radar profile and the two most stylistically similar historical F1 drivers, retrieved by "
    "cosine similarity over normalised telemetry dimensions.\n\n"
    "Write 4-6 sentences explaining WHY this driver's style resembles these two legends. "
    "Be specific: reference at least 2 dimension values from the data.\n\n"
    "Rules: No bullet points, headers, or markdown. Plain prose only. "
    "Do not speculate beyond the provided numbers."
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

# ── Tool definitions (OpenAI function-calling schema) ────────────────────────

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

# ── Core LLM helpers ─────────────────────────────────────────────────────────

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


# ── Chat agent guardrail helpers ─────────────────────────────────────────────

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


# ── Chat agent tool executors ────────────────────────────────────────────────

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
