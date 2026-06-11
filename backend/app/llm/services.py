"""The five Phase-9 LLM features.

Each function loads data from Postgres, builds a prompt, calls OpenAI,
records the call to ``llm_audit``, and returns a structured payload to
the router. No prompts or guardrails live in the router — they're all
here so they can be unit-tested independently.

Patterns represented (matches the Streamlit app):
  1. **Reflexion** — Analyst → Critic → conditional revision (style_analyst)
  2. **RAG** — Cosine retrieval over 5 historical-style profiles (dna_match)
  3. **Structured Output** — JSON mode + schema validation (report_card)
  4. **Single-shot** — One-pass narration with validated inputs (xai_explain)
  5. **ReAct** — Tool-calling loop with streaming (race_chat — see sse.py)
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import NotFoundError
from app.db.models import (
    Driver,
    DriverStats,
    Season,
)
from app.llm import openai_client
from app.llm.audit import record_llm_call

# ---------------------------------------------------------------------------
# Tiny RAG knowledge base: 5 historical-style archetypes with 4-dim vectors
# (consistency, aggression, tyre-care, qualifying-edge — all in [0, 1]).
# Real telemetry-derived 12-dim vectors live in src/llm_layer for Streamlit;
# the REST endpoint uses this lighter aggregate from per-driver DB stats.
# ---------------------------------------------------------------------------

_PROFILES: list[dict] = [
    {"name": "The Surgeon",     "era": "modern",  "desc": "Precise, error-free, conserves tyres.",
     "vector": [0.95, 0.40, 0.90, 0.65]},
    {"name": "The Berserker",   "era": "v6 turbo","desc": "Late-braking, aggressive, racy.",
     "vector": [0.55, 0.95, 0.30, 0.70]},
    {"name": "The Qualifier",   "era": "V10 era", "desc": "Single-lap specialist, Saturday hero.",
     "vector": [0.70, 0.65, 0.50, 0.95]},
    {"name": "The Diesel",      "era": "modern",  "desc": "Long-stint master, kind to rubber.",
     "vector": [0.80, 0.30, 0.95, 0.40]},
    {"name": "The Showman",     "era": "all-time","desc": "High-variance, brilliant when it clicks.",
     "vector": [0.40, 0.85, 0.45, 0.80]},
]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Shared DB lookup
# ---------------------------------------------------------------------------


async def _load_driver_stats(
    db: AsyncSession, *, driver_id: int, season: int
) -> tuple[Driver, DriverStats]:
    drv = await db.get(Driver, driver_id)
    if drv is None:
        raise NotFoundError("driver", driver_id)
    stats = await db.scalar(
        select(DriverStats)
        .join(Season, Season.id == DriverStats.season_id)
        .where(DriverStats.driver_id == driver_id, Season.year == season)
    )
    if stats is None:
        raise NotFoundError("driver_stats", f"driver={driver_id} season={season}")
    return drv, stats


def _vector_from_stats(stats: DriverStats) -> list[float]:
    """Project the driver's seasonal stats onto the 4-dim style space.

    All four dimensions are normalised to [0, 1] using rough F1 heuristics —
    the LLM only needs comparable magnitudes, not absolute calibration.
    """
    wins = stats.wins or 0
    podiums = stats.podiums or 0
    poles = stats.poles or 0
    dnfs = stats.dnfs or 0
    races_est = max(podiums + dnfs + 4, 1)
    return [
        max(0.0, min(1.0, 1.0 - dnfs / max(races_est, 1))),     # consistency
        max(0.0, min(1.0, (poles + wins) / 25.0)),              # aggression proxy
        max(0.0, min(1.0, podiums / max(races_est, 1))),        # tyre-care proxy
        max(0.0, min(1.0, poles / 20.0)),                       # qualifying edge
    ]


# ===========================================================================
# 1. Reflexion — style_analyst
# ===========================================================================


@dataclass
class StyleAnalystResult:
    analysis: str
    critique: dict | None
    revised: str | None
    confidence: int
    rounds: int


_RA_ANALYST_SYS = (
    "You are an F1 analyst writing concise driving-style narratives based on "
    "season stats. 4–6 sentences. No bullet lists. Cite specific numbers."
)
_RA_CRITIC_SYS = (
    "You critique F1 narratives for factual accuracy against the stats. "
    "Return ONLY a JSON object: "
    '{"confidence": <int 1-10>, "issues": [<string>...]}.'
)


async def run_style_analyst(
    db: AsyncSession,
    *,
    driver_id: int,
    season: int,
    request_id: str | None,
    user_session_id: uuid.UUID | None = None,
) ->StyleAnalystResult:
    drv, stats = await _load_driver_stats(db, driver_id=driver_id, season=season)
    data_block = (
        f"Driver: {drv.full_name} ({drv.code}), {season} season\n"
        f"Wins: {stats.wins}, Podiums: {stats.podiums}, Poles: {stats.poles}, "
        f"DNFs: {stats.dnfs}, Points: {stats.points}, Avg finish: {stats.avg_finish}"
    )

    # Round 1 — Analyst
    user_r1 = f"Analyse this driver:\n\n{data_block}\n\nWrite the narrative now."
    analysis, err, usage = await openai_client.chat_completion(
        system_msg=_RA_ANALYST_SYS, user_msg=user_r1,
        max_tokens=400, temperature=0.4,
    )
    await record_llm_call(
        db, feature="style_analyst.analyst", model=openai_client.MODEL,
        input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"],
        latency_ms=usage["latency_ms"],
        status="success" if not err else "error", error_type=err,
        request_id=request_id, user_session_id=user_session_id,
    )
    if err or not analysis:
        raise NotFoundError("openai_response", err or "empty")

    # Critic
    critic_user = (
        f"NARRATIVE:\n{analysis}\n\nSTATS:\n{data_block}\n\n"
        "Return the JSON object now."
    )
    critic_raw, c_err, c_usage = await openai_client.chat_completion(
        system_msg=_RA_CRITIC_SYS, user_msg=critic_user,
        max_tokens=300, temperature=0.0, json_mode=True,
    )
    await record_llm_call(
        db, feature="style_analyst.critic", model=openai_client.MODEL,
        input_tokens=c_usage["input_tokens"], output_tokens=c_usage["output_tokens"],
        latency_ms=c_usage["latency_ms"],
        status="success" if not c_err else "error", error_type=c_err,
        request_id=request_id, user_session_id=user_session_id,
    )

    critique: dict | None = None
    confidence = 10
    if critic_raw:
        try:
            critique = json.loads(critic_raw)
            confidence = int(critique.get("confidence", 10))
        except (json.JSONDecodeError, ValueError, TypeError):
            critique = None

    # Round 2 — revise only if critic was unhappy
    revised: str | None = None
    rounds = 1
    if critique and confidence < 7:
        issues = critique.get("issues", []) or []
        bullet = "\n".join(f"  - {i}" for i in issues) or "  - (none specified)"
        revise_user = (
            f"Revise the narrative below to address these critic notes.\n\n"
            f"ORIGINAL:\n{analysis}\n\n"
            f"CRITIC NOTES (confidence={confidence}/10):\n{bullet}\n\n"
            f"STATS:\n{data_block}\n\n"
            "Write the improved 4-6 sentence narrative now."
        )
        revised, r_err, r_usage = await openai_client.chat_completion(
            system_msg=_RA_ANALYST_SYS, user_msg=revise_user,
            max_tokens=400, temperature=0.4,
        )
        await record_llm_call(
            db, feature="style_analyst.revise", model=openai_client.MODEL,
            input_tokens=r_usage["input_tokens"], output_tokens=r_usage["output_tokens"],
            latency_ms=r_usage["latency_ms"],
            status="success" if not r_err else "error", error_type=r_err,
            request_id=request_id, user_session_id=user_session_id,
        )
        rounds = 2

    return StyleAnalystResult(
        analysis=analysis,
        critique=critique,
        revised=revised,
        confidence=confidence,
        rounds=rounds,
    )


# ===========================================================================
# 2. RAG — dna_match
# ===========================================================================


@dataclass
class DNAMatch:
    name: str
    era: str
    description: str
    similarity: float


@dataclass
class DNAMatchResult:
    driver_code: str
    vector: list[float]
    matches: list[DNAMatch]
    narrative: str


_RAG_SYS = (
    "You explain F1 driving-style similarity to historical archetypes. "
    "4–6 sentences, reference the specific dimensions that drove the match."
)


async def run_dna_match(
    db: AsyncSession,
    *,
    driver_id: int,
    season: int,
    request_id: str | None,
    user_session_id: uuid.UUID | None = None,
) ->DNAMatchResult:
    drv, stats = await _load_driver_stats(db, driver_id=driver_id, season=season)
    vec = _vector_from_stats(stats)

    scored = sorted(
        (
            {**p, "similarity": _cosine(vec, p["vector"])}
            for p in _PROFILES
        ),
        key=lambda p: p["similarity"],
        reverse=True,
    )
    top2 = scored[:2]

    def _fmt(v: list[float]) -> str:
        labels = ["consistency", "aggression", "tyre_care", "qualifying_edge"]
        return ", ".join(f"{lab}={x:.2f}" for lab, x in zip(labels, v))

    user_msg = (
        f"CURRENT DRIVER: {drv.full_name} ({drv.code}), {season}\n"
        f"  Style vector: {_fmt(vec)}\n\n"
        + "".join(
            f"MATCH #{i}: {m['name']} ({m['era']}) — sim={m['similarity']:.3f}\n"
            f"  Style: {m['desc']}\n  Vector: {_fmt(m['vector'])}\n\n"
            for i, m in enumerate(top2, 1)
        )
        + f"Explain why {drv.code}'s style is closest to {top2[0]['name']} "
          f"and {top2[1]['name']}."
    )

    text, err, usage = await openai_client.chat_completion(
        system_msg=_RAG_SYS, user_msg=user_msg,
        max_tokens=400, temperature=0.4,
    )
    await record_llm_call(
        db, feature="dna_match", model=openai_client.MODEL,
        input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"],
        latency_ms=usage["latency_ms"],
        status="success" if not err else "error", error_type=err,
        request_id=request_id, user_session_id=user_session_id,
    )
    if err or not text:
        raise NotFoundError("openai_response", err or "empty")

    return DNAMatchResult(
        driver_code=drv.code,
        vector=vec,
        matches=[
            DNAMatch(name=m["name"], era=m["era"], description=m["desc"],
                     similarity=round(m["similarity"], 4))
            for m in top2
        ],
        narrative=text,
    )


# ===========================================================================
# 3. Structured Output — report_card
# ===========================================================================


_RC_SYS = (
    "You are an F1 season report-card generator. Return ONLY a JSON object "
    "matching this schema:\n"
    "{\n"
    '  "grade": "A+|A|B+|B|C+|C|D",\n'
    '  "headline": "<one sentence summary>",\n'
    '  "strengths": [<string>, ...],   # 2-4 items\n'
    '  "weaknesses": [<string>, ...],  # 1-3 items\n'
    '  "verdict": "<2-3 sentences>"\n'
    "}\n"
    "No prose outside the JSON."
)

_RC_GRADES = {"A+", "A", "B+", "B", "C+", "C", "D"}


def _validate_report_card(data: dict) -> tuple[bool, str]:
    if not isinstance(data, dict):
        return False, "not an object"
    for k in ("grade", "headline", "strengths", "weaknesses", "verdict"):
        if k not in data:
            return False, f"missing {k}"
    if data["grade"] not in _RC_GRADES:
        return False, f"invalid grade: {data['grade']!r}"
    if not isinstance(data["strengths"], list) or not 2 <= len(data["strengths"]) <= 4:
        return False, "strengths must be a list of 2-4 strings"
    if not isinstance(data["weaknesses"], list) or not 1 <= len(data["weaknesses"]) <= 3:
        return False, "weaknesses must be a list of 1-3 strings"
    return True, ""


@dataclass
class ReportCardResult:
    grade: str
    headline: str
    strengths: list[str]
    weaknesses: list[str]
    verdict: str
    raw: dict


async def run_report_card(
    db: AsyncSession,
    *,
    driver_id: int,
    season: int,
    request_id: str | None,
    user_session_id: uuid.UUID | None = None,
) ->ReportCardResult:
    drv, stats = await _load_driver_stats(db, driver_id=driver_id, season=season)
    user_msg = (
        f"Driver: {drv.full_name} ({drv.code}), {season}\n"
        f"Wins: {stats.wins}, Podiums: {stats.podiums}, Poles: {stats.poles}, "
        f"DNFs: {stats.dnfs}, Points: {stats.points}, Avg finish: {stats.avg_finish}\n\n"
        "Return the JSON report card now."
    )

    text, err, usage = await openai_client.chat_completion(
        system_msg=_RC_SYS, user_msg=user_msg,
        max_tokens=500, temperature=0.3, json_mode=True,
    )
    await record_llm_call(
        db, feature="report_card", model=openai_client.MODEL,
        input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"],
        latency_ms=usage["latency_ms"],
        status="success" if not err else "error", error_type=err,
        request_id=request_id, user_session_id=user_session_id,
    )
    if err or not text:
        raise NotFoundError("openai_response", err or "empty")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise NotFoundError("openai_response", f"invalid JSON: {e}") from e

    ok, why = _validate_report_card(data)
    if not ok:
        raise NotFoundError("openai_response", f"schema violation: {why}")

    return ReportCardResult(
        grade=data["grade"],
        headline=data["headline"],
        strengths=list(data["strengths"]),
        weaknesses=list(data["weaknesses"]),
        verdict=data["verdict"],
        raw=data,
    )


# ===========================================================================
# 4. Single-shot narration — xai_explain
# ===========================================================================


_XAI_SYS = (
    "You explain ML model predictions for F1 driver identification in plain "
    "English. Lead with the top 2 contributing features, then summarise."
)


@dataclass
class XAIExplainResult:
    explanation: str
    top_features: list[dict]


async def run_xai_explain(
    db: AsyncSession,
    *,
    predicted_driver_code: str,
    feature_contributions: list[dict],
    request_id: str | None,
    user_session_id: uuid.UUID | None = None,
) ->XAIExplainResult:
    # feature_contributions = [{feature, value, percentile, shap}, ...]
    if not feature_contributions:
        raise NotFoundError("feature_contributions", "empty list")

    top = sorted(
        feature_contributions, key=lambda f: abs(float(f.get("shap", 0))), reverse=True
    )[:4]
    bullet = "\n".join(
        f"  - {f.get('feature')}: value={f.get('value')}, "
        f"percentile={f.get('percentile')}, shap={f.get('shap')}"
        for f in top
    )
    user_msg = (
        f"Predicted driver: {predicted_driver_code}\n"
        f"Top SHAP contributions:\n{bullet}\n\n"
        "Write a 3-4 sentence plain-English explanation."
    )

    text, err, usage = await openai_client.chat_completion(
        system_msg=_XAI_SYS, user_msg=user_msg,
        max_tokens=300, temperature=0.3,
    )
    await record_llm_call(
        db, feature="xai_explain", model=openai_client.MODEL,
        input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"],
        latency_ms=usage["latency_ms"],
        status="success" if not err else "error", error_type=err,
        request_id=request_id, user_session_id=user_session_id,
    )
    if err or not text:
        raise NotFoundError("openai_response", err or "empty")

    return XAIExplainResult(explanation=text, top_features=top)
