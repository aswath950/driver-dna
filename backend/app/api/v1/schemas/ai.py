"""Request + response schemas for the /api/v1/ai/* endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.api.v1.schemas.common import ORMModel

# ---------------------------------------------------------------------------
# Style Analyst (Reflexion)
# ---------------------------------------------------------------------------


class StyleAnalystRequest(BaseModel):
    driver_id: int
    season: int = Field(..., examples=[2024])


class StyleAnalystResponse(ORMModel):
    analysis: str
    critique: dict | None = None
    revised: str | None = None
    confidence: int
    rounds: int


# ---------------------------------------------------------------------------
# DNA Match (RAG)
# ---------------------------------------------------------------------------


class DNAMatchRequest(BaseModel):
    driver_id: int
    season: int = Field(..., examples=[2024])


class DNAMatchMatchOut(BaseModel):
    name: str
    era: str
    description: str
    similarity: float


class DNAMatchResponse(ORMModel):
    driver_code: str
    vector: list[float]
    matches: list[DNAMatchMatchOut]
    narrative: str


# ---------------------------------------------------------------------------
# Report Card (Structured Output)
# ---------------------------------------------------------------------------


class ReportCardRequest(BaseModel):
    driver_id: int
    season: int = Field(..., examples=[2024])


class ReportCardResponse(ORMModel):
    grade: str
    headline: str
    strengths: list[str]
    weaknesses: list[str]
    verdict: str


# ---------------------------------------------------------------------------
# XAI Explain (Single-shot)
# ---------------------------------------------------------------------------


class FeatureContribution(BaseModel):
    feature: str
    value: float
    percentile: float
    shap: float


class XAIExplainRequest(BaseModel):
    predicted_driver_code: str
    feature_contributions: list[FeatureContribution] = Field(..., min_length=1)


class XAIExplainResponse(ORMModel):
    explanation: str
    top_features: list[dict]


# ---------------------------------------------------------------------------
# Race Chat (ReAct, SSE)
# ---------------------------------------------------------------------------


class RaceChatRequest(BaseModel):
    session_id: int
    message: str = Field(..., min_length=1, max_length=2000)
