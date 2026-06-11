"""/api/v1/ai/* — five LLM-backed endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.api.v1.schemas.ai import (
    DNAMatchMatchOut,
    DNAMatchRequest,
    DNAMatchResponse,
    RaceChatRequest,
    ReportCardRequest,
    ReportCardResponse,
    StyleAnalystRequest,
    StyleAnalystResponse,
    XAIExplainRequest,
    XAIExplainResponse,
)
from app.core.deps import DB, RequestID, UserSessionID
from app.core.errors import ErrorEnvelope
from app.llm import race_chat, services

router = APIRouter(prefix="/ai", tags=["ai"])

_ERR_RESPS = {404: {"model": ErrorEnvelope}, 503: {"model": ErrorEnvelope}}


@router.post(
    "/style-analyst",
    response_model=StyleAnalystResponse,
    responses=_ERR_RESPS,
    summary="Reflexion: analyst → critic → conditional revision narrative.",
)
async def style_analyst(
    body: StyleAnalystRequest, db: DB, request_id: RequestID, sid: UserSessionID,
) -> StyleAnalystResponse:
    r = await services.run_style_analyst(
        db, driver_id=body.driver_id, season=body.season,
        request_id=request_id, user_session_id=sid,
    )
    return StyleAnalystResponse(
        analysis=r.analysis,
        critique=r.critique,
        revised=r.revised,
        confidence=r.confidence,
        rounds=r.rounds,
    )


@router.post(
    "/dna-match",
    response_model=DNAMatchResponse,
    responses=_ERR_RESPS,
    summary="RAG: cosine retrieve top-2 historical archetypes + narrative.",
)
async def dna_match(
    body: DNAMatchRequest, db: DB, request_id: RequestID, sid: UserSessionID,
) -> DNAMatchResponse:
    r = await services.run_dna_match(
        db, driver_id=body.driver_id, season=body.season,
        request_id=request_id, user_session_id=sid,
    )
    return DNAMatchResponse(
        driver_code=r.driver_code,
        vector=r.vector,
        matches=[
            DNAMatchMatchOut(name=m.name, era=m.era, description=m.description,
                             similarity=m.similarity)
            for m in r.matches
        ],
        narrative=r.narrative,
    )


@router.post(
    "/report-card",
    response_model=ReportCardResponse,
    responses=_ERR_RESPS,
    summary="Structured output: JSON-mode report card with schema validation.",
)
async def report_card(
    body: ReportCardRequest, db: DB, request_id: RequestID, sid: UserSessionID,
) -> ReportCardResponse:
    r = await services.run_report_card(
        db, driver_id=body.driver_id, season=body.season,
        request_id=request_id, user_session_id=sid,
    )
    return ReportCardResponse(
        grade=r.grade,
        headline=r.headline,
        strengths=r.strengths,
        weaknesses=r.weaknesses,
        verdict=r.verdict,
    )


@router.post(
    "/xai-explain",
    response_model=XAIExplainResponse,
    responses=_ERR_RESPS,
    summary="Plain-English narration of SHAP contributions for one prediction.",
)
async def xai_explain(
    body: XAIExplainRequest, db: DB, request_id: RequestID, sid: UserSessionID,
) -> XAIExplainResponse:
    r = await services.run_xai_explain(
        db,
        predicted_driver_code=body.predicted_driver_code,
        feature_contributions=[fc.model_dump() for fc in body.feature_contributions],
        request_id=request_id,
        user_session_id=sid,
    )
    return XAIExplainResponse(explanation=r.explanation, top_features=r.top_features)


@router.post(
    "/race-chat/stream",
    responses={**_ERR_RESPS, 200: {"content": {"text/event-stream": {}}}},
    summary="ReAct tool-loop chat over race analytics; streams via SSE.",
)
async def race_chat_stream(
    body: RaceChatRequest, db: DB, request_id: RequestID, sid: UserSessionID,
    request: Request,
) -> StreamingResponse:
    gen = race_chat.race_chat_stream(
        db, session_id=body.session_id, user_message=body.message,
        request_id=request_id, user_session_id=sid,
    )
    return StreamingResponse(gen, media_type="text/event-stream")
