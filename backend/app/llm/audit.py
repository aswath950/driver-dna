"""Persist every LLM call to the ``llm_audit`` table.

Token rates for gpt-4o-mini as of 2026-05; bump these here if the price
changes — callers don't compute cost themselves.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import LLMAudit

# USD per 1M tokens (gpt-4o-mini)
_RATES = {
    "gpt-4o-mini": (Decimal("0.150"), Decimal("0.600")),
}


def _cost_usd(model: str, input_tokens: int, output_tokens: int) -> Decimal:
    rate = _RATES.get(model, (Decimal("0"), Decimal("0")))
    return (
        rate[0] * Decimal(input_tokens) + rate[1] * Decimal(output_tokens)
    ) / Decimal("1000000")


async def record_llm_call(
    db: AsyncSession,
    *,
    feature: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: int,
    status: str,
    error_type: str | None = None,
    request_id: str | None = None,
    user_session_id: uuid.UUID | None = None,
) -> None:
    """Persist one llm_audit row. Best-effort: on failure we swallow so an
    audit-store outage never propagates into a user-facing 500."""
    row = LLMAudit(
        request_id=request_id,
        feature=feature,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        cost_usd=_cost_usd(model, input_tokens, output_tokens),
        status=status,
        error_type=error_type,
        user_session_id=user_session_id,
    )
    try:
        db.add(row)
        await db.commit()
    except Exception:
        await db.rollback()
