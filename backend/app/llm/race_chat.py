"""Race-chat ReAct loop with SSE streaming.

The user asks a free-form question about a race. We expose a small set of
tools backed by the existing analytics service from Phase 7. The OpenAI
model decides which tool(s) to call (up to ``MAX_ROUNDS * MAX_TOOLS_PER_ROUND``),
then synthesises a final answer. Every token of the final answer streams
out as SSE ``token`` events; intermediate tool decisions stream as
``tool_call`` / ``tool_result`` events.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.repositories import sessions as sessions_repo
from app.llm import openai_client
from app.llm.audit import record_llm_call
from app.llm.sse import sse_event
from app.services import analytics_service

MAX_ROUNDS = 3
MAX_TOOLS_PER_ROUND = 3


# ---------------------------------------------------------------------------
# Tool catalogue exposed to the model
# ---------------------------------------------------------------------------


def _tool_specs() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_rolling_pace_top",
                "description": "Return the top-N fastest drivers by 5-lap rolling pace average.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "top_n": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_leader_gap_summary",
                "description": "Return median gap-to-leader per driver across the race.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_tyre_degradation_summary",
                "description": "Return mean degradation (sec/lap) per compound across all drivers.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


async def _build_analyser_and_map(db: AsyncSession, session_id: int):
    """Helper: build the per-request analyser + driver map once per round."""
    drv_map = await analytics_service._load_car_number_map(db, session_id)
    analyser = await analytics_service.build_analyser(db, session_id)
    return analyser, drv_map


async def _tool_rolling_pace(db: AsyncSession, session_id: int, top_n: int = 5) -> str:
    analyser, drv_map = await _build_analyser_and_map(db, session_id)
    rows = analytics_service.rolling_pace_rows(analyser, window=5, drv_map=drv_map)
    by_driver: dict[int, list[float]] = {}
    for r in rows:
        by_driver.setdefault(r["driver_id"], []).append(r["rolling_sec"])
    medians = [(d, sorted(v)[len(v) // 2]) for d, v in by_driver.items()]
    medians.sort(key=lambda x: x[1])
    return json.dumps([
        {"driver_id": d, "median_rolling_sec": round(p, 3)}
        for d, p in medians[: max(1, min(top_n, 10))]
    ])


async def _tool_leader_gap(db: AsyncSession, session_id: int) -> str:
    analyser, drv_map = await _build_analyser_and_map(db, session_id)
    rows = analytics_service.gap_to_leader_rows(analyser, drv_map=drv_map)
    by_driver: dict[int, list[float]] = {}
    for r in rows:
        by_driver.setdefault(r["driver_id"], []).append(r["gap_sec"])
    medians = [
        {"driver_id": d, "median_gap_sec": round(sorted(v)[len(v) // 2], 3)}
        for d, v in by_driver.items()
    ]
    medians.sort(key=lambda x: x["median_gap_sec"])
    return json.dumps(medians)


async def _tool_tyre_deg(db: AsyncSession, session_id: int) -> str:
    analyser, drv_map = await _build_analyser_and_map(db, session_id)
    rows = analytics_service.degradation_rows(analyser, drv_map=drv_map)
    by_compound: dict[str, list[float]] = {}
    for r in rows:
        by_compound.setdefault(r["compound"], []).append(r["deg_sec_per_lap"])
    summary = [
        {"compound": c, "mean_deg_sec_per_lap": round(sum(v) / len(v), 4), "n": len(v)}
        for c, v in by_compound.items()
    ]
    summary.sort(key=lambda x: x["mean_deg_sec_per_lap"])
    return json.dumps(summary)


async def _dispatch(
    db: AsyncSession, session_id: int, name: str, args: dict
) -> str:
    if name == "get_rolling_pace_top":
        return await _tool_rolling_pace(db, session_id, int(args.get("top_n", 5)))
    if name == "get_leader_gap_summary":
        return await _tool_leader_gap(db, session_id)
    if name == "get_tyre_degradation_summary":
        return await _tool_tyre_deg(db, session_id)
    return json.dumps({"error": f"unknown tool: {name}"})


# ---------------------------------------------------------------------------
# Synchronous OpenAI calls (tools + streaming) wrapped via to_thread
# ---------------------------------------------------------------------------


def _sync_tool_round(messages: list[dict]) -> tuple[dict, dict[str, int]]:
    """One tool-calling round. Returns (assistant_message_dict, usage)."""
    from openai import OpenAI

    t0 = time.perf_counter()
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=openai_client.MODEL,
        messages=messages,
        tools=_tool_specs(),
        tool_choice="auto",
        max_tokens=400,
        temperature=0.2,
    )
    msg = resp.choices[0].message
    usage = resp.usage
    return (
        {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in (msg.tool_calls or [])
            ],
        },
        {
            "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        },
    )


def _sync_stream_final(messages: list[dict]):
    """Yield content chunks from a streaming final-answer call."""
    from openai import OpenAI

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    return client.chat.completions.create(
        model=openai_client.MODEL,
        messages=messages,
        max_tokens=600,
        temperature=0.3,
        stream=True,
    )


# ---------------------------------------------------------------------------
# Public: SSE generator
# ---------------------------------------------------------------------------


_SYS = (
    "You are an F1 race-analytics assistant. Use the provided tools to fetch "
    "race data BEFORE asserting facts. Be concise — 3–6 sentences. Never "
    "invent driver IDs or numbers."
)


async def race_chat_stream(
    db: AsyncSession,
    *,
    session_id: int,
    user_message: str,
    request_id: str | None,
    user_session_id: uuid.UUID | None = None,
) -> AsyncIterator[str]:
    """Async generator producing SSE-encoded events for the race-chat endpoint."""
    if await sessions_repo.get_session(db, session_id) is None:
        yield sse_event("error", {"type": "not_found", "detail": f"session {session_id} not found"})
        return

    messages: list[dict] = [
        {"role": "system", "content": _SYS},
        {"role": "user", "content": f"[session_id={session_id}] {user_message}"},
    ]
    total_usage = {"input_tokens": 0, "output_tokens": 0}

    for round_idx in range(MAX_ROUNDS):
        assistant_msg, usage = await asyncio.to_thread(_sync_tool_round, messages)
        total_usage["input_tokens"] += usage["input_tokens"]
        total_usage["output_tokens"] += usage["output_tokens"]
        await record_llm_call(
            db, feature=f"race_chat.round_{round_idx + 1}",
            model=openai_client.MODEL,
            input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"],
            latency_ms=usage["latency_ms"],
            status="success", request_id=request_id, user_session_id=user_session_id,
        )

        tool_calls = assistant_msg.get("tool_calls") or []
        if not tool_calls:
            # Model decided to answer directly — send its content as one token.
            if assistant_msg.get("content"):
                yield sse_event("token", {"delta": assistant_msg["content"]})
            yield sse_event("done", {
                "input_tokens": total_usage["input_tokens"],
                "output_tokens": total_usage["output_tokens"],
                "rounds": round_idx + 1,
            })
            return

        # Otherwise, append assistant turn + run each tool call.
        messages.append(assistant_msg)
        for tc in tool_calls[:MAX_TOOLS_PER_ROUND]:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            yield sse_event("tool_call", {"tool": name, "args": args})
            result = await _dispatch(db, session_id, name, args)
            yield sse_event("tool_result", {"tool": name, "summary": result[:400]})
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": name,
                "content": result,
            })

    # If we hit the loop cap without a content-only round, force synthesis.
    messages.append({
        "role": "user",
        "content": "Based on the tool results above, write the final answer now (no more tools).",
    })
    stream = await asyncio.to_thread(_sync_stream_final, messages)
    final_chunks: list[str] = []
    for chunk in stream:
        delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
        if delta:
            final_chunks.append(delta)
            yield sse_event("token", {"delta": delta})
    await record_llm_call(
        db, feature="race_chat.final", model=openai_client.MODEL,
        input_tokens=0, output_tokens=sum(len(c) for c in final_chunks) // 4,
        latency_ms=0, status="success", request_id=request_id, user_session_id=user_session_id,
    )
    yield sse_event("done", {
        "input_tokens": total_usage["input_tokens"],
        "output_tokens": total_usage["output_tokens"],
        "rounds": MAX_ROUNDS,
    })
