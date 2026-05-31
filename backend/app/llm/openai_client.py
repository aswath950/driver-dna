"""Thin async wrapper around the OpenAI chat-completions API.

Every call returns ``(text, error, usage_dict)``. ``usage_dict`` has
``input_tokens``, ``output_tokens``, ``latency_ms`` so the caller can
audit it regardless of success/failure path.

We use ``asyncio.to_thread`` to bridge the synchronous OpenAI SDK rather
than the async client — this keeps the dep surface small and matches
how src/llm_layer.py uses the SDK.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from app.core.config import settings

MODEL = "gpt-4o-mini"


def _sync_call(
    *,
    system_msg: str,
    user_msg: str,
    max_tokens: int,
    temperature: float,
    json_mode: bool,
) -> tuple[str | None, str | None, dict[str, int]]:
    """Synchronous OpenAI call. Returns (text, error, usage)."""
    t0 = time.perf_counter()
    try:
        from openai import (
            APIError,
            AuthenticationError,
            OpenAI,
            RateLimitError,
        )
    except ImportError:
        return None, "openai SDK not installed", {
            "input_tokens": 0, "output_tokens": 0, "latency_ms": 0,
        }

    api_key = settings.OPENAI_API_KEY
    if not api_key:
        return None, "OPENAI_API_KEY is not configured", {
            "input_tokens": 0, "output_tokens": 0, "latency_ms": 0,
        }

    try:
        client = OpenAI(api_key=api_key)
        kwargs: dict[str, Any] = dict(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        text = (resp.choices[0].message.content or "").strip() or None
        usage = resp.usage
        return text, (None if text else "empty response"), {
            "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }
    except AuthenticationError:
        return None, "invalid API key", _err_usage(t0)
    except RateLimitError:
        return None, "rate limit exceeded", _err_usage(t0)
    except APIError as e:
        return None, f"OpenAI APIError: {type(e).__name__}", _err_usage(t0)
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}", _err_usage(t0)


def _err_usage(t0: float) -> dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "latency_ms": int((time.perf_counter() - t0) * 1000),
    }


async def chat_completion(
    *,
    system_msg: str,
    user_msg: str,
    max_tokens: int = 500,
    temperature: float = 0.4,
    json_mode: bool = False,
) -> tuple[str | None, str | None, dict[str, int]]:
    """Async-friendly wrapper around the sync OpenAI call."""
    return await asyncio.to_thread(
        _sync_call,
        system_msg=system_msg,
        user_msg=user_msg,
        max_tokens=max_tokens,
        temperature=temperature,
        json_mode=json_mode,
    )
