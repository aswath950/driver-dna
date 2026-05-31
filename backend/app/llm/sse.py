"""Server-Sent Events helpers for the race-chat streaming endpoint."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator


def sse_event(event: str, data: dict | str) -> str:
    """Format a single SSE event per the W3C spec.

    Produces ``event: <name>\\ndata: <json>\\n\\n``.
    """
    payload = data if isinstance(data, str) else json.dumps(data, separators=(",", ":"))
    # SSE requires each data line to start with "data:"; payloads with newlines
    # must split. Our payloads are JSON so no embedded newlines — safe.
    return f"event: {event}\ndata: {payload}\n\n"


async def merge_streams(*streams: AsyncIterator[str]) -> AsyncIterator[str]:
    """Yield events from multiple async iterators sequentially."""
    for s in streams:
        async for chunk in s:
            yield chunk
