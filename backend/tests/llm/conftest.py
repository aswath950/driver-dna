"""Reuse the API client fixture so we get the NullPool override + seed.

Adds a ``mock_openai`` autouse fixture that patches the OpenAI SDK with
deterministic stub responses — no real API key is ever consulted.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass

import pytest

# Re-export the API fixtures so pytest can discover them in this package.
from tests.api.conftest import _ensure_seeded, client  # noqa: F401

# ---------------------------------------------------------------------------
# OpenAI mock — replaces ``openai.OpenAI`` for the duration of each test.
# ---------------------------------------------------------------------------


@dataclass
class _Usage:
    prompt_tokens: int = 42
    completion_tokens: int = 84


@dataclass
class _Choice:
    message: object


@dataclass
class _Resp:
    choices: list[_Choice]
    usage: _Usage


class _ToolCallFn:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, id: str, name: str, arguments: str) -> None:
        self.id = id
        self.type = "function"
        self.function = _ToolCallFn(name, arguments)


class _Msg:
    def __init__(self, content: str | None, tool_calls: list | None = None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class _FakeChat:
    def __init__(self, queue: list) -> None:
        self._queue = queue

    def create(self, **kwargs):  # noqa: ANN001 ANN201
        # Pop the next pre-canned response off the queue.
        if not self._queue:
            raise RuntimeError("mock OpenAI: no more queued responses")
        item = self._queue.pop(0)
        if kwargs.get("stream"):
            # item must be a list of delta strings
            return _stream_iter(item)
        msg = _Msg(content=item.get("content"), tool_calls=item.get("tool_calls", []))
        return _Resp(choices=[_Choice(message=msg)], usage=_Usage())


def _stream_iter(deltas: list[str]):
    """Yield OpenAI-shaped streaming chunks."""
    from types import SimpleNamespace
    for d in deltas:
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content=d))]
        )


class _FakeCompletions:
    def __init__(self, queue: list) -> None:
        self.completions = _FakeChat(queue)


class _FakeOpenAI:
    def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002 ANN003
        # The queue is set on the class itself by the fixture below.
        self.chat = _FakeCompletions(_FakeOpenAI._queue)  # type: ignore[attr-defined]

    _queue: list = []  # filled per-test


@pytest.fixture()
def openai_queue(monkeypatch: pytest.MonkeyPatch) -> Iterator[list]:
    """Per-test queue of stub responses. Tests append to it before calling
    the endpoint. Each tool-round consumes ONE item.

    Items shape:
        {"content": str, "tool_calls": [_ToolCall, ...]}   for non-stream
        [str, str, ...]                                     for stream=True
    """
    queue: list = []
    _FakeOpenAI._queue = queue  # type: ignore[attr-defined]

    import openai
    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)

    # Also patch the symbol re-imported inside the LLM modules.

    # The modules import OpenAI lazily inside functions, so the monkeypatch
    # on ``openai.OpenAI`` is sufficient.
    # Ensure a non-empty API key so the gate passes.
    from app.core.config import settings
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "sk-test-dummy")

    yield queue
    _FakeOpenAI._queue = []  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Helpers for tests to enqueue stubs
# ---------------------------------------------------------------------------


def enqueue_text(queue: list, text: str) -> None:
    queue.append({"content": text, "tool_calls": []})


def enqueue_json(queue: list, payload: dict) -> None:
    queue.append({"content": json.dumps(payload), "tool_calls": []})


def enqueue_tool_call(queue: list, name: str, args: dict, call_id: str = "call_1") -> None:
    queue.append({
        "content": None,
        "tool_calls": [_ToolCall(call_id, name, json.dumps(args))],
    })


def enqueue_stream(queue: list, deltas: list[str]) -> None:
    queue.append(deltas)
