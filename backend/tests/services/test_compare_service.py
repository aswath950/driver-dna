"""Unit tests for compare_service session-key resolution.

``_resolve_session_key`` must fail fast with a clear ``UpstreamError`` when the
stored ``openf1_session_key`` is stale (OpenF1 no longer recognizes it), rather
than letting the live compare path return confusing empty traces / a 503.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.errors import UpstreamError
from app.services import compare_service


def _fake_db(session_row: object | None) -> MagicMock:
    db = MagicMock()
    db.get = AsyncMock(return_value=session_row)
    return db


async def test_stale_key_raises_upstream_error() -> None:
    db = _fake_db(SimpleNamespace(openf1_session_key=960))
    with patch.object(compare_service, "OpenF1Client") as client_cls:
        client_cls.return_value.session_exists.return_value = False
        with pytest.raises(UpstreamError, match="stale key"):
            await compare_service._resolve_session_key(db, session_id=46)


async def test_valid_key_returns_key() -> None:
    db = _fake_db(SimpleNamespace(openf1_session_key=9472))
    with patch.object(compare_service, "OpenF1Client") as client_cls:
        client_cls.return_value.session_exists.return_value = True
        key = await compare_service._resolve_session_key(db, session_id=46)
    assert key == 9472


async def test_missing_key_raises_before_openf1_call() -> None:
    db = _fake_db(SimpleNamespace(openf1_session_key=None))
    with patch.object(compare_service, "OpenF1Client") as client_cls:
        with pytest.raises(UpstreamError, match="no openf1_session_key"):
            await compare_service._resolve_session_key(db, session_id=46)
        client_cls.assert_not_called()
