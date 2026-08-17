"""Unit tests for compare_service session-key resolution and laps fetching.

``_resolve_session_key`` must return the stored key WITHOUT a live OpenF1 call,
so a fully-cached compare never depends on upstream availability. The stale-key
diagnosis is deferred to ``_fetch_laps_or_raise`` (the cache-miss path), which
must distinguish a genuinely-stale key from a transient upstream failure rather
than blaming the key for both.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from app.core.errors import NotFoundError, UpstreamError
from app.services import compare_service


def _fake_db(session_row: object | None) -> MagicMock:
    db = MagicMock()
    db.get = AsyncMock(return_value=session_row)
    return db


# ---------------------------------------------------------------------------
# _resolve_session_key — no live OpenF1 call
# ---------------------------------------------------------------------------


async def test_resolve_returns_key_without_openf1_call() -> None:
    db = _fake_db(SimpleNamespace(openf1_session_key=9472))
    with patch.object(compare_service, "OpenF1Client") as client_cls:
        key = await compare_service._resolve_session_key(db, session_id=46)
    assert key == 9472
    # Resolution must never touch OpenF1 — cached requests depend on this.
    client_cls.assert_not_called()


async def test_resolve_missing_session_raises_not_found() -> None:
    db = _fake_db(None)
    with pytest.raises(NotFoundError):
        await compare_service._resolve_session_key(db, session_id=46)


async def test_resolve_missing_key_raises_upstream_error() -> None:
    db = _fake_db(SimpleNamespace(openf1_session_key=None))
    with patch.object(compare_service, "OpenF1Client") as client_cls:
        with pytest.raises(UpstreamError, match="no openf1_session_key"):
            await compare_service._resolve_session_key(db, session_id=46)
        client_cls.assert_not_called()


# ---------------------------------------------------------------------------
# _fetch_laps_or_raise — stale vs transient disambiguation on a cache miss
# ---------------------------------------------------------------------------


def _client_with(laps: pd.DataFrame, status: str = "exists") -> MagicMock:
    client = MagicMock()
    client.get_laps.return_value = laps
    client.session_status.return_value = status
    return client


def test_laps_returns_df_without_status_check() -> None:
    laps = pd.DataFrame({"driver_number": [1], "lap_number": [1]})
    client = _client_with(laps)
    out = compare_service._fetch_laps_or_raise(
        client, session_key=960, session_id=46
    )
    assert out is laps
    # A valid laps response must not trigger a redundant /sessions call.
    client.session_status.assert_not_called()


def test_empty_laps_stale_key_raises_stale_error() -> None:
    client = _client_with(pd.DataFrame(), status="not_found")
    with pytest.raises(UpstreamError, match="stale key"):
        compare_service._fetch_laps_or_raise(client, session_key=960, session_id=46)


def test_empty_laps_transient_failure_is_not_blamed_on_key() -> None:
    client = _client_with(pd.DataFrame(), status="unknown")
    with pytest.raises(UpstreamError, match="transient") as exc:
        compare_service._fetch_laps_or_raise(client, session_key=960, session_id=46)
    # Crucially, a transient failure must NOT be reported as a stale key.
    assert "stale key" not in str(exc.value)


def test_empty_laps_valid_key_reports_no_laps() -> None:
    client = _client_with(pd.DataFrame(), status="exists")
    with pytest.raises(UpstreamError, match="no laps"):
        compare_service._fetch_laps_or_raise(client, session_key=960, session_id=46)
