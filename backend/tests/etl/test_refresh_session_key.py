"""Tests for the refresh-session-key ETL: re-resolve a stale OpenF1 key in place.

OpenF1 HTTP is mocked with ``responses``. Because both the staleness probe and
the meeting→session lookup hit ``/v1/sessions``, we disambiguate registrations
with query-param matchers. Runs against the real Postgres DB via ``clean_db``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import responses
import sqlalchemy as sa
from responses import matchers
from sqlalchemy.orm import Session

from app.db.models import Circuit, Event, Season, SessionType
from app.db.models import Session as SessionRow
from app.etl import refresh_session_key

OPENF1_BASE = "https://api.openf1.org/v1"

_GP = "Synthetic Grand Prix"
_YEAR = 2099


def _seed_session(engine: sa.Engine, *, session_key: int) -> int:
    with Session(engine) as db:
        season = Season(year=_YEAR)
        circuit = Circuit(name="Synthetic Circuit")
        db.add_all([season, circuit])
        db.flush()
        event = Event(
            season_id=season.id, circuit_id=circuit.id, round=1, name=_GP,
        )
        db.add(event)
        db.flush()
        srow = SessionRow(
            event_id=event.id, type=SessionType.R,
            date_start=datetime(_YEAR, 4, 1, 15, 0, tzinfo=timezone.utc),
            openf1_session_key=session_key,
        )
        db.add(srow)
        db.flush()
        sid = srow.id
        db.commit()
    return sid


def _key_in_db(engine: sa.Engine, session_id: int) -> int | None:
    with Session(engine) as db:
        return db.get(SessionRow, session_id).openf1_session_key


def _mock_stale_probe(session_key: int) -> None:
    """/sessions?session_key=K returns empty → session_exists() is False."""
    responses.add(
        responses.GET, f"{OPENF1_BASE}/sessions", json=[], status=200,
        match=[matchers.query_param_matcher({"session_key": str(session_key)})],
    )


def _mock_resolution(new_key: int) -> None:
    """get_sessions(year, gp): meeting lookup then sessions-by-meeting."""
    responses.add(
        responses.GET, f"{OPENF1_BASE}/meetings",
        json=[{"meeting_key": 1234, "meeting_name": _GP}],
        match=[matchers.query_param_matcher({"year": str(_YEAR), "meeting_name": _GP})],
    )
    responses.add(
        responses.GET, f"{OPENF1_BASE}/sessions",
        json=[{
            "session_key": new_key, "session_name": "Race", "session_type": "Race",
            "date_start": f"{_YEAR}-04-01T15:00:00+00:00",
            "date_end": f"{_YEAR}-04-01T17:00:00+00:00",
        }],
        match=[matchers.query_param_matcher({"meeting_key": "1234"})],
    )


@responses.activate
def test_stale_key_updated_in_place(clean_db: sa.Engine) -> None:
    sid = _seed_session(clean_db, session_key=960)
    _mock_stale_probe(960)        # 960 no longer resolves
    _mock_resolution(9472)        # current key for the same race

    (res,) = refresh_session_key.run(session_id=sid)

    assert res.status == "updated"
    assert res.old_key == 960
    assert res.new_key == 9472
    assert _key_in_db(clean_db, sid) == 9472   # persisted, same row


@responses.activate
def test_valid_key_left_unchanged(clean_db: sa.Engine) -> None:
    sid = _seed_session(clean_db, session_key=9472)
    # The stored key still resolves → no re-resolution, no meetings call.
    responses.add(
        responses.GET, f"{OPENF1_BASE}/sessions",
        json=[{"session_key": 9472, "session_name": "Race"}], status=200,
        match=[matchers.query_param_matcher({"session_key": "9472"})],
    )

    (res,) = refresh_session_key.run(session_id=sid)

    assert res.status == "ok"
    assert res.new_key == 9472
    assert _key_in_db(clean_db, sid) == 9472


@responses.activate
def test_unresolvable_session_reported(clean_db: sa.Engine) -> None:
    sid = _seed_session(clean_db, session_key=960)
    _mock_stale_probe(960)
    # Meeting not found on OpenF1 → get_sessions returns empty → unresolved.
    responses.add(
        responses.GET, f"{OPENF1_BASE}/meetings", json=[], status=200,
        match=[matchers.query_param_matcher({"year": str(_YEAR), "meeting_name": _GP})],
    )

    (res,) = refresh_session_key.run(session_id=sid)

    assert res.status == "unresolved"
    assert _key_in_db(clean_db, sid) == 960    # untouched
    assert res.status in refresh_session_key._FAILURE_STATUSES
