"""Re-resolve a session's current OpenF1 ``session_key`` and update it in place.

OpenF1 periodically renumbers its sessions, leaving ``sessions.openf1_session_key``
pointing at a key that now 404s on every endpoint (see ``fetch_telemetry``'s
stale-key diagnostic). This command finds the *current* key for the same
real-world session — by ``(season.year, event.name, session.type)``, exactly how
``hydrate`` first resolved it — and writes it back onto the existing row.

Why an in-place UPDATE rather than re-hydrate: ``_upsert_session`` conflicts on
``openf1_session_key``, so re-hydrating a renumbered weekend INSERTs a *new*
session row and orphans the stale one. Updating the key on the existing row keeps
the session id (and all its ``car_telemetry`` / ``lap_times`` FKs) intact.

CLI usage::

    python -m app.etl refresh-session-key --session-id 46
    python -m app.etl refresh-session-key --all     # scan & fix every stale key
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import create_engine, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import Event, Season
from app.db.models import Session as SessionRow
from app.db.models import SessionType
from app.etl.hydrate_session import _to_session_type
from src.openf1 import OpenF1Client

logger = logging.getLogger(__name__)


@dataclass
class RefreshResult:
    session_id: int
    status: str = "error"          # updated | ok | unchanged | unresolved | conflict | not_found | error
    old_key: int | None = None
    new_key: int | None = None
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "status": self.status,
            "old_key": self.old_key,
            "new_key": self.new_key,
            "message": self.message,
        }


# Statuses that did NOT leave the row in a usable state — drive the CLI exit code.
_FAILURE_STATUSES = {"unresolved", "conflict", "not_found", "error"}


def _resolve_current_key(
    client: OpenF1Client, *, year: int, grand_prix: str, wanted: SessionType
) -> int | None:
    """Return the live OpenF1 session_key for (year, grand_prix, session type)."""
    df = client.get_sessions(year, grand_prix)
    if df.empty:
        return None
    matches: list[int] = []
    for _, row in df.iterrows():
        kind = _to_session_type(row.get("session_name")) or _to_session_type(
            row.get("session_type")
        )
        if kind == wanted and row.get("session_key") is not None:
            matches.append(int(row["session_key"]))
    if not matches:
        return None
    if len(matches) > 1:
        logger.warning(
            "%s %s: %d sessions matched type %s — using the first (%s)",
            grand_prix, year, len(matches), wanted.value, matches[0],
        )
    return matches[0]


def _refresh_one(db: Session, session_id: int, client: OpenF1Client) -> RefreshResult:
    res = RefreshResult(session_id=session_id)

    srow = db.get(SessionRow, session_id)
    if srow is None:
        res.status = "not_found"
        res.message = f"session {session_id} not found in DB"
        return res
    res.old_key = int(srow.openf1_session_key) if srow.openf1_session_key is not None else None

    # If the stored key still resolves, there's nothing to fix.
    if res.old_key is not None and client.session_exists(res.old_key):
        res.status = "ok"
        res.new_key = res.old_key
        res.message = "current openf1_session_key still valid — no change"
        return res

    event = db.get(Event, srow.event_id)
    season = db.get(Season, event.season_id) if event is not None else None
    if event is None or season is None:
        res.message = f"session {session_id} missing event/season — cannot re-resolve"
        return res

    new_key = _resolve_current_key(
        client, year=season.year, grand_prix=event.name, wanted=srow.type
    )
    if new_key is None:
        res.status = "unresolved"
        res.message = (
            f"no {srow.type.value} session found on OpenF1 for "
            f"{event.name!r} {season.year} (check event name / circuit alias)"
        )
        return res

    if res.old_key is not None and new_key == res.old_key:
        res.status = "unchanged"
        res.new_key = new_key
        res.message = "re-resolved to the same key — OpenF1 data genuinely absent"
        return res

    # Update in place, isolating a unique-constraint clash to a SAVEPOINT so a
    # conflict on one session doesn't roll back others in an --all run.
    try:
        with db.begin_nested():
            db.execute(
                text("UPDATE sessions SET openf1_session_key = :k WHERE id = :sid"),
                {"k": new_key, "sid": session_id},
            )
    except IntegrityError:
        res.status = "conflict"
        res.message = (
            f"new key {new_key} already belongs to another session row (likely a "
            "duplicate created by a prior re-hydrate) — resolve manually"
        )
        return res

    res.status = "updated"
    res.new_key = new_key
    res.message = f"openf1_session_key {res.old_key} → {new_key}"
    logger.info("session %d: %s", session_id, res.message)
    return res


def run(
    *, session_id: int | None = None, all_sessions: bool = False
) -> list[RefreshResult]:
    """Refresh one session (``session_id``) or scan every session (``all_sessions``).

    Commits once at the end so all in-place updates land atomically.
    """
    if not all_sessions and session_id is None:
        raise ValueError("pass either session_id or all_sessions=True")

    engine = create_engine(settings.DATABASE_URL_SYNC, future=True)
    with Session(engine) as db:
        client = OpenF1Client(mode="historical")
        if all_sessions:
            ids = [
                int(r) for r in db.execute(
                    select(SessionRow.id).order_by(SessionRow.id)
                ).scalars()
            ]
        else:
            ids = [int(session_id)]  # type: ignore[arg-type]

        results = [_refresh_one(db, sid, client) for sid in ids]
        try:
            db.commit()
        except Exception:
            db.rollback()
            raise
    return results
