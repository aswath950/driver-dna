"""Recompute the ``driver_stats`` aggregate table from ``race_results``.

Pure SQL — one ``INSERT ... ON CONFLICT DO UPDATE``. Targets a single
season by default; pass ``season_year=None`` to refresh all known seasons.

CLI:

    python -m app.etl refresh-stats --season 2024
    python -m app.etl refresh-stats --all-seasons
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.core.config import settings

logger = logging.getLogger(__name__)


# CTE counts wins/podiums/poles/dnfs/points over the Race + Sprint sessions
# in the given season, then upserts into driver_stats. Sprint (`'S'`) is
# included so the points totals match real championship tallies.
_REFRESH_SQL = text(
    """
    WITH agg AS (
        SELECT
            rr.driver_id,
            s_evt.season_id,
            COUNT(*) FILTER (WHERE rr.position = 1)                     AS wins,
            COUNT(*) FILTER (WHERE rr.position BETWEEN 1 AND 3)         AS podiums,
            COUNT(*) FILTER (WHERE rr.grid = 1 AND s.type = 'Q')        AS poles,
            COUNT(*) FILTER (
                WHERE rr.status IS NOT NULL AND rr.status NOT IN ('Finished', 'finished')
            )                                                            AS dnfs,
            COALESCE(SUM(rr.points), 0)::numeric(7,2)                   AS points,
            CASE WHEN COUNT(rr.position) > 0
                 THEN AVG(rr.position)::numeric(4,2)
                 ELSE NULL
            END                                                          AS avg_finish
        FROM race_results rr
        JOIN sessions s        ON s.id = rr.session_id
        JOIN events s_evt      ON s_evt.id = s.event_id
        JOIN seasons sn        ON sn.id = s_evt.season_id
        WHERE (:year IS NULL OR sn.year = :year)
        GROUP BY rr.driver_id, s_evt.season_id
    )
    INSERT INTO driver_stats (
        driver_id, season_id, wins, podiums, poles, dnfs, points, avg_finish, updated_at
    )
    SELECT driver_id, season_id, wins, podiums, poles, dnfs, points, avg_finish, now()
    FROM agg
    ON CONFLICT (driver_id, season_id) DO UPDATE SET
        wins       = EXCLUDED.wins,
        podiums    = EXCLUDED.podiums,
        poles      = EXCLUDED.poles,
        dnfs       = EXCLUDED.dnfs,
        points     = EXCLUDED.points,
        avg_finish = EXCLUDED.avg_finish,
        updated_at = now()
    """
)


@dataclass
class StatsRefreshResult:
    season_year: int | None
    rows_affected: int

    def as_dict(self) -> dict[str, Any]:
        return {"season_year": self.season_year, "rows_affected": self.rows_affected}


def run(*, season_year: int | None = None) -> StatsRefreshResult:
    engine = create_engine(settings.DATABASE_URL_SYNC, future=True)
    with Session(engine) as db:
        try:
            cursor = db.execute(_REFRESH_SQL, {"year": season_year})
            n = cursor.rowcount or 0
            db.commit()
        except Exception:
            db.rollback()
            raise
    logger.info("etl.refresh_stats.done year=%s rows=%d", season_year, n)
    return StatsRefreshResult(season_year=season_year, rows_affected=n)
