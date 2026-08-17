"""
openf1.py — OpenF1 API client for historical and live F1 data.

Provides a unified interface to the OpenF1 REST API with two modes:
  - historical: query past sessions by year/grand_prix/session_key
  - live: poll the latest session for real-time lap, stint, and position data

All public methods return clean pandas DataFrames.

API docs: https://openf1.org
"""

from __future__ import annotations

import logging
import time
from typing import Literal

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openf1.org/v1"

# Tri-state result of a session-key liveness check (see OpenF1Client.session_status).
SessionStatus = Literal["exists", "not_found", "unknown"]

# Columns to retain and their expected dtypes for each endpoint.
# Any extra columns returned by the API are kept but not type-cast.
_LAP_COLS = [
    "driver_number", "lap_number", "lap_duration",
    "is_pit_out_lap", "st_speed", "session_key", "date_start",
]
_STINT_COLS = [
    "driver_number", "stint_number", "compound",
    "tyre_age_at_start", "lap_start", "lap_end", "session_key",
]
_POSITION_COLS = [
    "driver_number", "position", "date", "session_key", "x", "y",
]
_LOCATION_COLS = ["driver_number", "date", "x", "y", "z", "session_key"]
_CAR_DATA_COLS = [
    "driver_number", "date", "speed", "throttle", "brake",
    "n_gear", "rpm", "drs", "session_key",
]


class OpenF1AuthError(Exception):
    """Raised when the OpenF1 API returns 401 Unauthorized."""


class OpenF1UnavailableError(Exception):
    """Raised by a strict client when a request fails after every retry.

    Distinct from an empty result: this means we never got an answer (429, 5xx,
    timeout), so the caller must NOT read the empty DataFrame as "no such data".
    Only strict clients raise it — see :class:`OpenF1Client`.
    """


def validate_dataframe(df: pd.DataFrame, required_cols: list[str], context: str = "") -> pd.DataFrame:
    """Check that *df* contains all *required_cols*.

    Missing columns are added as NaN and a warning is logged so the
    downstream consumer never hits a silent KeyError.
    """
    if df.empty:
        return df
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        label = f" ({context})" if context else ""
        logger.warning("OpenF1%s: missing columns %s — filled with NaN", label, missing)
        for col in missing:
            df[col] = pd.NA
    return df


class OpenF1Client:
    """
    Client for the OpenF1 REST API.

    Parameters
    ----------
    mode : 'historical' | 'live'
        In *historical* mode every query requires an explicit session_key
        (or year + grand_prix to discover one).
        In *live* mode the convenience methods ``get_live_*`` automatically
        target ``session_key=latest`` and track which laps have already been
        seen so that only new data is returned on each poll.
    timeout : int
        HTTP request timeout in seconds (default 10).
    strict : bool
        When False (default) a request that fails every retry returns an empty
        DataFrame, so interactive callers degrade gracefully rather than raising
        in the middle of a page render.

        When True those exhausted retries raise :class:`OpenF1UnavailableError`
        instead. Batch/ETL callers want this: an empty DataFrame is
        indistinguishable from "this race has no data", so a rate-limited run
        would otherwise write 0 rows and report success.
    """

    def __init__(
        self,
        mode: Literal["historical", "live"] = "historical",
        timeout: int = 10,
        strict: bool = False,
    ) -> None:
        if mode not in ("historical", "live"):
            raise ValueError(f"mode must be 'historical' or 'live', got {mode!r}")
        self.mode = mode
        self.timeout = timeout
        self.strict = strict
        self._session = requests.Session()

        # Live-mode watermarks — track the last-seen timestamps so
        # successive polls only return genuinely new rows.
        self._last_lap_ts: str | None = None
        self._last_stint_ts: str | None = None
        self._last_position_ts: str | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(
        self,
        endpoint: str,
        params: dict | None = None,
        retries: int = 3,
        backoff: float = 2.0,
    ) -> pd.DataFrame:
        """Fire a GET request against the OpenF1 API and return a DataFrame.

        Retries up to *retries* times with exponential backoff.

        An empty DataFrame means the API answered successfully with no rows —
        a definitive "no such data". What happens when the request never
        succeeds depends on the client's ``strict`` flag: a lenient client
        returns an empty DataFrame too (callers never need to catch
        transport-level exceptions), while a strict client raises
        :class:`OpenF1UnavailableError` so the two cases stay distinguishable.
        """
        url = f"{BASE_URL}/{endpoint}"
        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                resp = self._session.get(
                    url, params=params or {}, timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    return pd.DataFrame()
                return pd.DataFrame(data)
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 401:
                    raise OpenF1AuthError(
                        "OpenF1 API returned 401 Unauthorized. "
                        "The API may now require authentication."
                    ) from exc
                # 404 is a definitive "no such resource" — OpenF1 answers this
                # way for an unknown meeting_name rather than 200 with an empty
                # body. Short-circuit: retrying cannot change it, and a strict
                # caller must see genuine absence, not an outage. Mirrors the
                # 404 handling in session_status.
                if exc.response is not None and exc.response.status_code == 404:
                    return pd.DataFrame()
                last_exc = exc
                if attempt < retries:
                    delay = backoff ** attempt
                    logger.warning(
                        "OpenF1 %s attempt %d/%d failed (%s), retrying in %.1fs",
                        endpoint, attempt, retries, exc, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.warning(
                        "OpenF1 %s failed after %d attempts: %s",
                        endpoint, retries, exc,
                    )
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                if attempt < retries:
                    delay = backoff ** attempt
                    logger.warning(
                        "OpenF1 %s attempt %d/%d failed (%s), retrying in %.1fs",
                        endpoint, attempt, retries, exc, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.warning(
                        "OpenF1 %s failed after %d attempts: %s",
                        endpoint, retries, exc,
                    )
        # Every attempt failed. A strict caller must not mistake this for
        # "no data" — see the class docstring.
        if self.strict:
            raise OpenF1UnavailableError(
                f"OpenF1 /{endpoint} failed after {retries} attempts: {last_exc}"
            ) from last_exc
        return pd.DataFrame()

    @staticmethod
    def _clean(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
        """Keep all columns but ensure *key_cols* exist (fill with NaN if missing)."""
        for col in key_cols:
            if col not in df.columns:
                df[col] = pd.NA
        return df

    # ------------------------------------------------------------------
    # Historical mode
    # ------------------------------------------------------------------

    def get_sessions(self, year: int, grand_prix: str) -> pd.DataFrame:
        """
        Return available sessions for a race weekend.

        The OpenF1 API separates meetings from sessions.  ``meeting_name``
        is not a valid filter on ``/v1/sessions``, so we first resolve the
        ``meeting_key`` via ``/v1/meetings``, then fetch sessions for it.

        Parameters
        ----------
        year : int
            Season year, e.g. 2024.
        grand_prix : str
            Meeting name as it appears on the OpenF1 API,
            e.g. ``'Italian Grand Prix'``.

        Returns
        -------
        DataFrame with columns including ``session_key``, ``session_name``,
        ``session_type``, ``date_start``, ``date_end``.
        """
        meetings = self._get("meetings", {
            "year": year,
            "meeting_name": grand_prix,
        })
        if meetings.empty or "meeting_key" not in meetings.columns:
            logger.warning("OpenF1: no meeting found for '%s' %s", grand_prix, year)
            return pd.DataFrame()

        meeting_key = int(meetings.iloc[0]["meeting_key"])
        df = self._get("sessions", {"meeting_key": meeting_key})
        if df.empty:
            return df
        if "date_start" in df.columns:
            df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")
        if "date_end" in df.columns:
            df["date_end"] = pd.to_datetime(df["date_end"], errors="coerce")
        return df

    def session_status(
        self, session_key: int, retries: int = 3, backoff: float = 2.0
    ) -> SessionStatus:
        """Tri-state check of whether OpenF1 recognizes this ``session_key``.

        Queries ``/v1/sessions?session_key=X`` directly (not via :meth:`_get`,
        which collapses every failure into an empty DataFrame and so cannot tell
        a genuinely-stale key from a call that merely failed). Distinguishes:

        - ``"exists"``    — OpenF1 returned the session; the key is valid.
        - ``"not_found"`` — OpenF1 answered definitively that no such session
          exists (HTTP 404, or a 200 with an empty result set). A genuine stale
          key — safe to tell the caller to re-resolve it.
        - ``"unknown"``   — the check could not be completed (timeout, connection
          error, or 429/5xx after retries). NOT proof the key is stale; the
          caller should treat this as a transient upstream failure, not blame the
          session key.

        Only transient failures are retried; a definitive 404 short-circuits
        immediately (no wasted backoff on a key we already know is gone).
        """
        url = f"{BASE_URL}/sessions"
        params = {"session_key": session_key}
        for attempt in range(1, retries + 1):
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 401:
                    raise OpenF1AuthError(
                        "OpenF1 API returned 401 Unauthorized. "
                        "The API may now require authentication."
                    )
                if resp.status_code == 404:
                    return "not_found"
                resp.raise_for_status()
                data = resp.json()
                # A successful, empty response is a definitive "no such session".
                return "exists" if data else "not_found"
            except OpenF1AuthError:
                raise
            except (requests.RequestException, ValueError) as exc:
                # Non-404/401 HTTP errors (429, 5xx) and transport/parse failures
                # are transient — retry, then fall through to "unknown".
                if attempt < retries:
                    delay = backoff ** attempt
                    logger.warning(
                        "OpenF1 session_status attempt %d/%d failed (%s), "
                        "retrying in %.1fs",
                        attempt, retries, exc, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.warning(
                        "OpenF1 session_status failed after %d attempts: %s",
                        retries, exc,
                    )
        return "unknown"

    def session_exists(self, session_key: int) -> bool:
        """Return True only if OpenF1 definitively still recognizes ``session_key``.

        Thin boolean wrapper over :meth:`session_status`. Both ``"not_found"``
        (genuine stale key) and ``"unknown"`` (could not confirm) map to
        ``False``, so callers that need to tell those two apart — e.g. to avoid
        misreporting a transient failure as a stale key — should call
        :meth:`session_status` directly.
        """
        return self.session_status(session_key) == "exists"

    def get_drivers(self, session_key: int) -> pd.DataFrame:
        """
        Return driver info for a session, including 3-letter acronyms.

        Returns
        -------
        DataFrame with at least: ``driver_number``, ``name_acronym``,
        ``full_name``, ``team_name``.
        """
        df = self._get("drivers", {"session_key": session_key})
        if df.empty:
            return df
        for col in ["driver_number", "name_acronym", "full_name", "team_name", "team_colour"]:
            if col not in df.columns:
                df[col] = pd.NA
        return df

    def get_live_drivers(self) -> pd.DataFrame:
        """Poll ``/v1/drivers?session_key=latest`` for the current session."""
        self._require_live()
        df = self._get("drivers", {"session_key": "latest"})
        if df.empty:
            return df
        for col in ["driver_number", "name_acronym", "full_name", "team_name", "team_colour"]:
            if col not in df.columns:
                df[col] = pd.NA
        return df

    def get_laps(self, session_key: int) -> pd.DataFrame:
        """
        Return all lap data for a given session.

        Returns
        -------
        DataFrame with at least: ``driver_number``, ``lap_number``,
        ``lap_duration``, ``is_pit_out_lap``, ``st_speed``, ``session_key``.
        """
        df = self._get("laps", {"session_key": session_key})
        df = self._clean(df, _LAP_COLS)
        return validate_dataframe(df, _LAP_COLS, "get_laps")

    def get_stints(self, session_key: int) -> pd.DataFrame:
        """
        Return tyre stint data per driver for a given session.

        Returns
        -------
        DataFrame with at least: ``driver_number``, ``stint_number``,
        ``compound``, ``tyre_age_at_start``, ``lap_start``, ``lap_end``,
        ``session_key``.
        """
        df = self._get("stints", {"session_key": session_key})
        df = self._clean(df, _STINT_COLS)
        return validate_dataframe(df, _STINT_COLS, "get_stints")

    def get_car_data(
        self,
        session_key: int | str,
        driver_number: int | None = None,
        date_gte: str | None = None,
        date_lte: str | None = None,
    ) -> pd.DataFrame:
        """
        Return high-frequency car telemetry for a specific driver and time window.

        The ``/v1/car_data`` endpoint does not support ``lap_number`` filtering.
        Always provide ``driver_number`` plus a ``date_gte``/``date_lte`` window
        (ISO-8601 strings) to keep the response small and avoid 422 errors.

        Returns
        -------
        DataFrame with at least: ``driver_number``, ``date``, ``speed``,
        ``throttle``, ``brake``, ``n_gear``, ``rpm``, ``session_key``.
        """
        params: dict = {"session_key": session_key}
        if driver_number is not None:
            params["driver_number"] = driver_number
        if date_gte is not None:
            params["date>"] = date_gte
        if date_lte is not None:
            params["date<"] = date_lte
        df = self._get("car_data", params)
        df = self._clean(df, _CAR_DATA_COLS)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        return validate_dataframe(df, _CAR_DATA_COLS, "get_car_data")

    def get_position(self, session_key: int) -> pd.DataFrame:
        """
        Return position data per driver per lap for a given session.

        Returns
        -------
        DataFrame with at least: ``driver_number``, ``position``, ``date``,
        ``session_key``.
        """
        df = self._get("position", {"session_key": session_key})
        df = self._clean(df, _POSITION_COLS)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return validate_dataframe(df, _POSITION_COLS, "get_position")

    def get_location(
        self,
        session_key: int,
        driver_number: int | None = None,
        date_gte: str | None = None,
        date_lte: str | None = None,
    ) -> pd.DataFrame:
        """Return car track coordinates (x/y/z) from /v1/location.

        Returns
        -------
        DataFrame with at least: ``driver_number``, ``date``, ``x``, ``y``,
        ``z``, ``session_key``.
        """
        params: dict = {"session_key": session_key}
        if driver_number is not None:
            params["driver_number"] = driver_number
        if date_gte is not None:
            params["date>"] = date_gte
        if date_lte is not None:
            params["date<"] = date_lte
        df = self._get("location", params)
        df = self._clean(df, _LOCATION_COLS)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        return validate_dataframe(df, _LOCATION_COLS, "get_location")

    # ------------------------------------------------------------------
    # Live mode
    # ------------------------------------------------------------------

    def _require_live(self) -> None:
        if self.mode != "live":
            raise RuntimeError(
                "Live methods are only available when mode='live'. "
                "Create the client with OpenF1Client(mode='live')."
            )

    def get_live_laps(self) -> pd.DataFrame:
        """
        Poll ``/v1/laps?session_key=latest`` and return only rows newer
        than the last call.
        """
        self._require_live()
        params: dict = {"session_key": "latest"}
        if self._last_lap_ts:
            params["date_gt"] = self._last_lap_ts
        df = self._get("laps", params)
        if df.empty:
            return df
        df = self._clean(df, _LAP_COLS)
        if "date_gt" not in params and "date" in df.columns:
            # first call — no watermark yet, return everything
            pass
        if "date" in df.columns and not df["date"].isna().all():
            self._last_lap_ts = str(df["date"].max())
        return validate_dataframe(df, _LAP_COLS, "get_live_laps")

    def get_live_stints(self) -> pd.DataFrame:
        """
        Poll ``/v1/stints?session_key=latest`` and return only rows newer
        than the last call.
        """
        self._require_live()
        params: dict = {"session_key": "latest"}
        if self._last_stint_ts:
            params["date_gt"] = self._last_stint_ts
        df = self._get("stints", params)
        if df.empty:
            return df
        df = self._clean(df, _STINT_COLS)
        if "date" in df.columns and not df["date"].isna().all():
            self._last_stint_ts = str(df["date"].max())
        return validate_dataframe(df, _STINT_COLS, "get_live_stints")

    def get_live_car_data(self) -> pd.DataFrame:
        """Poll ``/v1/car_data?session_key=latest`` for the current session."""
        self._require_live()
        df = self._get("car_data", {"session_key": "latest"})
        df = self._clean(df, _CAR_DATA_COLS)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
        return validate_dataframe(df, _CAR_DATA_COLS, "get_live_car_data")

    def get_live_position(self) -> pd.DataFrame:
        """
        Poll ``/v1/position?session_key=latest`` and return only rows newer
        than the last call.
        """
        self._require_live()
        params: dict = {"session_key": "latest"}
        if self._last_position_ts:
            params["date_gt"] = self._last_position_ts
        df = self._get("position", params)
        if df.empty:
            return df
        df = self._clean(df, _POSITION_COLS)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            if not df["date"].isna().all():
                self._last_position_ts = str(df["date"].max())
        return validate_dataframe(df, _POSITION_COLS, "get_live_position")
