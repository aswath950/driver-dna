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

# Columns to retain and their expected dtypes for each endpoint.
# Any extra columns returned by the API are kept but not type-cast.
_LAP_COLS = [
    "driver_number", "lap_number", "lap_duration",
    "is_pit_out_lap", "st_speed", "session_key",
]
_STINT_COLS = [
    "driver_number", "stint_number", "compound",
    "tyre_age_at_start", "lap_start", "lap_end", "session_key",
]
_POSITION_COLS = [
    "driver_number", "position", "date", "session_key",
]
_CAR_DATA_COLS = [
    "driver_number", "date", "speed", "throttle", "brake",
    "n_gear", "rpm", "session_key",
]


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
    """

    def __init__(
        self,
        mode: Literal["historical", "live"] = "historical",
        timeout: int = 10,
    ) -> None:
        if mode not in ("historical", "live"):
            raise ValueError(f"mode must be 'historical' or 'live', got {mode!r}")
        self.mode = mode
        self.timeout = timeout
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
        Returns an empty DataFrame on final failure — callers never need
        to catch transport-level exceptions.
        """
        url = f"{BASE_URL}/{endpoint}"
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
            except (requests.RequestException, ValueError) as exc:
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
        for col in ["driver_number", "name_acronym", "full_name", "team_name"]:
            if col not in df.columns:
                df[col] = pd.NA
        return df

    def get_live_drivers(self) -> pd.DataFrame:
        """Poll ``/v1/drivers?session_key=latest`` for the current session."""
        self._require_live()
        df = self._get("drivers", {"session_key": "latest"})
        if df.empty:
            return df
        for col in ["driver_number", "name_acronym", "full_name", "team_name"]:
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
        session_key: int,
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
