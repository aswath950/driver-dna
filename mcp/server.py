"""
mcp/server.py — Driver DNA MCP Server

Exposes fastest-lap telemetry comparison capabilities from the driver-dna
dashboard as MCP tools, so any MCP client (Claude Desktop, custom tools, LLMs)
can query F1 session data, driver lists, and telemetry channel comparisons
without the Streamlit UI.

Tools
-----
list_sessions          — available sessions for a race weekend
list_drivers           — drivers in a specific session
get_fastest_lap_data   — full telemetry dataset for one driver's fastest lap
get_channel_comparison — two-driver channel comparison with raw data + Plotly JSON

Usage
-----
Run via stdio (default MCP transport):
    python mcp/server.py

Inspect with the MCP Inspector:
    npx @modelcontextprotocol/inspector python mcp/server.py

Add to Claude Desktop (claude_desktop_config.json):
    {
      "mcpServers": {
        "driver-dna": {
          "command": "python",
          "args": ["/absolute/path/to/mcp/server.py"]
        }
      }
    }
"""

from __future__ import annotations

import json
import pathlib
import sys

# Expose src/ modules without modifying the existing source tree
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from mcp.server.fastmcp import FastMCP
from openf1 import OpenF1Client
from viz import (
    _build_time_delta_fig,
    _build_track_map_fig,
    _fetch_fastest_lap_all_openf1,
    _fetch_fastest_lap_openf1,
    _resolve_pair_colours,
)

# Circuit XY coordinates for the Track Map visualisation — loaded once at startup.
# Keys match the race_name strings used in OpenF1 (e.g. "Italian Grand Prix").
_CIRCUITS_PATH = pathlib.Path(__file__).parent.parent / "data" / "circuits.json"
CIRCUITS: dict = json.loads(_CIRCUITS_PATH.read_text()) if _CIRCUITS_PATH.exists() else {}

N_POINTS = 200
VALID_CHANNELS = {"Speed", "Throttle", "Brake", "Time Delta", "Track Map"}

mcp = FastMCP("driver-dna-telemetry")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _resolve_session(year: int, race_name: str, session_type: str) -> tuple[int, OpenF1Client]:
    """Return (session_key, client) for the given year / race_name / session_type."""
    client = OpenF1Client(mode="historical")
    sessions = client.get_sessions(year, race_name)
    if sessions.empty:
        raise ValueError(
            f"No sessions found for '{race_name}' {year}. "
            "Check that the race name matches the OpenF1 API exactly "
            "(e.g. 'Italian Grand Prix', not 'Monza')."
        )
    if "session_type" not in sessions.columns:
        raise ValueError("Unexpected OpenF1 response: 'session_type' column missing.")
    match = sessions[sessions["session_type"].str.lower() == session_type.lower()]
    if match.empty:
        available = sessions["session_type"].tolist()
        raise ValueError(
            f"Session '{session_type}' not found for '{race_name}' {year}. "
            f"Available sessions: {available}"
        )
    return int(match.iloc[0]["session_key"]), client


def _acronym_map(drivers_df: pd.DataFrame) -> dict[int, str]:
    """Return {driver_number: name_acronym} from a drivers DataFrame."""
    result: dict[int, str] = {}
    for _, row in drivers_df.iterrows():
        num = row.get("driver_number")
        acronym = row.get("name_acronym", "UNK")
        if pd.notna(num):
            result[int(num)] = str(acronym) if pd.notna(acronym) else "UNK"
    return result


def _colour_map(drivers_df: pd.DataFrame) -> dict[int, str]:
    """Return {driver_number: hex_colour} from a drivers DataFrame."""
    result: dict[int, str] = {}
    if "team_colour" not in drivers_df.columns:
        return result
    for _, row in drivers_df.iterrows():
        num = row.get("driver_number")
        colour = row.get("team_colour")
        if pd.notna(num) and pd.notna(colour):
            col_str = str(colour)
            result[int(num)] = col_str if col_str.startswith("#") else f"#{col_str}"
    return result


def _to_list(a: np.ndarray | None) -> list[float]:
    """Convert a numpy array to a plain Python float list (JSON-serialisable)."""
    if a is None:
        return []
    return [float(v) for v in a]


# ── MCP Tools ─────────────────────────────────────────────────────────────────

@mcp.tool()
def list_sessions(year: int, race_name: str) -> list[dict]:
    """
    List all available sessions for a Formula 1 race weekend.

    Parameters
    ----------
    year : int
        Season year, e.g. 2024.
    race_name : str
        Race weekend name as it appears on OpenF1, e.g. 'Italian Grand Prix'.

    Returns
    -------
    List of dicts, each with: session_key (int), session_name (str),
    session_type (str), date_start (ISO-8601 string or null).
    Pass session_type values from this result to the other tools.
    """
    client = OpenF1Client(mode="historical")
    sessions = client.get_sessions(year, race_name)
    if sessions.empty:
        return []

    keep = [c for c in ["session_key", "session_name", "session_type", "date_start"]
            if c in sessions.columns]
    rows = []
    for _, row in sessions[keep].iterrows():
        entry: dict = {}
        for col in keep:
            val = row[col]
            if isinstance(val, pd.Timestamp):
                entry[col] = val.isoformat()
            elif pd.isna(val):
                entry[col] = None
            else:
                entry[col] = val
        rows.append(entry)
    return rows


@mcp.tool()
def list_drivers(year: int, race_name: str, session_type: str) -> list[dict]:
    """
    List all drivers that participated in a specific race session.

    Parameters
    ----------
    year : int
        Season year, e.g. 2024.
    race_name : str
        Race weekend name, e.g. 'Italian Grand Prix'.
    session_type : str
        Session type as returned by list_sessions, e.g. 'Race', 'Qualifying',
        'Practice 1'. Case-insensitive.

    Returns
    -------
    List of dicts, each with: driver_number (int), name_acronym (str),
    full_name (str), team_name (str).
    Use driver_number values when calling get_fastest_lap_data or
    get_channel_comparison.
    """
    session_key, client = _resolve_session(year, race_name, session_type)
    drivers_df = client.get_drivers(session_key)
    if drivers_df.empty:
        return []

    keep = [c for c in ["driver_number", "name_acronym", "full_name", "team_name"]
            if c in drivers_df.columns]
    rows = []
    for _, row in drivers_df[keep].iterrows():
        entry: dict = {}
        for col in keep:
            val = row[col]
            entry[col] = None if (not isinstance(val, str) and pd.isna(val)) else val
        rows.append(entry)
    return rows


@mcp.tool()
def get_fastest_lap_data(
    year: int,
    race_name: str,
    session_type: str,
    driver_number: int,
) -> dict:
    """
    Retrieve all telemetry channels for a driver's fastest lap in a session.

    Returns 200-point distance-normalised traces for speed, throttle, brake,
    and cumulative elapsed time. The 200 points span the full lap from start
    to finish at evenly spaced distance intervals.

    Parameters
    ----------
    year : int
        Season year, e.g. 2024.
    race_name : str
        Race weekend name, e.g. 'Italian Grand Prix'.
    session_type : str
        Session type, e.g. 'Race', 'Qualifying', 'Practice 1'.
    driver_number : int
        OpenF1 driver number (use list_drivers to look these up),
        e.g. 1 for Verstappen, 44 for Hamilton.

    Returns
    -------
    Dict with keys:
      driver_number (int), acronym (str), lap_time (float, seconds),
      lap_number (int or null),
      speed (200 floats, km/h), throttle (200 floats, 0–100 %),
      brake (200 floats, 0–100), cumtime (200 floats, elapsed seconds).
    """
    session_key, client = _resolve_session(year, race_name, session_type)
    laps_df = client.get_laps(session_key)
    drivers_df = client.get_drivers(session_key)

    acr = _acronym_map(drivers_df).get(driver_number, "UNK")
    data = _fetch_fastest_lap_all_openf1(session_key, driver_number, laps_df)
    if data is None:
        raise ValueError(
            f"Could not fetch telemetry for driver {driver_number} ({acr}) "
            f"in {session_type} at {race_name} {year}. "
            "The driver may not have completed a timed lap in this session."
        )

    return {
        "driver_number": driver_number,
        "acronym": acr,
        "lap_time": float(data["lap_time"]),
        "lap_number": data["lap_number"],
        "speed":    _to_list(data["speed"]),
        "throttle": _to_list(data["throttle"]),
        "brake":    _to_list(data["brake"]),
        "cumtime":  _to_list(data["cumtime"]),
    }


@mcp.tool()
def get_channel_comparison(
    year: int,
    race_name: str,
    session_type: str,
    driver_a: int,
    driver_b: int,
    channel: str,
) -> dict:
    """
    Compare two drivers' fastest laps on a selected telemetry channel.

    Returns raw data arrays for both drivers plus an interactive Plotly figure
    serialised as JSON.

    Parameters
    ----------
    year : int
        Season year, e.g. 2024.
    race_name : str
        Race weekend name, e.g. 'Italian Grand Prix'.
    session_type : str
        Session type, e.g. 'Race', 'Qualifying', 'Practice 1'.
    driver_a : int
        OpenF1 driver number for Driver A (use list_drivers to look up).
    driver_b : int
        OpenF1 driver number for Driver B.
    channel : str
        One of: 'Speed', 'Throttle', 'Brake', 'Time Delta', 'Track Map'.
        - Speed / Throttle / Brake: overlaid 200-point distance traces.
        - Time Delta: cumulative gap in seconds over the lap distance.
        - Track Map: circuit coloured by which driver is faster per microsector.

    Returns
    -------
    Dict with keys:
      driver_a (dict): telemetry data (same shape as get_fastest_lap_data)
      driver_b (dict): telemetry data
      channel (str): echoed back
      figure_json (str): Plotly figure as a JSON string.
                         Use plotly.io.from_json() or JSON.parse() to render.
    """
    if channel not in VALID_CHANNELS:
        raise ValueError(
            f"Invalid channel '{channel}'. Must be one of: {sorted(VALID_CHANNELS)}"
        )

    session_key, client = _resolve_session(year, race_name, session_type)
    laps_df = client.get_laps(session_key)
    drivers_df = client.get_drivers(session_key)

    acr_map = _acronym_map(drivers_df)
    col_map = _colour_map(drivers_df)
    acr_a = acr_map.get(driver_a, "DRV_A")
    acr_b = acr_map.get(driver_b, "DRV_B")
    color_a, color_b = _resolve_pair_colours(driver_a, driver_b, col_map)

    if channel in ("Track Map", "Time Delta"):
        data_a = _fetch_fastest_lap_all_openf1(session_key, driver_a, laps_df)
        data_b = _fetch_fastest_lap_all_openf1(session_key, driver_b, laps_df)

        if data_a is None:
            raise ValueError(f"No telemetry for driver {driver_a} ({acr_a}).")
        if data_b is None:
            raise ValueError(f"No telemetry for driver {driver_b} ({acr_b}).")

        if channel == "Track Map":
            circuit = CIRCUITS.get(race_name, {})
            if circuit:
                data_a["x"] = circuit.get("x")
                data_a["y"] = circuit.get("y")
                data_b["x"] = circuit.get("x")
                data_b["y"] = circuit.get("y")
            fig = _build_track_map_fig(data_a, acr_a, color_a, data_b, acr_b, color_b)
            if fig is None:
                available = list(CIRCUITS.keys())
                raise ValueError(
                    f"Track map unavailable: no circuit XY data found for '{race_name}'. "
                    f"Available circuits: {available}"
                )
        else:
            fig = _build_time_delta_fig(data_a, acr_a, color_a, data_b, acr_b, color_b)

        result_a: dict = {
            "driver_number": driver_a, "acronym": acr_a,
            "lap_time": float(data_a["lap_time"]), "lap_number": data_a["lap_number"],
            "speed":    _to_list(data_a["speed"]),
            "throttle": _to_list(data_a["throttle"]),
            "brake":    _to_list(data_a["brake"]),
            "cumtime":  _to_list(data_a["cumtime"]),
        }
        result_b: dict = {
            "driver_number": driver_b, "acronym": acr_b,
            "lap_time": float(data_b["lap_time"]), "lap_number": data_b["lap_number"],
            "speed":    _to_list(data_b["speed"]),
            "throttle": _to_list(data_b["throttle"]),
            "brake":    _to_list(data_b["brake"]),
            "cumtime":  _to_list(data_b["cumtime"]),
        }

    else:  # Speed / Throttle / Brake — single-channel line overlay
        trace_a, lap_time_a, lap_num_a = _fetch_fastest_lap_openf1(
            session_key, driver_a, laps_df, channel
        )
        trace_b, lap_time_b, lap_num_b = _fetch_fastest_lap_openf1(
            session_key, driver_b, laps_df, channel
        )
        if trace_a is None:
            raise ValueError(f"No {channel} telemetry for driver {driver_a} ({acr_a}).")
        if trace_b is None:
            raise ValueError(f"No {channel} telemetry for driver {driver_b} ({acr_b}).")

        x_axis = list(range(N_POINTS))
        label_a = f"Lap {lap_num_a}, {lap_time_a:.3f}s" if lap_num_a else f"{lap_time_a:.3f}s"
        label_b = f"Lap {lap_num_b}, {lap_time_b:.3f}s" if lap_num_b else f"{lap_time_b:.3f}s"

        channel_units = {"Speed": "km/h", "Throttle": "%", "Brake": "%"}
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_axis, y=_to_list(trace_a),
            mode="lines", name=f"{acr_a} ({label_a})",
            line=dict(color=color_a, width=2),
        ))
        fig.add_trace(go.Scatter(
            x=x_axis, y=_to_list(trace_b),
            mode="lines", name=f"{acr_b} ({label_b})",
            line=dict(color=color_b, width=2),
        ))
        fig.update_layout(
            title=f"Fastest Lap {channel} — {acr_a} vs {acr_b}",
            xaxis_title="Normalised Lap Distance (0 = start, 199 = end)",
            yaxis_title=f"{channel} ({channel_units.get(channel, '')})",
            height=420,
        )

        result_a = {
            "driver_number": driver_a, "acronym": acr_a,
            "lap_time": float(lap_time_a), "lap_number": lap_num_a,
            channel.lower(): _to_list(trace_a),
        }
        result_b = {
            "driver_number": driver_b, "acronym": acr_b,
            "lap_time": float(lap_time_b), "lap_number": lap_num_b,
            channel.lower(): _to_list(trace_b),
        }

    return {
        "driver_a": result_a,
        "driver_b": result_b,
        "channel": channel,
        "figure_json": fig.to_json(),
    }


if __name__ == "__main__":
    mcp.run()
