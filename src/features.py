"""features.py — Feature engineering constants and pure computation helpers.

No Streamlit UI code; the cache decorators (st.cache_data / st.cache_resource)
are the only Streamlit dependency here, making the business logic independently
testable by calling the unwrapped function directly.
"""

import hashlib
import pickle

import numpy as np
import pandas as pd
import streamlit as st


def _hash_df(df: pd.DataFrame) -> str:
    return hashlib.md5(pickle.dumps(df)).hexdigest()

# ── Distance-normalised grid size ────────────────────────────────────────────
N_POINTS = 200
TRACE_COLS = {"Speed": "speed_trace", "Throttle": "throttle_trace", "Brake": "brake_trace"}
# OpenF1 car_data column names for the same channels
OPENF1_TRACE_COLS = {"Speed": "speed", "Throttle": "throttle", "Brake": "brake"}
# Channels that require special rendering (not a simple 1D line overlay)
SPECIAL_CHANNELS = ["Track Map", "Time Delta"]
RADAR_FEATURES = ["mean_speed", "throttle_mean", "throttle_std", "brake_events", "steer_std", "gear_changes"]
EXTENDED_RADAR_FEATURES = [
    "mean_speed", "max_speed", "min_speed",
    "throttle_mean", "throttle_std", "throttle_pickup_pct",
    "brake_mean", "brake_events", "trail_brake_score",
    "steer_std", "gear_changes", "coasting_pct",
]
FEATURE_LABELS = {
    "mean_speed":           "Avg Speed",
    "max_speed":            "Top Speed",
    "min_speed":            "Corner Speed",
    "throttle_mean":        "Throttle Input",
    "throttle_std":         "Throttle Variation",
    "brake_mean":           "Brake Pressure",
    "brake_events":         "Brake Frequency",
    "steer_std":            "Steering Precision",
    "gear_changes":         "Gear Shifts",
    "coasting_pct":         "Coasting %",
    "throttle_pickup_pct":  "Full Throttle %",
    "trail_brake_score":    "Trail Braking",
}

_FEATURE_BOUNDS: dict = {
    "lap_time_seconds":  (60.0,  200.0),
    "mean_speed":        (50.0,  380.0),
    "max_speed":         (80.0,  420.0),
    "min_speed":         (0.0,   250.0),
    "throttle_mean":     (0.0,   100.0),
    "throttle_std":      (0.0,    60.0),
    "brake_mean":        (0.0,   100.0),
    "brake_events":      (0.0,   200.0),
    "gear_changes":      (0.0,  1500.0),
    "steer_std":         (0.0,   100.0),
}


@st.cache_data(hash_funcs={pd.DataFrame: _hash_df})
def compute_extended_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute coasting_pct, throttle_pickup_pct, trail_brake_score per lap from traces."""
    records = []
    for _, row in df.iterrows():
        try:
            thr = np.array(row["throttle_trace"], dtype=float)
            brk = np.array(row["brake_trace"], dtype=float)
            records.append({
                "driver": row["driver"],
                "lap_number": row.get("lap_number", 0),
                "coasting_pct": float(np.nanmean((thr < 2) & (brk < 1))),
                "throttle_pickup_pct": float(np.nanmean(thr > 80)),
                "trail_brake_score": float(np.nanmean((thr > 0) & (brk > 0))),
            })
        except Exception:
            continue
    if not records:
        df["coasting_pct"] = np.nan
        df["throttle_pickup_pct"] = np.nan
        df["trail_brake_score"] = np.nan
        return df
    ext = pd.DataFrame(records)
    return df.merge(ext, on=["driver", "lap_number"], how="left")


def compute_mean_trace(df: pd.DataFrame, driver: str, col: str) -> np.ndarray:
    """Return mean of a 200-point trace column across all laps for a driver."""
    rows = df[df["driver"] == driver][col]
    stack = np.stack([np.array(r, dtype=float) for r in rows])
    return np.nanmean(stack, axis=0)


def compute_sector_means(df: pd.DataFrame, driver: str, col: str) -> list[float]:
    """Split 200-point trace into three equal zones and return mean per zone."""
    stack = np.stack([np.array(r, dtype=float) for r in df[df["driver"] == driver][col]])
    return [float(np.nanmean(stack[:, s])) for s in [slice(0, 67), slice(67, 134), slice(134, 200)]]


@st.cache_resource
def get_shap_explainer(_clf):
    """Cache a shap.TreeExplainer. Underscore prefix prevents Streamlit hashing clf."""
    import shap
    return shap.TreeExplainer(_clf)


def compute_shap_for_prediction(
    explainer, X_input: np.ndarray, pred_class_idx: int
) -> np.ndarray:
    """Return 1-D SHAP values (n_features,) for the predicted class only.

    X_input always has shape (1, n_features), so the first axis of the raw
    SHAP output is a reliable discriminator:
      - New SHAP (>=0.41): (n_samples=1, n_features, n_classes) → shape[0] == 1
      - Old SHAP:          (n_classes, n_samples=1, n_features) → shape[0] == n_classes
    """
    raw = explainer.shap_values(X_input)
    arr = np.array(raw)
    if arr.ndim == 2:           # binary: (1, n_features)
        return arr[0]
    elif arr.ndim == 3:
        if arr.shape[0] == 1:   # new SHAP: (1, n_features, n_classes)
            return arr[0, :, pred_class_idx]
        else:                   # old SHAP: (n_classes, 1, n_features)
            return arr[pred_class_idx, 0, :]
    return np.abs(arr).mean(axis=tuple(range(arr.ndim - 1)))      # fallback


def _validate_feature_values(feature_dict: dict) -> tuple:
    """Check all values are finite floats within F1-reasonable bounds.
    Returns (True, '') on success or (False, reason) on failure."""
    import math
    for feat, val in feature_dict.items():
        try:
            fval = float(val)
        except (TypeError, ValueError):
            return False, f"Feature '{feat}' is not numeric."
        if not math.isfinite(fval):
            return False, f"Feature '{feat}' is not finite (got {val})."
        lo, hi = _FEATURE_BOUNDS.get(feat, (-1e9, 1e9))
        if not (lo <= fval <= hi):
            return False, f"Feature '{feat}' value {fval:.2f} outside expected range [{lo}, {hi}]."
    return True, ""


def _sanitize_driver_name(name: str, known_drivers: list) -> str:
    """Validate driver name is in the known list before prompt interpolation.
    Raises ValueError if not found — prevents prompt injection."""
    if name not in known_drivers:
        raise ValueError(f"Unknown driver '{name}' — not in model's class list.")
    return name


def classify_archetype(norm_means: dict) -> tuple[str, str]:
    """Assign a driving style archetype from normalised (0-1) feature means."""
    if norm_means.get("brake_events", 0) > 0.65 and norm_means.get("brake_mean", 0) > 0.60:
        return "Aggressive Braker", "💥"
    if norm_means.get("trail_brake_score", 0) > 0.55 and norm_means.get("steer_std", 0) > 0.60:
        return "Trail Braking Specialist", "🎯"
    if norm_means.get("mean_speed", 0) > 0.70 and norm_means.get("coasting_pct", 0) < 0.35:
        return "High Entry Speed", "🚀"
    if norm_means.get("throttle_pickup_pct", 0) > 0.60 and norm_means.get("throttle_std", 0) < 0.40:
        return "Smooth Operator", "🧊"
    if norm_means.get("gear_changes", 0) > 0.65 and norm_means.get("steer_std", 0) > 0.55:
        return "Technical Driver", "⚙️"
    return "Balanced Driver", "⚖️"
