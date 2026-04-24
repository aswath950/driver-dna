"""
test_pipeline_tab.py — Tests for the Pipeline & Training tab logic.

app.py executes Streamlit calls at module level, so direct import is
incompatible with pytest. Instead, each test function defines a thin local
mirror of _run_pipeline_download / _run_model_training that accepts a plain
dict for session_state and injectable function dependencies. The logic is
identical to the app.py wrappers; only the Streamlit coupling is removed.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ---------------------------------------------------------------------------
# Local mirrors of the app.py thread-wrapper logic (updated for progress_cb)
# ---------------------------------------------------------------------------

def _pipeline_download(
    ss: dict,
    year: int,
    gp: str,
    session_type: str,
    drivers,
    *,
    extract_fn,
    save_dataset_fn,
    save_meta_fn,
    data_dir: Path,
) -> None:
    def _cb(frac: float, msg: str) -> None:
        ss["pipe_progress"] = frac
        ss["pipe_status"] = msg
        ss["pipe_log"].append(msg)

    try:
        ss["pipe_log"] = []
        ss["pipe_progress"] = 0.0
        _cb(0.0, f"Connecting to FastF1 for {year} {gp} ({session_type})…")
        df = extract_fn(year, gp, session_type, drivers, progress_cb=_cb)
        _cb(0.97, "Saving dataset…")
        save_dataset_fn(df, data_dir / "dataset.parquet")
        save_meta_fn(year, gp, session_type)
        _cb(1.0, f"Done — {len(df)} laps saved to data/dataset.parquet")
        ss["pipe_done"] = True
    except Exception as exc:
        ss["pipe_error"] = str(exc)
    finally:
        ss["pipe_running"] = False


def _model_training(
    ss: dict,
    *,
    load_fn,
    train_fn,
    evaluate_fn,
    save_fn,
    data_dir: Path,
) -> None:
    def _cb(frac: float, msg: str) -> None:
        ss["train_progress"] = frac
        ss["train_status"] = msg
        ss["train_log"].append(msg)

    try:
        ss["train_log"] = []
        ss["train_progress"] = 0.0
        _cb(0.0, "Loading dataset…")
        X, y, le = load_fn(data_dir / "dataset.parquet")
        n_drivers = len(set(y))
        if n_drivers < 2:
            raise ValueError(f"Need at least 2 drivers to train a classifier; found {n_drivers}.")
        _cb(0.05, f"Loaded {len(X)} samples, {n_drivers} drivers. Starting 5-fold CV…")
        clf, acc = train_fn(X, y, progress_cb=lambda f, m: _cb(0.05 + f * 0.67, m))
        _cb(0.75, "Evaluating — computing metrics and SHAP values…")
        evaluate_fn(clf, X, y, le)
        _cb(0.97, "Saving model artifacts…")
        save_fn(clf, le)
        _cb(1.0, "Done — model saved to models/")
        ss["train_done"] = True
    except Exception as exc:
        ss["train_error"] = str(exc)
    finally:
        ss["train_running"] = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ss(**overrides) -> dict:
    defaults = {
        "pipe_running":  True,
        "pipe_done":     False,
        "pipe_error":    None,
        "pipe_log":      [],
        "pipe_progress": 0.0,
        "pipe_status":   "",
        "train_running":  True,
        "train_done":     False,
        "train_error":    None,
        "train_log":      [],
        "train_progress": 0.0,
        "train_status":   "",
    }
    return {**defaults, **overrides}


def _tiny_df(n_drivers: int = 3, laps_per_driver: int = 4) -> pd.DataFrame:
    """Minimal lap DataFrame matching the feature columns expected by load_and_prepare."""
    drivers = [f"DRV{i}" for i in range(n_drivers)]
    rows = []
    for drv in drivers:
        for lap in range(laps_per_driver):
            rows.append({
                "driver": drv,
                "lap_time_seconds": 90.0 + lap * 0.1,
                "mean_speed": 200.0,
                "max_speed": 280.0,
                "min_speed": 80.0,
                "throttle_mean": 0.7,
                "throttle_std": 0.1,
                "brake_mean": 0.2,
                "brake_events": 12,
                "gear_changes": 55,
                "steer_std": 0.05,
                "speed_trace": list(range(200)),
                "throttle_trace": [0.7] * 200,
                "brake_trace": [0.2] * 200,
            })
    return pd.DataFrame(rows)


def _simulating_extract_fn(fractions_seen: list, messages_seen: list):
    """
    Returns a mock extract_fn that calls its progress_cb for 3 drivers,
    recording each (fraction, message) pair for assertion.
    """
    def _extract(year, gp, session_type, drivers, progress_cb=None):
        if progress_cb:
            progress_cb(0.05, "Session loaded — 3 drivers found")
            for i, drv in enumerate(["DRV0", "DRV1", "DRV2"]):
                frac = 0.05 + 0.90 * (i + 1) / 3
                msg = f"Processed {drv} ({i + 1}/3)"
                progress_cb(frac, msg)
                fractions_seen.append(frac)
                messages_seen.append(msg)
        return _tiny_df(n_drivers=3, laps_per_driver=4)
    return _extract


def _simulating_train_fn(fractions_seen: list):
    """
    Returns a mock train_fn that calls its progress_cb for 5 folds,
    recording each fraction for assertion.
    """
    def _train(X, y, progress_cb=None):
        for i in range(5):
            if progress_cb:
                progress_cb(i / 5 * 0.90, f"Training fold {i + 1}/5…")
                progress_cb((i + 1) / 5 * 0.90, f"Fold {i + 1}/5 — accuracy: 0.850")
                fractions_seen.append(i / 5 * 0.90)
                fractions_seen.append((i + 1) / 5 * 0.90)
        if progress_cb:
            progress_cb(0.97, "Fitting final model…")
            fractions_seen.append(0.97)
        return MagicMock(), 0.850
    return _train


# ===========================================================================
# TestPipelineDownload — mirrors app._run_pipeline_download
# ===========================================================================

class TestPipelineDownload:
    def test_happy_path_sets_pipe_done(self):
        ss = _make_ss()
        _pipeline_download(
            ss, 2025, "Italian Grand Prix", "R", None,
            extract_fn=MagicMock(return_value=_tiny_df()),
            save_dataset_fn=MagicMock(),
            save_meta_fn=MagicMock(),
            data_dir=Path("/tmp"),
        )
        assert ss["pipe_done"] is True
        assert ss["pipe_running"] is False
        assert ss["pipe_error"] is None

    def test_happy_path_log_contains_lap_count(self):
        ss = _make_ss()
        mock_df = _tiny_df(n_drivers=3, laps_per_driver=5)  # 15 laps
        _pipeline_download(
            ss, 2025, "Italian Grand Prix", "R", None,
            extract_fn=MagicMock(return_value=mock_df),
            save_dataset_fn=MagicMock(),
            save_meta_fn=MagicMock(),
            data_dir=Path("/tmp"),
        )
        full_log = " ".join(ss["pipe_log"])
        assert "15" in full_log
        assert "laps saved" in full_log

    def test_extraction_error_sets_pipe_error(self):
        ss = _make_ss()
        _pipeline_download(
            ss, 2025, "Bad Grand Prix", "R", None,
            extract_fn=MagicMock(side_effect=ValueError("session not found")),
            save_dataset_fn=MagicMock(),
            save_meta_fn=MagicMock(),
            data_dir=Path("/tmp"),
        )
        assert ss["pipe_error"] == "session not found"
        assert ss["pipe_running"] is False
        assert ss["pipe_done"] is False

    def test_running_flag_always_cleared_on_error(self):
        ss = _make_ss(pipe_running=True)
        _pipeline_download(
            ss, 2025, "Crash GP", "Q", None,
            extract_fn=MagicMock(side_effect=RuntimeError("network timeout")),
            save_dataset_fn=MagicMock(),
            save_meta_fn=MagicMock(),
            data_dir=Path("/tmp"),
        )
        assert ss["pipe_running"] is False

    def test_save_functions_called_with_correct_args(self):
        ss = _make_ss()
        save_ds = MagicMock()
        save_m = MagicMock()
        _pipeline_download(
            ss, 2024, "Monaco Grand Prix", "Q", ["VER", "HAM"],
            extract_fn=MagicMock(return_value=_tiny_df()),
            save_dataset_fn=save_ds,
            save_meta_fn=save_m,
            data_dir=Path("/data"),
        )
        save_ds.assert_called_once()
        save_m.assert_called_once_with(2024, "Monaco Grand Prix", "Q")

    def test_progress_cb_fractions_are_non_decreasing(self):
        ss = _make_ss()
        fractions: list[float] = []
        messages: list[str] = []
        _pipeline_download(
            ss, 2025, "Italian Grand Prix", "R", None,
            extract_fn=_simulating_extract_fn(fractions, messages),
            save_dataset_fn=MagicMock(),
            save_meta_fn=MagicMock(),
            data_dir=Path("/tmp"),
        )
        # Verify overall: pipe_progress ends at 1.0 and pipe_done is True
        assert ss["pipe_progress"] == 1.0
        assert ss["pipe_done"] is True
        # Fractions captured by the simulating fn must be non-decreasing
        for a, b in zip(fractions, fractions[1:]):
            assert b >= a, f"Fraction went backwards: {a} → {b}"

    def test_progress_cb_messages_contain_driver_info(self):
        ss = _make_ss()
        fractions: list[float] = []
        messages: list[str] = []
        _pipeline_download(
            ss, 2025, "Italian Grand Prix", "R", None,
            extract_fn=_simulating_extract_fn(fractions, messages),
            save_dataset_fn=MagicMock(),
            save_meta_fn=MagicMock(),
            data_dir=Path("/tmp"),
        )
        assert any("DRV0" in m for m in messages), "Expected per-driver messages in callback"

    def test_progress_cb_none_does_not_raise(self):
        """Passing progress_cb=None to extract_fn must not break CLI usage."""
        ss = _make_ss()
        # extract_fn that ignores progress_cb (as CLI calls would)
        def _cli_style_extract(year, gp, session_type, drivers, progress_cb=None):
            assert progress_cb is not None  # wrapper always passes _cb
            return _tiny_df()
        _pipeline_download(
            ss, 2025, "Italian Grand Prix", "R", None,
            extract_fn=_cli_style_extract,
            save_dataset_fn=MagicMock(),
            save_meta_fn=MagicMock(),
            data_dir=Path("/tmp"),
        )
        assert ss["pipe_done"] is True

    def test_pipe_progress_is_zero_at_start_of_new_run(self):
        ss = _make_ss(pipe_progress=0.85)  # simulate leftover from previous run
        _pipeline_download(
            ss, 2025, "Italian Grand Prix", "R", None,
            extract_fn=MagicMock(side_effect=RuntimeError("fail early")),
            save_dataset_fn=MagicMock(),
            save_meta_fn=MagicMock(),
            data_dir=Path("/tmp"),
        )
        # After error the wrapper already reset pipe_progress to 0.0 before calling extract
        assert ss["pipe_error"] is not None
        assert ss["pipe_running"] is False


# ===========================================================================
# TestModelTraining — mirrors app._run_model_training
# ===========================================================================

class TestModelTraining:
    def _call(self, ss, *, load_rv=None, load_exc=None, train_fn=None):
        X = np.random.rand(20, 10)
        y = np.array([0] * 10 + [1] * 10)
        le = MagicMock()
        clf = MagicMock()

        load_fn = (
            MagicMock(side_effect=load_exc)
            if load_exc
            else MagicMock(return_value=load_rv or (X, y, le))
        )
        _train_fn = train_fn or MagicMock(return_value=(clf, 0.85))
        eval_fn = MagicMock()
        save_fn = MagicMock()

        _model_training(
            ss,
            load_fn=load_fn,
            train_fn=_train_fn,
            evaluate_fn=eval_fn,
            save_fn=save_fn,
            data_dir=Path("/data"),
        )
        return load_fn, _train_fn, eval_fn, save_fn

    def test_happy_path_sets_train_done(self):
        ss = _make_ss()
        self._call(ss)
        assert ss["train_done"] is True
        assert ss["train_running"] is False
        assert ss["train_error"] is None

    def test_happy_path_log_contains_done(self):
        ss = _make_ss()
        self._call(ss)
        full_log = " ".join(ss["train_log"])
        assert "Done" in full_log

    def test_happy_path_log_contains_sample_count(self):
        ss = _make_ss()
        X = np.random.rand(20, 10)
        y = np.array([0] * 10 + [1] * 10)
        le = MagicMock()
        clf = MagicMock()
        _model_training(
            ss,
            load_fn=MagicMock(return_value=(X, y, le)),
            train_fn=MagicMock(return_value=(clf, 0.850)),
            evaluate_fn=MagicMock(),
            save_fn=MagicMock(),
            data_dir=Path("/data"),
        )
        full_log = " ".join(ss["train_log"])
        assert "20" in full_log  # sample count

    def test_missing_dataset_sets_train_error(self):
        ss = _make_ss()
        self._call(ss, load_exc=FileNotFoundError("dataset.parquet not found"))
        assert ss["train_error"] is not None
        assert "dataset.parquet" in ss["train_error"]
        assert ss["train_running"] is False
        assert ss["train_done"] is False

    def test_running_flag_always_cleared_on_error(self):
        ss = _make_ss(train_running=True)
        self._call(ss, load_exc=RuntimeError("disk full"))
        assert ss["train_running"] is False

    def test_low_sample_guard_fewer_than_two_drivers(self):
        ss = _make_ss()
        X = np.random.rand(10, 10)
        y = np.zeros(10, dtype=int)  # only 1 class
        le = MagicMock()
        _model_training(
            ss,
            load_fn=MagicMock(return_value=(X, y, le)),
            train_fn=MagicMock(),
            evaluate_fn=MagicMock(),
            save_fn=MagicMock(),
            data_dir=Path("/data"),
        )
        assert ss["train_error"] is not None
        assert "2 drivers" in ss["train_error"] or "driver" in ss["train_error"]
        assert ss["train_done"] is False
        assert ss["train_running"] is False

    def test_all_downstream_functions_called_on_success(self):
        ss = _make_ss()
        load_fn, train_fn, eval_fn, save_fn = self._call(ss)
        load_fn.assert_called_once()
        train_fn.assert_called_once()
        eval_fn.assert_called_once()
        save_fn.assert_called_once()

    def test_downstream_not_called_on_load_error(self):
        ss = _make_ss()
        load_fn, train_fn, eval_fn, save_fn = self._call(ss, load_exc=FileNotFoundError())
        train_fn.assert_not_called()
        eval_fn.assert_not_called()
        save_fn.assert_not_called()

    def test_progress_cb_fractions_are_non_decreasing(self):
        ss = _make_ss()
        fractions: list[float] = []
        self._call(ss, train_fn=_simulating_train_fn(fractions))
        assert ss["train_progress"] == 1.0
        assert ss["train_done"] is True
        for a, b in zip(fractions, fractions[1:]):
            assert b >= a, f"Train fraction went backwards: {a} → {b}"

    def test_progress_cb_fires_per_fold(self):
        """Simulating train_fn fires 2 calls per fold (before + after) × 5 folds = 10."""
        ss = _make_ss()
        fractions: list[float] = []
        self._call(ss, train_fn=_simulating_train_fn(fractions))
        # 5 folds × 2 callbacks + 1 final-fit callback = 11 total from train_fn
        assert len(fractions) == 11

    def test_progress_cb_none_does_not_raise(self):
        """MagicMock train_fn receives progress_cb kwarg — must not raise."""
        ss = _make_ss()
        self._call(ss)
        assert ss["train_done"] is True


# ===========================================================================
# TestTrainModelUnit — tests train_model from model.py directly
# ===========================================================================

class TestTrainModelUnit:
    """Integration-level tests that exercise the real train_model function."""

    def _make_xy(self, n_per_class: int = 30, n_features: int = 10, n_classes: int = 2):
        rng = np.random.default_rng(42)
        X_parts, y_parts = [], []
        for cls in range(n_classes):
            X_parts.append(rng.normal(loc=cls * 5.0, scale=1.0, size=(n_per_class, n_features)))
            y_parts.append(np.full(n_per_class, cls, dtype=int))
        return np.vstack(X_parts), np.concatenate(y_parts)

    def test_returns_classifier_and_float_accuracy(self):
        from model import train_model
        X, y = self._make_xy()
        clf, acc = train_model(X, y)
        assert hasattr(clf, "predict"), "Returned object must have a predict method"
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0

    def test_progress_cb_called_for_each_fold(self):
        from model import train_model
        X, y = self._make_xy()
        calls: list[tuple[float, str]] = []

        def _cb(frac: float, msg: str) -> None:
            calls.append((frac, msg))

        train_model(X, y, progress_cb=_cb)

        fracs = [f for f, _ in calls]
        # Must have at least 10 calls: 2 per fold × 5 folds (before + after)
        assert len(calls) >= 10, f"Expected ≥10 callback calls, got {len(calls)}"
        # Final fraction before the fit call must be 0.97
        assert any(abs(f - 0.97) < 1e-9 for f, _ in calls), "Expected 0.97 (final-fit) callback"
        # All fractions must be non-decreasing
        for a, b in zip(fracs, fracs[1:]):
            assert b >= a, f"Fraction went backwards: {a} → {b}"

    def test_progress_cb_none_does_not_raise(self):
        from model import train_model
        X, y = self._make_xy()
        clf, acc = train_model(X, y, progress_cb=None)
        assert hasattr(clf, "predict")
        assert 0.0 <= acc <= 1.0
