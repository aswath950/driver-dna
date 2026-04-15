"""
test_app_helpers.py — Tests for pure logic helpers from src/app.py

These functions are copied inline here to avoid triggering Streamlit's
module-level st.set_page_config() call at import time. After the planned
module split (llm_layer.py, features.py), this file will import directly
from the new modules instead.
"""

import json

import numpy as np
import pytest


# ===========================================================================
# Copied helpers — keep in sync with src/app.py until module split
# ===========================================================================

def _ra_parse_critique(raw_text: str) -> tuple:
    try:
        data = json.loads(raw_text)
    except (json.JSONDecodeError, TypeError):
        return None, "Critic returned non-JSON response."
    if not isinstance(data, dict):
        return None, "Critic response was not a JSON object."
    try:
        confidence = max(1, min(10, int(data.get("confidence", 10))))
    except (TypeError, ValueError):
        return None, "Critic 'confidence' field missing or invalid."
    factual_errors = data.get("factual_errors", [])
    improvements = data.get("suggested_improvements", [])
    return {
        "confidence": confidence,
        "factual_errors": [str(e) for e in (factual_errors if isinstance(factual_errors, list) else [])[:5]],
        "suggested_improvements": [str(s) for s in (improvements if isinstance(improvements, list) else [])[:3]],
    }, None


_RC_SCHEMA: dict = {
    "headline":           (str,  None, None),
    "strengths":          (list, 3,    3),
    "weaknesses":         (list, 2,    2),
    "race_craft_rating":  (int,  1,    10),
    "raw_pace_rating":    (int,  1,    10),
    "consistency_rating": (int,  1,    10),
    "tactical_tips":      (list, 2,    2),
    "style_tags":         (list, 3,    5),
}


def _rc_validate_report(data: dict) -> tuple:
    for key, (expected_type, lo, hi) in _RC_SCHEMA.items():
        if key not in data:
            return False, f"Missing required key: '{key}'"
        val = data[key]
        if expected_type == int:
            try:
                int_val = int(val)
            except (TypeError, ValueError):
                return False, f"Key '{key}' must be integer, got {type(val).__name__}"
            if lo is not None and not (lo <= int_val <= hi):
                return False, f"Key '{key}' value {int_val} outside range [{lo}, {hi}]"
            data[key] = int_val
        elif expected_type == list:
            if not isinstance(val, list):
                return False, f"Key '{key}' must be a list, got {type(val).__name__}"
            if lo is not None and len(val) < lo:
                return False, f"Key '{key}' has {len(val)} items, minimum is {lo}"
            if hi is not None and len(val) > hi:
                data[key] = val[:hi]
            for i, item in enumerate(data[key]):
                if not isinstance(item, str):
                    return False, f"Key '{key}[{i}]' must be a string"
                data[key][i] = item.replace("<", "").replace(">", "")
        elif expected_type == str:
            if not isinstance(val, str):
                return False, f"Key '{key}' must be a string, got {type(val).__name__}"
            data[key] = val.replace("<", "").replace(">", "")
    return True, ""


_CA_MAX_INPUT_LEN = 500

_CA_TOOL_ARG_SCHEMAS: dict[str, dict] = {
    "get_rolling_pace":        {"window": (int, 2, 15), "top_n": (int, 1, 20)},
    "detect_strategy_events":  {"lookahead": (int, 1, 10)},
    "project_finishing_order": {"laps_remaining": (int, 1, 80)},
}


def _ca_sanitize_input(text: str) -> tuple:
    text = text.strip()
    if not text:
        return "", "Please enter a question."
    if len(text) > _CA_MAX_INPUT_LEN:
        return "", f"Message too long ({len(text)} chars). Please keep it under {_CA_MAX_INPUT_LEN} characters."
    cleaned = "".join(ch for ch in text if ch in ("\n", "\t") or ord(ch) >= 0x20)
    if not cleaned:
        return "", "Message contained only invalid characters."
    return cleaned, None


def _ca_validate_tool_args(tool_name: str, args: dict) -> tuple:
    schema = _CA_TOOL_ARG_SCHEMAS.get(tool_name, {})
    cleaned: dict = {}
    for key, (typ, lo, hi) in schema.items():
        raw = args.get(key)
        if raw is None:
            cleaned[key] = lo
            continue
        try:
            val = typ(raw)
        except (ValueError, TypeError):
            return {}, f"Tool '{tool_name}': argument '{key}' must be {typ.__name__}, got {raw!r}"
        cleaned[key] = max(lo, min(val, hi))
    return cleaned, None


def _rag_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


# ===========================================================================
# _ra_parse_critique
# ===========================================================================

class TestRaParseCritique:
    def test_valid_json_returns_dict(self):
        raw = json.dumps({"confidence": 8, "factual_errors": [], "suggested_improvements": ["be more specific"]})
        result, err = _ra_parse_critique(raw)
        assert err is None
        assert result["confidence"] == 8
        assert result["suggested_improvements"] == ["be more specific"]

    def test_malformed_json_returns_error(self):
        result, err = _ra_parse_critique("{not valid json")
        assert result is None
        assert "non-JSON" in err

    def test_empty_string_returns_error(self):
        result, err = _ra_parse_critique("")
        assert result is None
        assert err is not None

    def test_confidence_clamped_to_maximum(self):
        raw = json.dumps({"confidence": 99, "factual_errors": [], "suggested_improvements": []})
        result, err = _ra_parse_critique(raw)
        assert err is None
        assert result["confidence"] == 10

    def test_confidence_clamped_to_minimum(self):
        raw = json.dumps({"confidence": -5, "factual_errors": [], "suggested_improvements": []})
        result, err = _ra_parse_critique(raw)
        assert err is None
        assert result["confidence"] == 1

    def test_factual_errors_capped_at_five(self):
        errors = [f"error {i}" for i in range(10)]
        raw = json.dumps({"confidence": 5, "factual_errors": errors, "suggested_improvements": []})
        result, err = _ra_parse_critique(raw)
        assert err is None
        assert len(result["factual_errors"]) == 5


# ===========================================================================
# _rc_validate_report
# ===========================================================================

class TestRcValidateReport:
    def _valid_report(self) -> dict:
        return {
            "headline": "Driver X delivers 190 km/h average with trail brake score 0.12.",
            "strengths": ["Trail braking 0.12", "Full throttle 85%", "Corner speed 150 km/h"],
            "weaknesses": ["Brake events 25/lap", "Coasting fraction 0.08"],
            "race_craft_rating": 8,
            "raw_pace_rating": 7,
            "consistency_rating": 9,
            "tactical_tips": ["Reduce coasting in sector 2", "Increase trail braking depth"],
            "style_tags": ["Late Braker", "Smooth", "High Speed"],
        }

    def test_valid_report_passes(self):
        valid, reason = _rc_validate_report(self._valid_report())
        assert valid is True
        assert reason == ""

    def test_missing_key_rejected(self):
        report = self._valid_report()
        del report["headline"]
        valid, reason = _rc_validate_report(report)
        assert valid is False
        assert "headline" in reason

    def test_out_of_range_integer_rejected(self):
        report = self._valid_report()
        report["race_craft_rating"] = 11  # max is 10
        valid, reason = _rc_validate_report(report)
        assert valid is False
        assert "race_craft_rating" in reason

    def test_list_too_short_rejected(self):
        report = self._valid_report()
        report["strengths"] = ["only one"]  # needs exactly 3
        valid, reason = _rc_validate_report(report)
        assert valid is False
        assert "strengths" in reason

    def test_list_trimmed_when_too_long(self):
        report = self._valid_report()
        report["style_tags"] = ["a", "b", "c", "d", "e", "f"]  # max is 5
        valid, reason = _rc_validate_report(report)
        assert valid is True
        assert len(report["style_tags"]) == 5

    def test_html_tags_stripped_from_strings(self):
        report = self._valid_report()
        report["headline"] = "Driver <script>alert(1)</script> is fast."
        valid, _ = _rc_validate_report(report)
        assert valid is True
        assert "<" not in report["headline"]
        assert ">" not in report["headline"]


# ===========================================================================
# _ca_sanitize_input
# ===========================================================================

class TestCaSanitizeInput:
    def test_valid_input_passes_through(self):
        text, err = _ca_sanitize_input("Who is the fastest driver?")
        assert err is None
        assert text == "Who is the fastest driver?"

    def test_over_limit_input_rejected(self):
        text, err = _ca_sanitize_input("x" * 501)
        assert text == ""
        assert err is not None
        assert "too long" in err

    def test_exactly_max_length_accepted(self):
        text, err = _ca_sanitize_input("a" * 500)
        assert err is None
        assert len(text) == 500

    def test_control_characters_stripped(self):
        text, err = _ca_sanitize_input("Hello\x01\x02World")
        assert err is None
        assert "\x01" not in text
        assert "\x02" not in text
        assert "Hello" in text and "World" in text

    def test_whitespace_only_rejected(self):
        text, err = _ca_sanitize_input("   \t  ")
        assert text == ""
        assert err is not None

    def test_newlines_and_tabs_preserved(self):
        text, err = _ca_sanitize_input("line1\nline2\ttabbed")
        assert err is None
        assert "\n" in text
        assert "\t" in text


# ===========================================================================
# _ca_validate_tool_args
# ===========================================================================

class TestCaValidateToolArgs:
    def test_clamps_window_above_maximum(self):
        cleaned, err = _ca_validate_tool_args("get_rolling_pace", {"window": 99, "top_n": 5})
        assert err is None
        assert cleaned["window"] == 15

    def test_clamps_window_below_minimum(self):
        cleaned, err = _ca_validate_tool_args("get_rolling_pace", {"window": 0, "top_n": 5})
        assert err is None
        assert cleaned["window"] == 2

    def test_valid_args_pass_through_unchanged(self):
        cleaned, err = _ca_validate_tool_args("get_rolling_pace", {"window": 5, "top_n": 8})
        assert err is None
        assert cleaned["window"] == 5
        assert cleaned["top_n"] == 8

    def test_unknown_tool_returns_empty_dict(self):
        cleaned, err = _ca_validate_tool_args("unknown_tool", {"any": "arg"})
        assert err is None
        assert cleaned == {}

    def test_missing_arg_defaults_to_lower_bound(self):
        cleaned, err = _ca_validate_tool_args("get_rolling_pace", {})
        assert err is None
        assert cleaned["window"] == 2   # lower bound
        assert cleaned["top_n"] == 1   # lower bound

    def test_non_integer_arg_returns_error(self):
        cleaned, err = _ca_validate_tool_args("get_rolling_pace", {"window": "fast"})
        assert cleaned == {}
        assert err is not None


# ===========================================================================
# _rag_cosine_similarity
# ===========================================================================

class TestRagCosineSimilarity:
    def test_identical_vectors_return_one(self):
        v = np.array([1.0, 0.5, 0.3, 0.7])
        assert abs(_rag_cosine_similarity(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors_return_zero(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert abs(_rag_cosine_similarity(v1, v2)) < 1e-9

    def test_zero_vector_returns_zero(self):
        v1 = np.zeros(4)
        v2 = np.array([1.0, 0.5, 0.3, 0.7])
        assert _rag_cosine_similarity(v1, v2) == 0.0

    def test_similarity_is_symmetric(self):
        v1 = np.array([0.8, 0.3, 0.5])
        v2 = np.array([0.4, 0.9, 0.2])
        diff = abs(_rag_cosine_similarity(v1, v2) - _rag_cosine_similarity(v2, v1))
        assert diff < 1e-9

    def test_similarity_is_within_valid_range(self):
        v1 = np.array([0.8, 0.3, 0.5, 0.2])
        v2 = np.array([0.4, 0.9, 0.2, 0.7])
        sim = _rag_cosine_similarity(v1, v2)
        assert 0.0 <= sim <= 1.0
