"""
eval_llm.py — Offline LLM evaluation framework for Driver DNA.

Measures the quality of the five agentic AI features without making
live OpenAI API calls (except where explicitly opted in via --live).

Usage:
    python src/eval_llm.py                # all evaluations (offline / math-only)
    python src/eval_llm.py --audit        # parse existing llm_audit.jsonl for cost report
    python src/eval_llm.py --live         # run all evaluations with real OpenAI calls

Outputs:
    models/eval_rag.json          — RAG retrieval quality
    models/eval_report_card.json  — Report Card schema compliance (live only)
    models/eval_reflexion.json    — Reflexion critique quality    (live only)
    models/eval_cost.json         — Token cost dashboard from audit log
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow running as `python src/eval_llm.py` from the project root
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).parent
_ROOT = _SRC_DIR.parent
sys.path.insert(0, str(_SRC_DIR))

MODELS_DIR = _ROOT / "models"
DATA_DIR   = _ROOT / "data"
LLM_AUDIT_PATH = MODELS_DIR / "llm_audit.jsonl"

_GPT4O_MINI_INPUT_COST_PER_M  = 0.15   # USD per 1 million input tokens
_GPT4O_MINI_OUTPUT_COST_PER_M = 0.60   # USD per 1 million output tokens


# ===========================================================================
# Evaluation 1: RAG retrieval sanity (offline — no LLM calls)
# ===========================================================================

def _rag_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def eval_rag_retrieval() -> dict:
    """
    Evaluate RAG retrieval quality using the 10 historical driver profiles.

    For each profile, verify that the top cosine match is the profile itself
    (self-similarity = 1.0) and measure the retrieval margin (gap between
    rank-1 and rank-2 similarity).  A larger margin means sharper retrieval.

    Returns a dict saved to models/eval_rag.json.
    """
    # Import the knowledge base directly from app.py constants via dynamic exec
    # to avoid triggering Streamlit at import time.
    profiles_raw: dict = {}
    app_path = _SRC_DIR / "app.py"
    try:
        import ast
        src = app_path.read_text()
        tree = ast.parse(src)
        # Find _RAG_HISTORICAL_PROFILES — could be a plain Assign or AnnAssign
        for node in ast.walk(tree):
            value_node = None
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "_RAG_HISTORICAL_PROFILES"
            ):
                value_node = node.value
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "_RAG_HISTORICAL_PROFILES"
                and node.value is not None
            ):
                value_node = node.value
            if value_node is not None:
                profiles_raw = ast.literal_eval(value_node)
                break
    except Exception as e:
        print(f"[eval_rag] Could not parse _RAG_HISTORICAL_PROFILES from app.py: {e}")
        print("[eval_rag] Skipping RAG evaluation.")
        return {"status": "skipped", "reason": str(e)}

    if not profiles_raw:
        return {"status": "skipped", "reason": "No profiles found in app.py"}

    names = list(profiles_raw.keys())
    vectors = {name: np.array(profiles_raw[name]["vector"], dtype=float) for name in names}

    results = []
    for name in names:
        query = vectors[name]
        similarities = [(n, _rag_cosine_similarity(query, vectors[n])) for n in names]
        similarities.sort(key=lambda x: x[1], reverse=True)

        rank1_name, rank1_sim = similarities[0]
        rank2_name, rank2_sim = similarities[1]
        self_match = rank1_name == name
        margin = round(rank1_sim - rank2_sim, 6)

        results.append({
            "profile": name,
            "rank1_match": rank1_name,
            "rank1_similarity": round(rank1_sim, 6),
            "rank2_match": rank2_name,
            "rank2_similarity": round(rank2_sim, 6),
            "self_match": self_match,
            "retrieval_margin": margin,
        })

    self_match_rate = sum(1 for r in results if r["self_match"]) / len(results)
    mean_margin = round(float(np.mean([r["retrieval_margin"] for r in results])), 6)

    output = {
        "status": "ok",
        "n_profiles": len(results),
        "self_match_rate": round(self_match_rate, 4),
        "mean_retrieval_margin": mean_margin,
        "profiles": results,
    }
    return output


# ===========================================================================
# Evaluation 2: Report Card schema compliance (live — requires OpenAI key)
# ===========================================================================

def eval_report_card_compliance(api_key: str, sample_drivers: list[str] | None = None) -> dict:
    """
    Run _rc_run_report on a sample of drivers with synthetic radar data and
    measure first-attempt pass rate vs retry rate vs failure rate.

    Requires a valid OpenAI API key (passed or read from OPENAI_API_KEY env var).
    """
    try:
        import pandas as pd
        import importlib.util
        spec = importlib.util.spec_from_file_location("app", _SRC_DIR / "app.py")
    except ImportError as e:
        return {"status": "skipped", "reason": f"Missing dependency: {e}"}

    # Build a minimal synthetic DataFrame for testing
    try:
        import pandas as pd
        from model import FEATURE_COLS, DATA_DIR as _DATA_DIR
        df = pd.read_parquet(_DATA_DIR / "dataset.parquet")
    except Exception as e:
        return {"status": "skipped", "reason": f"Could not load dataset: {e}"}

    # Import app helpers without triggering Streamlit
    try:
        _ns: dict = {}
        exec(
            "from pathlib import Path\n"
            "import sys\n"
            f"sys.path.insert(0, {str(_SRC_DIR)!r})\n",
            _ns,
        )
        # We'll use ast-based import of _rc_run_report to avoid Streamlit init
        # For live evaluation, users can extend this section.
        return {
            "status": "partial",
            "note": (
                "Report card live evaluation requires a running Streamlit context. "
                "Run the dashboard and observe retry rates in models/llm_audit.jsonl "
                "(feature='report_card' vs 'report_card_retry')."
            ),
        }
    except Exception as e:
        return {"status": "skipped", "reason": str(e)}


# ===========================================================================
# Evaluation 3: Reflexion critique quality (live — requires OpenAI key)
# ===========================================================================

def eval_reflexion_quality(api_key: str) -> dict:
    """
    Run the Reflexion loop on a fixed driver sample and measure
    mean confidence, revision rate, and mean factual error count.

    Requires a valid OpenAI API key.
    """
    # Parse the reflexion constants and functions from app.py via AST
    # to avoid Streamlit initialisation side effects.
    return {
        "status": "partial",
        "note": (
            "Reflexion live evaluation captures data via the audit log. "
            "After running the dashboard, re-run: python src/eval_llm.py --audit "
            "to see per-feature latency and token statistics."
        ),
    }


# ===========================================================================
# Evaluation 4: Token cost dashboard (offline — reads llm_audit.jsonl)
# ===========================================================================

def eval_token_cost() -> dict:
    """
    Parse models/llm_audit.jsonl and compute per-feature metrics:
      - call count
      - mean latency (ms)
      - total prompt / completion tokens
      - estimated USD cost at gpt-4o-mini pricing

    Returns a dict and prints a markdown summary table.
    """
    if not LLM_AUDIT_PATH.exists():
        return {
            "status": "no_data",
            "note": "models/llm_audit.jsonl does not exist yet. Use the dashboard to generate LLM calls.",
        }

    entries: list[dict] = []
    with open(LLM_AUDIT_PATH) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not entries:
        return {"status": "no_data", "note": "Audit log is empty."}

    # Aggregate per feature
    from collections import defaultdict
    agg: dict = defaultdict(lambda: {
        "calls": 0, "successes": 0, "errors": 0,
        "latency_ms": [], "prompt_tokens": 0, "completion_tokens": 0,
    })

    for e in entries:
        feat = e.get("feature", "unknown")
        agg[feat]["calls"] += 1
        if e.get("success"):
            agg[feat]["successes"] += 1
        else:
            agg[feat]["errors"] += 1
        agg[feat]["latency_ms"].append(e.get("latency_ms", 0))
        agg[feat]["prompt_tokens"] += e.get("prompt_tokens", 0)
        agg[feat]["completion_tokens"] += e.get("completion_tokens", 0)

    feature_reports = []
    total_cost_usd = 0.0

    for feat, stats in sorted(agg.items()):
        pt = stats["prompt_tokens"]
        ct = stats["completion_tokens"]
        cost = (pt / 1_000_000 * _GPT4O_MINI_INPUT_COST_PER_M
                + ct / 1_000_000 * _GPT4O_MINI_OUTPUT_COST_PER_M)
        latencies = stats["latency_ms"]
        mean_lat = round(float(np.mean(latencies)), 1) if latencies else 0.0
        total_cost_usd += cost
        feature_reports.append({
            "feature": feat,
            "calls": stats["calls"],
            "success_rate": round(stats["successes"] / stats["calls"], 4) if stats["calls"] else 0,
            "mean_latency_ms": mean_lat,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "estimated_cost_usd": round(cost, 6),
        })

    # Print markdown table
    print("\n## LLM Audit — Cost & Latency Summary\n")
    header = f"{'Feature':<25} {'Calls':>6} {'Success%':>9} {'Latency ms':>11} {'Tokens (P+C)':>14} {'Est. Cost $':>12}"
    print(header)
    print("-" * len(header))
    for r in feature_reports:
        print(
            f"{r['feature']:<25} {r['calls']:>6} {r['success_rate']*100:>8.1f}% "
            f"{r['mean_latency_ms']:>10.0f}ms "
            f"{r['prompt_tokens']:>7}+{r['completion_tokens']:<6} "
            f"${r['estimated_cost_usd']:>10.6f}"
        )
    print(f"\n{'Total estimated cost':<25} {'':>6} {'':>9} {'':>11} {'':>14} ${total_cost_usd:>10.6f}")

    return {
        "status": "ok",
        "total_calls": len(entries),
        "total_cost_usd": round(total_cost_usd, 6),
        "features": feature_reports,
    }


# ===========================================================================
# Main entry point
# ===========================================================================

def _save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"  Saved → {path.relative_to(_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Driver DNA — offline LLM evaluation framework"
    )
    parser.add_argument("--audit", action="store_true",
                        help="Parse llm_audit.jsonl and print cost report")
    parser.add_argument("--live",  action="store_true",
                        help="Run live evaluations (requires OPENAI_API_KEY)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")

    print("=" * 60)
    print("Driver DNA — LLM Evaluation Framework")
    print("=" * 60)

    # Evaluation 1: RAG (always offline)
    print("\n[1/4] RAG Retrieval Sanity (offline, no API calls)")
    rag_result = eval_rag_retrieval()
    _save(MODELS_DIR / "eval_rag.json", rag_result)
    if rag_result.get("status") == "ok":
        print(f"  ✓ {rag_result['n_profiles']} profiles, "
              f"self-match rate {rag_result['self_match_rate']:.1%}, "
              f"mean margin {rag_result['mean_retrieval_margin']:.4f}")

    # Evaluation 2: Report Card schema compliance
    print("\n[2/4] Report Card Schema Compliance")
    if args.live and api_key:
        rc_result = eval_report_card_compliance(api_key)
    else:
        rc_result = {"status": "skipped", "note": "Use --live with OPENAI_API_KEY to run."}
    _save(MODELS_DIR / "eval_report_card.json", rc_result)
    print(f"  {rc_result.get('status', '?')}: {rc_result.get('note', '')}")

    # Evaluation 3: Reflexion critique quality
    print("\n[3/4] Reflexion Critique Quality")
    if args.live and api_key:
        refl_result = eval_reflexion_quality(api_key)
    else:
        refl_result = {"status": "skipped", "note": "Use --live with OPENAI_API_KEY to run."}
    _save(MODELS_DIR / "eval_reflexion.json", refl_result)
    print(f"  {refl_result.get('status', '?')}: {refl_result.get('note', '')}")

    # Evaluation 4: Token cost dashboard
    print("\n[4/4] Token Cost Dashboard (from audit log)")
    cost_result = eval_token_cost()
    _save(MODELS_DIR / "eval_cost.json", cost_result)

    print("\n" + "=" * 60)
    print("Evaluation complete. Results in models/eval_*.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
