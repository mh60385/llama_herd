#!/usr/bin/env python3
"""Compare LLM agents against baseline agents.

Generates comparison metrics to quantify whether LLM agents
are performing better than simple baselines.

Usage:
    python scripts/compare_baselines.py --episodes 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baselines import BASELINES, get_baseline_queries
from src.drift_analysis import (
    _drift_trend,
    _jaccard_similarity,
    _mean_drift,
    _pairwise_drift_scores,
    load_jsonl,
)
from src.config import data_path
from src.utils import read_json


def compute_drift_scores(texts: list[str]) -> tuple[float, str]:
    """Compute mean drift score and trend for a list of texts."""
    scores = _pairwise_drift_scores(texts)
    mean_drift = _mean_drift(scores)
    trend = _drift_trend(mean_drift)
    return mean_drift, trend


def get_agent_queries(agent_id: str) -> list[str]:
    """Get search queries from episode logs for a specific agent."""
    episodes = load_jsonl(data_path("logs", "episodes.jsonl"))
    return [e.get("search_query", "") for e in episodes if e.get("agent_id") == agent_id]


def run_comparison(episodes: int = 20) -> dict[str, Any]:
    """Run baseline comparison against actual agents.
    
    Args:
        episodes: Number of episodes to compare
        
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    # Get actual agent queries
    profiles = list(data_path("profiles").glob("*.json"))
    for profile_path in profiles:
        agent_id = profile_path.stem
        queries = get_agent_queries(agent_id)
        if len(queries) >= 2:
            drift, trend = compute_drift_scores(queries[:episodes])
            results[agent_id] = {
                "type": "llm",
                "query_count": len(queries),
                "drift_score": drift,
                "trend": trend,
                "queries": queries[:5] + ["..."] if len(queries) > 5 else queries,
            }
    
    # Run baselines
    baseline_seeds = {
        "random": "baseline_random:llama-herd:2026-01-01",
        "static": "baseline_static:llama-herd:2026-01-01",
        "echo": "baseline_echo:llama-herd:2026-01-01",
        "repeating": "baseline_repeating:llama-herd:2026-01-01",
    }
    
    for baseline_type, seed in baseline_seeds.items():
        queries = get_baseline_queries(baseline_type, "baseline_agent", seed, episodes)
        drift, trend = compute_drift_scores(queries)
        results[f"baseline_{baseline_type}"] = {
            "type": "baseline",
            "baseline_type": baseline_type,
            "query_count": len(queries),
            "drift_score": drift,
            "trend": trend,
            "queries": queries[:5] + ["..."] if len(queries) > 5 else queries,
        }
    
    return results


def print_comparison_table(results: dict[str, Any]) -> None:
    """Print comparison results as a formatted table."""
    print("\n" + "=" * 100)
    print("BASELINE COMPARISON (Semantic Drift Scores)")
    print("=" * 100)
    print()
    
    # Sort by drift score descending
    sorted_results = sorted(results.items(), key=lambda x: x[1]["drift_score"], reverse=True)
    
    print(f"{'Agent':<25} {'Type':<12} {'Drift':<12} {'Trend':<10} {'Queries'}")
    print("-" * 100)
    
    for name, data in sorted_results:
        agent_type = data["type"]
        drift = f"{data['drift_score']:.3f}"
        trend = data["trend"]
        query_preview = ", ".join(data["queries"][:2])
        if len(query_preview) > 40:
            query_preview = query_preview[:37] + "..."
        print(f"{name:<25} {agent_type:<12} {drift:<12} {trend:<10} {query_preview}")
    
    print()
    print("Interpretation:")
    print("  - stuck (<0.3): Agent is repeating/reusing queries")
    print("  - exploring (0.3-0.7): Normal topic evolution")
    print("  - diverse (>0.7): Agent is jumping between unrelated topics")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare LLM agents against baseline agents."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of episodes to compare (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (optional)",
    )
    args = parser.parse_args()
    
    results = run_comparison(args.episodes)
    print_comparison_table(results)
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
