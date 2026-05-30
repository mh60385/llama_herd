from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .config import data_path
from .utils import read_json


def _jaccard_similarity(s1: str, s2: str) -> float:
    """Placeholder for semantic similarity - uses Jaccard on word sets.
    
    TODO: Replace with sentence-transformers when ready:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode([s1, s2])
        return float(embeddings[0].dot(embeddings[1]))
    """
    set1 = set(s1.lower().split())
    set2 = set(s2.lower().split())
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def _pairwise_drift_scores(texts: list[str]) -> list[float]:
    """Calculate drift scores between consecutive texts.
    Drift = 1 - similarity, so higher = more different."""
    if len(texts) < 2:
        return []
    scores = []
    for i in range(len(texts) - 1):
        similarity = _jaccard_similarity(texts[i], texts[i + 1])
        drift = 1.0 - similarity
        scores.append(drift)
    return scores


def _mean_drift(scores: list[float]) -> float:
    """Mean drift score across all consecutive pairs."""
    return sum(scores) / len(scores) if scores else 0.0


def _drift_trend(drift: float) -> str:
    """Classify drift trend based on corrected thresholds."""
    if drift < 0.3:
        return "stuck"  # Low drift = repeating, stuck in loop
    elif drift < 0.7:
        return "exploring"  # Medium drift = normal exploration
    else:
        return "diverse"  # High drift = jumping between unrelated topics


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def analyze() -> str:
    profiles = sorted(data_path("profiles").glob("*.json"))
    episodes = load_jsonl(data_path("logs", "episodes.jsonl"))
    lines = ["World Model Drift Summary", ""]
    lines.append("Current interests by agent:")
    for path in profiles:
        profile = read_json(path, {})
        lines.append(f"- {profile.get('agent_id')}: {profile.get('current_interests', [])}")
    lines.append("")
    lines.append("Preferred sources by agent:")
    for path in profiles:
        profile = read_json(path, {})
        lines.append(f"- {profile.get('agent_id')}: {profile.get('preferred_sources', [])}")
    
    # Extract data for semantic analysis
    queries = [row.get("search_query", "") for row in episodes if row.get("search_query")]
    diaries = []
    for row in episodes:
        de = row.get("diary_entry", {})
        if de:
            # Use diary_summary as primary, fall back to any available field
            summary = de.get("diary_summary", "") or de.get("what_caught_attention", "") or ""
            if summary:
                diaries.append(summary)
    
    # Calculate semantic drift metrics
    query_scores = _pairwise_drift_scores(queries)
    diary_scores = _pairwise_drift_scores(diaries)
    query_drift = _mean_drift(query_scores)
    diary_drift = _mean_drift(diary_scores)
    
    terms = Counter(word.lower().strip(".,:;!?()[]") for query in queries for word in query.split() if len(word) > 3)
    sources = Counter()
    errors = Counter()
    versions = Counter()
    for row in episodes:
        versions[row.get("agent_id", "unknown")] = max(
            versions[row.get("agent_id", "unknown")], int(row.get("profile_version_after") or 0)
        )
        for result in row.get("search_results", []):
            source = result.get("source") or result.get("backend") or "unknown"
            sources[source] += 1
        for error in row.get("errors", []):
            errors[error.get("stage", "unknown")] += 1
    
    # Semantic drift section
    lines.extend([
        "",
        "--- Semantic Drift Analysis (placeholder: Jaccard similarity) ---",
        f"Query drift score: {query_drift:.3f} [{_drift_trend(query_drift)}]",
        f"Diary drift score: {diary_drift:.3f} [{_drift_trend(diary_drift)}]",
        "",
        "  Interpretation:",
        "  - stuck (<0.3): Agent is repeating/reusing queries",
        "  - exploring (0.3-0.7): Normal topic evolution", 
        "  - diverse (>0.7): Agent is jumping between unrelated topics",
        "",
    ])
    
    lines.extend(
        [
            "--- Traditional Metrics ---",
            f"Repeated search terms: {terms.most_common(15)}",
            f"Number of profile versions: {dict(versions)}",
            f"Most common sources: {sources.most_common(10)}",
            f"Error counts: {dict(errors)}",
            "",
            "Rough divergence notes:",
        ]
    )
    if len(profiles) <= 1:
        lines.append("- Single-profile mode: inspect topic and source drift across versions rather than between agents.")
    else:
        lines.append("- Compare interests and preferred sources above for between-agent divergence.")
    if queries:
        lines.append(f"- Recorded {len(queries)} search queries; repeated terms indicate topic stickiness.")
    else:
        lines.append("- No completed search queries yet.")
    
    # TODO replacement note
    lines.extend([
        "",
        "Semantic drift TODO: Replace Jaccard with sentence-transformers for production use.",
    ])
    
    return "\n".join(lines)
