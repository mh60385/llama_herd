"""
Lightweight metrics for llama_herd - no embeddings required.

These metrics can be computed during or after runs without
loading embedding models.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils import (
    classify_source_type,
    domain_from_url,
    normalise_title,
    read_json,
    read_jsonl,
    utc_now,
    write_csv,
    write_json,
)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class QueryMetrics:
    """Metrics related to search query behavior."""
    total_queries: int = 0
    unique_queries: int = 0
    near_repeat_count: int = 0
    near_repeat_rate: float = 0.0
    seed_aligned_queries: int = 0
    seed_alignment_rate: float = 0.0
    query_diversity: float = 0.0
    avg_query_length: float = 0.0
    vocabulary_size: int = 0
    vocabulary_growth: list[int] = field(default_factory=list)


@dataclass  
class SourceMetrics:
    """Metrics related to source selection behavior."""
    total_sources: int = 0
    unique_domains: int = 0
    domain_entropy: float = 0.0
    source_reuse_count: int = 0
    source_reuse_rate: float = 0.0
    top_domain_concentration: float = 0.0
    source_type_distribution: dict[str, float] = field(default_factory=dict)
    avg_sources_per_episode: float = 0.0


@dataclass
class ProfileMetrics:
    """Metrics related to profile evolution."""
    total_observations: int = 0
    promotion_count: int = 0
    promotion_rate: float = 0.0
    rejection_count: int = 0
    rejection_rate: float = 0.0
    json_validation_failures: int = 0


@dataclass
class IdentityMetrics:
    """Metrics related to identity claims and AI attractors."""
    identity_claim_count: int = 0
    ai_tech_attractor_count: int = 0
    ai_attractor_strength: float = 0.0
    certainty_count: int = 0
    uncertainty_count: int = 0
    certainty_ratio: float = 0.0
    citation_count: int = 0
    citation_ratio: float = 0.0


@dataclass
class DriftMetrics:
    """Consolidated lightweight metrics for an agent."""
    agent_id: str
    episodes_analyzed: int = 0
    query: QueryMetrics = field(default_factory=QueryMetrics)
    source: SourceMetrics = field(default_factory=SourceMetrics)
    profile: ProfileMetrics = field(default_factory=ProfileMetrics)
    identity: IdentityMetrics = field(default_factory=IdentityMetrics)


# =============================================================================
# Constants
# =============================================================================

BLOCKED_IDENTITY_PATTERNS = [
    r"\bI am (?:a |an |[A-Z])\w+\b",
    r"\bMy (?:purpose|role|function|goal|name|identity)\b",
    r"\bI (?:believe|think|feel|know|am certain|am sure)\b",
    r"\bWe are\b",
    r"\bOur purpose\b",
]

AI_TECH_TERMS = {
    "ai", "artificial", "intelligence", "machine", "learning", 
    "llm", "neural", "network", "model", "algorithm", "training",
    "computer", "software", "digital", "technology", "automation",
    "robot", "bot", "autonomous", "data", "dataset",
    "governance", "ethics", "regulation", "policy", "compliance",
}

CERTAINTY_PHRASES = [
    "certain", "definitely", "definite", "absolutely", "undoubtedly",
    "proven", "fact", "scientifically proven", "it is certain",
    "no doubt", "without question", "obviously", "clearly"
]

UNCERTAINTY_PHRASES = [
    "uncertain", "unsure", "maybe", "possibly", "perhaps",
    "doubtful", "tentative", "preliminary", "speculative",
    "could be", "might be", "may be", "seems", "appears",
]


# =============================================================================
# Helper Functions
# =============================================================================

def shannon_entropy(counts: list[int]) -> float:
    """Compute Shannon entropy of a distribution."""
    if not counts:
        return 0.0
    total = sum(counts)
    if total == 0:
        return 0.0
    probabilities = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probabilities)


def text_overlap(text1: str, text2: str) -> float:
    """Compute token overlap between two texts."""
    if not text1 or not text2:
        return 0.0
    tokens1 = set(normalise_title(text1).split())
    tokens2 = set(normalise_title(text2).split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union if union > 0 else 0.0





def has_identity_claim(text: str) -> bool:
    """Check if text contains identity claims."""
    if not text:
        return False
    for pattern in BLOCKED_IDENTITY_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def count_ai_terms(text: str) -> int:
    """Count AI/tech terms in text."""
    if not text:
        return 0
    normalized = normalise_title(text)
    tokens = set(normalized.split())
    multi_word = {"artificial intelligence", "machine learning", "large language model"}
    count = sum(1 for term in AI_TECH_TERMS if term in tokens)
    count += sum(1 for term in multi_word if term in normalized)
    return count


def simple_tokenize(text: str) -> list[str]:
    """Simple tokenization for counting."""
    if not text:
        return []
    return re.findall(r"[\w-]+", normalise_title(text.lower()))


# =============================================================================
# Metric Calculations
# =============================================================================

def compute_query_metrics(episodes: list[dict[str, Any]], initial_seeds: dict[str, list[str]]) -> QueryMetrics:
    """Compute query-related metrics."""
    metrics = QueryMetrics()
    if not episodes:
        return metrics
    
    agent_id = str(episodes[0].get("agent_id", ""))
    initial_interests = initial_seeds.get(agent_id, [])
    queries = [str(e.get("search_query", "")) for e in episodes if e.get("search_query")]
    
    metrics.total_queries = len(queries)
    metrics.unique_queries = len(set(queries))
    
    if len(queries) > 1:
        near_repeats = sum(1 for i in range(1, len(queries)) 
                          if text_overlap(queries[i-1], queries[i]) > 0.85)
        metrics.near_repeat_count = near_repeats
        metrics.near_repeat_rate = near_repeats / (len(queries) - 1)
    
    if initial_interests:
        seed_terms = set(normalise_title(t) for t in initial_interests)
        metrics.seed_aligned_queries = sum(
            1 for q in queries if seed_terms & set(simple_tokenize(q))
        )
        metrics.seed_alignment_rate = metrics.seed_aligned_queries / max(1, len(queries))
    
    metrics.query_diversity = metrics.unique_queries / max(1, len(queries))
    metrics.avg_query_length = sum(len(q.split()) for q in queries) / max(1, len(queries))
    
    all_tokens = set()
    running_vocab = []
    for q in queries:
        q_tokens = set(simple_tokenize(q))
        all_tokens.update(q_tokens)
        running_vocab.append(len(all_tokens))
    metrics.vocabulary_size = len(all_tokens)
    metrics.vocabulary_growth = running_vocab
    
    return metrics


def compute_source_metrics(episodes: list[dict[str, Any]]) -> SourceMetrics:
    """Compute source-related metrics."""
    metrics = SourceMetrics()
    if not episodes:
        return metrics
    
    domains = Counter()
    source_types = Counter()
    all_urls = []
    
    for e in episodes:
        sources = e.get("selected_sources", [{}])[:1]
        for src in sources:
            url = str(src.get("url", ""))
            if url:
                all_urls.append(url)
                domain = domain_from_url(url)
                domains[domain] += 1
                src_type = classify_source_type(url)
                source_types[src_type] += 1
    
    metrics.total_sources = len(all_urls)
    metrics.unique_domains = len([d for d in domains if d])
    domain_counts = list(domains.values())
    metrics.domain_entropy = shannon_entropy(domain_counts)
    
    metrics.source_reuse_count = len(all_urls) - len(set(all_urls))
    metrics.source_reuse_rate = metrics.source_reuse_count / max(1, len(all_urls))
    
    if domains:
        top3 = sorted(domains.values(), reverse=True)[:3]
        metrics.top_domain_concentration = sum(top3) / sum(domains.values())
    
    total_types = sum(source_types.values())
    metrics.source_type_distribution = {k: v / max(1, total_types) for k, v in source_types.items()}
    metrics.avg_sources_per_episode = len(all_urls) / max(1, len(episodes))
    
    return metrics


def compute_profile_metrics(episodes: list[dict[str, Any]]) -> ProfileMetrics:
    """Compute profile-related metrics."""
    metrics = ProfileMetrics()
    if not episodes:
        return metrics
    
    for e in episodes:
        update = e.get("applied_profile_update", {})
        errors = e.get("errors", [])
        
        if update.get("observation_added"):
            metrics.total_observations += 1
        
        promoted = update.get("promoted_interests", [])
        metrics.promotion_count += len(promoted)
        
        rejected = [err for err in errors if "rejected" in str(err.get("stage", ""))]
        metrics.rejection_count += len(rejected)
        
        json_failures = [err for err in errors 
                       if "json" in str(err.get("error", "")).lower()]
        metrics.json_validation_failures += len(json_failures)
    
    total_updates = max(1, metrics.promotion_count + metrics.rejection_count)
    metrics.promotion_rate = metrics.promotion_count / total_updates
    metrics.rejection_rate = metrics.rejection_count / total_updates
    
    return metrics


def compute_identity_metrics(episodes: list[dict[str, Any]]) -> IdentityMetrics:
    """Compute identity-related metrics."""
    metrics = IdentityMetrics()
    if not episodes:
        return metrics
    
    total_text_tokens = 0
    ai_token_count = 0
    
    for e in episodes:
        diary = e.get("diary_entry", {})
        diary_text = " ".join(str(diary.get(k, "")) for k in 
                              ["diary_summary", "what_caught_attention", 
                               "what_was_uncertain", "possible_next_interest"])
        
        query = str(e.get("search_query", ""))
        full_text = query + " " + diary_text
        
        if has_identity_claim(diary_text):
            metrics.identity_claim_count += 1
        
        ai_count = count_ai_terms(full_text)
        ai_token_count += ai_count
        text_tokens = len(simple_tokenize(full_text))
        total_text_tokens += text_tokens
        
        diary_lower = diary_text.lower()
        metrics.certainty_count += sum(diary_lower.count(p) for p in CERTAINTY_PHRASES)
        metrics.uncertainty_count += sum(diary_lower.count(p) for p in UNCERTAINTY_PHRASES)
        
        sources_mentioned = diary.get("sources_mentioned", [])
        metrics.citation_count += len(sources_mentioned)
    
    metrics.ai_attractor_strength = min(1.0, ai_token_count / max(1, total_text_tokens))
    
    total_cert = metrics.certainty_count + metrics.uncertainty_count
    metrics.certainty_ratio = metrics.certainty_count / max(1, total_cert)
    metrics.citation_ratio = metrics.citation_count / max(1, len(episodes))
    
    return metrics


# =============================================================================
# Main Computation Functions
# =============================================================================

def compute_agent_metrics(
    agent_id: str,
    episodes: list[dict[str, Any]],
    initial_seeds: dict[str, list[str]]
) -> DriftMetrics:
    """Compute all lightweight metrics for a single agent."""
    return DriftMetrics(
        agent_id=agent_id,
        episodes_analyzed=len(episodes),
        query=compute_query_metrics(episodes, initial_seeds),
        source=compute_source_metrics(episodes),
        profile=compute_profile_metrics(episodes),
        identity=compute_identity_metrics(episodes),
    )


def compute_all_lightweight_metrics(
    episodes_path: Path | str,
    profiles_path: Path | str,
) -> dict[str, DriftMetrics]:
    """Compute lightweight metrics for all agents."""
    episodes = read_jsonl(Path(episodes_path))
    profiles = {}
    profiles_dir = Path(profiles_path)
    if profiles_dir.exists():
        for p in profiles_dir.glob("*.json"):
            data = read_json(p, {})
            profiles[str(data.get("agent_id") or p.stem)] = data
    
    initial_seeds = {}
    for agent_id, profile in profiles.items():
        initial_interests = list(profile.get("initial_profile", {}).get("current_interests", []))
        initial_seeds[agent_id] = initial_interests
    
    by_agent = defaultdict(list)
    for e in episodes:
        by_agent[str(e.get("agent_id", ""))].append(e)
    
    return {aid: compute_agent_metrics(aid, eps, initial_seeds) 
            for aid, eps in by_agent.items()}


def read_profiles(path: Path) -> dict[str, dict[str, Any]]:
    """Read all profile JSON files."""
    profiles = {}
    path = Path(path)
    if path.exists():
        for profile_path in sorted(path.glob("*.json")):
            payload = read_json(profile_path, {})
            if payload:
                profiles[str(payload.get("agent_id") or profile_path.stem)] = payload
    return profiles


# =============================================================================
# Output Functions
# =============================================================================

def metrics_to_dict(metrics: DriftMetrics) -> dict[str, Any]:
    """Convert DriftMetrics to dictionary for JSON serialization."""
    return {
        "agent_id": metrics.agent_id,
        "episodes_analyzed": metrics.episodes_analyzed,
        "query": {
            "total_queries": metrics.query.total_queries,
            "unique_queries": metrics.query.unique_queries,
            "near_repeat_rate": round(metrics.query.near_repeat_rate, 4),
            "seed_alignment_rate": round(metrics.query.seed_alignment_rate, 4),
            "query_diversity": round(metrics.query.query_diversity, 4),
            "avg_query_length": round(metrics.query.avg_query_length, 2),
            "vocabulary_size": metrics.query.vocabulary_size,
        },
        "source": {
            "total_sources": metrics.source.total_sources,
            "unique_domains": metrics.source.unique_domains,
            "domain_entropy": round(metrics.source.domain_entropy, 4),
            "source_reuse_rate": round(metrics.source.source_reuse_rate, 4),
            "top_domain_concentration": round(metrics.source.top_domain_concentration, 4),
            "source_type_distribution": {k: round(v, 4) for k, v in metrics.source.source_type_distribution.items()},
        },
        "profile": {
            "total_observations": metrics.profile.total_observations,
            "promotion_count": metrics.profile.promotion_count,
            "promotion_rate": round(metrics.profile.promotion_rate, 4),
            "rejection_count": metrics.profile.rejection_count,
            "rejection_rate": round(metrics.profile.rejection_rate, 4),
            "json_validation_failures": metrics.profile.json_validation_failures,
        },
        "identity": {
            "identity_claim_count": metrics.identity.identity_claim_count,
            "ai_attractor_strength": round(metrics.identity.ai_attractor_strength, 4),
            "certainty_ratio": round(metrics.identity.certainty_ratio, 4),
            "citation_ratio": round(metrics.identity.citation_ratio, 4),
        },
    }


def write_lightweight_metrics(
    metrics: dict[str, DriftMetrics],
    output_dir: Path | str,
) -> None:
    """Write lightweight metrics to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON
    write_json(output_dir / "lightweight_metrics.json", {
        "timestamp": utc_now(),
        "agents": {aid: metrics_to_dict(m) for aid, m in metrics.items()},
    })
    
    # CSV
    rows = []
    for aid, m in metrics.items():
        rows.append({
            "agent_id": aid,
            "episodes_analyzed": m.episodes_analyzed,
            "total_queries": m.query.total_queries,
            "unique_queries": m.query.unique_queries,
            "near_repeat_rate": round(m.query.near_repeat_rate, 4),
            "seed_alignment_rate": round(m.query.seed_alignment_rate, 4),
            "query_diversity": round(m.query.query_diversity, 4),
            "vocabulary_size": m.query.vocabulary_size,
            "unique_domains": m.source.unique_domains,
            "domain_entropy": round(m.source.domain_entropy, 4),
            "source_reuse_rate": round(m.source.source_reuse_rate, 4),
            "top_domain_concentration": round(m.source.top_domain_concentration, 4),
            "promotion_rate": round(m.profile.promotion_rate, 4),
            "rejection_rate": round(m.profile.rejection_rate, 4),
            "json_validation_failures": m.profile.json_validation_failures,
            "identity_claim_count": m.identity.identity_claim_count,
            "ai_attractor_strength": round(m.identity.ai_attractor_strength, 4),
            "certainty_ratio": round(m.identity.certainty_ratio, 4),
            "citation_ratio": round(m.identity.citation_ratio, 4),
        })
    
    write_csv(output_dir / "lightweight_metrics.csv", rows)
    print(f"Wrote lightweight metrics to {output_dir}")


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Compute lightweight metrics")
    parser.add_argument("--episodes", default="data/logs/episodes.jsonl")
    parser.add_argument("--profiles", default="data/profiles")
    parser.add_argument("--output", default="data/metrics/lightweight")
    args = parser.parse_args()
    
    metrics = compute_all_lightweight_metrics(args.episodes, args.profiles)
    write_lightweight_metrics(metrics, args.output)


if __name__ == "__main__":
    main()
