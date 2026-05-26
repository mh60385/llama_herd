#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import data_path
from src.utils import utc_now, write_json


EMBODIED_PATTERNS = [
    r"\bi visited\b",
    r"\bi went\b",
    r"\bi traveled\b",
    r"\bi travelled\b",
    r"\bi attended\b",
    r"\bi saw\b",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Write research-ready metrics from llama_herd episode logs.")
    parser.add_argument("--episodes", default=str(data_path("logs", "episodes.jsonl")))
    parser.add_argument("--out-dir", default=str(data_path("metrics")))
    args = parser.parse_args()

    episodes = read_jsonl(Path(args.episodes))
    metrics = build_metrics(episodes)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "research_metrics.json"
    md_path = out_dir / "research_metrics.md"
    write_json(json_path, metrics)
    md_path.write_text(render_markdown(metrics), encoding="utf-8")
    print(json_path)
    print(md_path)


def build_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        by_agent[str(episode.get("agent_id", ""))].append(episode)
    return {
        "timestamp": utc_now(),
        "total_episodes": len(episodes),
        "agents": {agent_id: agent_metrics(agent_id, rows) for agent_id, rows in sorted(by_agent.items())},
    }


def agent_metrics(agent_id: str, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    selected_domains = Counter()
    source_types = Counter()
    error_stages = Counter()
    query_scores = []
    promoted = Counter()
    rejected_saved = Counter()
    embodied_count = 0
    empty_diary_count = 0
    exact_facts = Counter()
    raw_failures = 0
    raw_repairs = 0

    previous_query = ""
    for episode in episodes:
        selected = (episode.get("selected_sources") or [{}])[0]
        selected_domains[domain_from_url(selected.get("url", ""))] += 1
        source_type = episode.get("source_selection", {}).get("selected_source_type")
        if not source_type:
            source_type = (episode.get("source_summaries") or [{}])[0].get("source_type", "unknown")
        if not source_type or source_type == "unknown":
            source_type = source_type_for_url(selected.get("url", ""))
        source_types[str(source_type or "unknown")] += 1

        query = str(episode.get("search_query", ""))
        if previous_query and query:
            query_scores.append(text_overlap(previous_query, query))
        if query:
            previous_query = query

        diary = episode.get("diary_entry") or {}
        diary_text = " ".join(str(diary.get(key, "")) for key in diary)
        empty_diary_count += int(not str(diary.get("diary_summary", "")).strip())
        embodied_count += int(has_embodied_language(diary_text))
        for fact in number_heavy_sentences(diary_text):
            exact_facts[fact] += 1

        update = episode.get("applied_profile_update") or {}
        promoted.update(str(item) for item in update.get("promoted_interests", []) if str(item).strip())
        for item in update.get("rejected_candidate_interests", []):
            rejected_saved[str(item.get("reason", "unknown"))] += 1
        for item in update.get("single_source_facts", []):
            exact_facts[str(item.get("fact", "")).strip()] += 1

        for error in episode.get("errors", []):
            stage = str(error.get("stage", "unknown"))
            error_stages[stage] += 1
            if stage == "profile_interest_rejected":
                rejected_saved[str(error.get("reason", "unknown"))] += 1

        for raw in episode.get("raw_model_outputs", []):
            raw_failures += int(bool(raw.get("error")))
            raw_repairs += int("repair_for" in raw)

    episode_count = len(episodes)
    repeat_total = sum(count - 1 for domain, count in selected_domains.items() if domain and count > 1)
    repeated_facts = Counter({fact: count for fact, count in exact_facts.items() if count > 1})
    return {
        "episodes": episode_count,
        "unique_selected_domains": len([domain for domain in selected_domains if domain]),
        "selected_domain_entropy_bits": round(entropy(selected_domains), 3),
        "source_repeat_rate": round(repeat_total / episode_count, 3) if episode_count else 0,
        "source_type_counts": dict(source_types.most_common()),
        "top_selected_domains": dict(selected_domains.most_common(12)),
        "avg_adjacent_query_similarity": round(sum(query_scores) / len(query_scores), 3) if query_scores else 0,
        "max_adjacent_query_similarity": round(max(query_scores), 3) if query_scores else 0,
        "accepted_saved_interests": dict(promoted.most_common()),
        "rejected_saved_interests": dict(rejected_saved.most_common()),
        "embodied_diary_language_count": embodied_count,
        "embodied_diary_language_rate": round(embodied_count / episode_count, 3) if episode_count else 0,
        "repeated_exact_fact_count": len(repeated_facts),
        "top_repeated_exact_facts": dict(repeated_facts.most_common(8)),
        "empty_diary_count": empty_diary_count,
        "source_extraction_failure_count": error_stages.get("source_reader", 0),
        "source_extraction_failure_rate": round(error_stages.get("source_reader", 0) / episode_count, 3) if episode_count else 0,
        "model_output_failure_count": raw_failures,
        "model_json_repair_count": raw_repairs,
        "error_stage_counts": dict(error_stages.most_common()),
    }


def render_markdown(metrics: dict[str, Any]) -> str:
    lines = [
        "# llama_herd Research Metrics",
        "",
        f"Updated: `{metrics['timestamp']}`",
        f"Total episodes: `{metrics['total_episodes']}`",
        "",
    ]
    for agent_id, values in metrics["agents"].items():
        lines.extend(
            [
                f"## {agent_id}",
                "",
                f"- Episodes: `{values['episodes']}`",
                f"- Unique selected domains: `{values['unique_selected_domains']}`",
                f"- Selected-domain entropy: `{values['selected_domain_entropy_bits']}` bits",
                f"- Source repeat rate: `{values['source_repeat_rate']}`",
                f"- Average adjacent query similarity: `{values['avg_adjacent_query_similarity']}`",
                f"- Embodied diary language: `{values['embodied_diary_language_count']}` (`{values['embodied_diary_language_rate']}`)",
                f"- Empty diaries: `{values['empty_diary_count']}`",
                f"- Source extraction failures: `{values['source_extraction_failure_count']}` (`{values['source_extraction_failure_rate']}`)",
                f"- Model output failures: `{values['model_output_failure_count']}`",
                f"- Model JSON repairs: `{values['model_json_repair_count']}`",
                f"- Source types: {inline_counter(values['source_type_counts'])}",
                f"- Accepted saved interests: {inline_counter(values['accepted_saved_interests'])}",
                f"- Rejected saved interests: {inline_counter(values['rejected_saved_interests'])}",
                f"- Repeated exact facts: {inline_counter(values['top_repeated_exact_facts'])}",
                "",
            ]
        )
    return "\n".join(lines)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def domain_from_url(url: str) -> str:
    domain = urlparse(str(url or "")).netloc.lower()
    return domain.removeprefix("www.")


def source_type_for_url(url: str) -> str:
    domain = domain_from_url(url)
    path = urlparse(str(url or "")).path.lower()
    if not domain:
        return "unknown"
    if domain in {"facebook.com", "reddit.com", "x.com", "twitter.com", "instagram.com", "youtube.com"}:
        return "social"
    if any(part in domain for part in ["jstor.org", "springer.com", "sciencedirect.com", "ncbi.nlm.nih.gov"]):
        return "academic"
    if domain.endswith(".edu") or ".edu." in domain or domain.endswith(".ac.uk"):
        return "academic"
    if domain.endswith(".gov") or ".gov." in domain or domain.endswith(".int"):
        return "official"
    if any(part in domain for part in ["museum", "unesco.org", "kew.org", "gbif.org", "bgbm.org", "rbge.org.uk"]):
        return "official"
    if any(part in domain for part in ["wikipedia.org", "britannica.com", "ebsco.com"]):
        return "reference"
    if any(part in domain for part in ["news", "magazine", "times", "guardian", "bbc.", "vulture.com"]):
        return "news_or_magazine"
    if any(part in domain for part in ["amazon.", "shop", "store"]) or "/product" in path:
        return "commercial"
    return "unknown"


def tokens(text: str) -> set[str]:
    stop = {"the", "and", "for", "with", "from", "into", "that", "this", "what", "how", "why", "are", "was", "were"}
    return {token for token in re.findall(r"[a-z0-9]{3,}", text.lower()) if token not in stop}


def text_overlap(left: str, right: str) -> float:
    left_tokens = tokens(left)
    right_tokens = tokens(right)
    return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if not total:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counter.values() if count)


def has_embodied_language(text: str) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in EMBODIED_PATTERNS)


def number_heavy_sentences(text: str) -> list[str]:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    return [
        sentence[:240]
        for sentence in sentences
        if re.search(r"\d{1,3}(?:,\d{3})+|\b\d{4,}\b", sentence)
    ]


def inline_counter(values: dict[str, Any]) -> str:
    clean = [(key, value) for key, value in values.items() if str(key).strip()]
    if not clean:
        return "`none`"
    return ", ".join(f"`{key}`: `{value}`" for key, value in clean[:8])


if __name__ == "__main__":
    main()
