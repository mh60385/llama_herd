#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import data_path
from src.llm_client import LLMClient
from src.utils import append_jsonl, normalise_title, utc_now, write_json


SEEDS = [f"loose_seed_{index:02d}:public-web-curiosity" for index in range(1, 11)]
REQUIRED_FIELDS = {"topics"}
CLUSTER_TERMS = {
    "ai": ["ai", "artificial intelligence", "machine learning", "llm", "automation"],
    "tech": ["software", "computer", "digital", "technology", "app", "data"],
    "governance": ["governance", "ethics", "regulation", "policy"],
    "climate": ["climate", "sustainability", "renewable", "carbon"],
}


def main() -> None:
    client = LLMClient()
    ok, detail = client.healthcheck()
    if not ok:
        raise RuntimeError(f"LLM server unavailable: {detail}")

    jsonl_path = data_path("logs", "loose_seed_topic_smoke.jsonl")
    summary_path = data_path("logs", "loose_seed_topic_smoke_summary.json")
    jsonl_path.unlink(missing_ok=True)

    rows = []
    for seed in SEEDS:
        response = client.chat(loose_topic_prompt(seed), temperature=0, top_p=1, expect_json=True, max_tokens=180)
        topics = [str(item).strip() for item in response.get("topics", []) if str(item).strip()]
        row = {
            "timestamp": utc_now(),
            "model": client.model,
            "seed": seed,
            "topics": topics,
            "valid_json": not response.get("error") and REQUIRED_FIELDS.issubset(response) and len(topics) >= 3,
            "clusters": cluster_hits(topics),
            "raw_response": response,
        }
        rows.append(row)
        append_jsonl(jsonl_path, row)
        print(f"{seed}: valid={row['valid_json']} clusters={row['clusters']} topics={topics}", flush=True)

    summary = build_summary(rows, client.model)
    write_json(summary_path, summary)
    print(f"\nWrote {jsonl_path}")
    print(f"Wrote {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


def loose_topic_prompt(seed: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "Return only compact valid JSON. No markdown. No explanation.",
        },
        {
            "role": "user",
            "content": (
                "Use this seed as random input to create 5 varied public-web curiosity topics. "
                "Do not create a persona. Do not mention the seed. Do not explain. "
                "Topics can be about ordinary life, places, arts, sports, nature, history, food, language, "
                "tools, education, transport, games, rituals, architecture, or any other public-web subject. "
                "Return concrete topic phrases, not categories.\n"
                f"seed: {seed}\n"
                'Return JSON: {"topics":["...","...","...","...","..."]}'
            ),
        },
    ]


def cluster_hits(topics: list[str]) -> dict[str, list[str]]:
    combined = normalise_title(" | ".join(topics))
    token_set = set(combined.replace("-", " ").split())
    hits: dict[str, list[str]] = {}
    for cluster, terms in CLUSTER_TERMS.items():
        found = []
        for term in terms:
            if " " in term:
                if term in combined:
                    found.append(term)
            elif term in token_set:
                found.append(term)
        if found:
            hits[cluster] = found
    return hits


def build_summary(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    cluster_counts: dict[str, int] = {}
    unique_topics = set()
    for row in rows:
        unique_topics.update(normalise_title(topic) for topic in row["topics"])
        for cluster in row["clusters"]:
            cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
    return {
        "timestamp": utc_now(),
        "model": model,
        "total_seeds": len(rows),
        "valid_json_count": sum(1 for row in rows if row["valid_json"]),
        "unique_topic_count": len(unique_topics),
        "cluster_counts": cluster_counts,
        "rows_with_any_cluster": sum(1 for row in rows if row["clusters"]),
    }


if __name__ == "__main__":
    main()
