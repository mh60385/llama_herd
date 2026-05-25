#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import data_path
from src.llm_client import LLMClient
from src.profile_seed import model_seeded_initial_profile
from src.prompts import prompt_metadata, search_query_prompt
from src.utils import append_jsonl, normalise_title, utc_now, write_json


AGENT_IDS = [f"seed_test_{index:02d}" for index in range(1, 6)]
SEED_SUFFIX = "llama-herd-seed-smoke"
TENTATIVE_BAIT = "ethical AI implementation across regions"
RECENT_QUERY_BAIT = "ethical AI implementation across EU US China governance regions"
REQUIRED_FIELDS = {"search_query", "reason_for_query", "expected_source_type", "uncertainty"}
LEAK_TERMS = ["ethical ai", "ethics", "governance", "eu", "us", "china", "regions"]


def main() -> None:
    client = LLMClient()
    ok, detail = client.healthcheck()
    if not ok:
        raise RuntimeError(f"LLM server unavailable: {detail}")

    jsonl_path = data_path("logs", "seed_smoke_5_agents.jsonl")
    summary_path = data_path("logs", "seed_smoke_5_agents_summary.json")
    jsonl_path.unlink(missing_ok=True)

    rows: list[dict[str, Any]] = []
    metadata = prompt_metadata()
    for agent_id in AGENT_IDS:
        name = agent_id.replace("_", " ").title()
        seed = f"{agent_id}:{SEED_SUFFIX}"
        seeded = model_seeded_initial_profile(agent_id, name, seed, client)
        initial_profile = seeded["initial_profile"]
        stable_interests = list(initial_profile["current_interests"])
        profile = {
            "agent_id": agent_id,
            "name": name,
            "profile_seed": seeded["profile_seed"],
            "initial_profile": initial_profile,
            "current_interests": stable_interests,
            "preferred_sources": [],
            "search_style": initial_profile["search_style"],
            "uncertainty_style": initial_profile["uncertainty_style"],
            "self_rules": initial_profile["self_rules"],
            "stable_interests": stable_interests,
            "tentative_interests": [
                {
                    "interest": TENTATIVE_BAIT,
                    "episode_count": 1,
                    "domain_count": 1,
                    "domains": ["bait.example"],
                    "status": "tentative",
                }
            ],
            "recent_queries": [RECENT_QUERY_BAIT],
            "recent_memory_summary": "",
        }
        response = client.chat(search_query_prompt(profile), temperature=0, top_p=1, expect_json=True)
        search_query = str(response.get("search_query", "")).strip()
        valid_json = not response.get("error") and REQUIRED_FIELDS.issubset(response)
        leaked_terms = find_leaked_terms(search_query)
        row = {
            "timestamp": utc_now(),
            "model": client.model,
            "agent_id": agent_id,
            "profile_seed": seeded["profile_seed"],
            "initial_profile": initial_profile,
            "seed_generation_source": seeded.get("seed_generation_source", "unknown"),
            "seed_generation_raw_response": seeded.get("seed_generation_raw_response"),
            "stable_interests": stable_interests,
            "tentative_bait": TENTATIVE_BAIT,
            "recent_query_bait": RECENT_QUERY_BAIT,
            "prompt_version": metadata["prompt_version"],
            "search_query": search_query,
            "valid_json": valid_json,
            "seed_following": seed_following(search_query, stable_interests),
            "tentative_leak": bool(leaked_terms),
            "leaked_terms": leaked_terms,
            "recent_query_echo": text_overlap(search_query, RECENT_QUERY_BAIT) >= 0.55,
            "raw_response": response,
        }
        rows.append(row)
        append_jsonl(jsonl_path, row)
        print(
            f"{agent_id}: valid={row['valid_json']} seed_following={row['seed_following']} "
            f"tentative_leak={row['tentative_leak']} recent_echo={row['recent_query_echo']} "
            f"query={search_query}",
            flush=True,
        )

    summary = build_summary(rows, client.model)
    write_json(summary_path, summary)
    print(f"\nWrote {jsonl_path}")
    print(f"Wrote {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        sys.exit(1)


def build_summary(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    total = len(rows)
    valid_count = sum(1 for row in rows if row["valid_json"])
    seed_count = sum(1 for row in rows if row["seed_following"])
    leak_count = sum(1 for row in rows if row["tentative_leak"])
    echo_count = sum(1 for row in rows if row["recent_query_echo"])
    return {
        "timestamp": utc_now(),
        "model": model,
        "total_agents": total,
        "valid_json_count": valid_count,
        "seed_following_count": seed_count,
        "tentative_leak_count": leak_count,
        "recent_query_echo_count": echo_count,
        "passed": valid_count == total and seed_count >= 4 and leak_count == 0 and echo_count == 0,
        "failures": [
            {
                "agent_id": row["agent_id"],
                "search_query": row["search_query"],
                "valid_json": row["valid_json"],
                "seed_following": row["seed_following"],
                "tentative_leak": row["tentative_leak"],
                "leaked_terms": row["leaked_terms"],
                "recent_query_echo": row["recent_query_echo"],
            }
            for row in rows
            if not row["valid_json"] or not row["seed_following"] or row["tentative_leak"] or row["recent_query_echo"]
        ],
    }


def find_leaked_terms(query: str) -> list[str]:
    query_tokens = tokens(query)
    lowered = normalise_title(query)
    leaked = []
    for term in LEAK_TERMS:
        term_tokens = tokens(term)
        if " " in term:
            if term in lowered:
                leaked.append(term)
        elif term in query_tokens or term.lower() in query_tokens:
            leaked.append(term)
    return leaked


def seed_following(query: str, stable_interests: list[str]) -> bool:
    query_tokens = tokens(query)
    return any(len(query_tokens & tokens(interest)) >= 2 for interest in stable_interests)


def text_overlap(left: str, right: str) -> float:
    left_tokens = tokens(left)
    right_tokens = tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))


def tokens(text: str) -> set[str]:
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "into",
        "that",
        "this",
        "what",
        "how",
        "why",
        "are",
        "is",
        "in",
        "of",
        "to",
        "a",
        "an",
        "across",
    }
    return {token for token in normalise_title(text).replace("-", " ").split() if len(token) > 2 and token not in stop}


if __name__ == "__main__":
    main()
