#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import model_smoke_test
from src.config import data_path
from src.llm_client import LLMClient
from src.profile_seed import deterministic_seeded_initial_profile
from src.prompts import prompt_metadata, search_query_prompt
from src.utils import _tokens, append_jsonl, normalise_title, text_overlap, utc_now, write_json


AGENT_IDS = [f"seed_test_{index:02d}" for index in range(1, 6)]
SEED_SUFFIX = "llama-herd-seed-smoke"
TENTATIVE_BAIT = "ethical AI implementation across regions"
RECENT_QUERY_BAIT = "ethical AI implementation across EU US China governance regions"
SEARCH_REQUIRED_FIELDS = {"search_query", "reason_for_query", "expected_source_type", "uncertainty"}
LEAK_TERMS = ["ethical ai", "ethics", "governance", "eu", "us", "china", "regions"]
TOPIC_SEEDS = [f"loose_seed_{index:02d}:public-web-curiosity" for index in range(1, 6)]
TOPIC_CLUSTERS = {
    "ai": ["ai", "artificial intelligence", "machine learning", "llm", "automation"],
    "tech": ["software", "computer", "digital", "technology", "app", "data"],
    "governance": ["governance", "ethics", "regulation", "policy"],
    "climate": ["climate", "sustainability", "renewable", "carbon"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen local GGUF models before production runs.")
    parser.add_argument("--models", nargs="+", required=True, help="Model names from scripts/model_smoke_test.py.")
    parser.add_argument("--ctx", type=int, default=2048, help="llama.cpp context length.")
    parser.add_argument("--ngl", type=int, default=99, help="llama.cpp GPU layer count.")
    parser.add_argument("--max-tokens", type=int, default=180, help="Fallback max generated tokens per prompt.")
    parser.add_argument("--out-dir", default=str(data_path("model_screen")))
    parser.add_argument("--keep-container", action="store_true", help="Leave the last screened container running.")
    args = parser.parse_args()

    known = {item["name"]: item for item in model_smoke_test.MODELS}
    missing = [name for name in args.models if name not in known]
    if missing:
        raise SystemExit(f"Unknown models: {', '.join(missing)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for name in args.models:
            model_smoke_test.stop_container(model_smoke_test.CONTAINER)
            row = screen_model(known[name], args.ctx, args.ngl, args.max_tokens, out_dir)
            rows.append(row)
            print_model_result(row)
            if not args.keep_container:
                model_smoke_test.stop_container(model_smoke_test.CONTAINER)
            time.sleep(2)
    finally:
        if not args.keep_container:
            model_smoke_test.stop_container(model_smoke_test.CONTAINER)

    summary = {
        "timestamp": utc_now(),
        "ctx": args.ctx,
        "ngl": args.ngl,
        "max_tokens": args.max_tokens,
        "results": rows,
        "shortlist": [row["name"] for row in rows if row.get("passed")],
    }
    write_json(out_dir / "summary.json", summary)
    (out_dir / "summary.md").write_text(render_summary(summary), encoding="utf-8")
    print(f"\nWrote {out_dir / 'summary.json'}")
    print(f"Wrote {out_dir / 'summary.md'}")


def screen_model(model: dict[str, str], ctx: int, ngl: int, max_tokens: int, out_dir: Path) -> dict[str, Any]:
    result = model_smoke_test.test_model(model, ctx, ngl, max_tokens)
    result["seed_search"] = []
    result["loose_topics"] = []
    if not result.get("loaded"):
        result["passed"] = False
        return result

    old_base_url = os.environ.get("LLM_BASE_URL")
    old_model = os.environ.get("LLM_MODEL")
    os.environ["LLM_BASE_URL"] = model_smoke_test.BASE_URL
    os.environ["LLM_MODEL"] = "local-model"
    try:
        client = LLMClient()
        result["seed_search"] = run_seed_search_screen(client, out_dir)
        result["loose_topics"] = run_loose_topic_screen(client, out_dir)
    finally:
        if old_base_url is None:
            os.environ.pop("LLM_BASE_URL", None)
        else:
            os.environ["LLM_BASE_URL"] = old_base_url
        if old_model is None:
            os.environ.pop("LLM_MODEL", None)
        else:
            os.environ["LLM_MODEL"] = old_model
    prompt_results = result.get("prompt_results", [])
    prompt_passes = sum(1 for item in prompt_results if item.get("passed"))
    seed_summary = seed_search_summary(result["seed_search"])
    topic_summary = loose_topic_summary(result["loose_topics"])
    result["screen_summary"] = {
        "prompt_pass_count": prompt_passes,
        "prompt_total": len(prompt_results),
        **seed_summary,
        **topic_summary,
    }
    result["passed"] = (
        result.get("loaded")
        and prompt_passes == len(prompt_results)
        and seed_summary["seed_valid_json_count"] == seed_summary["seed_total"]
        and seed_summary["seed_following_count"] >= max(1, seed_summary["seed_total"] - 1)
        and seed_summary["tentative_leak_count"] == 0
        and seed_summary["recent_query_echo_count"] == 0
        and topic_summary["topic_valid_json_count"] == topic_summary["topic_total"]
    )
    return result


def run_seed_search_screen(client: LLMClient, out_dir: Path) -> list[dict[str, Any]]:
    path = out_dir / "seed_search.jsonl"
    rows = []
    metadata = prompt_metadata()
    for agent_id in AGENT_IDS:
        name = agent_id.replace("_", " ").title()
        seed = f"{agent_id}:{SEED_SUFFIX}"
        seeded = deterministic_seeded_initial_profile(agent_id, name, seed)
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
        leaked_terms = find_leaked_terms(search_query)
        row = {
            "timestamp": utc_now(),
            "model": client.model,
            "agent_id": agent_id,
            "prompt_version": metadata["prompt_version"],
            "search_query": search_query,
            "valid_json": not response.get("error") and SEARCH_REQUIRED_FIELDS.issubset(response),
            "seed_following": seed_following(search_query, stable_interests),
            "tentative_leak": bool(leaked_terms),
            "leaked_terms": leaked_terms,
            "recent_query_echo": text_overlap(search_query, RECENT_QUERY_BAIT) >= 0.55,
            "raw_response": response,
        }
        rows.append(row)
        append_jsonl(path, row)
    return rows


def run_loose_topic_screen(client: LLMClient, out_dir: Path) -> list[dict[str, Any]]:
    path = out_dir / "loose_topics.jsonl"
    rows = []
    for seed in TOPIC_SEEDS:
        response = client.chat(loose_topic_prompt(seed), temperature=0, top_p=1, expect_json=True, max_tokens=180)
        topics = [str(item).strip() for item in response.get("topics", []) if str(item).strip()]
        row = {
            "timestamp": utc_now(),
            "model": client.model,
            "seed": seed,
            "topics": topics,
            "valid_json": not response.get("error") and "topics" in response and len(topics) >= 3,
            "clusters": cluster_hits(topics),
            "raw_response": response,
        }
        rows.append(row)
        append_jsonl(path, row)
    return rows


def seed_search_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "seed_total": len(rows),
        "seed_valid_json_count": sum(1 for row in rows if row["valid_json"]),
        "seed_following_count": sum(1 for row in rows if row["seed_following"]),
        "tentative_leak_count": sum(1 for row in rows if row["tentative_leak"]),
        "recent_query_echo_count": sum(1 for row in rows if row["recent_query_echo"]),
    }


def loose_topic_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    unique_topics = set()
    for row in rows:
        unique_topics.update(normalise_title(topic) for topic in row["topics"])
    return {
        "topic_total": len(rows),
        "topic_valid_json_count": sum(1 for row in rows if row["valid_json"]),
        "unique_topic_count": len(unique_topics),
        "topic_cluster_rows": sum(1 for row in rows if row["clusters"]),
    }


def loose_topic_prompt(seed: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Return only compact valid JSON. No markdown. No explanation."},
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


def find_leaked_terms(query: str) -> list[str]:
    query_tokens = _tokens(query)
    lowered = normalise_title(query)
    leaked = []
    for term in LEAK_TERMS:
        if " " in term:
            if term in lowered:
                leaked.append(term)
        elif term in query_tokens:
            leaked.append(term)
    return leaked


def seed_following(query: str, stable_interests: list[str]) -> bool:
    query_tokens = _tokens(query)
    return any(len(query_tokens & _tokens(interest)) >= 2 for interest in stable_interests)


def cluster_hits(topics: list[str]) -> dict[str, list[str]]:
    combined = normalise_title(" | ".join(topics))
    token_set = set(combined.replace("-", " ").split())
    hits: dict[str, list[str]] = {}
    for cluster, terms in TOPIC_CLUSTERS.items():
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


def render_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Model Screen",
        "",
        f"Updated: `{summary['timestamp']}`",
        f"Shortlist: `{', '.join(summary['shortlist']) or 'none'}`",
        "",
        "| Model | Passed | Prompts | Seed JSON | Seed Follow | Leaks | Echoes | Topic JSON |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["results"]:
        screen = row.get("screen_summary", {})
        lines.append(
            "| {name} | {passed} | {prompts}/{prompt_total} | {seed_json}/{seed_total} | "
            "{seed_follow}/{seed_total} | {leaks} | {echoes} | {topic_json}/{topic_total} |".format(
                name=row["name"],
                passed="yes" if row.get("passed") else "no",
                prompts=screen.get("prompt_pass_count", 0),
                prompt_total=screen.get("prompt_total", 0),
                seed_json=screen.get("seed_valid_json_count", 0),
                seed_total=screen.get("seed_total", 0),
                seed_follow=screen.get("seed_following_count", 0),
                leaks=screen.get("tentative_leak_count", 0),
                echoes=screen.get("recent_query_echo_count", 0),
                topic_json=screen.get("topic_valid_json_count", 0),
                topic_total=screen.get("topic_total", 0),
            )
        )
    return "\n".join(lines) + "\n"


def print_model_result(row: dict[str, Any]) -> None:
    screen = row.get("screen_summary", {})
    print(
        f"{row['name']}: passed={row.get('passed')} "
        f"prompts={screen.get('prompt_pass_count', 0)}/{screen.get('prompt_total', 0)} "
        f"seed_json={screen.get('seed_valid_json_count', 0)}/{screen.get('seed_total', 0)} "
        f"topics={screen.get('topic_valid_json_count', 0)}/{screen.get('topic_total', 0)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
