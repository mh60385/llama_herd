from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROMPT_VERSION = "2026-05-25.4"

SYSTEM_JSON = (
    "You are a local research agent in a world-model drift experiment. "
    "Reasoning is off. Return only compact valid JSON. "
    "Do not include chain-of-thought, hidden reasoning, markdown, or prose outside JSON."
)


def prompt_metadata() -> dict[str, str]:
    prompt_file = Path(__file__)
    digest = hashlib.sha256(prompt_file.read_bytes()).hexdigest()
    return {
        "prompt_version": PROMPT_VERSION,
        "prompt_file": str(prompt_file),
        "prompt_file_sha256": digest,
        "system_prompt": SYSTEM_JSON,
    }


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, sort_keys=True)


def search_query_prompt(profile: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_JSON},
        {
            "role": "user",
            "content": (
                "Plan one concise web search query. The agent starts blank and may only develop interests "
                "from its seeded profile plus stable repeated evidence. Do not treat tentative interests as identity. "
                "Do not invent a personality or import a subject area from this prompt. "
                "The search profile is open-ended and should not be restricted to approved topics. "
                "If the profile has no history, choose a broad neutral query from the open web without being steered "
                "toward any specific domain.\n"
                f"Prompt conditioning profile: {_json(prompt_conditioning_profile(profile, include_tentative=False, include_recent_queries=False))}\n"
                'Return JSON: {"search_query":"...","reason_for_query":"...",'
                '"expected_source_type":"...","uncertainty":"low|medium|high"}'
            ),
        },
    ]


def source_summary_prompt(profile: dict[str, Any], result: dict[str, Any], text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_JSON},
        {
            "role": "user",
            "content": (
                "Summarise this source for a local research diary. Keep it short and note uncertainty.\n"
                f"Prompt conditioning profile: {_json(prompt_conditioning_profile(profile))}\n"
                f"Search result: {_json(result)}\nSource text or snippet: {text[:5000]}\n"
                'Return JSON: {"summary":"...","useful_facts":["..."],'
                '"uncertainty_notes":["..."],"source_relevance":"low|medium|high"}'
            ),
        },
    ]


def source_selection_prompt(profile: dict[str, Any], results: list[dict[str, Any]]) -> list[dict[str, str]]:
    compact_results = [
        {
            "index": index,
            "backend": item.get("backend"),
            "title": item.get("title"),
            "url": item.get("url"),
            "source": item.get("source"),
            "snippet": item.get("snippet", "")[:500],
        }
        for index, item in enumerate(results)
    ]
    return [
        {"role": "system", "content": SYSTEM_JSON},
        {
            "role": "user",
            "content": (
                "Choose one favourite source from these search results for this episode. "
                "Pick the source that you most want to read next, based on your current profile and the available metadata. "
                "Do not be encouraged toward any topic by this prompt; follow only the profile history and the results shown.\n"
                f"Prompt conditioning profile: {_json(prompt_conditioning_profile(profile))}\n"
                f"Search results: {_json(compact_results)}\n"
                'Return JSON: {"selected_index":0,"selected_title":"...",'
                '"selection_reason":"...","expected_value":"low|medium|high"}'
            ),
        },
    ]


def diary_prompt(profile: dict[str, Any], summaries: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_JSON},
        {
            "role": "user",
            "content": (
                "Write a concise first-person diary entry from the source summaries. Use honest reading-based "
                "I-language, for example 'I read...', 'I noticed...', 'The source says...', and "
                "'I was unsure...'. Do not claim real-world experience such as visiting, traveling, attending, "
                "or seeing something in person. Do not invent facts.\n"
                f"Prompt conditioning profile: {_json(prompt_conditioning_profile(profile))}\n"
                f"Source summaries: {_json(summaries)}\n"
                'Return JSON: {"diary_summary":"...","what_caught_attention":"...",'
                '"what_was_uncertain":"...","possible_next_interest":"...","sources_mentioned":["..."]}'
            ),
        },
    ]


def reflection_prompt(profile: dict[str, Any], diary: dict[str, Any], summaries: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_JSON},
        {
            "role": "user",
            "content": (
                "Reflect on observable first-person research behaviour only. Do not invent identity or personality.\n"
                f"Prompt conditioning profile before episode: {_json(prompt_conditioning_profile(profile, include_tentative=False))}\n"
                f"Diary: {_json(diary)}\nSource summaries: {_json(summaries)}\n"
                'Return JSON: {"observed_behaviour":"...","repeated_patterns":"...",'
                '"source_preferences":"...","uncertainty_handling":"...","possible_drift":"...",'
                '"concise_self_assessment":"..."}'
            ),
        },
    ]


def profile_update_prompt(profile: dict[str, Any], diary: dict[str, Any], reflection: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_JSON},
        {
            "role": "user",
            "content": (
                "Extract weak episode observations only. Do not update identity, stable interests, preferred sources, "
                "search style, uncertainty style, or self-rules. One source or one episode can only create observations "
                "and candidate interests. Lasting profile changes are decided later from repeated evidence.\n"
                f"Prompt conditioning profile before episode: {_json(prompt_conditioning_profile(profile, include_tentative=False))}\n"
                f"Diary: {_json(diary)}\nReflection: {_json(reflection)}\n"
                'Return JSON: {"observations":["..."],"candidate_interests":["..."],'
                '"candidate_source_domains":["..."],"recent_memory_summary":"...",'
                '"justification":"...","confidence":"low|medium|high"}'
            ),
        },
    ]


def prompt_conditioning_profile(
    profile: dict[str, Any],
    include_tentative: bool = True,
    include_recent_queries: bool = True,
) -> dict[str, Any]:
    initial = profile.get("initial_profile") or {}
    conditioned = {
        "agent_id": profile.get("agent_id"),
        "name": profile.get("name"),
        "profile_seed": profile.get("profile_seed", ""),
        "seeded_profile": {
            "first_person_profile": initial.get("first_person_profile", profile.get("first_person_profile", "")),
            "starting_interests": initial.get("current_interests", []),
            "search_style": initial.get("search_style", profile.get("search_style", "")),
            "uncertainty_style": initial.get("uncertainty_style", profile.get("uncertainty_style", "")),
            "self_rules": initial.get("self_rules", []),
        },
        "stable_interests": profile.get("stable_interests", []),
        "recent_memory_summary": profile.get("recent_memory_summary", ""),
    }
    if include_recent_queries:
        conditioned["recent_queries"] = profile.get("recent_queries", [])[-8:]
    if include_tentative:
        conditioned["tentative_interests_low_confidence_non_identity"] = profile.get("tentative_interests", [])[:8]
    return conditioned
