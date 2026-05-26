from __future__ import annotations

import hashlib
import random
from typing import Any

from .llm_client import LLMClient
from .utils import clamp_list


TRAIT_SEEDS = [
    ("broad, evidence-seeking, contrastive", "state uncertainty plainly"),
    ("curious, source-diverse, concise", "separate evidence from speculation"),
    ("methodical, novelty-seeking, skeptical", "name weak evidence clearly"),
]

STARTING_INTERESTS = [
    "open-ended web research methods",
    "how evidence quality changes across sources",
    "comparisons between practical systems",
    "long-running local AI experiments",
    "uncertainty in public information",
]

PUBLIC_WORLD_FALLBACK_INTERESTS = [
    "news",
    "technology",
    "health",
    "food",
    "travel",
    "movies",
    "television",
    "music",
    "sports",
    "books",
    "science",
    "weather",
    "money",
    "jobs",
    "education",
    "housing",
    "transport",
    "shopping",
    "family",
    "relationships",
    "pets",
    "fitness",
    "games",
    "fashion",
    "history",
    "nature",
    "cars",
    "home improvement",
    "social media",
    "local events",
]


def deterministic_public_world_fallback_profile(agent_id: str, name: str, seed: str) -> dict[str, object]:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    rng = random.Random(digest)
    interests = rng.sample(PUBLIC_WORLD_FALLBACK_INTERESTS, k=3)
    return {
        "profile_seed": seed,
        "seed_generation_source": "deterministic_public_world_fallback",
        "seed_generation_error": "",
        "initial_profile": {
            "first_person_profile": (
                f"I am {name}, a seeded local research profile using public-world source material. "
                "Lasting interests require repeated evidence."
            ),
            "current_interests": interests,
            "preferred_sources": [],
            "search_style": rng.choice(["source-diverse and concrete", "curious and contrastive", "broad but evidence-seeking"]),
            "uncertainty_style": rng.choice(["state uncertainty plainly", "separate weak signals from evidence"]),
            "self_rules": ["treat seed material as a starting point"],
        },
    }


def deterministic_seeded_initial_profile(agent_id: str, name: str, seed: str) -> dict[str, object]:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    rng = random.Random(digest)
    search_style, uncertainty_style = rng.choice(TRAIT_SEEDS)
    interests = rng.sample(STARTING_INTERESTS, k=2)
    return {
        "profile_seed": seed,
        "initial_profile": {
            "first_person_profile": (
                f"I am {name}, a seeded local research profile. I can start with light preferences, "
                "but lasting interests require repeated evidence."
            ),
            "current_interests": interests,
            "preferred_sources": [],
            "search_style": search_style,
            "uncertainty_style": uncertainty_style,
            "self_rules": ["treat one episode as evidence, not identity"],
        },
    }


def model_seeded_initial_profile(
    agent_id: str,
    name: str,
    seed: str,
    client: LLMClient,
) -> dict[str, object]:
    payload = client.chat(seed_profile_prompt(agent_id, name, seed), temperature=0, top_p=1, expect_json=True, max_tokens=140)
    validation_error = seed_payload_error(payload)
    if validation_error:
        retry_payload = client.chat(
            seed_profile_retry_prompt(agent_id, name, seed, validation_error),
            temperature=0,
            top_p=1,
            expect_json=True,
            max_tokens=140,
        )
        if not retry_payload.get("error") and not seed_payload_error(retry_payload):
            payload = retry_payload
    if payload.get("error"):
        fallback = deterministic_seeded_initial_profile(agent_id, name, seed)
        fallback["seed_generation_error"] = payload
        fallback["seed_generation_source"] = "deterministic_fallback"
        return fallback

    initial = sanitize_initial_profile(agent_id, name, payload)
    return {
        "profile_seed": seed,
        "initial_profile": initial,
        "seed_generation_raw_response": payload,
        "seed_generation_source": "llm",
    }


def seed_profile_prompt(agent_id: str, name: str, seed: str) -> list[dict[str, str]]:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    codes = [str(int(digest[index : index + 4], 16) % 97) for index in range(0, 12, 4)]
    anchors = seed_anchors(digest)
    return [
        {
            "role": "system",
            "content": (
                "Return only compact valid JSON. Do not include markdown, chain-of-thought, hidden reasoning, "
                "or prose outside JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "Generate a reproducible initial research-agent profile from the given seed string. "
                "Use the seed hash codes as random-looking constraints, not as topics. The interests for different "
                "hash codes must be substantially different. Create broad, concrete, non-identical starting interests "
                "suitable for open web research. Each interest must be a specific researchable topic phrase of 4-9 words, "
                "not a hobby, not a generic activity, and not a one-word category. Use these seed anchors to force "
                "variety; combine or reinterpret them, but do not output the same topic for different seeds. "
                f"Seed anchors: {', '.join(anchors)}. "
                "Avoid biography, demographics, protected "
                "attributes, fixed personality identity, politics-as-identity, medical identity, and rules that "
                "say always/never/must/only. Do not mention the seed or agent id as a research topic. "
                "Return 3 current_interests, no preferred_sources, 1-2 gentle self_rules. Keep every string short. "
                "No explanations.\n"
                f"agent_id: {agent_id}\nname: {name}\nseed: {seed}\nseed_hash: {digest}\n"
                f"random_codes: {', '.join(codes)}\n"
                'Return JSON: {"first_person_profile":"...",'
                '"current_interests":["...","...","..."],'
                '"preferred_sources":[],"search_style":"...",'
                '"uncertainty_style":"...","self_rules":["..."]}'
            ),
        },
    ]


def seed_profile_retry_prompt(agent_id: str, name: str, seed: str, validation_error: str) -> list[dict[str, str]]:
    messages = seed_profile_prompt(agent_id, name, seed)
    messages[1]["content"] += (
        "\nYour previous profile failed validation: "
        f"{validation_error}. Return a corrected profile with specific, diverse research topics."
    )
    return messages


def seed_anchors(digest: str) -> list[str]:
    pools = [
        ["housing", "water", "libraries", "microgrids", "shipping", "language", "soil", "craft"],
        ["maintenance", "mapping", "forecasting", "training", "inspection", "archiving", "routing", "measurement"],
        ["cities", "farms", "schools", "clinics", "workshops", "coasts", "markets", "museums"],
    ]
    anchors = []
    for index, pool in enumerate(pools):
        value = int(digest[index * 4 : index * 4 + 4], 16)
        anchors.append(pool[value % len(pool)])
    return anchors


def seed_payload_error(payload: dict[str, Any]) -> str:
    if payload.get("error"):
        return str(payload.get("error"))
    interests = [str(item).strip() for item in payload.get("current_interests", []) if str(item).strip()]
    if len(interests) < 3:
        return "fewer than 3 interests"
    one_word = [item for item in interests if len(item.split()) < 4]
    if one_word:
        return f"interests too generic: {', '.join(one_word)}"
    unique = {item.lower() for item in interests}
    if len(unique) < len(interests):
        return "duplicate interests"
    return ""


def sanitize_initial_profile(agent_id: str, name: str, payload: dict[str, Any]) -> dict[str, object]:
    fallback = deterministic_seeded_initial_profile(agent_id, name, f"{agent_id}:fallback")["initial_profile"]
    interests = [
        item
        for item in (sanitize_text(value, "") for value in payload.get("current_interests", []))
        if item
    ]
    rules = [
        item
        for item in (sanitize_text(value, "") for value in payload.get("self_rules", []))
        if item
    ]
    return {
        "first_person_profile": sanitize_text(
            payload.get("first_person_profile"),
            str(fallback["first_person_profile"]),
            limit=360,
        ),
        "current_interests": clamp_list(interests or list(fallback["current_interests"]), 3),
        "preferred_sources": [],
        "search_style": sanitize_text(payload.get("search_style"), str(fallback["search_style"]), limit=180),
        "uncertainty_style": sanitize_text(
            payload.get("uncertainty_style"),
            str(fallback["uncertainty_style"]),
            limit=180,
        ),
        "self_rules": clamp_list(rules or list(fallback["self_rules"]), 3),
    }


def sanitize_text(value: Any, fallback: str, limit: int = 180) -> str:
    text = str(value or "").strip()
    lowered = text.lower()
    blocked_terms = {
        "i was born",
        "my childhood",
        "my family",
        "my race",
        "my religion",
        "my gender",
        "my nationality",
        "my disability",
        "my diagnosis",
        "my political party",
    }
    strict_terms = {"always", "never", "avoid", "must", "only", "forbidden", "require", "refuse"}
    if any(term in lowered for term in blocked_terms):
        return fallback
    if any(term in lowered for term in strict_terms):
        return fallback
    return text[:limit] or fallback
