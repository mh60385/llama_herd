from __future__ import annotations

import hashlib
import random
from typing import Any, Literal

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

PUBLIC_WORLD_INTERESTS = [
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

SeedingStrategy = Literal["public_world", "seeded"]


def create_initial_profile(
    agent_id: str,
    name: str,
    seed: str,
    strategy: SeedingStrategy = "public_world",
) -> dict[str, object]:
    """Create initial profile using specified seeding strategy.
    
    Args:
        agent_id: Agent identifier
        name: Agent name
        seed: Seed string for reproducibility
        strategy: Seeding strategy - "public_world" or "seeded"
    
    Returns:
        Initial profile dictionary with seed metadata
    """
    if strategy == "seeded":
        return _create_deterministic_seeded_profile(agent_id, name, seed, use_trait_seeds=True)
    # public_world
    return _create_deterministic_seeded_profile(agent_id, name, seed, use_trait_seeds=False)


def _create_deterministic_seeded_profile(
    agent_id: str, name: str, seed: str, use_trait_seeds: bool
) -> dict[str, object]:
    """Create deterministic profile from seed."""
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    rng = random.Random(digest)
    
    if use_trait_seeds:
        interest_pool = STARTING_INTERESTS
        num_interests = 2
        search_style, uncertainty_style = rng.choice(TRAIT_SEEDS)
        self_rules = ["treat one episode as evidence, not identity"]
        profile_text = (
            f"I am {name}, a seeded local research profile. I can start with light preferences, "
            "but lasting interests require repeated evidence."
        )
        source_label = "deterministic_seeded"
    else:
        interest_pool = PUBLIC_WORLD_INTERESTS
        num_interests = 3
        search_style = rng.choice(["source-diverse and concrete", "curious and contrastive", "broad but evidence-seeking"])
        uncertainty_style = rng.choice(["state uncertainty plainly", "separate weak signals from evidence"])
        self_rules = ["treat seed material as a starting point"]
        profile_text = (
            f"I am {name}, a seeded local research profile using public-world source material. "
            "Lasting interests require repeated evidence."
        )
        source_label = "deterministic_public_world"
    
    interests = rng.sample(interest_pool, k=num_interests)
    
    return {
        "profile_seed": seed,
        "seed_generation_source": source_label,
        "seed_generation_error": "",
        "initial_profile": {
            "first_person_profile": profile_text,
            "current_interests": interests,
            "preferred_sources": [],
            "search_style": search_style,
            "uncertainty_style": uncertainty_style,
            "self_rules": self_rules,
        },
    }


# Backwards compatibility aliases
def deterministic_public_world_fallback_profile(agent_id: str, name: str, seed: str) -> dict[str, object]:
    return create_initial_profile(agent_id, name, seed, strategy="public_world")


def deterministic_seeded_initial_profile(agent_id: str, name: str, seed: str) -> dict[str, object]:
    return create_initial_profile(agent_id, name, seed, strategy="seeded")


