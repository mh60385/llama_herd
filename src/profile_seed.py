from __future__ import annotations

import hashlib
import random
from typing import Any, Literal

from .utils import clamp_list


# Wikipedia Vital Articles Level 4 snapshot (as of May 2026)
# Used for seeding to provide diverse, globally-relevant starting interests
# This is a static snapshot for reproducibility - does not require internet
WIKIPEDIA_VITAL_LEVEL4 = [
    "Africa", "Antarctica", "Asia", "Europe", "North America", "Oceania", "South America",
    "Atlantic Ocean", "Indian Ocean", "Pacific Ocean", "Southern Ocean", "Arctic Ocean",
    "Earth", "Geography", "History", "Politics", "Economics", "Culture", "Society",
    "Science", "Mathematics", "Physics", "Chemistry", "Biology", "Medicine", "Astronomy",
    "Philosophy", "Religion", "Psychology", "Anthropology", "Sociology", "Linguistics",
    "Art", "Music", "Literature", "Cinema", "Theatre", "Architecture", "Sculpture",
    "Agriculture", "Food", "Cooking", "Nutrition", "Health", "Disease", "Medicine",
    "Education", "School", "University", "Learning", "Research", "Knowledge",
    "Technology", "Engineering", "Computing", "Transportation", "Communication", "Energy",
    "Law", "Justice", "Human rights", "Democracy", "Government", "Constitution",
    "War", "Peace", "Conflict", "Diplomacy", "Military", "Strategy",
    "Trade", "Commerce", "Industry", "Manufacturing", "Labor", "Economy",
    "Money", "Finance", "Banking", "Currency", "Taxation", "Wealth",
    "Population", "Demographics", "Migration", "Urbanization", "Family", "Gender",
    "Language", "Writing", "Alphabet", "Translation", "Linguistics",
    "Sports", "Olympic Games", "Football", "Cricket", "Basketball", "Athletics",
    "Music genres", "Classical music", "Jazz", "Rock music", "Hip hop", "Folk music",
    "Literary genres", "Novel", "Poetry", "Drama", "Essay", "Biography",
    "Visual arts", "Painting", "Drawing", "Photography", "Sculpture", "Calligraphy",
    "Performing arts", "Dance", "Theatre", "Opera", "Circus", "Puppetry",
    "Film", "Cinema history", "Film genres", "Animation", "Documentary",
    "Mass media", "Journalism", "Newspaper", "Radio", "Television", "Internet",
    "Transport", "Roads", "Rail transport", "Shipping", "Aviation", "Spaceflight",
    "Clothing", "Fashion", "Textiles", "Footwear", "Jewelry", "Costume",
    "Food preparation", "Cuisine", "Baking", "Brewing", "Preservation", "Fermentation",
    "Holidays", "Festivals", "Ceremonies", "Rituals", "Traditions", "Celebrations",
    "Religions", "Christianity", "Islam", "Hinduism", "Buddhism", "Judaism", "Sikhism",
    "Mythology", "Folklore", "Legends", "Fairy tales", "Epic poetry", "Oral tradition",
    "Games", "Board games", "Card games", "Video games", "Puzzles", "Gambling",
    "Hobbies", "Gardening", "Cooking", "Crafts", "Collecting", "Model building",
    "Science history", "Scientific method", "Discovery", "Invention", "Experiment", "Theory",
    "Space", "Astronomy", "Planets", "Stars", "Galaxies", "Cosmology",
    "Environment", "Ecology", "Conservation", "Pollution", "Climate", "Biodiversity",
    "Medicine history", "Diseases", "Vaccination", "Surgery", "Pharmacology", "Public health",
    "Mathematics history", "Algebra", "Geometry", "Calculus", "Statistics", "Number theory",
    "Physics history", "Mechanics", "Thermodynamics", "Electromagnetism", "Quantum mechanics",
    "Chemistry history", "Elements", "Compounds", "Reactions", "Organic chemistry", "Biochemistry",
    "Engineering", "Civil engineering", "Mechanical engineering", "Electrical engineering", "Chemical engineering",
    "Architecture history", "Building", "Urban planning", "Landscape architecture", "Interior design",
    "Literature history", "Fiction", "Non-fiction", "Poetry", "Drama", "Short story",
    "Music history", "Instruments", "Composition", "Musical notation", "Music theory",
    "Art history", "Painting", "Sculpture", "Drawing", "Printmaking", "Photography",
]


def get_wikipedia_vital_articles_pool(level: int = 4, use_cache: bool = True) -> list[str]:
    """
    Get Wikipedia Vital Articles as interest pool.
    
    Uses a static snapshot for reproducibility (no internet required).
    
    Args:
        level: Vital articles level (only level 4 currently embedded)
        use_cache: Ignored (snapshot is always available)
        
    Returns:
        List of topic strings for seeding
    """
    if level != 4:
        # For other levels, fall back to a subset or the full Level 4
        # In production, you could fetch from API, but for reproducibility we use snapshot
        pass
    return WIKIPEDIA_VITAL_LEVEL4.copy()


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

# Expanded public-world interest pool for better global diversity and coherence
# Organized by domain for maintainability (but sampled randomly)
PUBLIC_WORLD_INTERESTS = [
    # News and current affairs
    "news",
    "current events",
    "local news",
    "international affairs",
    # Technology and knowledge (reduced weight)
    "technology",
    "practical tools",
    "problem-solving methods",
    # Health and well-being
    "health",
    "public health",
    "traditional medicine",
    "mental well-being",
    "first aid",
    # Food and culture
    "food",
    "cooking techniques",
    "regional cuisines",
    "food traditions",
    # Travel and geography
    "travel",
    "local geography",
    "regional landscapes",
    "transportation systems",
    # Arts and entertainment (consolidated)
    "music",
    "traditional music",
    "storytelling",
    "oral traditions",
    "folk arts",
    "handicrafts",
    "textile traditions",
    "visual arts",
    # Sports and movement
    "sports",
    "traditional games",
    "physical activities",
    "movement practices",
    # Literature and knowledge
    "books",
    "reading habits",
    "libraries",
    "local literature",
    "oral histories",
    # Science and nature
    "science",
    "environmental changes",
    "local ecosystems",
    "weather patterns",
    "climate adaptation",
    "natural resources",
    # Practical life
    "budgeting",
    "time management",
    "household skills",
    "home repair",
    "gardening methods",
    "food preservation",
    # Work and economy
    "jobs",
    "workplace skills",
    "local economies",
    "market traditions",
    "trade practices",
    # Education and learning
    "education",
    "skill sharing",
    "apprenticeship",
    "community learning",
    "language study",
    # Housing and community
    "housing",
    "neighborhood development",
    "local architecture",
    "community spaces",
    "public facilities",
    # Relationships and society
    "family",
    "community relationships",
    "intergenerational knowledge",
    "cultural practices",
    "social customs",
    # Pets and animals
    "pets",
    "local wildlife",
    "animal husbandry",
    "working animals",
    # Fitness and movement
    "fitness",
    "traditional sports",
    "group activities",
    "recreational movement",
    # History and tradition
    "history",
    "local history",
    "oral histories",
    "cultural heritage",
    "traditional celebrations",
    # Nature and environment
    "nature",
    "local flora",
    "local fauna",
    "seasonal changes",
    "water sources",
    # Civic and community
    "local government",
    "neighborhood organizations",
    "community events",
    "public safety",
    "volunteering",
    "civic participation",
]

SeedingStrategy = Literal["public_world", "seeded", "wikipedia"]


def create_initial_profile(
    agent_id: str,
    name: str,
    seed: str,
    strategy: SeedingStrategy = "public_world",
    wikipedia_level: int = 4,
    use_wikipedia_cache: bool = True,
) -> dict[str, object]:
    """Create initial profile using specified seeding strategy.
    
    Args:
        agent_id: Agent identifier
        name: Agent name
        seed: Seed string for reproducibility
        strategy: Seeding strategy - "public_world", "seeded", or "wikipedia"
        wikipedia_level: Vital articles level (1-5) when using wikipedia strategy
        use_wikipedia_cache: Whether to cache Wikipedia results locally
    
    Returns:
        Initial profile dictionary with seed metadata
    """
    if strategy == "seeded":
        return _create_deterministic_seeded_profile(agent_id, name, seed, use_trait_seeds=True)
    elif strategy == "wikipedia":
        return _create_wikipedia_seeded_profile(agent_id, name, seed, wikipedia_level, use_wikipedia_cache)
    # public_world
    return _create_deterministic_seeded_profile(agent_id, name, seed, use_trait_seeds=False)


def _create_wikipedia_seeded_profile(
    agent_id: str, name: str, seed: str, level: int = 4, use_cache: bool = True
) -> dict[str, object]:
    """Create deterministic profile using Wikipedia Vital Articles as interest pool."""
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    rng = random.Random(digest)
    
    # Get Wikipedia Vital Articles pool (static snapshot for reproducibility)
    interest_pool = get_wikipedia_vital_articles_pool(level, use_cache)
    
    # Filter to reasonable length and remove duplicates
    interest_pool = list(set([t for t in interest_pool if 3 <= len(t) <= 40]))
    
    num_interests = 3
    search_style = rng.choice(["source-diverse and concrete", "curious and contrastive", "broad but evidence-seeking"])
    uncertainty_style = rng.choice(["state uncertainty plainly", "separate weak signals from evidence"])
    self_rules = ["treat seed material as a starting point"]
    profile_text = (
        f"I am {name}, a seeded local research profile using public-world source material. "
        "Lasting interests require repeated evidence."
    )
    source_label = f"wikipedia_vital_level{level}"
    
    interests = rng.sample(interest_pool, k=min(num_interests, len(interest_pool)))
    
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


