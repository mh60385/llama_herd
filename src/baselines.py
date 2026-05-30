"""Baseline agents for comparison against LLM-driven agents.

Provides research-grade baselines to quantify whether LLM agents
are actually performing better than simple alternatives.

Baselines:
- Random: Random queries from seed pool
- Static: Fixed interests, no adaptation
- Echo: Repeats last query (worst case, detects anti-anchoring)
"""

from __future__ import annotations

import random
from typing import Any

from .profile_seed import PUBLIC_WORLD_INTERESTS, WIKIPEDIA_VITAL_LEVEL4


def random_query_baseline(agent_id: str, seed: str, episode_count: int) -> list[str]:
    """Generate random queries from public world interest pool.
    
    This baseline tests whether the LLM is doing better than random selection.
    
    Args:
        agent_id: Agent identifier for reproducibility
        seed: Seed string for deterministic randomness
        episode_count: Number of episodes to generate queries for
        
    Returns:
        List of random queries (one per episode)
    """
    digest = seed + agent_id
    rng = random.Random(digest)
    
    # Combine both interest pools for diversity
    pool = list(set(PUBLIC_WORLD_INTERESTS + WIKIPEDIA_VITAL_LEVEL4))
    pool = [t for t in pool if t and len(t) > 2]
    
    queries = []
    for i in range(episode_count):
        # Pick 2-4 random interests and combine them
        selected = rng.sample(pool, k=min(4, len(pool)))
        query = ", ".join(selected)
        queries.append(query)
    
    return queries


def static_interest_baseline(agent_id: str, seed: str, episode_count: int, 
                             interests: list[str] | None = None) -> list[str]:
    """Generate queries from fixed interests (no adaptation).
    
    This baseline tests whether profile adaptation adds value.
    If interests not provided, uses first 3 from PUBLIC_WORLD_INTERESTS.
    
    Args:
        agent_id: Agent identifier
        seed: Seed for reproducibility
        episode_count: Number of episodes
        interests: Fixed interests to use (optional)
        
    Returns:
        List of queries derived from fixed interests
    """
    digest = seed + agent_id
    rng = random.Random(digest)
    
    if interests is None or len(interests) == 0:
        interests = PUBLIC_WORLD_INTERESTS[:3]
    
    queries = []
    for i in range(episode_count):
        # Rotate through interests with slight variations
        base = interests[i % len(interests)]
        # Add random suffix sometimes to avoid exact repetition
        if rng.random() > 0.7:
            suffix_pool = ["perspectives", "analysis", "overview", "study", "research"]
            base = f"{base} {rng.choice(suffix_pool)}"
        queries.append(base)
    
    return queries


def echo_baseline(agent_id: str, seed: str, episode_count: int, 
                  initial_query: str = "public world research topics") -> list[str]:
    """Repeat the same query every time (worst case).
    
    This baseline tests anti-anchoring effectiveness.
    If anti-anchoring works, this agent should show HIGH drift scores.
    
    Args:
        agent_id: Agent identifier
        seed: Seed (unused for echo, but included for API consistency)
        episode_count: Number of episodes
        initial_query: Query to repeat
        
    Returns:
        List of identical queries
    """
    return [initial_query] * episode_count


def repeating_interest_baseline(agent_id: str, seed: str, episode_count: int) -> list[str]:
    """Cycle through a small set of interests repeatedly.
    
    Simulates an agent that gets stuck in a loop.
    
    Args:
        agent_id: Agent identifier
        seed: Seed for reproducibility
        episode_count: Number of episodes
        
    Returns:
        List of queries cycling through a small interest set
    """
    digest = seed + agent_id
    rng = random.Random(digest)
    
    # Small set of related interests (simulating a stuck agent)
    loop_interests = ["Visual arts", "Gender", "Non-fiction", "Activism"]
    
    queries = []
    for i in range(episode_count):
        interest = loop_interests[i % len(loop_interests)]
        queries.append(interest)
    
    return queries


# Baseline registry for easy access
BASELINES = {
    "random": random_query_baseline,
    "static": static_interest_baseline,
    "echo": echo_baseline,
    "repeating": repeating_interest_baseline,
}


def get_baseline_queries(
    baseline_type: str,
    agent_id: str,
    seed: str,
    episode_count: int,
    **kwargs: Any
) -> list[str]:
    """Get queries for a baseline agent.
    
    Args:
        baseline_type: One of 'random', 'static', 'echo', 'repeating'
        agent_id: Agent identifier
        seed: Seed for reproducibility
        episode_count: Number of episodes
        **kwargs: Additional arguments for specific baselines
        
    Returns:
        List of queries
    """
    if baseline_type not in BASELINES:
        raise ValueError(f"Unknown baseline: {baseline_type}. Available: {list(BASELINES.keys())}")
    
    return BASELINES[baseline_type](agent_id, seed, episode_count, **kwargs)
