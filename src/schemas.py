from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(value).strip()] if str(value).strip() else []


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "; ".join(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, dict):
        return "; ".join(
            f"{key}: {item}".strip()
            for key, item in value.items()
            if str(item).strip()
        )
    return str(value).strip()


class AgentProfile(BaseModel):
    agent_id: str
    name: str
    profile_seed: str = ""
    seed_generation_source: str = ""
    seed_generation_error: str = ""
    initial_profile: dict[str, Any] = Field(default_factory=dict)
    first_person_profile: str = "I am a seeded local research profile. My stable interests require repeated evidence."
    current_interests: list[str] = Field(default_factory=list)
    preferred_sources: list[str] = Field(default_factory=list)
    search_style: str = "blank, concise, evidence-seeking"
    uncertainty_style: str = "state uncertainty plainly"
    self_rules: list[str] = Field(default_factory=list)
    recent_memory_summary: str = ""
    observations: list[dict[str, Any]] = Field(default_factory=list)
    tentative_interests: list[dict[str, Any]] = Field(default_factory=list)
    stable_interests: list[str] = Field(default_factory=list)
    recent_queries: list[str] = Field(default_factory=list)
    version: int = 1
    created_at: str
    updated_at: str

    @field_validator("current_interests", "preferred_sources", "self_rules", "stable_interests", "recent_queries", mode="before")
    @classmethod
    def coerce_profile_lists(cls, value: Any) -> list[str]:
        return _string_list(value)


class SearchResult(BaseModel):
    backend: str
    title: str = ""
    url: str = ""
    snippet: str = ""
    source: str = ""
    published_date: Optional[str] = None
    authors: list[str] = Field(default_factory=list)
    doi: Optional[str] = None
    score: float = 0.0
    raw: dict[str, Any] = Field(default_factory=dict)

    @field_validator("authors", mode="before")
    @classmethod
    def coerce_authors(cls, value: Any) -> list[str]:
        return _string_list(value)


class SourceSummary(BaseModel):
    title: str = ""
    url: str = ""
    backend: str = ""
    source_type: str = "unknown"
    summary: str = ""
    useful_facts: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    extraction_error: Optional[str] = None

    @field_validator("useful_facts", "uncertainty_notes", mode="before")
    @classmethod
    def coerce_summary_lists(cls, value: Any) -> list[str]:
        return _string_list(value)


class DiaryEntry(BaseModel):
    agent_id: str
    episode_id: str
    diary_summary: str = ""
    what_caught_attention: str = ""
    what_was_uncertain: str = ""
    possible_next_interest: str = ""
    sources_mentioned: list[str] = Field(default_factory=list)

    @field_validator("sources_mentioned", mode="before")
    @classmethod
    def coerce_sources(cls, value: Any) -> list[str]:
        return _string_list(value)

    @field_validator("diary_summary", "what_caught_attention", "what_was_uncertain", "possible_next_interest", mode="before")
    @classmethod
    def coerce_diary_text(cls, value: Any) -> str:
        return _text(value)


class Reflection(BaseModel):
    agent_id: str
    episode_id: str
    observed_behaviour: str = ""
    repeated_patterns: str = ""
    source_preferences: str = ""
    uncertainty_handling: str = ""
    possible_drift: str = ""
    concise_self_assessment: str = ""

    @field_validator(
        "observed_behaviour",
        "repeated_patterns",
        "source_preferences",
        "uncertainty_handling",
        "possible_drift",
        "concise_self_assessment",
        mode="before",
    )
    @classmethod
    def coerce_reflection_text(cls, value: Any) -> str:
        return _text(value)


class ProfileUpdateProposal(BaseModel):
    agent_id: str
    episode_id: str
    observations: list[str] = Field(default_factory=list)
    candidate_interests: list[str] = Field(default_factory=list)
    candidate_source_domains: list[str] = Field(default_factory=list)
    recent_memory_summary: str = ""
    justification: str = ""
    confidence: str = "low"

    @field_validator("observations", "candidate_interests", "candidate_source_domains", mode="before")
    @classmethod
    def coerce_proposal_lists(cls, value: Any) -> list[str]:
        return _string_list(value)

    @field_validator("recent_memory_summary", "justification", "confidence", mode="before")
    @classmethod
    def coerce_proposal_text(cls, value: Any) -> str:
        return _text(value)


class EpisodeLog(BaseModel):
    episode_id: str
    timestamp: str
    agent_id: str
    profile_version_before: int
    profile_version_after: int
    search_query: str = ""
    prompt_metadata: dict[str, Any] = Field(default_factory=dict)
    source_selection: dict[str, Any] = Field(default_factory=dict)
    search_results: list[SearchResult] = Field(default_factory=list)
    selected_sources: list[SearchResult] = Field(default_factory=list)
    source_summaries: list[SourceSummary] = Field(default_factory=list)
    diary_entry: Optional[DiaryEntry] = None
    reflection: Optional[Reflection] = None
    profile_update_proposal: Optional[ProfileUpdateProposal] = None
    applied_profile_update: dict[str, Any] = Field(default_factory=dict)
    raw_model_outputs: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[dict[str, Any]] = Field(default_factory=list)
