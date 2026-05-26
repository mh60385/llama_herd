from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import data_path
from .profile_seed import deterministic_public_world_fallback_profile
from .schemas import AgentProfile, EpisodeLog
from .utils import append_jsonl, read_json, write_json


DATA_DIRS = [
    data_path("profiles"),
    data_path("logs"),
    data_path("sources"),
    data_path("sources", "search_cache"),
]


class Storage:
    def __init__(self) -> None:
        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        for path in DATA_DIRS:
            path.mkdir(parents=True, exist_ok=True)

    def profile_path(self, agent_id: str) -> Path:
        return data_path("profiles", f"{agent_id}.json")

    def load_profile(self, agent_id: str) -> AgentProfile:
        payload = read_json(self.profile_path(agent_id))
        if not payload:
            raise FileNotFoundError(f"Profile not found for {agent_id}. Run scripts/init_experiment.py first.")
        payload = self._migrate_profile_payload(payload)
        payload = self._clean_profile_payload(payload)
        return AgentProfile(**payload)

    def _migrate_profile_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("profile_seed") and payload.get("initial_profile"):
            migrated = dict(payload)
            migrated.setdefault("seed_generation_source", "")
            migrated.setdefault("seed_generation_error", "")
            return migrated
        agent_id = str(payload.get("agent_id", "agent"))
        name = str(payload.get("name", agent_id))
        seed = f"{agent_id}:llama-herd:2026-05-25"
        seeded = deterministic_public_world_fallback_profile(agent_id, name, seed)
        initial = seeded["initial_profile"]
        migrated = dict(payload)
        migrated["profile_seed"] = seeded["profile_seed"]
        migrated["seed_generation_source"] = str(seeded.get("seed_generation_source", ""))
        migrated["seed_generation_error"] = str(seeded.get("seed_generation_error", ""))
        migrated["initial_profile"] = initial
        migrated["first_person_profile"] = initial["first_person_profile"]
        migrated["current_interests"] = list(initial["current_interests"])
        migrated["preferred_sources"] = list(initial["preferred_sources"])
        migrated["search_style"] = initial["search_style"]
        migrated["uncertainty_style"] = initial["uncertainty_style"]
        migrated["self_rules"] = list(initial["self_rules"])
        migrated.setdefault("observations", [])
        migrated.setdefault("tentative_interests", [])
        migrated["stable_interests"] = list(initial["current_interests"])
        migrated.setdefault("recent_queries", [])
        return migrated

    def _clean_profile_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        cleaned = dict(payload)
        cleaned["tentative_interests"] = [
            item
            for item in cleaned.get("tentative_interests", [])
            if isinstance(item, dict) and self._valid_interest_text(str(item.get("interest", "")))
        ]
        observations = []
        for observation in cleaned.get("observations", []):
            if not isinstance(observation, dict):
                continue
            copy = dict(observation)
            copy["candidate_interests"] = [
                interest
                for interest in copy.get("candidate_interests", [])
                if self._valid_interest_text(str(interest))
            ]
            observations.append(copy)
        cleaned["observations"] = observations
        return cleaned

    def _valid_interest_text(self, value: str) -> bool:
        text = str(value or "").strip()
        lowered = text.lower()
        if not text:
            return False
        if text.startswith(("{", "[")) or "'interest':" in lowered or '"interest":' in lowered:
            return False
        return True

    def save_profile(self, profile: AgentProfile) -> None:
        write_json(self.profile_path(profile.agent_id), profile.model_dump())

    def save_episode(self, episode: EpisodeLog) -> None:
        payload = episode.model_dump()
        append_jsonl(data_path("logs", "episodes.jsonl"), payload)
        append_jsonl(data_path("logs", f"{episode.agent_id}.jsonl"), payload)
