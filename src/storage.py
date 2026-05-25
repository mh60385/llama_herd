from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .config import data_path
from .llm_client import LLMClient
from .profile_seed import deterministic_seeded_initial_profile, model_seeded_initial_profile
from .schemas import AgentProfile, EpisodeLog
from .utils import append_jsonl, read_json, write_json


DATA_DIRS = [
    data_path("profiles"),
    data_path("logs"),
    data_path("sources"),
    data_path("sources", "search_cache"),
    data_path("db"),
]


class Storage:
    def __init__(self) -> None:
        self.db_path = data_path("db", "world_model_lab.sqlite")
        self.ensure_dirs()
        self.init_db()

    def ensure_dirs(self) -> None:
        for path in DATA_DIRS:
            path.mkdir(parents=True, exist_ok=True)

    def init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(
                """
                create table if not exists agents (
                    agent_id text primary key,
                    name text,
                    created_at text,
                    updated_at text
                );
                create table if not exists profile_versions (
                    agent_id text,
                    version integer,
                    profile_json text,
                    created_at text,
                    primary key(agent_id, version)
                );
                create table if not exists episodes (
                    episode_id text primary key,
                    timestamp text,
                    agent_id text,
                    profile_version_before integer,
                    profile_version_after integer,
                    search_query text,
                    diary_summary text,
                    raw_json text
                );
                create table if not exists search_results (
                    episode_id text,
                    agent_id text,
                    backend text,
                    title text,
                    url text,
                    source text,
                    score real,
                    raw_json text
                );
                create table if not exists source_summaries (
                    episode_id text,
                    agent_id text,
                    title text,
                    url text,
                    backend text,
                    summary text,
                    raw_json text
                );
                create table if not exists errors (
                    episode_id text,
                    agent_id text,
                    stage text,
                    error text,
                    raw_json text
                );
                """
            )

    def profile_path(self, agent_id: str) -> Path:
        return data_path("profiles", f"{agent_id}.json")

    def load_profile(self, agent_id: str) -> AgentProfile:
        payload = read_json(self.profile_path(agent_id))
        if not payload:
            raise FileNotFoundError(f"Profile not found for {agent_id}. Run scripts/init_experiment.py first.")
        payload = self._migrate_profile_payload(payload)
        return AgentProfile(**payload)

    def _migrate_profile_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("profile_seed") and payload.get("initial_profile"):
            return payload
        agent_id = str(payload.get("agent_id", "agent"))
        name = str(payload.get("name", agent_id))
        seed = f"{agent_id}:llama-herd:2026-05-25"
        try:
            seeded = model_seeded_initial_profile(agent_id, name, seed, LLMClient())
        except Exception:
            seeded = deterministic_seeded_initial_profile(agent_id, name, seed)
        initial = seeded["initial_profile"]
        migrated = dict(payload)
        migrated["profile_seed"] = seeded["profile_seed"]
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

    def save_profile(self, profile: AgentProfile) -> None:
        payload = profile.model_dump()
        write_json(self.profile_path(profile.agent_id), payload)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "insert or replace into agents(agent_id, name, created_at, updated_at) values (?, ?, ?, ?)",
                (profile.agent_id, profile.name, profile.created_at, profile.updated_at),
            )
            conn.execute(
                """
                insert or replace into profile_versions(agent_id, version, profile_json, created_at)
                values (?, ?, ?, ?)
                """,
                (profile.agent_id, profile.version, json.dumps(payload, sort_keys=True), profile.updated_at),
            )

    def save_episode(self, episode: EpisodeLog) -> None:
        payload = episode.model_dump()
        append_jsonl(data_path("logs", "episodes.jsonl"), payload)
        append_jsonl(data_path("logs", f"{episode.agent_id}.jsonl"), payload)
        diary_summary = episode.diary_entry.diary_summary if episode.diary_entry else ""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                insert or replace into episodes(
                    episode_id, timestamp, agent_id, profile_version_before,
                    profile_version_after, search_query, diary_summary, raw_json
                ) values (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode.episode_id,
                    episode.timestamp,
                    episode.agent_id,
                    episode.profile_version_before,
                    episode.profile_version_after,
                    episode.search_query,
                    diary_summary,
                    json.dumps(payload, sort_keys=True),
                ),
            )
            for result in episode.search_results:
                conn.execute(
                    """
                    insert into search_results(episode_id, agent_id, backend, title, url, source, score, raw_json)
                    values (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        episode.episode_id,
                        episode.agent_id,
                        result.backend,
                        result.title,
                        result.url,
                        result.source,
                        result.score,
                        result.model_dump_json(),
                    ),
                )
            for summary in episode.source_summaries:
                conn.execute(
                    """
                    insert into source_summaries(episode_id, agent_id, title, url, backend, summary, raw_json)
                    values (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        episode.episode_id,
                        episode.agent_id,
                        summary.title,
                        summary.url,
                        summary.backend,
                        summary.summary,
                        summary.model_dump_json(),
                    ),
                )
            for error in episode.errors:
                conn.execute(
                    "insert into errors(episode_id, agent_id, stage, error, raw_json) values (?, ?, ?, ?, ?)",
                    (
                        episode.episode_id,
                        episode.agent_id,
                        str(error.get("stage", "")),
                        str(error.get("error") or error.get("detail") or error),
                        json.dumps(error, sort_keys=True),
                    ),
                )

    def recent_rows(self, table: str, limit: int = 10) -> list[dict[str, Any]]:
        allowed = {"episodes", "errors", "search_results", "source_summaries"}
        if table not in allowed:
            raise ValueError(f"Unsupported table: {table}")
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(f"select * from {table} order by rowid desc limit ?", (limit,)).fetchall()
        return [dict(row) for row in rows]
