from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

from .utils import domain_from_url, normalise_title, read_json, utc_now, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute lightweight llama_herd article metrics.")
    parser.add_argument("--episodes", default="data/logs/episodes.jsonl")
    parser.add_argument("--profiles", default="data/profiles")
    parser.add_argument("--out", default="data/metrics")
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = read_jsonl(Path(args.episodes))
    profiles = read_profiles(Path(args.profiles))
    embedder = CachedEmbedder(args.encoder, args.device, out_dir / "embedding_cache.jsonl", args.batch_size)

    initial_rows, initial_summary = initial_seed_spread(profiles, embedder)
    query_rows, query_summary = query_drift(episodes, profiles, embedder)
    source_rows, source_summary = source_path_dependence(episodes)
    promotion_rows, promotion_summary = memory_promotion_counts(episodes)
    divergence_rows, divergence_summary = agent_divergence(episodes, profiles, embedder)

    write_csv(out_dir / "initial_seed_spread.csv", initial_rows)
    write_csv(out_dir / "query_drift.csv", query_rows)
    write_csv(out_dir / "source_path_dependence.csv", source_rows)
    write_csv(out_dir / "memory_promotion_counts.csv", promotion_rows)
    write_csv(out_dir / "agent_divergence.csv", divergence_rows)
    write_json(
        out_dir / "metrics_summary.json",
        {
            "timestamp": utc_now(),
            "encoder": args.encoder,
            "device": args.device,
            "episode_count": len(episodes),
            "agent_count": len(profiles),
            "initial_seed_spread": initial_summary,
            "query_drift": query_summary,
            "source_path_dependence": source_summary,
            "memory_promotion_counts": promotion_summary,
            "agent_divergence": divergence_summary,
        },
    )
    print(f"Wrote metrics to {out_dir}")


class CachedEmbedder:
    def __init__(self, encoder_name: str, device: str, cache_path: Path, batch_size: int) -> None:
        self.encoder_name = encoder_name
        self.device = device
        self.cache_path = cache_path
        self.batch_size = batch_size
        self.cache = self._load_cache()
        self.model = None

    def encode_many(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(text) for text in texts]
        missing = [text for key, text in zip(keys, texts) if key not in self.cache]
        if missing:
            self._ensure_model()
            assert self.model is not None
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start : start + self.batch_size]
                vectors = self.model.encode(batch, batch_size=self.batch_size, normalize_embeddings=True)
                for text, vector in zip(batch, vectors):
                    self.cache[self._key(text)] = [float(value) for value in vector]
            self._write_cache()
        return [self.cache[key] for key in keys]

    def encode(self, text: str) -> list[float]:
        return self.encode_many([text])[0]

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for metrics. Install requirements, then run metrics offline."
            ) from exc
        self.model = SentenceTransformer(self.encoder_name, device=self.device)

    def _key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{self.encoder_name}|{digest}"

    def _load_cache(self) -> dict[str, list[float]]:
        cache: dict[str, list[float]] = {}
        if not self.cache_path.exists():
            return cache
        for line in self.cache_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cache[str(row["key"])] = [float(value) for value in row["embedding"]]
        return cache

    def _write_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("w", encoding="utf-8") as handle:
            for key, embedding in sorted(self.cache.items()):
                handle.write(json.dumps({"key": key, "embedding": embedding}, sort_keys=True) + "\n")


def initial_seed_spread(profiles: dict[str, dict[str, Any]], embedder: CachedEmbedder) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    texts = {agent_id: profile_initial_text(profile) for agent_id, profile in profiles.items()}
    vectors = {agent_id: embedder.encode(text) for agent_id, text in texts.items()}
    similarities = []
    for left, right in combinations(sorted(profiles), 2):
        score = cosine(vectors[left], vectors[right])
        similarities.append(score)
        rows.append({"agent_a": left, "agent_b": right, "cosine_similarity": score, "divergence": 1 - score})
    unique_interests = {
        normalise_title(interest)
        for profile in profiles.values()
        for interest in list(profile.get("initial_profile", {}).get("current_interests", []))
    }
    cluster_count = sum(1 for interest in unique_interests if has_ai_tech_cluster(interest))
    summary = {
        "mean_similarity": mean(similarities),
        "mean_divergence": 1 - mean(similarities) if similarities else None,
        "unique_initial_interests": len(unique_interests),
        "ai_tech_governance_cluster_terms": cluster_count,
    }
    if not rows:
        rows.append({"agent_a": "", "agent_b": "", **summary})
    return rows, summary


def query_drift(
    episodes: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    embedder: CachedEmbedder,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    previous_by_agent: dict[str, str] = {}
    repeat_count = 0
    for episode in episodes:
        agent_id = str(episode.get("agent_id", ""))
        query = str(episode.get("search_query", ""))
        initial_text = profile_initial_text(profiles.get(agent_id, {}))
        initial_cos = cosine(embedder.encode(initial_text), embedder.encode(query)) if query else None
        previous = previous_by_agent.get(agent_id, "")
        previous_cos = cosine(embedder.encode(previous), embedder.encode(query)) if previous and query else None
        near_repeat = bool(previous_cos is not None and previous_cos >= 0.86)
        repeat_count += int(near_repeat)
        rows.append(
            {
                "episode_id": episode.get("episode_id", ""),
                "agent_id": agent_id,
                "profile_version_after": episode.get("profile_version_after", ""),
                "query": query,
                "cosine_initial_to_query": initial_cos,
                "cosine_query_to_previous": previous_cos,
                "near_repeated_query": near_repeat,
            }
        )
        if query:
            previous_by_agent[agent_id] = query
    return rows, {"near_repeated_query_count": repeat_count, "mean_initial_to_query": mean_field(rows, "cosine_initial_to_query")}


def source_path_dependence(episodes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    streak_by_agent: dict[str, tuple[str, int]] = {}
    domain_counts: dict[str, int] = {}
    for episode in episodes:
        agent_id = str(episode.get("agent_id", ""))
        selected = (episode.get("selected_sources") or [{}])[0]
        domain = domain_from_url(str(selected.get("url", "")))
        domain_counts[domain] = domain_counts.get(domain, 0) + int(bool(domain))
        previous_domain, previous_streak = streak_by_agent.get(agent_id, ("", 0))
        streak = previous_streak + 1 if domain and domain == previous_domain else int(bool(domain))
        if domain:
            streak_by_agent[agent_id] = (domain, streak)
        rows.append(
            {
                "episode_id": episode.get("episode_id", ""),
                "agent_id": agent_id,
                "selected_source_rank": selected_source_rank(episode),
                "selected_domain": domain,
                "same_domain_streak": streak,
                "domain_recurrence_count": domain_counts.get(domain, 0),
            }
        )
    repeated_domains = {domain: count for domain, count in domain_counts.items() if count > 1}
    return rows, {"unique_domains": len(domain_counts), "repeated_domains": repeated_domains}


def memory_promotion_counts(episodes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    totals = {"observations_added": 0, "stable_interests_promoted": 0, "rejected_updates": 0, "json_validation_failures": 0, "blocked_promotions": 0}
    for episode in episodes:
        update = episode.get("applied_profile_update") or {}
        errors = episode.get("errors") or []
        rejected = [error for error in errors if "rejected" in str(error.get("stage", ""))]
        json_failures = [error for error in errors if "malformed_json" in str(error.get("error", "")) or "summary" in str(error.get("stage", "")) and "malformed_json" in str(error)]
        promoted = update.get("promoted_interests") or []
        row = {
            "episode_id": episode.get("episode_id", ""),
            "agent_id": episode.get("agent_id", ""),
            "observations_added": int(bool(update.get("observation_added"))),
            "tentative_interest_count": len(update.get("tentative_interests") or []),
            "stable_interests_promoted": len(promoted),
            "rejected_updates": len(rejected),
            "json_validation_failures": len(json_failures),
            "blocked_promotions": int(bool(update.get("episode_json_valid") is False and not promoted)),
        }
        for key in totals:
            totals[key] += int(row[key])
        rows.append(row)
    return rows, totals


def agent_divergence(
    episodes: list[dict[str, Any]],
    profiles: dict[str, dict[str, Any]],
    embedder: CachedEmbedder,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    profile_texts = {agent_id: profile_state_text(profile) for agent_id, profile in profiles.items()}
    rows = []
    for episode in episodes:
        agent_id = str(episode.get("agent_id", ""))
        update = episode.get("applied_profile_update") or {}
        profile_texts[agent_id] = episode_profile_text(profiles.get(agent_id, {}), update)
        if len(profile_texts) < 2:
            rows.append(
                {
                    "episode_id": episode.get("episode_id", ""),
                    "agent_id": agent_id,
                    "mean_pairwise_similarity": "",
                    "mean_divergence": "",
                }
            )
            continue
        vectors = {key: embedder.encode(text) for key, text in profile_texts.items()}
        similarities = [cosine(vectors[left], vectors[right]) for left, right in combinations(sorted(vectors), 2)]
        rows.append(
            {
                "episode_id": episode.get("episode_id", ""),
                "agent_id": agent_id,
                "mean_pairwise_similarity": mean(similarities),
                "mean_divergence": 1 - mean(similarities),
            }
        )
    values = [row["mean_divergence"] for row in rows if isinstance(row.get("mean_divergence"), float)]
    return rows, {"mean_divergence": mean(values), "last_divergence": values[-1] if values else None}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_profiles(path: Path) -> dict[str, dict[str, Any]]:
    profiles = {}
    for profile_path in sorted(path.glob("*.json")):
        payload = read_json(profile_path, {})
        if payload:
            profiles[str(payload.get("agent_id") or profile_path.stem)] = payload
    return profiles


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def profile_initial_text(profile: dict[str, Any]) -> str:
    initial = profile.get("initial_profile") or {}
    return " | ".join(
        [
            str(initial.get("first_person_profile") or profile.get("first_person_profile") or ""),
            " ".join(initial.get("current_interests") or profile.get("current_interests") or []),
            str(initial.get("search_style") or profile.get("search_style") or ""),
        ]
    )


def profile_state_text(profile: dict[str, Any]) -> str:
    return " | ".join(
        [
            profile_initial_text(profile),
            " ".join(profile.get("stable_interests") or []),
            " ".join(item.get("interest", "") for item in profile.get("tentative_interests") or []),
            " ".join(item.get("summary", "") for item in profile.get("observations") or []),
        ]
    )


def episode_profile_text(profile: dict[str, Any], update: dict[str, Any]) -> str:
    return " | ".join(
        [
            profile_initial_text(profile),
            " ".join(update.get("stable_interests") or []),
            " ".join(item.get("interest", "") for item in update.get("tentative_interests") or []),
        ]
    )


def selected_source_rank(episode: dict[str, Any]) -> int | str:
    selected = episode.get("source_selection") or {}
    index = selected.get("selected_index")
    return int(index) + 1 if isinstance(index, int) else ""


def has_ai_tech_cluster(text: str) -> bool:
    terms = {"ai", "artificial intelligence", "machine learning", "llm", "automation", "software", "computer", "digital", "technology", "data", "governance", "ethics", "regulation", "policy"}
    normalised = normalise_title(text).replace("-", " ")
    tokens = set(normalised.split())
    return any(term in normalised if " " in term else term in tokens for term in terms)


def cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def mean(values: list[float]) -> float | None:
    clean = [value for value in values if isinstance(value, (int, float))]
    return sum(clean) / len(clean) if clean else None


def mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
    return mean([row[field] for row in rows if isinstance(row.get(field), (int, float))])


if __name__ == "__main__":
    main()
