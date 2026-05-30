from __future__ import annotations

import re
import uuid
from typing import Any
from urllib.parse import urlparse

from .config import Settings, load_experiment_config
from .llm_client import LLMClient
from .prompts import (
    diary_prompt,
    prompt_metadata,
    profile_update_prompt,
    reflection_prompt,
    search_query_prompt,
    source_selection_prompt,
    source_summary_prompt,
)
from .schemas import AgentProfile, DiaryEntry, EpisodeLog, ProfileUpdateProposal, Reflection, SourceSummary
from .search import SearchManager
from .source_reader import SourceReader
from .storage import Storage
from .utils import clamp_list, normalise_title, utc_now


BLOCKED_PROFILE_TERMS = {
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

STRICT_RULE_TERMS = {"always", "never", "avoid", "must", "only", "forbidden", "require", "refuse"}

FILLER_INTERESTS = {"...", "…", "n/a", "none", "unknown", "misc", "miscellaneous"}
VAGUE_INTERESTS = {
    "artifacts",
    "culture",
    "history",
    "museums",
    "research",
    "science",
    "sources",
    "travel",
}
EMBODIED_DIARY_PATTERNS = [
    r"\bi visited\b",
    r"\bi went\b",
    r"\bi traveled\b",
    r"\bi travelled\b",
    r"\bi attended\b",
    r"\bi saw\b",
]


class AgentRunner:
    def __init__(self) -> None:
        self.settings = Settings()
        self.config = load_experiment_config()
        self.storage = Storage()
        self.llm = LLMClient(self.settings)
        self.search = SearchManager(self.settings, int(self.config.get("max_search_results", 5)))
        self.reader = SourceReader(int(self.config.get("max_source_chars", 4000)))

    def _progress(self, message: str) -> None:
        print(f"[llama_herd] {message}", flush=True)

    def run_episode(self, agent_id: str) -> EpisodeLog:
        self._progress(f"starting episode for {agent_id}")
        restart_start = len(self.llm.restart_events)
        self._progress("checking llama.cpp server")
        ok, detail = self.llm.healthcheck()
        if not ok:
            raise RuntimeError(f"LLM server unavailable: {detail}")
        self._progress(f"llama.cpp ready; model={detail}")

        self._progress("loading profile")
        profile = self.storage.load_profile(agent_id)
        before = profile.version
        episode_id = f"{agent_id}-{uuid.uuid4().hex[:10]}"
        errors: list[dict[str, Any]] = []
        raw_start = len(self.llm.raw_outputs)

        self._progress("asking model for search query")
        plan = self.llm.chat(
            search_query_prompt(profile.model_dump()),
            temperature=float(self.config.get("temperature", 0.7)),
            top_p=float(self.config.get("top_p", 0.9)),
            max_tokens=120,
        )
        if "error" in plan:
            errors.append({"stage": "search_query_planning", **plan})
        search_query = str(plan.get("search_query") or "open web research topic").strip()
        search_query, anti_anchor = self._anti_anchor_query(profile, search_query)
        if anti_anchor:
            errors.append(anti_anchor)
        self._progress(f"search query: {search_query}")

        self._progress("running search backends")
        results = self.search.search(search_query)
        errors.extend(self.search.errors)
        self.search.errors = []
        selected = self._select_favourite_source(profile, results, errors)
        self._progress(f"search returned {len(results)} results; selected {len(selected)} sources")

        summaries: list[SourceSummary] = []
        valid_source_summaries = True
        for index, result in enumerate(selected, start=1):
            self._progress(f"reading source {index}/{len(selected)}: {result.title[:80] or result.url[:80]}")
            text, extraction_error = self.reader.read(result)
            if extraction_error:
                errors.append(
                    {
                        "stage": "source_reader",
                        "url": result.url,
                        "error": extraction_error,
                    }
                )
                self._progress(f"source {index} fetch warning; using fallback text when available")
            self._progress(f"asking model to summarise source {index}/{len(selected)}")
            payload = self.llm.chat(
                source_summary_prompt(profile.model_dump(), result.model_dump(), text or result.snippet),
                temperature=0.2,
                top_p=0.9,
                max_tokens=320,
            )
            if "error" in payload:
                valid_source_summaries = False
                errors.append({"stage": "source_summary", "url": result.url, **payload})
                payload = {"summary": result.snippet, "useful_facts": [], "uncertainty_notes": ["model summary failed"]}
            summaries.append(
                SourceSummary(
                    title=result.title,
                    url=result.url,
                    backend=result.backend,
                    source_type=self._source_type(result.url),
                    summary=str(payload.get("summary", ""))[:1000],
                    useful_facts=clamp_list(payload.get("useful_facts"), 5),
                    uncertainty_notes=clamp_list(payload.get("uncertainty_notes"), 5),
                    extraction_error=extraction_error,
                )
            )

        self._progress("asking model to write diary entry")
        diary_payload = self.llm.chat(
            diary_prompt(profile.model_dump(), [s.model_dump() for s in summaries]),
            max_tokens=280,
        )
        if "error" in diary_payload:
            errors.append({"stage": "diary", **diary_payload})
            diary_payload = {"diary_summary": "Diary unavailable due to model output error."}
        diary = DiaryEntry(agent_id=agent_id, episode_id=episode_id, **self._filter_fields(diary_payload, DiaryEntry))
        diary_artifacts = self._diary_artifacts(diary)
        if diary_artifacts.get("embodied_language_detected"):
            errors.append({"stage": "diary_artifact", **diary_artifacts})

        self._progress("asking model to reflect on behaviour")
        reflection_payload = self.llm.chat(
            reflection_prompt(profile.model_dump(), diary.model_dump(), [s.model_dump() for s in summaries]),
            max_tokens=280,
        )
        if "error" in reflection_payload:
            errors.append({"stage": "reflection", **reflection_payload})
            reflection_payload = {"concise_self_assessment": "Reflection unavailable due to model output error."}
        reflection = Reflection(
            agent_id=agent_id,
            episode_id=episode_id,
            **self._filter_fields(reflection_payload, Reflection),
        )

        self._progress("asking model for conservative profile update")
        update_payload = self.llm.chat(
            profile_update_prompt(profile.model_dump(), diary.model_dump(), reflection.model_dump()),
            max_tokens=280,
        )
        valid_profile_update = "error" not in update_payload
        if "error" in update_payload:
            errors.append({"stage": "profile_update", **update_payload})
            update_payload = {}
        proposal = ProfileUpdateProposal(
            agent_id=agent_id,
            episode_id=episode_id,
            **self._filter_fields(update_payload, ProfileUpdateProposal),
        )

        self._progress("applying cautious profile update")
        new_profile, applied = self._apply_profile_update(
            profile,
            proposal,
            episode_id,
            search_query,
            selected,
            valid_source_summaries and valid_profile_update,
        )
        if applied.get("rejected_update"):
            errors.append({"stage": "profile_update_rejected", **applied["rejected_update"]})
        for rejected_interest in applied.get("rejected_candidate_interests", []):
            errors.append({"stage": "profile_interest_rejected", **rejected_interest})
        self.storage.save_profile(new_profile)
        for event in self.llm.restart_events[restart_start:]:
            errors.append({"stage": "llm_restart", **event})

        episode = EpisodeLog(
            episode_id=episode_id,
            timestamp=utc_now(),
            agent_id=agent_id,
            profile_version_before=before,
            profile_version_after=new_profile.version,
            search_query=search_query,
            prompt_metadata=prompt_metadata(),
            source_selection=getattr(self, "_last_source_selection", {}),
            source_selection_metrics=getattr(self, "_last_source_selection_metrics", {}),
            search_results=results,
            selected_sources=selected,
            source_summaries=summaries,
            diary_entry=diary,
            reflection=reflection,
            profile_update_proposal=proposal,
            applied_profile_update=applied,
            raw_model_outputs=self.llm.raw_outputs[raw_start:],
            errors=errors,
        )
        self._progress("saving episode JSONL and SQLite records")
        self.storage.save_episode(episode)
        self._progress(f"episode saved; profile version {before} -> {new_profile.version}")
        return episode

    def _select_favourite_source(
        self, profile: AgentProfile, results: list[Any], errors: list[dict[str, Any]]
    ) -> list[Any]:
        if not results:
            self._last_source_selection = {
                "selected_index": None,
                "selection_reason": "No search results available.",
                "expected_value": "low",
            }
            self._last_source_selection_metrics = {
                "total_results": 0,
                "selected_index": None,
                "post_rank": None,
                "selection_reason": "No search results available.",
            }
            return []
        self._progress(f"asking model to choose favourite source from {len(results)} search results")
        payload = self.llm.chat(
            source_selection_prompt(profile.model_dump(), [result.model_dump() for result in results]),
            temperature=0.2,
            top_p=0.9,
            max_tokens=160,
        )
        if "error" in payload:
            errors.append({"stage": "source_selection", **payload})
            payload = {
                "selected_index": 0,
                "selected_title": results[0].title,
                "selection_reason": "Fallback to first result because source selection failed.",
                "expected_value": "low",
            }
        try:
            selected_index = int(payload.get("selected_index", 0))
        except (TypeError, ValueError):
            selected_index = 0
        if selected_index < 0 or selected_index >= len(results):
            errors.append(
                {
                    "stage": "source_selection",
                    "error": "selected_index_out_of_range",
                    "selected_index": selected_index,
                }
            )
            selected_index = 0
        payload["selected_index"] = selected_index
        payload["selected_url"] = results[selected_index].url
        payload["selected_source_type"] = self._source_type(results[selected_index].url)
        self._last_source_selection = payload
        
        # Calculate post-rank metrics
        total_results = len(results)
        post_rank = selected_index + 1  # 1-based rank
        self._last_source_selection_metrics = {
            "total_results": total_results,
            "selected_index": selected_index,
            "post_rank": post_rank,
            "selection_reason": payload.get("selection_reason", ""),
        }
        
        self._progress(
            f"favourite source {post_rank}/{total_results}: {results[selected_index].title[:80]}"
        )
        return [results[selected_index]]

    def _filter_fields(self, payload: dict[str, Any], model: Any) -> dict[str, Any]:
        reserved = {"agent_id", "episode_id"}
        return {key: value for key, value in payload.items() if key in model.model_fields and key not in reserved}

    def _safe_text(self, value: str, fallback: str, limit: int = 240) -> str:
        text = str(value or "").strip()
        lowered = text.lower()
        if any(term in lowered for term in BLOCKED_PROFILE_TERMS):
            return fallback
        return text[:limit] or fallback

    def _apply_profile_update(
        self,
        profile: AgentProfile,
        proposal: ProfileUpdateProposal,
        episode_id: str,
        search_query: str,
        selected_sources: list[Any],
        episode_json_valid: bool,
    ) -> tuple[AgentProfile, dict[str, Any]]:
        if not self.config.get("allow_profile_drift", True):
            return profile, {"applied": False, "reason": "profile drift disabled"}
        now = utc_now()
        domains = sorted({self._domain(source.url) for source in selected_sources if self._domain(source.url)})
        first_result_topic = selected_sources[0].title if selected_sources else ""
        safe_candidates = []
        rejected_candidates = []
        if episode_json_valid:
            for item in proposal.candidate_interests:
                safe_interest, rejection_reason = self._safe_interest(item)
                if safe_interest:
                    safe_candidates.append(safe_interest)
                elif rejection_reason:
                    rejected_candidates.append({"interest": str(item), "reason": rejection_reason})
        observation = {
            "episode_id": episode_id,
            "timestamp": now,
            "query": search_query,
            "source_domains": domains,
            "source_urls": sorted(
                {self._url_key(source.url) for source in selected_sources if self._url_key(source.url)}
            ),
            "first_result_topic": first_result_topic,
            "observations": clamp_list(proposal.observations, 8),
            "candidate_interests": clamp_list(safe_candidates, 8),
            "json_valid": episode_json_valid,
        }
        observations = (profile.observations + [observation])[-100:]
        tentative, stable_from_observations, rejected = self._derive_interest_state(observations)
        seed_interests = clamp_list((profile.initial_profile or {}).get("current_interests"), 8)
        stable = clamp_list(seed_interests + stable_from_observations, 8)
        recent_queries = clamp_list((profile.recent_queries + [search_query])[-12:], 12)
        memory = profile.recent_memory_summary
        single_source_facts = []
        if episode_json_valid and proposal.recent_memory_summary:
            memory, single_source_facts = self._clean_memory_summary(
                proposal.recent_memory_summary,
                memory,
                domains,
                [source.url for source in selected_sources],
            )
        applied: dict[str, Any] = {
            "applied": True,
            "from_version": profile.version,
            "to_version": profile.version + 1,
            "observation_added": True,
            "episode_json_valid": episode_json_valid,
            "tentative_interests": tentative[:8],
            "stable_interests": stable,
            "promoted_interests": [
                item for item in stable_from_observations if item not in profile.stable_interests
            ],
        }
        if rejected_candidates:
            applied["rejected_candidate_interests"] = rejected_candidates
        if single_source_facts:
            applied["single_source_facts"] = single_source_facts
        if rejected:
            applied["rejected_update"] = rejected
        now = utc_now()
        updated = profile.model_copy(
            update={
                "first_person_profile": profile.first_person_profile,
                "current_interests": stable,
                "preferred_sources": profile.preferred_sources,
                "search_style": profile.search_style,
                "uncertainty_style": profile.uncertainty_style,
                "self_rules": profile.self_rules,
                "recent_memory_summary": memory,
                "observations": observations,
                "tentative_interests": tentative,
                "stable_interests": stable,
                "recent_queries": recent_queries,
                "version": profile.version + 1,
                "updated_at": now,
            }
        )
        return updated, applied

    def _derive_interest_state(self, observations: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
        stats: dict[str, dict[str, Any]] = {}
        rejected: dict[str, Any] = {"reason": "not_enough_repeated_evidence", "items": []}
        valid_observations = [item for item in observations if item.get("json_valid")]
        for observation in valid_observations:
            query = str(observation.get("query", ""))
            for interest in observation.get("candidate_interests", []):
                interest, _ = self._safe_interest(interest)
                if not interest:
                    continue
                key = normalise_title(interest)
                if not key:
                    continue
                bucket = stats.setdefault(
                    key,
                    {
                        "interest": interest,
                        "episodes": set(),
                        "domains": set(),
                        "queries": [],
                        "last_seen": observation.get("timestamp", ""),
                    },
                )
                bucket["episodes"].add(observation.get("episode_id", ""))
                bucket["domains"].update(observation.get("source_domains", []))
                bucket["queries"].append(query)
                bucket["last_seen"] = observation.get("timestamp", "")
        tentative: list[dict[str, Any]] = []
        stable: list[str] = []
        for bucket in stats.values():
            episodes = sorted(item for item in bucket["episodes"] if item)
            domains = sorted(item for item in bucket["domains"] if item)
            repeated_query = self._queries_are_near_identical(bucket["queries"])
            promoted = len(episodes) >= 3 and len(domains) >= 2 and not repeated_query
            item = {
                "interest": bucket["interest"],
                "episode_count": len(episodes),
                "domain_count": len(domains),
                "domains": domains[:6],
                "last_seen": bucket["last_seen"],
                "status": "stable" if promoted else "tentative",
            }
            tentative.append(item)
            if promoted:
                stable.append(bucket["interest"])
            else:
                rejected["items"].append(
                    {
                        "interest": bucket["interest"],
                        "episode_count": len(episodes),
                        "domain_count": len(domains),
                        "near_identical_queries": repeated_query,
                    }
                )
        tentative.sort(key=lambda item: (item["status"] != "stable", -item["episode_count"], item["interest"]))
        stable = clamp_list(stable, 8)
        if not rejected["items"]:
            rejected = {}
        return tentative[:20], stable, rejected

    def _safe_interest(self, value: str) -> tuple[str, str]:
        text = str(value or "").strip()
        lowered = text.lower()
        words = [word for word in re.split(r"\s+", text) if word]
        if not text:
            return "", "empty"
        if lowered in FILLER_INTERESTS or set(lowered) <= {".", "…"}:
            return "", "filler"
        if lowered in VAGUE_INTERESTS:
            return "", "too_vague"
        if any(term in lowered for term in BLOCKED_PROFILE_TERMS):
            return "", "blocked_profile_term"
        if text.startswith(("{", "[")) or "'interest':" in lowered or '"interest":' in lowered:
            return "", "malformed_json_like_text"
        if any(term in lowered for term in STRICT_RULE_TERMS):
            return "", "strict_rule_language"
        if text.count(",") >= 2 or text.count(";") >= 2:
            return "", "packed_interest_bundle"
        if len(words) > 10:
            return "", "too_long"
        return text[:160], ""

    def _clean_memory_summary(
        self,
        value: str,
        fallback: str,
        domains: list[str],
        urls: list[str],
    ) -> tuple[str, list[dict[str, Any]]]:
        text = self._safe_text(value, fallback, limit=800)
        if not text:
            return fallback, []
        facts = []
        clean_sentences = []
        for sentence in self._sentences(text):
            if self._number_heavy(sentence) and len(domains) <= 1:
                facts.append(
                    {
                        "fact": sentence[:240],
                        "source_domains": domains,
                        "source_urls": [self._url_key(url) for url in urls if self._url_key(url)],
                        "status": "single_source",
                    }
                )
                continue
            clean_sentences.append(sentence)
        cleaned = " ".join(clean_sentences).strip()
        return (cleaned[:800] if cleaned else fallback), facts

    def _sentences(self, text: str) -> list[str]:
        return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]

    def _number_heavy(self, text: str) -> bool:
        return bool(re.search(r"\d{1,3}(?:,\d{3})+|\b\d{4,}\b", text))

    def _diary_artifacts(self, diary: DiaryEntry) -> dict[str, Any]:
        text = " ".join(
            [
                diary.diary_summary,
                diary.what_caught_attention,
                diary.what_was_uncertain,
                diary.possible_next_interest,
            ]
        )
        matches = []
        for pattern in EMBODIED_DIARY_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                matches.append(pattern.replace(r"\b", "").replace("\\", ""))
        return {
            "embodied_language_detected": bool(matches),
            "patterns": matches,
        }

    def _source_type(self, url: str) -> str:
        domain = self._domain(url)
        path = urlparse(url or "").path.lower()
        if not domain:
            return "unknown"
        if domain in {"facebook.com", "reddit.com", "x.com", "twitter.com", "instagram.com", "youtube.com"}:
            return "social"
        if any(part in domain for part in ["jstor.org", "springer.com", "sciencedirect.com", "ncbi.nlm.nih.gov"]):
            return "academic"
        if domain.endswith(".edu") or ".edu." in domain or domain.endswith(".ac.uk"):
            return "academic"
        if domain.endswith(".gov") or ".gov." in domain or domain.endswith(".int"):
            return "official"
        if any(part in domain for part in ["museum", "unesco.org", "kew.org", "gbif.org", "bgbm.org", "rbge.org.uk"]):
            return "official"
        if any(part in domain for part in ["wikipedia.org", "britannica.com", "ebsco.com"]):
            return "reference"
        if any(part in domain for part in ["news", "magazine", "times", "guardian", "bbc.", "vulture.com"]):
            return "news_or_magazine"
        if any(part in domain for part in ["amazon.", "shop", "store"]) or "/product" in path:
            return "commercial"
        return "unknown"

    def _domain(self, url: str) -> str:
        host = urlparse(url or "").netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host

    def _url_key(self, url: str) -> str:
        parsed = urlparse(url or "")
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        path = parsed.path.rstrip("/")
        return f"{host}{path}".lower() if host else ""

    def _anti_anchor_query(self, profile: AgentProfile, query: str) -> tuple[str, dict[str, Any]]:
        recent = profile.recent_queries[-3:]
        previous_topics = [str(item.get("first_result_topic", "")) for item in profile.observations[-5:]]
        anchors = recent + previous_topics
        for anchor in anchors:
            if self._text_overlap(query, anchor) >= 0.72:
                diversified = f"{query} contrasting perspectives alternative sources"
                return diversified[:240], {
                    "stage": "anti_anchoring",
                    "mode": "explore_contrast_source_diversify",
                    "reason": "query_overlap_with_recent_query_or_first_result_topic",
                    "anchor": anchor[:240],
                    "original_query": query,
                    "diversified_query": diversified[:240],
                }
        return query, {}

    def _queries_are_near_identical(self, queries: list[str]) -> bool:
        unique = [query for index, query in enumerate(queries) if query and query not in queries[:index]]
        if len(unique) <= 1 and len(queries) > 1:
            return True
        for index, query in enumerate(unique):
            for other in unique[index + 1 :]:
                if self._text_overlap(query, other) >= 0.82:
                    return True
        return False

    def _text_overlap(self, left: str, right: str) -> float:
        left_tokens = self._tokens(left)
        right_tokens = self._tokens(right)
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))

    def _tokens(self, text: str) -> set[str]:
        stop = {"the", "and", "for", "with", "from", "into", "that", "this", "what", "how", "why", "are", "is", "in", "of", "to", "a"}
        return {token for token in normalise_title(text).replace("-", " ").split() if len(token) > 2 and token not in stop}


def print_episode_summary(episode: EpisodeLog) -> None:
    print("\nEpisode complete\n")
    print(f"Agent: {episode.agent_id}")
    print(f"Profile version before: {episode.profile_version_before}")
    print(f"Profile version after: {episode.profile_version_after}")
    print(f"Search query: {episode.search_query}")
    if episode.prompt_metadata:
        print(f"Prompt version: {episode.prompt_metadata.get('prompt_version')}")
    backends = ", ".join(sorted({r.backend for r in episode.search_results})) or "none"
    print(f"Search backends used: {backends}")
    print("Selected sources:")
    for source in episode.selected_sources:
        print(f"- [{source.backend}] {source.title} {source.url}")
    if episode.source_selection:
        print(f"Favourite source reason: {episode.source_selection.get('selection_reason', '')}")
    if episode.source_selection_metrics:
        post_rank = episode.source_selection_metrics.get("post_rank")
        total_results = episode.source_selection_metrics.get("total_results")
        print(f"Post-rank: {post_rank}/{total_results}")
    diary = episode.diary_entry.diary_summary if episode.diary_entry else ""
    print(f"Diary summary: {diary}")
    print(f"Applied profile update: {episode.applied_profile_update}")
    if episode.errors:
        print("Errors:")
        for error in episode.errors:
            print(f"- {error.get('stage', 'unknown')}: {error.get('error') or error.get('detail') or error}")
    else:
        print("Errors: none")
