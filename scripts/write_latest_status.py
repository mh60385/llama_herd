#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import data_path
from src.utils import read_json, utc_now


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a compact Markdown status report for the latest run.")
    parser.add_argument("--episodes", default=str(data_path("logs", "episodes.jsonl")))
    parser.add_argument("--profiles", default=str(data_path("profiles")))
    parser.add_argument("--out", default=str(data_path("logs", "latest_status.md")))
    parser.add_argument("--last", type=int, default=12)
    args = parser.parse_args()

    episodes = read_jsonl(Path(args.episodes))
    profiles = read_profiles(Path(args.profiles))
    report = build_report(episodes, profiles, args.last)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(out)


def build_report(episodes: list[dict], profiles: dict[str, dict], last_count: int) -> str:
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for episode in episodes:
        by_agent[str(episode.get("agent_id", ""))].append(episode)

    lines = [
        "# llama_herd Latest Status",
        "",
        f"Updated: `{utc_now()}`",
        "",
        "## Run Summary",
        "",
        f"- Episodes logged: `{len(episodes)}`",
        f"- Profiles found: `{len(profiles)}`",
        "",
        "## Agents",
        "",
    ]

    for agent_id in sorted(profiles):
        profile = profiles[agent_id]
        agent_episodes = by_agent.get(agent_id, [])
        last_episode = agent_episodes[-1] if agent_episodes else {}
        stable = profile.get("stable_interests") or []
        tentative = profile.get("tentative_interests") or []
        lines.extend(
            [
                f"### {agent_id}",
                "",
                f"- Profile version: `{profile.get('version', '')}`",
                f"- Episodes logged: `{len(agent_episodes)}`",
                f"- Last query: `{last_episode.get('search_query', '')}`",
                f"- Stable interests: {inline_list(stable)}",
                f"- Tentative interests: `{len(tentative)}`",
                "",
            ]
        )

    lines.extend(["## Recent Episodes", ""])
    for episode in episodes[-last_count:]:
        errors = [str(error.get("stage", "error")) for error in episode.get("errors", [])]
        promoted = episode.get("applied_profile_update", {}).get("promoted_interests") or []
        selected = (episode.get("selected_sources") or [{}])[0]
        lines.extend(
            [
                f"### {episode.get('agent_id', '')} / profile v{episode.get('profile_version_after', '')}",
                "",
                f"- Query: `{episode.get('search_query', '')}`",
                f"- Selected: `{selected.get('title', '')}`",
                f"- Domain: `{domain_from_url(selected.get('url', ''))}`",
                f"- Promoted: {inline_list(promoted)}",
                f"- Errors: {inline_list(errors)}",
                "",
            ]
        )

    lines.extend(["## Source Domains", ""])
    domains = Counter(
        domain_from_url((episode.get("selected_sources") or [{}])[0].get("url", ""))
        for episode in episodes
    )
    for domain, count in domains.most_common(12):
        if domain:
            lines.append(f"- `{domain}`: `{count}`")
    lines.append("")
    return "\n".join(lines)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_profiles(path: Path) -> dict[str, dict]:
    profiles = {}
    for profile_path in sorted(path.glob("*.json")):
        payload = read_json(profile_path, {})
        if payload:
            profiles[str(payload.get("agent_id") or profile_path.stem)] = payload
    return profiles


def inline_list(values: list[str]) -> str:
    clean = [str(value).strip() for value in values if str(value).strip()]
    if not clean:
        return "`none`"
    return ", ".join(f"`{value}`" for value in clean[:8])


def domain_from_url(url: str) -> str:
    return urlparse(str(url)).netloc.lower().removeprefix("www.")


if __name__ == "__main__":
    main()
