from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .config import data_path
from .utils import read_json


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def analyze() -> str:
    profiles = sorted(data_path("profiles").glob("*.json"))
    episodes = load_jsonl(data_path("logs", "episodes.jsonl"))
    lines = ["World Model Drift Summary", ""]
    lines.append("Current interests by agent:")
    for path in profiles:
        profile = read_json(path, {})
        lines.append(f"- {profile.get('agent_id')}: {profile.get('current_interests', [])}")
    lines.append("")
    lines.append("Preferred sources by agent:")
    for path in profiles:
        profile = read_json(path, {})
        lines.append(f"- {profile.get('agent_id')}: {profile.get('preferred_sources', [])}")
    queries = [row.get("search_query", "") for row in episodes if row.get("search_query")]
    terms = Counter(word.lower().strip(".,:;!?()[]") for query in queries for word in query.split() if len(word) > 3)
    sources = Counter()
    errors = Counter()
    versions = Counter()
    for row in episodes:
        versions[row.get("agent_id", "unknown")] = max(
            versions[row.get("agent_id", "unknown")], int(row.get("profile_version_after") or 0)
        )
        for result in row.get("search_results", []):
            source = result.get("source") or result.get("backend") or "unknown"
            sources[source] += 1
        for error in row.get("errors", []):
            errors[error.get("stage", "unknown")] += 1
    lines.extend(
        [
            "",
            f"Repeated search terms: {terms.most_common(15)}",
            f"Number of profile versions: {dict(versions)}",
            f"Most common sources: {sources.most_common(10)}",
            f"Error counts: {dict(errors)}",
            "",
            "Rough divergence notes:",
        ]
    )
    if len(profiles) <= 1:
        lines.append("- Single-profile mode: inspect topic and source drift across versions rather than between agents.")
    else:
        lines.append("- Compare interests and preferred sources above for between-agent divergence.")
    if queries:
        lines.append(f"- Recorded {len(queries)} search queries; repeated terms indicate topic stickiness.")
    else:
        lines.append("- No completed search queries yet.")
    return "\n".join(lines)
