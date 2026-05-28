from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


# Blocked terms for profile validation
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
    "artifacts", "culture", "history", "museums", "research", "science", "sources", "travel",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL file into list of dicts."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write list of dicts to CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def slugify(value: str, fallback: str = "item") -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug[:80] or fallback


def normalise_title(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def clamp_list(values: list[str] | None, limit: int) -> list[str]:
    seen: set[str] = set()
    return [
        str(v).strip()[:160]
        for v in values or []
        if str(v).strip() and (key := str(v).strip().lower()) not in seen and not seen.add(key)
    ][:limit]


def domain_from_url(url: str) -> str:
    """Extract domain from URL, removing www. prefix."""
    if not url:
        return ""
    host = urlparse(url).netloc.lower()
    return host.removeprefix("www.")


def url_key(url: str) -> str:
    """Create a normalized URL key for deduplication."""
    parsed = urlparse(url or "")
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path.rstrip("/")
    return f"{host}{path}".lower() if host else ""


def classify_source_type(url: str) -> str:
    """Classify source type based on URL domain."""
    domain = domain_from_url(url)
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
    if any(part in domain for part in ["amazon.", "shop", "store"]) or "/product" in url:
        return "commercial"
    return "unknown"


def sanitize_text(value: Any, fallback: str = "", limit: int = 180) -> str:
    """Sanitize text for profile fields, filtering blocked and strict rule terms."""
    text = str(value or "").strip()
    lowered = text.lower()
    if any(term in lowered for term in BLOCKED_PROFILE_TERMS):
        return fallback
    if any(term in lowered for term in STRICT_RULE_TERMS):
        return fallback
    return text[:limit] or fallback


def _tokens(text: str) -> set[str]:
    """Tokenize text for overlap calculations."""
    stop = {"the", "and", "for", "with", "from", "into", "that", "this", "what", "how", "why", "are", "is", "in", "of", "to", "a"}
    return {token for token in normalise_title(text).replace("-", " ").split() if len(token) > 2 and token not in stop}


def text_overlap(left: str, right: str) -> float:
    """Calculate token overlap between two texts."""
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
