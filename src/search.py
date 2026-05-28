from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

from .config import SearchConfig, data_path
from .schemas import SearchResult
from .utils import normalise_title, slugify, utc_now, write_json


class SearchManager:
    def __init__(self, search_config: SearchConfig | None = None, max_results: int = 5) -> None:
        self.config = search_config or SearchConfig()
        self.max_results = max_results
        self.errors: list[dict[str, Any]] = []
        self.cache_dir = data_path("sources", "search_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.headers = {"User-Agent": "llama-herd-drift-lab/0.1 (local research; contact: local)"}

    def search(self, query: str) -> list[SearchResult]:
        backend_results: list[list[SearchResult]] = []
        for backend in self._enabled_backends():
            try:
                results = backend(query)
                print(f"[llama_herd] search backend {backend.__name__} returned {len(results)} results", flush=True)
                backend_results.append(results)
            except Exception as exc:
                print(f"[llama_herd] search backend warning: {backend.__name__}: {exc}", flush=True)
                self.errors.append(
                    {"stage": "search", "backend": backend.__name__, "error": str(exc), "timestamp": utc_now()}
                )
        return self._balanced_dedupe(backend_results)[: self.max_results]

    def _enabled_backends(self):
        backends = [self._search_searxng]
        if self.config.crossref_mailto and "example.com" not in self.config.crossref_mailto:
            backends.append(self._search_crossref)
        if self.config.gdelt_enabled:
            backends.append(self._search_gdelt)
        return backends

    def _cache(self, backend: str, query: str, payload: Any) -> None:
        digest = hashlib.sha256(f"{backend}:{query}".encode("utf-8")).hexdigest()[:16]
        write_json(self.cache_dir / f"{utc_now()[:10]}-{backend}-{digest}.json", payload)

    def _search_searxng(self, query: str) -> list[SearchResult]:
        params = {"q": query, "format": "json"}
        url = self.config.searxng_url
        response = requests.get(url, params=params, timeout=20, headers=self.headers)
        if response.status_code == 403:
            raise RuntimeError("SearXNG returned 403; JSON format may be disabled or blocked")
        response.raise_for_status()
        payload = response.json()
        self._cache("searxng", query, payload)
        results = []
        for item in payload.get("results", []):
            engines = item.get("engines") or [item.get("engine") or "searxng"]
            results.append(
                SearchResult(
                    backend="searxng",
                    title=item.get("title") or "",
                    url=item.get("url") or "",
                    snippet=item.get("content") or "",
                    source=", ".join(str(e) for e in engines if e),
                    published_date=item.get("publishedDate"),
                    score=float(item.get("score") or 0.0),
                    raw=item,
                )
            )
        return results

    def _search_crossref(self, query: str) -> list[SearchResult]:
        params = {"query": query, "rows": self.max_results, "mailto": self.config.crossref_mailto}
        response = requests.get("https://api.crossref.org/works", params=params, timeout=20, headers=self.headers)
        response.raise_for_status()
        payload = response.json()
        self._cache("crossref", query, payload)
        results = []
        for item in payload.get("message", {}).get("items", []):
            title = (item.get("title") or [""])[0]
            authors = [
                " ".join(part for part in [a.get("given"), a.get("family")] if part)
                for a in item.get("author", [])[:5]
            ]
            year_parts = item.get("published-print") or item.get("published-online") or {}
            year = (year_parts.get("date-parts") or [[None]])[0][0]
            results.append(
                SearchResult(
                    backend="crossref",
                    title=title,
                    url=item.get("URL") or "",
                    snippet=item.get("abstract") or "",
                    source="Crossref",
                    published_date=str(year) if year else None,
                    authors=authors,
                    doi=item.get("DOI"),
                    score=float(item.get("score") or 0.0),
                    raw=item,
                )
            )
        return results

    def _search_gdelt(self, query: str) -> list[SearchResult]:
        params = {"query": query, "mode": "ArtList", "format": "json", "maxrecords": self.max_results}
        response = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=20, headers=self.headers)
        response.raise_for_status()
        payload = response.json()
        self._cache("gdelt", query, payload)
        return [
            SearchResult(
                backend="gdelt",
                title=item.get("title") or "",
                url=item.get("url") or "",
                snippet=item.get("seendate") or "",
                source=item.get("sourceCountry") or item.get("domain") or "GDELT",
                published_date=item.get("seendate"),
                raw=item,
            )
            for item in payload.get("articles", [])
        ]

    def _balanced_dedupe(self, backend_results: list[list[SearchResult]]) -> list[SearchResult]:
        seen: set[str] = set()
        deduped: list[SearchResult] = []
        max_len = max((len(results) for results in backend_results), default=0)
        for index in range(max_len):
            for results in backend_results:
                if index >= len(results):
                    continue
                self._append_unique(results[index], seen, deduped)
        return deduped

    def _append_unique(self, result: SearchResult, seen: set[str], deduped: list[SearchResult]) -> None:
            keys = [result.url.strip().lower(), (result.doi or "").strip().lower(), normalise_title(result.title)]
            key = next((k for k in keys if k), "")
            if not key or key in seen:
                return
            seen.add(key)
            deduped.append(result)


def cache_source_text(url: str, text: str) -> Path:
    name = slugify(url, "source")
    path = data_path("sources", f"{name}.txt")
    path.write_text(text, encoding="utf-8")
    return path
