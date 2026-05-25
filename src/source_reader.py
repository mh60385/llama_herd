from __future__ import annotations

from typing import Any

import requests
from bs4 import BeautifulSoup

from .schemas import SearchResult
from .search import cache_source_text


class SourceReader:
    def __init__(self, max_chars: int = 4000) -> None:
        self.max_chars = max_chars

    def read(self, result: SearchResult) -> tuple[str, str | None]:
        if not result.url:
            text = result.snippet or ""
            if text:
                cache_source_text(result.title or "snippet", text)
            return text[: self.max_chars], "missing_url"
        try:
            response = requests.get(
                result.url,
                timeout=20,
                headers={"User-Agent": "llama-herd-drift-lab/0.1"},
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                tag.decompose()
            text = " ".join(soup.get_text(" ").split())[: self.max_chars]
            if not text:
                raise RuntimeError("empty extracted text")
            cache_source_text(result.url, text)
            return text, None
        except Exception as exc:
            fallback = result.snippet or ""
            if fallback:
                cache_source_text(result.url or result.title, fallback)
            return fallback[: self.max_chars], str(exc)
