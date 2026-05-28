#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import data_path, get_llm_config, get_search_config
from src.llm_client import LLMClient
from src.storage import Storage


def _mem_available_mb() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        return -1
    return -1


def main() -> None:
    llm_config = get_llm_config()
    search_config = get_search_config()
    print("Jetson Local LLM Drift Lab Infrastructure Check\n")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Machine: {__import__('platform').machine()}")
    print(f"LLM base URL: {llm_config.base_url}")
    print(f"SearXNG URL: {search_config.searxng_url}")
    print(f"Available memory: {_mem_available_mb()} MB")
    usage = shutil.disk_usage(Path.cwd())
    print(f"Free disk: {usage.free // (1024 ** 3)} GB")

    Storage()
    print(f"Data directories: ok ({data_path()})")

    llm = LLMClient()
    ok, detail = llm.healthcheck()
    print(f"llama.cpp /v1/models: {'ok' if ok else 'failed'} ({detail})")

    try:
        response = requests.get(search_config.searxng_url, params={"q": "jetson orin nano", "format": "json"}, timeout=10)
        print(f"SearXNG JSON: HTTP {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"SearXNG result count: {len(data.get('results', []))}")
    except Exception as exc:
        print(f"SearXNG JSON: warning ({exc})")

    print("\nPreflight policy: LLM calls are retried; if configured, llama.cpp is restarted once before failing.")


if __name__ == "__main__":
    main()
