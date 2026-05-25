from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]


class Settings:
    def __init__(self) -> None:
        load_dotenv(ROOT / ".env")
        self.root = ROOT
        self.llm_base_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:10000/v1").rstrip("/")
        self.llm_api_key = os.getenv("LLM_API_KEY", "not-needed")
        self.llm_model = os.getenv("LLM_MODEL", "local-model")
        self.llm_max_tokens = int(os.getenv("LLM_MAX_TOKENS", "700"))
        self.llm_repair_max_tokens = int(os.getenv("LLM_REPAIR_MAX_TOKENS", "300"))
        self.llm_retry_attempts = int(os.getenv("LLM_RETRY_ATTEMPTS", "3"))
        self.llm_retry_initial_delay = float(os.getenv("LLM_RETRY_INITIAL_DELAY", "1.0"))
        self.llm_retry_max_delay = float(os.getenv("LLM_RETRY_MAX_DELAY", "8.0"))
        self.llm_restart_command = os.getenv("LLM_RESTART_COMMAND", "").strip()
        self.llm_stop_command = os.getenv("LLM_STOP_COMMAND", "").strip()
        self.llm_restart_wait = float(os.getenv("LLM_RESTART_WAIT", "10.0"))
        self.llm_restart_log = os.getenv(
            "LLM_RESTART_LOG",
            str(ROOT / "data" / "logs" / "llama_cpp_server.log"),
        )
        self.searxng_url = os.getenv("SEARXNG_URL", "http://127.0.0.1:8888/search")
        self.wikipedia_enabled = os.getenv("WIKIPEDIA_ENABLED", "true").lower() == "true"
        self.gdelt_enabled = os.getenv("GDELT_ENABLED", "false").lower() == "true"
        self.crossref_mailto = os.getenv("CROSSREF_MAILTO", "")


def load_yaml(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or default


def load_experiment_config() -> dict[str, Any]:
    return load_yaml(ROOT / "configs" / "experiment.yaml", {})


def load_agents_config() -> dict[str, Any]:
    return load_yaml(ROOT / "configs" / "agents.yaml", {"agents": []})


def data_path(*parts: str) -> Path:
    return ROOT / "data" / Path(*parts)
