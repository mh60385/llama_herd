from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class LLMConfig:
    """LLM server configuration."""

    base_url: str = "http://127.0.0.1:10000/v1"
    api_key: str = "not-needed"
    model: str = "local-model"
    max_tokens: int = 300
    repair_max_tokens: int = 300
    retry_attempts: int = 3
    retry_initial_delay: float = 1.0
    retry_max_delay: float = 8.0
    restart_command: str = ""
    stop_command: str = ""
    restart_wait: float = 10.0
    restart_log: str = ""

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create from environment variables with LLM_ prefix."""
        return cls(
            base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:10000/v1").rstrip("/"),
            api_key=os.getenv("LLM_API_KEY", "not-needed"),
            model=os.getenv("LLM_MODEL", "local-model"),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "300")),
            repair_max_tokens=int(os.getenv("LLM_REPAIR_MAX_TOKENS", "300")),
            retry_attempts=int(os.getenv("LLM_RETRY_ATTEMPTS", "3")),
            retry_initial_delay=float(os.getenv("LLM_RETRY_INITIAL_DELAY", "1.0")),
            retry_max_delay=float(os.getenv("LLM_RETRY_MAX_DELAY", "8.0")),
            restart_command=os.getenv("LLM_RESTART_COMMAND", "").strip(),
            stop_command=os.getenv("LLM_STOP_COMMAND", "").strip(),
            restart_wait=float(os.getenv("LLM_RESTART_WAIT", "10.0")),
            restart_log=os.getenv("LLM_RESTART_LOG", ""),
        )


@dataclass
class SearchConfig:
    """Search backend configuration."""

    searxng_url: str = "http://127.0.0.1:8888/search"
    gdelt_enabled: bool = False
    crossref_mailto: str = ""
    wikipedia_enabled: bool = False

    @classmethod
    def from_env(cls) -> "SearchConfig":
        """Create from environment variables."""
        return cls(
            searxng_url=os.getenv("SEARXNG_URL", "http://127.0.0.1:8888/search"),
            gdelt_enabled=os.getenv("GDELT_ENABLED", "false").lower() == "true",
            crossref_mailto=os.getenv("CROSSREF_MAILTO", ""),
            wikipedia_enabled=os.getenv("WIKIPEDIA_ENABLED", "false").lower() == "true",
        )


@dataclass
class SystemMonitorConfig:
    """System monitoring thresholds."""

    memory_warn_available_mb: int = 1200
    memory_restart_available_mb: int = 800
    swap_warn_used_mb: int = 3200
    swap_stop_used_mb: int = 3700
    temperature_warn_c: float = 75.0
    temperature_pause_c: float = 78.0
    temperature_stop_c: float = 83.0
    thermal_cooldown_seconds: float = 90.0
    thermal_max_cooldown_cycles: int = 5
    llm_restart_every_episodes: int = 0
    llm_restart_if_memory_trending_down: bool = True
    llm_memory_trend_window: int = 4
    llm_min_restart_gap_episodes: int = 3


@dataclass
class ExperimentConfig:
    """Experiment configuration."""

    number_of_agents: int = 5
    episodes_per_run: int = 100
    max_search_results: int = 10
    max_source_chars: int = 4000
    temperature: float = 0.7
    top_p: float = 0.9
    inter_episode_delay_seconds: float = 2.0
    system_monitor_enabled: bool = True
    reasoning: str = "off"
    profile_update_mode: str = "proposal_then_apply"
    allow_profile_drift: bool = True
    search_failure_is_valid_episode_data: bool = True

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        """Create from YAML file."""
        data = load_yaml(path, {})
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AppConfig:
    """Main application configuration."""

    llm: LLMConfig = field(default_factory=LLMConfig.from_env)
    search: SearchConfig = field(default_factory=SearchConfig.from_env)
    monitor: SystemMonitorConfig = field(default_factory=SystemMonitorConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


@lru_cache()
def get_config() -> AppConfig:
    """Get cached application configuration."""
    load_dotenv(ROOT / ".env")
    return AppConfig()


def get_llm_config() -> LLMConfig:
    """Get LLM configuration for use in modules that need it."""
    return get_config().llm


def get_search_config() -> SearchConfig:
    """Get search configuration for use in modules that need it."""
    return get_config().search


def get_monitor_config() -> SystemMonitorConfig:
    """Get monitor configuration for use in modules that need it."""
    return get_config().monitor


def load_yaml(path: Path, default: Any = None) -> Any:
    if default is None:
        default = {}
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or default


def load_experiment_config() -> dict[str, Any]:
    """Load experiment config from YAML file."""
    yaml_config = load_yaml(ROOT / "configs" / "experiment.yaml", {})
    return yaml_config


def load_agents_config() -> dict[str, Any]:
    return load_yaml(ROOT / "configs" / "agents.yaml", {"agents": []})


def data_path(*parts: str) -> Path:
    return ROOT / "data" / Path(*parts)
