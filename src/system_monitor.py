from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import Any

from .config import SystemMonitorConfig, data_path, get_monitor_config
from .utils import append_jsonl, utc_now


class SystemMonitor:
    def __init__(self, monitor_config: SystemMonitorConfig | None = None, config: dict[str, Any] | None = None) -> None:
        self.config = monitor_config or get_monitor_config()
        self.experiment_config = config or {}
        self.enabled = bool(self.experiment_config.get("system_monitor_enabled", True))
        self.container_name = str(self.experiment_config.get("llm_container_name", "llama-server"))
        self.log_path = data_path("logs", "system_monitor.jsonl")
        self.memory_restart_mb = self.config.memory_restart_available_mb
        self.memory_warn_mb = self.config.memory_warn_available_mb
        self.swap_warn_mb = self.config.swap_warn_used_mb
        self.swap_stop_mb = self.config.swap_stop_used_mb
        self.temp_warn_c = self.config.temperature_warn_c
        self.temp_pause_c = self.config.temperature_pause_c
        self.temp_stop_c = self.config.temperature_stop_c
        self.cooldown_seconds = self.config.thermal_cooldown_seconds
        self.max_cooldown_cycles = self.config.thermal_max_cooldown_cycles
        self.restart_every_episodes = int(self.experiment_config.get("llm_restart_every_episodes", self.config.llm_restart_every_episodes))
        self.restart_if_memory_trending_down = bool(self.experiment_config.get("llm_restart_if_memory_trending_down", self.config.llm_restart_if_memory_trending_down))
        self.memory_trend_window = int(self.experiment_config.get("llm_memory_trend_window", self.config.llm_memory_trend_window))
        self.min_restart_gap_episodes = int(self.experiment_config.get("llm_min_restart_gap_episodes", self.config.llm_min_restart_gap_episodes))
        self._available_history: list[int] = []
        self._last_restart_episode = -10_000

    def pre_episode(self, episode_index: int, total_episodes: int, progress: Any = print) -> dict[str, Any]:
        snapshot = self.snapshot(episode_index, total_episodes)
        action = "continue"
        reasons: list[str] = []
        if not self.enabled:
            snapshot["action"] = "disabled"
            self._log(snapshot)
            return snapshot

        max_temp = snapshot.get("max_temp_c")
        if isinstance(max_temp, (int, float)) and max_temp >= self.temp_stop_c:
            snapshot["action"] = "stop"
            snapshot["reasons"] = [f"temperature_stop_c:{max_temp}"]
            self._log(snapshot)
            raise RuntimeError(f"System temperature too high to continue: {max_temp}C")

        swap_used = snapshot.get("swap_used_mb")
        if self.swap_stop_mb > 0 and isinstance(swap_used, int) and swap_used >= self.swap_stop_mb:
            snapshot["action"] = "stop"
            snapshot["reasons"] = [f"swap_used_mb_above_stop:{swap_used}"]
            self._log(snapshot)
            raise RuntimeError(f"System swap use too high to continue safely: {swap_used}MB")

        if isinstance(max_temp, (int, float)) and max_temp >= self.temp_pause_c:
            action = "thermal_pause"
            reasons.append(f"temperature_pause_c:{max_temp}")
            snapshot = self._cool_down(snapshot, episode_index, total_episodes, progress, reasons)

        available_mb = snapshot.get("available_mb")
        if isinstance(available_mb, int):
            self._available_history.append(available_mb)
            self._available_history = self._available_history[-max(2, self.memory_trend_window) :]

        restart_gap_ok = episode_index - self._last_restart_episode >= self.min_restart_gap_episodes
        memory_trending_down = self._memory_trending_down()
        should_scheduled_restart = (
            self.restart_every_episodes > 0
            and episode_index > 0
            and episode_index % self.restart_every_episodes == 0
            and restart_gap_ok
        )
        should_trend_restart = (
            self.restart_if_memory_trending_down
            and restart_gap_ok
            and memory_trending_down
            and isinstance(available_mb, int)
            and available_mb < self.memory_warn_mb
        )

        if isinstance(available_mb, int) and available_mb < self.memory_restart_mb and restart_gap_ok:
            action = "restart_llm"
            reasons.append(f"available_mb_below_restart:{available_mb}")
            self.restart_llm(progress)
            self._last_restart_episode = episode_index
            snapshot = self.snapshot(episode_index, total_episodes)
        elif should_scheduled_restart:
            action = "scheduled_restart_llm"
            reasons.append(f"scheduled_restart_every:{self.restart_every_episodes}")
            self.restart_llm(progress)
            self._last_restart_episode = episode_index
            snapshot = self.snapshot(episode_index, total_episodes)
        elif should_trend_restart:
            action = "trend_restart_llm"
            reasons.append(f"available_mb_trending_down:{self._available_history}")
            self.restart_llm(progress)
            self._last_restart_episode = episode_index
            snapshot = self.snapshot(episode_index, total_episodes)
        elif isinstance(available_mb, int) and available_mb < self.memory_warn_mb:
            reasons.append(f"available_mb_below_warn:{available_mb}")

        swap_used = snapshot.get("swap_used_mb")
        if isinstance(swap_used, int) and swap_used >= self.swap_warn_mb:
            reasons.append(f"swap_used_mb_above_warn:{swap_used}")

        if not reasons:
            reasons.append("ok")
        snapshot["action"] = action
        snapshot["reasons"] = reasons
        self._log(snapshot)
        return snapshot

    def restart_llm(self, progress: Any = print) -> None:
        from .config import get_llm_config
        llm_config = get_llm_config()
        progress("[llama_herd] system guard restarting llama.cpp server", flush=True)
        if llm_config.stop_command:
            subprocess.run(llm_config.stop_command, shell=True, check=False, timeout=45)
        if llm_config.restart_command:
            subprocess.run(llm_config.restart_command, shell=True, check=False, timeout=45)
        if llm_config.restart_wait > 0:
            time.sleep(llm_config.restart_wait)

    def _memory_trending_down(self) -> bool:
        if len(self._available_history) < max(3, self.memory_trend_window):
            return False
        return all(
            earlier > later
            for earlier, later in zip(self._available_history, self._available_history[1:])
        )

    def snapshot(self, episode_index: int, total_episodes: int) -> dict[str, Any]:
        mem = self._memory()
        temps = self._temperatures()
        container = self._container_memory()
        return {
            "timestamp": utc_now(),
            "episode_index": episode_index,
            "total_episodes": total_episodes,
            **mem,
            "temperatures_c": temps,
            "max_temp_c": max(temps.values()) if temps else None,
            "llm_container": self.container_name,
            "llm_container_memory": container,
        }

    def _cool_down(
        self,
        snapshot: dict[str, Any],
        episode_index: int,
        total_episodes: int,
        progress: Any,
        reasons: list[str],
    ) -> dict[str, Any]:
        cycles = 0
        current = snapshot
        while cycles < self.max_cooldown_cycles:
            max_temp = current.get("max_temp_c")
            if not isinstance(max_temp, (int, float)) or max_temp < self.temp_pause_c:
                break
            progress(
                f"[llama_herd] system guard cooling down; max temp {max_temp:.1f}C, sleeping {self.cooldown_seconds:.0f}s",
                flush=True,
            )
            self._log({**current, "action": "thermal_pause", "reasons": reasons, "cooldown_cycle": cycles + 1})
            time.sleep(self.cooldown_seconds)
            current = self.snapshot(episode_index, total_episodes)
            cycles += 1
            if isinstance(current.get("max_temp_c"), (int, float)) and current["max_temp_c"] >= self.temp_stop_c:
                current["action"] = "stop"
                current["reasons"] = [f"temperature_stop_c:{current['max_temp_c']}"]
                self._log(current)
                raise RuntimeError(f"System temperature too high to continue: {current['max_temp_c']}C")
        return current

    def _memory(self) -> dict[str, int]:
        values: dict[str, int] = {}
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                key, raw = line.split(":", 1)
                if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                    values[key] = int(raw.split()[0]) // 1024
        except OSError:
            pass
        swap_total = values.get("SwapTotal", 0)
        swap_free = values.get("SwapFree", 0)
        return {
            "mem_total_mb": values.get("MemTotal", -1),
            "available_mb": values.get("MemAvailable", -1),
            "swap_total_mb": swap_total,
            "swap_used_mb": max(0, swap_total - swap_free),
        }

    def _temperatures(self) -> dict[str, float]:
        temps: dict[str, float] = {}
        for zone in Path("/sys/devices/virtual/thermal").glob("thermal_zone*"):
            try:
                name = self._read_sysfs_text(zone.joinpath("type"))
                raw = self._read_sysfs_text(zone.joinpath("temp"))
                value = float(raw)
                if value > 1000:
                    value = value / 1000.0
                temps[name or zone.name] = round(value, 2)
            except OSError:
                continue
        if not temps:
            temps.update(self._tegrastats_temperatures())
        return temps

    def _read_sysfs_text(self, path: Path) -> str:
        data = path.read_bytes()
        if data is None:
            raise OSError(f"empty sysfs read: {path}")
        return data.decode("utf-8", errors="ignore").strip()

    def _tegrastats_temperatures(self) -> dict[str, float]:
        try:
            result = subprocess.run(["tegrastats", "--interval", "1000"], check=False, capture_output=True, text=True, timeout=2)
        except Exception:
            return {}
        matches = re.findall(r"([A-Za-z0-9_]+)@([0-9.]+)C", result.stdout)
        return {name: float(value) for name, value in matches}

    def _container_memory(self) -> str:
        result = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", self.container_name],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return result.stdout.strip() or result.stderr.strip()

    def _log(self, snapshot: dict[str, Any]) -> None:
        append_jsonl(self.log_path, snapshot)
