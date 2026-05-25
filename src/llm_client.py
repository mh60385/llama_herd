from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from .config import Settings


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.raw_outputs: list[dict[str, Any]] = []
        self.restart_events: list[dict[str, Any]] = []
        self.model = self._resolve_model(self.settings.llm_model)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }

    def _resolve_model(self, configured: str) -> str:
        if configured and configured != "local-model":
            return configured
        try:
            response = self._request(
                "get",
                "/models",
                timeout=8,
                allow_restart=False,
            )
            payload = response.json()
            data = payload.get("data") or payload.get("models") or []
            if data:
                first = data[0]
                return first.get("id") or first.get("model") or first.get("name") or configured
        except requests.RequestException:
            pass
        return configured

    def healthcheck(self) -> tuple[bool, str]:
        try:
            response = self._request(
                "get",
                "/models",
                timeout=8,
            )
            return True, self.model
        except Exception as exc:
            return False, str(exc)

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        expect_json: bool = True,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        content = ""
        try:
            response = self._request(
                "post",
                "/chat/completions",
                json=payload,
                timeout=90,
            )
            body = response.json()
            content = body["choices"][0]["message"]["content"]
            self.raw_outputs.append({"request": messages, "content": content, "error": None})
        except Exception as exc:
            error = {"error": "llm_request_failed", "detail": str(exc), "content": content}
            self.raw_outputs.append({"request": messages, "content": content, "error": error})
            return error

        if not expect_json:
            return {"content": content}

        parsed = self._parse_json(content)
        if parsed is not None:
            return parsed

        repaired = self._repair_json(content, temperature=0.0, top_p=1.0)
        if repaired is not None:
            return repaired

        return {"error": "malformed_json", "raw_content": content}

    def _parse_json(self, content: str) -> dict[str, Any] | None:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:].strip()
        try:
            value = json.loads(text)
            return value if isinstance(value, dict) else {"value": value}
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                try:
                    value = json.loads(text[start : end + 1])
                    return value if isinstance(value, dict) else {"value": value}
                except json.JSONDecodeError:
                    return None
        return None

    def _repair_json(self, content: str, temperature: float, top_p: float) -> dict[str, Any] | None:
        messages = [
            {
                "role": "system",
                "content": "Return only valid compact JSON. Do not explain. Do not include chain-of-thought.",
            },
            {"role": "user", "content": f"Repair this malformed JSON output:\n{content[:6000]}"},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        try:
            response = self._request(
                "post",
                "/chat/completions",
                json=payload,
                timeout=60,
            )
            body = response.json()
            repaired_content = body["choices"][0]["message"]["content"]
            self.raw_outputs.append({"request": messages, "content": repaired_content, "repair_for": content})
            return self._parse_json(repaired_content)
        except Exception as exc:
            self.raw_outputs.append({"request": messages, "content": "", "error": str(exc)})
            return None

    def _request(
        self,
        method: str,
        path: str,
        timeout: float,
        json: dict[str, Any] | None = None,
        allow_restart: bool = True,
    ) -> requests.Response:
        attempts = max(1, self.settings.llm_retry_attempts)
        restarted = False
        last_exc: Exception | None = None
        for cycle in range(2):
            for attempt in range(1, attempts + 1):
                try:
                    response = requests.request(
                        method,
                        f"{self.settings.llm_base_url}{path}",
                        headers=self._headers(),
                        json=json,
                        timeout=timeout,
                    )
                    response.raise_for_status()
                    return response
                except requests.HTTPError as exc:
                    last_exc = exc
                    status = exc.response.status_code if exc.response is not None else 0
                    if not self._retryable_status(status) or attempt == attempts:
                        break
                except requests.RequestException as exc:
                    last_exc = exc
                    if attempt == attempts:
                        break
                self._sleep_before_retry(attempt)
            if not allow_restart or restarted or not self.settings.llm_restart_command:
                break
            restarted = True
            self._restart_server(last_exc)
            self._sleep_after_restart()
        assert last_exc is not None
        raise last_exc

    def _retryable_status(self, status: int) -> bool:
        return status in {408, 409, 425, 429} or status >= 500

    def _sleep_before_retry(self, attempt: int) -> None:
        delay = min(
            self.settings.llm_retry_max_delay,
            self.settings.llm_retry_initial_delay * (2 ** max(0, attempt - 1)),
        )
        if delay > 0:
            time.sleep(delay)

    def _sleep_after_restart(self) -> None:
        if self.settings.llm_restart_wait > 0:
            time.sleep(self.settings.llm_restart_wait)

    def _restart_server(self, reason: Exception | None) -> None:
        event: dict[str, Any] = {"reason": str(reason or "unknown"), "stopped": False, "started": False}
        try:
            log_path = Path(self.settings.llm_restart_log)
            if self.settings.llm_restart_log:
                log_path = Path(self.settings.llm_restart_log)
                if not log_path.is_absolute():
                    log_path = self.settings.root / log_path
            log_path.parent.mkdir(parents=True, exist_ok=True)

            with log_path.open("ab") as log_handle:
                if self.settings.llm_stop_command:
                    stop = subprocess.run(
                        self.settings.llm_stop_command,
                        shell=True,
                        check=False,
                        timeout=30,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                    )
                    event["stopped"] = stop.returncode == 0
                    event["stop_returncode"] = stop.returncode
                else:
                    event["stopped"] = self._kill_process_on_base_url_port()

                subprocess.Popen(
                    self.settings.llm_restart_command,
                    shell=True,
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            event["started"] = True
            event["log"] = str(log_path)
        except Exception as exc:
            event["restart_error"] = str(exc)
        self.restart_events.append(event)

    def _kill_process_on_base_url_port(self) -> bool:
        parsed = urlparse(self.settings.llm_base_url)
        port = parsed.port
        if port is None:
            return False
        if shutil.which("fuser"):
            result = subprocess.run(["fuser", "-k", f"{port}/tcp"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return result.returncode == 0
        if shutil.which("lsof"):
            result = subprocess.run(["lsof", "-ti", f":{port}"], check=False, capture_output=True, text=True)
            pids = [pid.strip() for pid in result.stdout.splitlines() if pid.strip()]
            for pid in pids:
                subprocess.run(["kill", "-TERM", pid], check=False)
            return bool(pids)
        return False
