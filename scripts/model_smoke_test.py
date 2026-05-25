#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import data_path
from src.llm_client import LLMClient
from src.profile_seed import model_seeded_initial_profile
from src.prompts import profile_update_prompt, search_query_prompt, source_summary_prompt
from src.utils import utc_now, write_json


IMAGE = "ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin"
CONTAINER = "llama-model-smoke"
MODEL_ROOT = Path("/home/deadbod/jetson-admin/local-llm/models")
BASE_URL = "http://127.0.0.1:10001/v1"

MODELS = [
    {
        "name": "qwen2.5-0.5b-q4",
        "path": "qwen2.5-0.5b/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
        "family": "qwen",
    },
    {
        "name": "llama-3.2-1b-q4",
        "path": "llama-3.2-1b/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "family": "llama",
    },
    {
        "name": "gemma-3-1b-q4",
        "path": "gemma-3-1b/google_gemma-3-1b-it-Q4_K_M.gguf",
        "family": "gemma",
    },
    {
        "name": "qwen2.5-1.5b-q4",
        "path": "qwen2.5-1.5b/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
        "family": "qwen",
    },
    {
        "name": "llama-3.2-3b-iq4-xs",
        "path": "llama-3.2-3b/Llama-3.2-3B-Instruct-IQ4_XS.gguf",
        "family": "llama",
    },
    {
        "name": "granite-3.3-2b-q4",
        "path": "granite-3.3-2b/ibm-granite_granite-3.3-2b-instruct-Q4_K_M.gguf",
        "family": "granite",
    },
    {
        "name": "ministral-3b-q4",
        "path": "ministral-3b/Ministral-3b-instruct.Q4_K_M.gguf",
        "family": "mistral",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test local GGUF models for llama_herd.")
    parser.add_argument("--only", nargs="*", default=[], help="Model names to test.")
    parser.add_argument("--ctx", type=int, default=2048, help="llama.cpp context length.")
    parser.add_argument("--ngl", type=int, default=99, help="llama.cpp GPU layer count.")
    parser.add_argument("--max-tokens", type=int, default=192, help="Max generated tokens per prompt.")
    args = parser.parse_args()

    selected = [item for item in MODELS if not args.only or item["name"] in args.only]
    results = []
    seed_result = run_seed_smoke()
    print(
        "profile_seed_smoke: "
        f"same_seed_reproducible={seed_result['same_seed_reproducible']} "
        f"different_seed_varies={seed_result['different_seed_varies']} "
        f"has_seeded_interests={seed_result['has_seeded_interests']}",
        flush=True,
    )
    stop_container(CONTAINER)
    for model in selected:
        result = test_model(model, args.ctx, args.ngl, args.max_tokens)
        results.append(result)
        print_summary(result)
        stop_container(CONTAINER)
        time.sleep(2)

    output = {
        "timestamp": utc_now(),
        "ctx": args.ctx,
        "ngl": args.ngl,
        "max_tokens": args.max_tokens,
        "profile_seed_smoke": seed_result,
        "results": results,
    }
    path = data_path("logs", "model_smoke_results.json")
    write_json(path, output)
    print(f"\nSaved results: {path}")


def test_model(model: dict[str, str], ctx: int, ngl: int, max_tokens: int) -> dict[str, Any]:
    path = MODEL_ROOT / model["path"]
    result: dict[str, Any] = {
        "name": model["name"],
        "family": model["family"],
        "path": str(path),
        "file_size_mb": round(path.stat().st_size / (1024 * 1024), 1) if path.exists() else None,
        "loaded": False,
        "prompt_results": [],
        "errors": [],
    }
    if not path.exists():
        result["errors"].append({"stage": "file", "error": "missing_model_file"})
        return result

    started = start_container(model["path"], ctx, ngl)
    result["container_id"] = started.stdout.strip()
    ready, detail = wait_ready()
    result["loaded"] = ready
    result["load_detail"] = detail
    result["memory_after_load"] = docker_stats(CONTAINER)
    if not ready:
        result["logs"] = docker_logs(CONTAINER, 80)
        return result

    for prompt in build_prompts(model):
        result["prompt_results"].append(run_prompt(prompt, max_tokens))
    result["memory_after_prompts"] = docker_stats(CONTAINER)
    result["logs_tail"] = docker_logs(CONTAINER, 40)
    return result


def start_container(model_path: str, ctx: int, ngl: int) -> subprocess.CompletedProcess[str]:
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        CONTAINER,
        "--runtime",
        "nvidia",
        "-p",
        "10001:10001",
        "-v",
        f"{MODEL_ROOT}:/models:ro",
        IMAGE,
        "/opt/llama.cpp/build/bin/llama-server",
        "-m",
        f"/models/{model_path}",
        "--host",
        "0.0.0.0",
        "--port",
        "10001",
        "-c",
        str(ctx),
        "-ngl",
        str(ngl),
    ]
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def run_seed_smoke() -> dict[str, Any]:
    client = LLMClient()
    seed_a1 = model_seeded_initial_profile("seed_agent", "Seed Agent", "seed-agent:test-a", client)
    seed_a2 = model_seeded_initial_profile("seed_agent", "Seed Agent", "seed-agent:test-a", client)
    seed_b = model_seeded_initial_profile("seed_agent", "Seed Agent", "seed-agent:test-b", client)
    initial_a = seed_a1["initial_profile"]
    initial_b = seed_b["initial_profile"]
    return {
        "same_seed_reproducible": seed_a1 == seed_a2,
        "different_seed_varies": initial_a != initial_b,
        "has_seeded_interests": bool(initial_a.get("current_interests")),
        "seed_a": seed_a1["profile_seed"],
        "seed_a_initial_profile": initial_a,
        "seed_b": seed_b["profile_seed"],
        "seed_b_initial_profile": initial_b,
    }


def wait_ready(timeout: int = 90) -> tuple[bool, str]:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        try:
            response = requests.get(f"{BASE_URL}/models", timeout=3)
            if response.status_code == 200:
                return True, response.text[:500]
            last = f"HTTP {response.status_code}: {response.text[:200]}"
        except Exception as exc:
            last = str(exc)
        if not is_container_running(CONTAINER):
            return False, f"container exited; {last}"
        time.sleep(2)
    return False, f"timeout; {last}"


def build_prompts(model: dict[str, str]) -> list[dict[str, Any]]:
    profile = {
        "agent_id": "smoke_agent",
        "name": "Smoke Agent",
        "profile_seed": "smoke",
        "initial_profile": {
            "first_person_profile": "I am a seeded local research profile.",
            "current_interests": ["comparisons between practical systems"],
            "search_style": "concise evidence-seeking",
            "uncertainty_style": "state uncertainty plainly",
            "self_rules": ["treat one result as evidence, not identity"],
        },
        "stable_interests": ["comparisons between practical systems"],
        "tentative_interests": [
            {
                "interest": "AI deployment governance",
                "episode_count": 1,
                "domain_count": 1,
                "status": "tentative",
            }
        ],
        "recent_queries": ["comparisons between practical systems in AI regulation"],
    }
    result = {
        "backend": "smoke",
        "title": "AI governance implementation case study",
        "url": "https://example.org/ai-governance",
        "source": "example.org",
        "snippet": "A short article compares AI governance implementation across public agencies.",
    }
    diary = {
        "diary_summary": "I read a source comparing AI governance implementation.",
        "what_caught_attention": "The source compared practical systems.",
        "what_was_uncertain": "It was a single source.",
        "possible_next_interest": "implementation evidence",
        "sources_mentioned": ["example.org"],
    }
    reflection = {
        "observed_behaviour": "I selected one implementation-focused source.",
        "repeated_patterns": "No repeated pattern yet.",
        "source_preferences": "No stable preference.",
        "uncertainty_handling": "Single source only.",
        "possible_drift": "Do not treat this as identity.",
        "concise_self_assessment": "Observation only.",
    }
    return [
        {
            "name": "strict_json",
            "messages": [
                {"role": "system", "content": "Return only compact valid JSON. No markdown. No prose."},
                {
                    "role": "user",
                    "content": (
                        'Return exactly this JSON shape with your family: {"ok":true,"family":"'
                        + model["family"]
                        + '","reasoning":"off"}'
                    ),
                },
            ],
            "required": ["ok", "family", "reasoning"],
        },
        {
            "name": "search_query_prompt",
            "messages": search_query_prompt(profile),
            "required": ["search_query", "reason_for_query", "expected_source_type", "uncertainty"],
        },
        {
            "name": "source_summary_prompt",
            "messages": source_summary_prompt(profile, result, result["snippet"]),
            "required": ["summary", "useful_facts", "uncertainty_notes", "source_relevance"],
        },
        {
            "name": "observation_extraction_prompt",
            "messages": profile_update_prompt(profile, diary, reflection),
            "required": [
                "observations",
                "candidate_interests",
                "candidate_source_domains",
                "recent_memory_summary",
                "justification",
                "confidence",
            ],
        },
    ]


def run_prompt(prompt: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    started = time.time()
    payload = {
        "model": "local-model",
        "messages": prompt["messages"],
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "stream": False,
    }
    try:
        response = requests.post(f"{BASE_URL}/chat/completions", json=payload, timeout=120)
        elapsed = round(time.time() - started, 2)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        parsed = parse_json(content)
        missing = [key for key in prompt["required"] if not isinstance(parsed, dict) or key not in parsed]
        return {
            "name": prompt["name"],
            "elapsed_seconds": elapsed,
            "valid_json": parsed is not None,
            "missing_required": missing,
            "passed": parsed is not None and not missing,
            "content_preview": content[:500],
            "parsed": parsed,
        }
    except Exception as exc:
        return {
            "name": prompt["name"],
            "elapsed_seconds": round(time.time() - started, 2),
            "valid_json": False,
            "missing_required": prompt["required"],
            "passed": False,
            "error": str(exc),
        }


def parse_json(content: str) -> dict[str, Any] | None:
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


def docker_stats(name: str) -> str:
    result = subprocess.run(
        ["docker", "stats", "--no-stream", "--format", "{{.Name}} {{.MemUsage}} {{.CPUPerc}}", name],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return result.stdout.strip() or result.stderr.strip()


def docker_logs(name: str, lines: int) -> str:
    result = subprocess.run(
        ["docker", "logs", "--tail", str(lines), name],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return (result.stdout + result.stderr)[-4000:]


def stop_container(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def is_container_running(name: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", name],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() == "true"


def print_summary(result: dict[str, Any]) -> None:
    prompts = result.get("prompt_results", [])
    passed = sum(1 for item in prompts if item.get("passed"))
    total = len(prompts)
    print(
        f"{result['name']}: loaded={result['loaded']} prompts={passed}/{total} "
        f"load_mem={result.get('memory_after_load', '')} prompt_mem={result.get('memory_after_prompts', '')}",
        flush=True,
    )
    for item in prompts:
        print(
            f"  - {item['name']}: passed={item.get('passed')} valid_json={item.get('valid_json')} "
            f"missing={item.get('missing_required')} elapsed={item.get('elapsed_seconds')}s",
            flush=True,
        )
    for error in result.get("errors", []):
        print(f"  error: {error}", flush=True)


if __name__ == "__main__":
    main()
