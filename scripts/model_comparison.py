#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import model_smoke_test
from src.config import ROOT, data_path, load_agents_config
from src.utils import read_json, utc_now, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run short isolated model-comparison experiments.")
    parser.add_argument("--models", nargs="+", required=True, help="Model names from scripts/model_smoke_test.py.")
    parser.add_argument("--agents", type=int, default=3, help="Number of configured agents to include.")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per included agent.")
    parser.add_argument("--ctx", type=int, default=2048, help="llama.cpp context length.")
    parser.add_argument("--ngl", type=int, default=99, help="llama.cpp GPU layer count.")
    parser.add_argument("--out-dir", default=str(data_path("model_comparison")))
    parser.add_argument("--keep-containers", action="store_true", help="Leave the last comparison container running.")
    args = parser.parse_args()

    known = {item["name"]: item for item in model_smoke_test.MODELS}
    missing = [name for name in args.models if name not in known]
    if missing:
        raise SystemExit(f"Unknown models: {', '.join(missing)}")

    final_out_dir = Path(args.out_dir)
    work_out_dir = ROOT / f".model_comparison_work-{int(time.time())}"
    work_out_dir.mkdir(parents=True, exist_ok=True)
    original_data = ROOT / "data"
    saved_data = backup_current_data(original_data)
    results = []
    try:
        for name in args.models:
            result = run_model_case(known[name], args.agents, args.episodes, args.ctx, args.ngl, work_out_dir)
            results.append(result)
            if not args.keep_containers:
                model_smoke_test.stop_container(model_smoke_test.CONTAINER)
            time.sleep(2)
    finally:
        if not args.keep_containers:
            model_smoke_test.stop_container(model_smoke_test.CONTAINER)
        restore_data(original_data, saved_data)

    summary = {
        "timestamp": utc_now(),
        "agents": args.agents,
        "episodes_per_agent": args.episodes,
        "ctx": args.ctx,
        "ngl": args.ngl,
        "results": results,
    }
    write_json(work_out_dir / "summary.json", summary)
    (work_out_dir / "summary.md").write_text(render_summary(summary), encoding="utf-8")
    if final_out_dir.exists():
        shutil.rmtree(final_out_dir)
    final_out_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(work_out_dir), str(final_out_dir))
    print(final_out_dir / "summary.json")
    print(final_out_dir / "summary.md")


def run_model_case(
    model: dict[str, str],
    agents: int,
    episodes: int,
    ctx: int,
    ngl: int,
    out_dir: Path,
) -> dict[str, Any]:
    case_dir = out_dir / model["name"]
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_smoke_test.MODEL_ROOT / model["path"]
    result: dict[str, Any] = {
        "name": model["name"],
        "family": model["family"],
        "path": str(model_path),
        "status": "not_started",
        "errors": [],
    }
    if not model_path.exists():
        result["status"] = "missing_model_file"
        result["errors"].append({"stage": "file", "error": "missing_model_file"})
        return result

    model_smoke_test.stop_container(model_smoke_test.CONTAINER)
    started = model_smoke_test.start_container(model["path"], ctx, ngl)
    result["container_id"] = started.stdout.strip()
    ready, detail = model_smoke_test.wait_ready()
    result["load_detail"] = detail
    result["memory_after_load"] = model_smoke_test.docker_stats(model_smoke_test.CONTAINER)
    if not ready:
        result["status"] = "load_failed"
        result["errors"].append({"stage": "load", "error": detail})
        result["logs_tail"] = model_smoke_test.docker_logs(model_smoke_test.CONTAINER, 80)
        return result

    env = os.environ.copy()
    env["LLM_BASE_URL"] = model_smoke_test.BASE_URL
    env["LLM_MODEL"] = "local-model"
    env["LLM_STOP_COMMAND"] = ""
    env["LLM_RESTART_COMMAND"] = ""
    env["LLM_RESTART_WAIT"] = "0"

    init = run_cmd([".venv/bin/python", "scripts/init_experiment.py", "--reset"], env, timeout=600)
    result["init"] = command_result(init)
    if init.returncode != 0:
        result["status"] = "init_failed"
        result["errors"].append({"stage": "init", "error": init.stderr[-1000:] or init.stdout[-1000:]})
        snapshot_case_data(case_dir)
        return result

    selected_agents = read_selected_agents(agents)
    run_started = time.time()
    for agent_id in selected_agents:
        run = run_cmd(
            [
                ".venv/bin/python",
                "scripts/run_batch.py",
                "--agent",
                agent_id,
                "--episodes",
                str(episodes),
                "--no-system-monitor",
            ],
            env,
            timeout=max(900, episodes * 240),
        )
        result.setdefault("agent_runs", {})[agent_id] = command_result(run)
        if run.returncode != 0:
            result["errors"].append({"stage": "run", "agent_id": agent_id, "error": run.stderr[-2000:] or run.stdout[-2000:]})
            break
    result["elapsed_seconds"] = round(time.time() - run_started, 2)

    metrics_dir = case_dir / "metrics"
    metrics = run_cmd(
        [".venv/bin/python", "scripts/write_research_metrics.py", "--out-dir", str(metrics_dir)],
        env,
        timeout=120,
    )
    result["metrics_command"] = command_result(metrics)
    metrics_payload = read_json(metrics_dir / "research_metrics.json", {})
    result["metrics"] = compact_metrics(metrics_payload)
    result["memory_after_run"] = model_smoke_test.docker_stats(model_smoke_test.CONTAINER)
    result["logs_tail"] = model_smoke_test.docker_logs(model_smoke_test.CONTAINER, 80)
    snapshot_case_data(case_dir)
    result["status"] = "completed" if not result["errors"] else "completed_with_errors"
    return result


def backup_current_data(data_dir: Path) -> Path | None:
    if not data_dir.exists():
        return None
    backup = ROOT / f"data.precomparison-{int(time.time())}"
    shutil.move(str(data_dir), str(backup))
    return backup


def restore_data(data_dir: Path, backup: Path | None) -> None:
    if data_dir.exists():
        shutil.rmtree(data_dir)
    if backup and backup.exists():
        shutil.move(str(backup), str(data_dir))


def snapshot_case_data(case_dir: Path) -> None:
    current = ROOT / "data"
    if not current.exists():
        return
    dest = case_dir / "data"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(current, dest)


def read_selected_agents(count: int) -> list[str]:
    agents = load_agents_config().get("agents", [])
    return [str(item["agent_id"]) for item in agents[:count]]


def run_cmd(cmd: list[str], env: dict[str, str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, env=env, check=False, capture_output=True, text=True, timeout=timeout)


def command_result(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return {
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-3000:],
        "stderr_tail": result.stderr[-3000:],
    }


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    agents = metrics.get("agents", {}) if isinstance(metrics, dict) else {}
    totals = {
        "total_episodes": metrics.get("total_episodes", 0) if isinstance(metrics, dict) else 0,
        "model_output_failures": 0,
        "model_json_repairs": 0,
        "embodied_diary_language": 0,
        "empty_diaries": 0,
        "source_extraction_failures": 0,
        "rejected_saved_interests": 0,
        "unique_selected_domains": 0,
        "mean_adjacent_query_similarity": 0.0,
    }
    similarities = []
    for data in agents.values():
        totals["model_output_failures"] += int(data.get("model_output_failure_count", 0))
        totals["model_json_repairs"] += int(data.get("model_json_repair_count", 0))
        totals["embodied_diary_language"] += int(data.get("embodied_diary_language_count", 0))
        totals["empty_diaries"] += int(data.get("empty_diary_count", 0))
        totals["source_extraction_failures"] += int(data.get("source_extraction_failure_count", 0))
        totals["rejected_saved_interests"] += sum(int(v) for v in data.get("rejected_saved_interests", {}).values())
        totals["unique_selected_domains"] += int(data.get("unique_selected_domains", 0))
        similarities.append(float(data.get("avg_adjacent_query_similarity", 0)))
    if similarities:
        totals["mean_adjacent_query_similarity"] = round(sum(similarities) / len(similarities), 3)
    return totals


def render_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Model Comparison",
        "",
        f"Updated: `{summary['timestamp']}`",
        f"Design: `{summary['agents']}` agents x `{summary['episodes_per_agent']}` episodes",
        "",
        "| Model | Status | Episodes | JSON repairs | Model failures | Embodied diary | Empty diaries | Rejected saved interests | Query similarity |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in summary["results"]:
        metrics = result.get("metrics", {})
        lines.append(
            "| {name} | {status} | {episodes} | {repairs} | {failures} | {embodied} | {empty} | {rejected} | {similarity} |".format(
                name=result.get("name", ""),
                status=result.get("status", ""),
                episodes=metrics.get("total_episodes", 0),
                repairs=metrics.get("model_json_repairs", 0),
                failures=metrics.get("model_output_failures", 0),
                embodied=metrics.get("embodied_diary_language", 0),
                empty=metrics.get("empty_diaries", 0),
                rejected=metrics.get("rejected_saved_interests", 0),
                similarity=metrics.get("mean_adjacent_query_similarity", 0),
            )
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
