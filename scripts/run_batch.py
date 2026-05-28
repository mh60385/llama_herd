#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import AgentRunner, print_episode_summary
from src.config import data_path, load_agents_config, load_experiment_config
from src.system_monitor import SystemMonitor


def saved_episode_count(agent_id: str) -> int:
    path = data_path("logs", f"{agent_id}.jsonl")
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sequential drift episodes.")
    parser.add_argument("--agent", default="agent_01", help="Agent ID to run.")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes to run.")
    parser.add_argument("--all-agents", action="store_true", help="Run configured agents sequentially.")
    parser.add_argument("--resume", action="store_true", help="Skip episodes already saved in data/logs.")
    parser.add_argument("--no-system-monitor", action="store_true", help="Disable pre-episode system guard checks.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed progress for each stage.")
    args = parser.parse_args()

    config = load_experiment_config()
    if args.no_system_monitor:
        config["system_monitor_enabled"] = False
    runner = AgentRunner(verbose=args.verbose)
    monitor = SystemMonitor(None, config)
    if args.all_agents:
        agents = [item["agent_id"] for item in load_agents_config().get("agents", [])]
        episodes = args.episodes or int(config.get("episodes_per_run", 10))
        delay = float(config.get("inter_episode_delay_seconds", 0))
        total = len(agents) * episodes
        saved_counts = {agent_id: min(saved_episode_count(agent_id), episodes) for agent_id in agents}
        completed = sum(saved_counts.values()) if args.resume else 0
        if args.resume:
            for agent_id in agents:
                print(f"Resume: {agent_id} has {saved_counts[agent_id]}/{episodes} saved episodes")
        for agent_id in agents:
            start_index = saved_counts[agent_id] if args.resume else 0
            if start_index >= episodes:
                print(f"Resume: skipping {agent_id}; target already reached")
                continue
            for index in range(start_index, episodes):
                print(f"\nRunning episode {index + 1}/{episodes} for {agent_id} ({completed + 1}/{total} total)")
                monitor.pre_episode(completed, total)
                print_episode_summary(runner.run_episode(agent_id))
                completed += 1
                if delay > 0 and completed < total:
                    time.sleep(delay)
        return

    episodes = args.episodes or int(config.get("episodes_per_run", 10))
    delay = float(config.get("inter_episode_delay_seconds", 0))
    start_index = min(saved_episode_count(args.agent), episodes) if args.resume else 0
    if args.resume:
        print(f"Resume: {args.agent} has {start_index}/{episodes} saved episodes")
    for index in range(start_index, episodes):
        print(f"\nRunning episode {index + 1}/{episodes} for {args.agent}")
        monitor.pre_episode(index, episodes)
        print_episode_summary(runner.run_episode(args.agent))
        if delay > 0 and index + 1 < episodes:
            time.sleep(delay)


if __name__ == "__main__":
    main()
