#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import AgentRunner, print_episode_summary
from src.config import load_agents_config, load_experiment_config
from src.system_monitor import SystemMonitor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sequential drift episodes.")
    parser.add_argument("--agent", default="agent_01", help="Agent ID to run.")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes to run.")
    parser.add_argument("--all-agents", action="store_true", help="Run configured agents once each.")
    args = parser.parse_args()

    config = load_experiment_config()
    runner = AgentRunner()
    monitor = SystemMonitor(runner.settings, config)
    if args.all_agents:
        agents = [item["agent_id"] for item in load_agents_config().get("agents", [])]
        for index, agent_id in enumerate(agents):
            monitor.pre_episode(index, len(agents))
            print_episode_summary(runner.run_episode(agent_id))
        return

    episodes = args.episodes or int(config.get("episodes_per_run", 10))
    delay = float(config.get("inter_episode_delay_seconds", 0))
    for index in range(episodes):
        print(f"\nRunning episode {index + 1}/{episodes} for {args.agent}")
        monitor.pre_episode(index, episodes)
        print_episode_summary(runner.run_episode(args.agent))
        if delay > 0 and index + 1 < episodes:
            time.sleep(delay)


if __name__ == "__main__":
    main()
