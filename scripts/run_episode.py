#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import AgentRunner, print_episode_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one research episode for one profile.")
    parser.add_argument("--agent", default="agent_01", help="Agent profile ID to run.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed progress for each stage.")
    args = parser.parse_args()
    episode = AgentRunner(verbose=args.verbose).run_episode(args.agent)
    print_episode_summary(episode)


if __name__ == "__main__":
    main()
