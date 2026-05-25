#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import AgentRunner, print_episode_summary
from src.config import data_path
from src.drift_analysis import load_jsonl
from src.storage import Storage


def show_profiles() -> None:
    for path in sorted(data_path("profiles").glob("*.json")):
        print(path.name)
        print(path.read_text(encoding="utf-8"))


def show_latest_diaries() -> None:
    rows = load_jsonl(data_path("logs", "episodes.jsonl"))[-10:]
    for row in rows:
        diary = row.get("diary_entry") or {}
        print(f"- {row.get('timestamp')} {row.get('agent_id')}: {diary.get('diary_summary', '')}")


def show_recent_search_queries() -> None:
    rows = load_jsonl(data_path("logs", "episodes.jsonl"))[-20:]
    for row in rows:
        print(f"- {row.get('timestamp')} {row.get('agent_id')}: {row.get('search_query')}")


def show_recent_errors() -> None:
    rows = load_jsonl(data_path("logs", "episodes.jsonl"))[-20:]
    found = False
    for row in rows:
        for error in row.get("errors", []):
            found = True
            print(f"- {row.get('timestamp')} {row.get('agent_id')} {error}")
    if not found:
        print("No recent errors.")


def main() -> None:
    Storage()
    runner = AgentRunner()
    while True:
        print(
            """
World Model Lab

1. Run one agent for one episode
2. Run agent_01 for one search test
3. Show current profiles
4. Show latest diary entries
5. Show recent search queries
6. Show recent errors
7. Exit
"""
        )
        choice = input("Select: ").strip()
        if choice == "1":
            agent_id = input("Agent ID [agent_01]: ").strip() or "agent_01"
            print_episode_summary(runner.run_episode(agent_id))
        elif choice == "2":
            print("\nRunning one search test for agent_01")
            print_episode_summary(runner.run_episode("agent_01"))
        elif choice == "3":
            show_profiles()
        elif choice == "4":
            show_latest_diaries()
        elif choice == "5":
            show_recent_search_queries()
        elif choice == "6":
            show_recent_errors()
        elif choice == "7":
            break
        else:
            print("Unknown option.")


if __name__ == "__main__":
    main()
