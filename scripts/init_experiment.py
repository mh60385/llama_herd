#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import data_path, load_agents_config
from src.llm_client import LLMClient
from src.profile_seed import model_seeded_initial_profile
from src.schemas import AgentProfile
from src.storage import Storage
from src.utils import utc_now


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize the local drift experiment.")
    parser.add_argument("--reset", action="store_true", help="Remove existing experiment data first.")
    args = parser.parse_args()

    if args.reset and data_path().exists():
        shutil.rmtree(data_path())

    storage = Storage()
    llm = LLMClient()
    now = utc_now()
    agents = load_agents_config().get("agents", [])
    for item in agents:
        agent_id = item["agent_id"]
        path = storage.profile_path(agent_id)
        if path.exists() and not args.reset:
            print(f"Profile exists, keeping: {agent_id}")
            continue
        name = item.get("name", agent_id)
        seed = str(item.get("profile_seed") or f"{agent_id}:llama-herd:2026-05-25")
        seeded = model_seeded_initial_profile(agent_id, name, seed, llm)
        initial = seeded["initial_profile"]
        profile = AgentProfile(
            agent_id=agent_id,
            name=name,
            profile_seed=str(seeded["profile_seed"]),
            initial_profile=initial,
            first_person_profile=str(initial["first_person_profile"]),
            current_interests=list(initial["current_interests"]),
            preferred_sources=list(initial["preferred_sources"]),
            search_style=str(initial["search_style"]),
            uncertainty_style=str(initial["uncertainty_style"]),
            self_rules=list(initial["self_rules"]),
            stable_interests=list(initial["current_interests"]),
            created_at=now,
            updated_at=now,
        )
        storage.save_profile(profile)
        print(f"Created profile: {agent_id} seed={seed}")
    print(f"SQLite database: {storage.db_path}")


if __name__ == "__main__":
    main()
