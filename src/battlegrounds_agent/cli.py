from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .llm import OpenAICompatibleClient
from .planner import BattlegroundsAgent
from .vision_adapter import JsonStateAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan a Battlegrounds turn from normalized JSON state.")
    parser.add_argument("--state", required=True, help="Path to normalized game state JSON.")
    parser.add_argument("--use-llm", action="store_true", help="Call configured LLM API for reasoning.")
    parser.add_argument("--env-file", help="Optional .env file for LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL.")
    args = parser.parse_args()

    state = JsonStateAdapter.load(args.state)
    llm = OpenAICompatibleClient.from_env(args.env_file) if args.use_llm else None
    plan = BattlegroundsAgent(llm_client=llm).plan_turn(state, use_llm=args.use_llm)
    print(json.dumps(asdict(plan), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
