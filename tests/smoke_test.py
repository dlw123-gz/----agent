from __future__ import annotations

from pathlib import Path

from battlegrounds_agent.planner import BattlegroundsAgent
from battlegrounds_agent.vision_adapter import JsonStateAdapter


def test_sample_state() -> None:
    root = Path(__file__).resolve().parents[1]
    state = JsonStateAdapter.load(root / "examples" / "sample_state.json")
    plan = BattlegroundsAgent().plan_turn(state)

    assert plan.actions
    assert plan.risk_level == "low"
    assert {action.target_name for action in plan.actions[:3]} >= {"兽群护符", "兽栏管理员"}


if __name__ == "__main__":
    test_sample_state()
    print("smoke test passed")
