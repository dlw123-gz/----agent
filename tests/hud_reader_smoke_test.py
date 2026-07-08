from __future__ import annotations

from battlegrounds_agent.hud_reader import HudReading, merge_hud_options
from battlegrounds_agent.state_builder import BuildStateOptions


def test_hud_reading_overrides_state_options() -> None:
    options = BuildStateOptions(tavern_tier=1, health=40, armor=0, gold=3, turn=1, level_cost=None)
    hud = HudReading(tavern_tier=4, health=30, armor=10, gold=8, turn=7, level_cost=6, confidence=0.8)

    merged = merge_hud_options(options, hud)

    assert merged.tavern_tier == 4
    assert merged.health == 30
    assert merged.armor == 10
    assert merged.gold == 8
    assert merged.turn == 7
    assert merged.level_cost == 6


if __name__ == "__main__":
    test_hud_reading_overrides_state_options()
    print("hud reader smoke test passed")
