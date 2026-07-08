from __future__ import annotations

from pathlib import Path

from battlegrounds_agent.database import CardDatabase
from battlegrounds_agent.image_recognizer import RecognitionResult
from battlegrounds_agent.planner import BattlegroundsAgent
from battlegrounds_agent.screen_reader import SlotRecognition
from battlegrounds_agent.state_builder import BuildStateOptions, build_state_from_slots


def test_slots_can_build_state_and_plan() -> None:
    root = Path(__file__).resolve().parents[1]
    database = CardDatabase.load(root / "examples" / "data" / "cards.json", root / "examples" / "data" / "trinkets.json")
    slots = [
        _slot("shop_1", "card", "beast_buffer"),
        _slot("shop_2", "card", "mech_scaler"),
        _slot("board_1", "card", "beast_summoner"),
        _slot("trinket_1", "trinket", "beast_charm"),
    ]

    state = build_state_from_slots(
        slots,
        database,
        BuildStateOptions(
            tavern_tier=3,
            health=32,
            armor=4,
            gold=8,
            turn=6,
            available_tribes=("野兽", "机械"),
            level_cost=6,
        ),
    )
    plan = BattlegroundsAgent().plan_turn(state)

    assert state.shop.cards[0].id == "beast_buffer"
    assert state.player.board[0].card_id == "beast_summoner"
    assert state.offered_trinkets[0].id == "beast_charm"
    assert plan.actions


def _slot(slot_id: str, kind: str, match_id: str) -> SlotRecognition:
    return SlotRecognition(
        slot_id=slot_id,
        kind=kind,
        box_pixels=(0, 0, 10, 10),
        crop_path=None,
        matches=[
            RecognitionResult(
                id=match_id,
                kind=kind,
                path=f"{match_id}.png",
                score=0.99,
                hash_distance=0.01,
                histogram_distance=0.01,
            )
        ],
    )


if __name__ == "__main__":
    test_slots_can_build_state_and_plan()
    print("state builder smoke test passed")
