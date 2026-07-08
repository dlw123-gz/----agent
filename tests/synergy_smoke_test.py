from __future__ import annotations

from battlegrounds_agent.planner import BattlegroundsAgent
from battlegrounds_agent.schemas import BoardMinion, Card, GameState, PlayerState, ShopState
from battlegrounds_agent.state_summary import summarize_state
from battlegrounds_agent.synergy import score_card_synergy


def test_battlecry_synergy_affects_plan() -> None:
    state = GameState(
        player=PlayerState(
            tavern_tier=4,
            health=30,
            gold=8,
            board=[
                BoardMinion(
                    card_id="brann_like",
                    name="战吼放大器",
                    attack=2,
                    health=4,
                    position=0,
                    text="你的战吼会触发两次。",
                    enchantments=["double_battlecry"],
                )
            ],
        ),
        shop=ShopState(
            cards=[
                Card(
                    id="dragon_buffer",
                    name="龙人强化者",
                    tier=4,
                    attack=4,
                    health=4,
                    tribes=["龙"],
                    text="战吼：使一条友方龙获得+5生命值。",
                    tags=["battlecry", "dragon_buff"],
                )
            ],
            level_cost=7,
        ),
        available_tribes=["龙"],
    )

    score, reasons = score_card_synergy(state, state.shop.cards[0])
    plan = BattlegroundsAgent().plan_turn(state)
    summary = summarize_state(state)

    assert score > 0
    assert reasons
    assert plan.actions[0].target_id == "dragon_buffer"
    assert summary["synergies"]


if __name__ == "__main__":
    test_battlecry_synergy_affects_plan()
    print("synergy smoke test passed")
