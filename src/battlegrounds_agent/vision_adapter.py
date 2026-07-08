from __future__ import annotations

import json
from pathlib import Path

from .schemas import BoardMinion, Card, GameState, PlayerState, ShopState, Trinket


class JsonStateAdapter:
    """Adapter for OCR/vision output normalized as JSON."""

    @staticmethod
    def load(path: str | Path) -> GameState:
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)

        player_data = data["player"]
        shop_data = data["shop"]
        return GameState(
            player=PlayerState(
                tavern_tier=player_data["tavern_tier"],
                health=player_data["health"],
                armor=player_data.get("armor", 0),
                gold=player_data.get("gold", 0),
                turn=player_data.get("turn", 1),
                board=[BoardMinion(**item) for item in player_data.get("board", [])],
                hand=[Card(**item) for item in player_data.get("hand", [])],
            ),
            shop=ShopState(
                cards=[Card(**item) for item in shop_data.get("cards", [])],
                frozen=shop_data.get("frozen", False),
                roll_cost=shop_data.get("roll_cost", 1),
                level_cost=shop_data.get("level_cost"),
            ),
            available_tribes=data.get("available_tribes", []),
            offered_trinkets=[Trinket(**item) for item in data.get("offered_trinkets", [])],
            notes=data.get("notes", ""),
        )
