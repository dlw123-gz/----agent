from __future__ import annotations

from pathlib import Path

from battlegrounds_agent.asset_indexer import build_card_index, build_composition_image_index, build_trinket_index
from battlegrounds_agent.database import CardDatabase


def test_generated_asset_indexes_are_loadable() -> None:
    root = Path(__file__).resolve().parents[1]
    outputs = root.parent
    cards = build_card_index(outputs / "data card")
    trinkets = build_trinket_index(outputs / "data sp")
    compositions = build_composition_image_index(outputs / "data team")

    assert cards
    assert trinkets
    assert compositions
    assert cards[0]["id"]
    assert compositions[0]["rows"] == []

    db = CardDatabase.load(root / "examples" / "data" / "cards.generated.json", root / "examples" / "data" / "trinkets.generated.json")
    assert db.cards
    assert db.trinkets


if __name__ == "__main__":
    test_generated_asset_indexes_are_loadable()
    print("asset index smoke test passed")
