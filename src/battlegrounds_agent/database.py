from __future__ import annotations

import json
from pathlib import Path

from .schemas import Card, Trinket


class CardDatabase:
    def __init__(self, cards: list[Card], trinkets: list[Trinket] | None = None) -> None:
        self.cards = cards
        self.trinkets = trinkets or []
        self._cards_by_id = {card.id: card for card in self.cards}
        self._cards_by_name = {card.name: card for card in self.cards}

    @classmethod
    def load(cls, card_path: str | Path, trinket_path: str | Path | None = None) -> "CardDatabase":
        cards = [_card_from_json(item) for item in _read_json_list(card_path)]
        trinkets = [_trinket_from_json(item) for item in _read_json_list(trinket_path)] if trinket_path else []
        return cls(cards=cards, trinkets=trinkets)

    def card_by_id(self, card_id: str) -> Card | None:
        return self._cards_by_id.get(card_id)

    def card_by_name(self, name: str) -> Card | None:
        return self._cards_by_name.get(name)

    def legal_cards(self, available_tribes: list[str], max_tier: int | None = None) -> list[Card]:
        allowed = set(available_tribes)
        result: list[Card] = []
        for card in self.cards:
            if max_tier is not None and card.tier > max_tier:
                continue
            if not card.tribes or allowed.intersection(card.tribes):
                result.append(card)
        return result

    def legal_trinkets(self, available_tribes: list[str]) -> list[Trinket]:
        allowed = set(available_tribes)
        return [item for item in self.trinkets if not item.tribes or allowed.intersection(item.tribes)]


def _read_json_list(path: str | Path | None) -> list[dict]:
    if path is None:
        return []
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def _card_from_json(item: dict) -> Card:
    return Card(
        id=str(item["id"]),
        name=str(item.get("name") or item["id"]),
        tier=int(item.get("tier") or 1),
        attack=int(item.get("attack") or 0),
        health=int(item.get("health") or 0),
        tribes=list(item.get("tribes") or []),
        text=str(item.get("text") or ""),
        tags=list(item.get("tags") or []),
        version=item.get("version"),
        image=item.get("image"),
    )


def _trinket_from_json(item: dict) -> Trinket:
    return Trinket(
        id=str(item["id"]),
        name=str(item.get("name") or item["id"]),
        text=str(item.get("text") or ""),
        tier=item.get("tier"),
        tribes=list(item.get("tribes") or []),
        tags=list(item.get("tags") or []),
        image=item.get("image"),
    )
