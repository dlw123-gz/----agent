from __future__ import annotations

from dataclasses import asdict

from .schemas import GameState
from .screen_reader import SlotRecognition
from .synergy import explain_state_synergies, infer_card_tags


def summarize_recognition(slots: list[SlotRecognition], min_score: float = 0.58) -> dict:
    low_confidence = []
    recognized = []
    for slot in slots:
        best = slot.matches[0] if slot.matches else None
        if best is None or best.score < min_score:
            low_confidence.append(
                {
                    "slot_id": slot.slot_id,
                    "kind": slot.kind,
                    "best_id": best.id if best else None,
                    "best_score": best.score if best else None,
                    "crop_path": slot.crop_path,
                }
            )
            continue
        recognized.append(
            {
                "slot_id": slot.slot_id,
                "kind": slot.kind,
                "id": best.id,
                "score": best.score,
                "crop_path": slot.crop_path,
            }
        )
    return {
        "recognized_count": len(recognized),
        "low_confidence_count": len(low_confidence),
        "recognized": recognized,
        "low_confidence": low_confidence,
    }


def summarize_state(state: GameState) -> dict:
    unknown_cards = []
    cards = []
    for section, items in (
        ("shop", state.shop.cards),
        ("hand", state.player.hand),
    ):
        for card in items:
            row = {
                "section": section,
                "id": card.id,
                "name": card.name,
                "tribes": card.tribes,
                "tags": sorted(infer_card_tags(card)),
            }
            cards.append(row)
            if card.name == card.id or "未在 cards.json" in card.text:
                unknown_cards.append(row)

    for minion in state.player.board:
        row = {
            "section": "board",
            "id": minion.card_id,
            "name": minion.name,
            "tribes": minion.tribes,
            "tags": sorted(infer_card_tags(_minion_card_like(minion))),
        }
        cards.append(row)
        if minion.name == minion.card_id or "未在 cards.json" in minion.text:
            unknown_cards.append(row)

    return {
        "counts": {
            "shop_cards": len(state.shop.cards),
            "board_minions": len(state.player.board),
            "hand_cards": len(state.player.hand),
            "offered_trinkets": len(state.offered_trinkets),
            "unknown_cards": len(unknown_cards),
        },
        "cards": cards,
        "unknown_cards": unknown_cards,
        "synergies": [asdict(fact) for fact in explain_state_synergies(state)[:12]],
        "notes": state.notes,
    }


def _minion_card_like(minion):
    from .schemas import Card

    return Card(
        id=minion.card_id,
        name=minion.name,
        tier=1,
        attack=minion.attack,
        health=minion.health,
        tribes=list(minion.tribes),
        text=minion.text,
    )
