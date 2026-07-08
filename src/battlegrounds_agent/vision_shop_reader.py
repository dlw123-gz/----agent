from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .database import CardDatabase
from .llm import OpenAICompatibleClient
from .schemas import Card
from .screen_reader import SlotRecognition


@dataclass(frozen=True, slots=True)
class VisionShopReading:
    cards: list[Card]
    slots: list[dict[str, Any]]
    notes: str = ""


def read_shop_with_vision(
    image_path: str | Path,
    client: OpenAICompatibleClient,
    database: CardDatabase,
    local_slots: list[SlotRecognition] | None = None,
) -> VisionShopReading:
    prompt = _build_prompt(database, local_slots or [])
    raw = client.complete_with_image(prompt, image_path)
    data = _parse_json_object(raw)
    raw_slots = data.get("shop_slots")
    if not isinstance(raw_slots, list):
        raise ValueError("Vision shop response missing shop_slots list")

    cards: list[Card] = []
    slots: list[dict[str, Any]] = []
    notes: list[str] = []
    for index, item in enumerate(raw_slots[:7], start=1):
        if not isinstance(item, dict):
            continue
        slot = dict(item)
        slot.setdefault("slot", index)
        confidence = _float_or_default(slot.get("confidence"), 0.0)
        kind = str(slot.get("type") or "").lower()
        name = str(slot.get("name") or "").strip()
        if kind in {"spell", "tavern_spell"}:
            slot["accepted"] = False
            slot["reject_reason"] = "tavern spell is not a minion card"
            slots.append(slot)
            continue
        if kind not in {"minion", "card", ""}:
            slot["accepted"] = False
            slot["reject_reason"] = f"unsupported slot type: {kind}"
            slots.append(slot)
            continue
        card = _resolve_card(database, name, str(slot.get("id") or "").strip())
        if card is None:
            slot["accepted"] = False
            slot["reject_reason"] = "vision name did not match local card database"
            notes.append(f"shop_{index}: vision saw {name!r} but it was not found in card database")
            slots.append(slot)
            continue
        if confidence < 0.72:
            slot["accepted"] = False
            slot["reject_reason"] = f"vision confidence too low: {confidence:.2f}"
            notes.append(f"shop_{index}: low confidence vision match {card.name} confidence={confidence:.2f}")
            slots.append(slot)
            continue
        slot["accepted"] = True
        slot["id"] = card.id
        slot["name"] = card.name
        slot["tier"] = card.tier
        cards.append(card)
        slots.append(slot)
    return VisionShopReading(cards=cards, slots=slots, notes="; ".join(notes))


def _build_prompt(database: CardDatabase, local_slots: list[SlotRecognition]) -> str:
    local_candidates = []
    for slot in local_slots:
        if not slot.slot_id.startswith("shop_"):
            continue
        candidates = []
        for match in slot.matches[:5]:
            card = database.card_by_id(match.id)
            candidates.append(
                {
                    "id": match.id,
                    "name": card.name if card else match.id,
                    "tier": card.tier if card else match.tier,
                    "score": match.score,
                }
            )
        local_candidates.append({"slot": slot.slot_id, "candidates": candidates})

    payload = {
        "task": "Read the Hearthstone Battlegrounds tavern shop row from the screenshot.",
        "rules": [
            "Return exactly one JSON object.",
            "Identify shop slots from left to right, 1 through 7.",
            "For each slot, return type: minion, tavern_spell, empty, or unknown.",
            "For minions, return the exact Chinese card name if visible/recognizable.",
            "For tavern spells, do not guess a minion name; set type to tavern_spell.",
            "Use confidence 0-1. If unsure, use unknown and confidence below 0.5.",
            "Local candidates are only hints and may be wrong. Override them if the image clearly differs.",
        ],
        "local_candidate_hints": local_candidates,
        "output_schema": {
            "shop_slots": [
                {
                    "slot": 1,
                    "type": "minion | tavern_spell | empty | unknown",
                    "name": "exact Chinese card name or empty string",
                    "id": "optional card id if known",
                    "tier": "integer or null",
                    "attack": "integer or null",
                    "health": "integer or null",
                    "confidence": 0.0,
                    "notes": "short reason",
                }
            ],
            "notes": "short overall notes",
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _resolve_card(database: CardDatabase, name: str, card_id: str = "") -> Card | None:
    if card_id:
        card = database.card_by_id(card_id)
        if card is not None:
            return card
    if name:
        card = database.card_by_name(name)
        if card is not None:
            return card
        compact = _compact_name(name)
        for item in database.cards:
            item_name = _compact_name(item.name)
            if compact and (compact == item_name or compact in item_name or item_name in compact):
                return item
    return None


def _compact_name(value: str) -> str:
    return "".join(ch for ch in value.strip().lower() if not ch.isspace())


def _parse_json_object(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end < start:
            raise
        data = json.loads(raw[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("Vision shop response is not a JSON object")
    return data


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
