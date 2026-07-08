from __future__ import annotations

from dataclasses import dataclass

from .database import CardDatabase
from .schemas import BoardMinion, Card, GameState, PlayerState, ShopState, Trinket
from .screen_reader import SlotRecognition


@dataclass(frozen=True, slots=True)
class BuildStateOptions:
    tavern_tier: int = 1
    health: int = 40
    armor: int = 0
    gold: int = 3
    turn: int = 1
    available_tribes: tuple[str, ...] = ()
    roll_cost: int = 1
    level_cost: int | None = None
    min_score: float = 0.58
    low_confidence_score: float = 0.68


def build_state_from_slots(
    slots: list[SlotRecognition],
    database: CardDatabase,
    options: BuildStateOptions | None = None,
) -> GameState:
    opts = options or BuildStateOptions()
    shop_cards: list[Card] = []
    board: list[BoardMinion] = []
    hand: list[Card] = []
    trinkets: list[Trinket] = []
    notes: list[str] = []

    for slot in slots:
        scored_match = _best_match(slot, database, opts.min_score)
        if scored_match is None:
            feature_note = _feature_note(slot)
            notes.append(f"{slot.slot_id}: no confident match{feature_note}")
            continue
        match, effective_score, confidence_gap = scored_match

        if slot.kind == "card":
            card = database.card_by_id(match.id)
            if card is None:
                notes.append(f"{slot.slot_id}: recognized {match.id} but card is missing from the card database")
                card = _unknown_card(match.id)
            if effective_score < opts.low_confidence_score:
                notes.append(f"{slot.slot_id}: low confidence match {match.id} score={effective_score:.3f}{_feature_note(slot)}")
                continue
            if slot.slot_id.startswith("shop_") and card.tier > opts.tavern_tier:
                notes.append(
                    f"{slot.slot_id}: suspicious shop card {card.name} tier={card.tier} above tavern_tier={opts.tavern_tier}; check crop/profile"
                )
                continue
            if slot.slot_id.startswith("board_") and not _is_confident_board_match(effective_score, confidence_gap):
                gap_note = f", gap={confidence_gap:.3f}" if confidence_gap is not None else ""
                notes.append(f"{slot.slot_id}: low confidence board match {match.id} score={effective_score:.3f}{gap_note}{_feature_note(slot)}")
                continue

            if slot.slot_id.startswith("shop_"):
                shop_cards.append(card)
            elif slot.slot_id.startswith("board_"):
                board.append(_board_minion(card, _slot_index(slot.slot_id)))
            elif slot.slot_id.startswith("hand_"):
                hand.append(card)
            else:
                notes.append(f"{slot.slot_id}: recognized card {card.name} but slot prefix is unknown")
        elif slot.kind == "trinket":
            trinket = next((item for item in database.trinkets if item.id == match.id), None)
            if trinket is None:
                notes.append(f"{slot.slot_id}: recognized {match.id} but trinket is missing from the trinket database")
                trinket = _unknown_trinket(match.id)
            if effective_score < opts.low_confidence_score:
                notes.append(f"{slot.slot_id}: low confidence match {match.id} score={match.score:.3f}")
                continue
            trinkets.append(trinket)

    board.sort(key=lambda item: item.position)
    for position, minion in enumerate(board):
        minion.position = position
    return GameState(
        player=PlayerState(
            tavern_tier=opts.tavern_tier,
            health=opts.health,
            armor=opts.armor,
            gold=opts.gold,
            turn=opts.turn,
            board=board,
            hand=hand,
        ),
        shop=ShopState(cards=shop_cards, roll_cost=opts.roll_cost, level_cost=opts.level_cost),
        available_tribes=list(opts.available_tribes),
        offered_trinkets=trinkets,
        notes="; ".join(notes),
    )


def _best_match(slot: SlotRecognition, database: CardDatabase, min_score: float):
    if not slot.matches:
        return None
    if slot.kind != "card":
        match = slot.matches[0]
        return (match, match.score, None) if match.score >= min_score else None

    features = slot.features or {}
    is_shop_slot = slot.slot_id.startswith("shop_")
    is_board_slot = slot.slot_id.startswith("board_")
    use_slot_numbers = not is_shop_slot and not is_board_slot
    observed_attack = _int_or_none(features.get("attack")) if use_slot_numbers else None
    observed_health = _int_or_none(features.get("health")) if use_slot_numbers else None
    observed_tier = None if is_board_slot else _int_or_none(features.get("tier"))
    tier_confidence = _float_or_zero(features.get("tier_confidence"))
    if is_shop_slot:
        # Tiny shop badge reads are still noisy on compressed/video frames.
        # Use the player's tavern tier as the database upper bound instead.
        observed_tier = None
    if observed_tier is not None:
        tier_threshold = 0.70 if is_shop_slot else 0.40
        if tier_confidence < tier_threshold or not 1 <= observed_tier <= 6:
            observed_tier = None

    best = None
    best_score = -1.0
    scored_candidates: list[tuple[float, object]] = []
    for match in slot.matches:
        card = database.card_by_id(match.id)
        if card is None:
            continue
        score = match.score
        if observed_attack is not None:
            score += 0.09 if card.attack == observed_attack else -0.16
        if observed_health is not None:
            score += 0.09 if card.health == observed_health else -0.16
        if observed_tier is not None:
            score += _tier_score_adjustment(card.tier, observed_tier, tier_confidence, is_shop_slot)
        scored_candidates.append((score, match))
        if score > best_score:
            best = match
            best_score = score

    if best is None:
        return None
    confidence_gap = None
    if len(scored_candidates) >= 2:
        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        confidence_gap = scored_candidates[0][0] - scored_candidates[1][0]
    if is_shop_slot and confidence_gap is not None:
        if best_score < 0.70 and confidence_gap < 0.02:
            return None
    return (best, best_score, confidence_gap) if best_score >= min_score else None


def _is_confident_board_match(score: float, confidence_gap: float | None) -> bool:
    if score >= 0.86:
        return True
    if confidence_gap is None:
        return score >= 0.82
    return score >= 0.76 and confidence_gap >= 0.035


def _board_minion(card: Card, position: int) -> BoardMinion:
    return BoardMinion(
        card_id=card.id,
        name=card.name,
        attack=card.attack,
        health=card.health,
        position=position,
        tribes=list(card.tribes),
        text=card.text,
    )


def _unknown_card(card_id: str) -> Card:
    return Card(id=card_id, name=card_id, tier=1, attack=0, health=0, text="未在卡牌数据库中找到这张牌。")


def _unknown_trinket(trinket_id: str) -> Trinket:
    return Trinket(id=trinket_id, name=trinket_id, text="未在饰品数据库中找到这个饰品。")


def _slot_index(slot_id: str) -> int:
    suffix = slot_id.rsplit("_", 1)[-1]
    return int(suffix) - 1 if suffix.isdigit() else 0


def _feature_note(slot: SlotRecognition) -> str:
    features = slot.features or {}
    values = []
    for key in ("tier", "attack", "health", "presence_score"):
        value = features.get(key)
        if value is not None:
            values.append(f"{key}={value}")
    return f" ({', '.join(values)})" if values else ""


def _int_or_none(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_zero(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _tier_score_adjustment(card_tier: int, observed_tier: int, confidence: float, is_shop_slot: bool) -> float:
    if card_tier == observed_tier:
        return 0.08 if is_shop_slot else 0.05
    if is_shop_slot:
        # Shop badge reading is useful but still noisy on small animated cards,
        # so it should nudge close visual matches rather than override them.
        penalty = 0.04 if abs(card_tier - observed_tier) <= 1 else 0.07
        return -penalty * min(max(confidence, 0.0), 1.0)
    return -0.08
