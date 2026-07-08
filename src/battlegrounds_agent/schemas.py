from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class ActionType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    ROLL = "roll"
    LEVEL = "level"
    FREEZE = "freeze"
    PLAY = "play"
    REPOSITION = "reposition"
    PICK_TRINKET = "pick_trinket"
    HOLD = "hold"


@dataclass(slots=True)
class Card:
    id: str
    name: str
    tier: int
    attack: int
    health: int
    tribes: list[str] = field(default_factory=list)
    text: str = ""
    tags: list[str] = field(default_factory=list)
    version: str | None = None
    image: str | None = None


@dataclass(slots=True)
class Trinket:
    id: str
    name: str
    text: str
    tier: Literal["lesser", "greater"] | None = None
    tribes: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    image: str | None = None


@dataclass(slots=True)
class BoardMinion:
    card_id: str
    name: str
    attack: int
    health: int
    position: int
    tribes: list[str] = field(default_factory=list)
    text: str = ""
    enchantments: list[str] = field(default_factory=list)
    golden: bool = False


@dataclass(slots=True)
class PlayerState:
    tavern_tier: int
    health: int
    armor: int = 0
    gold: int = 0
    turn: int = 1
    board: list[BoardMinion] = field(default_factory=list)
    hand: list[Card] = field(default_factory=list)


@dataclass(slots=True)
class ShopState:
    cards: list[Card] = field(default_factory=list)
    frozen: bool = False
    roll_cost: int = 1
    level_cost: int | None = None


@dataclass(slots=True)
class GameState:
    player: PlayerState
    shop: ShopState
    available_tribes: list[str] = field(default_factory=list)
    offered_trinkets: list[Trinket] = field(default_factory=list)
    notes: str = ""


@dataclass(slots=True)
class CandidateAction:
    action_type: ActionType
    target_id: str | None = None
    target_name: str | None = None
    score: float = 0.0
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActionPlan:
    summary: str
    actions: list[CandidateAction] = field(default_factory=list)
    composition_goal: str | None = None
    buy_recommendation: str | None = None
    level_or_roll_recommendation: str | None = None
    potential_synergy_cards: list[dict[str, Any]] = field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "medium"
    llm_reasoning: str | None = None
