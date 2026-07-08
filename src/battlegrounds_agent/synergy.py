from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import Card, GameState, Trinket


MECHANIC_KEYWORDS = {
    "battlecry": ("战吼", "Battlecry"),
    "deathrattle": ("亡语", "Deathrattle"),
    "summon": ("召唤", "Summon"),
    "discover": ("发现", "Discover"),
    "buff": ("获得+", "获得 +", "使一个", "使其获得", "+"),
    "economy": ("铸币", "金币", "出售", "免费刷新"),
    "scaling": ("每当", "每回合", "永久", "额外获得", "在本局对战中"),
    "spell": ("法术", "Spell"),
    "tavern_spell": ("酒馆法术", "Tavern Spell"),
    "magnetic": ("磁力", "Magnetic"),
    "reborn": ("复生", "Reborn"),
    "divine_shield": ("圣盾", "Divine Shield"),
    "taunt": ("嘲讽", "Taunt"),
    "venomous": ("剧毒", "烈毒", "Venomous", "Poisonous"),
    "windfury": ("风怒", "Windfury"),
}

SYNERGY_RULES = (
    ("double_battlecry", "battlecry", 14.0, "战吼会被额外触发，战吼牌价值提高。"),
    ("deathrattle_multiplier", "deathrattle", 12.0, "亡语触发收益被放大，亡语牌价值提高。"),
    ("summon_payoff", "summon", 9.0, "召唤频率能触发当前成长/收益组件。"),
    ("buff_payoff", "buff", 8.0, "属性增益能喂给当前成长组件或补即时战力。"),
    ("economy_payoff", "economy", 7.0, "经济牌能支持更多买卖、升本和找核心。"),
    ("spell_payoff", "spell", 7.0, "法术相关收益会提高酒馆法术价值。"),
)

TAG_ALIASES = {
    "battlecry": "battlecry",
    "double_battlecry": "double_battlecry",
    "brann": "double_battlecry",
    "deathrattle": "deathrattle",
    "deathrattle_multiplier": "deathrattle_multiplier",
    "summon": "summon",
    "summon_payoff": "summon_payoff",
    "buff": "buff",
    "dragon_buff": "buff",
    "economy": "economy",
    "economy_payoff": "economy_payoff",
    "scaling": "scaling",
    "spell": "spell",
    "tavern_spell": "spell",
    "spell_payoff": "spell_payoff",
}


@dataclass(frozen=True, slots=True)
class SynergyFact:
    source_id: str
    source_name: str
    target_id: str
    target_name: str
    score: float
    reason: str
    tags: tuple[str, ...] = field(default_factory=tuple)


def infer_card_tags(card: Card | Trinket) -> set[str]:
    tags = {_normalize_tag(tag) for tag in card.tags}
    text = card.text or ""
    lower_text = text.lower()
    for tag, keywords in MECHANIC_KEYWORDS.items():
        if any(keyword.lower() in lower_text for keyword in keywords):
            tags.add(tag)
    for tribe in getattr(card, "tribes", []):
        tags.add(f"tribe:{tribe}")
    return {tag for tag in tags if tag}


def explain_state_synergies(state: GameState) -> list[SynergyFact]:
    sources = list(state.player.board) + list(state.player.hand)
    shop_cards = list(state.shop.cards)
    facts: list[SynergyFact] = []
    source_tags = [(source, infer_card_tags(_card_like(source))) for source in sources]

    for target in shop_cards:
        target_tags = infer_card_tags(target)
        for source, tags in source_tags:
            facts.extend(_match_synergy(_source_id(source), source.name, tags, target, target_tags))
            facts.extend(_match_tribe_synergy(_source_id(source), source.name, tags, target, target_tags))
    return sorted(facts, key=lambda item: item.score, reverse=True)


def score_card_synergy(state: GameState, card: Card) -> tuple[float, list[str]]:
    target_tags = infer_card_tags(card)
    score = 0.0
    reasons: list[str] = []
    for source in list(state.player.board) + list(state.player.hand):
        tags = infer_card_tags(_card_like(source))
        for fact in _match_synergy(_source_id(source), source.name, tags, card, target_tags):
            score += fact.score
            reasons.append(f"{source.name}: {fact.reason}")
        for fact in _match_tribe_synergy(_source_id(source), source.name, tags, card, target_tags):
            score += fact.score
            reasons.append(f"{source.name}: {fact.reason}")
    return score, reasons


def _match_synergy(source_id: str, source_name: str, source_tags: set[str], target: Card, target_tags: set[str]) -> list[SynergyFact]:
    facts: list[SynergyFact] = []
    for source_tag, target_tag, score, reason in SYNERGY_RULES:
        if source_tag in source_tags and target_tag in target_tags:
            facts.append(
                SynergyFact(
                    source_id=source_id,
                    source_name=source_name,
                    target_id=target.id,
                    target_name=target.name,
                    score=score,
                    reason=reason,
                    tags=(source_tag, target_tag),
                )
            )
    return facts


def _match_tribe_synergy(source_id: str, source_name: str, source_tags: set[str], target: Card, target_tags: set[str]) -> list[SynergyFact]:
    facts: list[SynergyFact] = []
    shared = sorted(tag.removeprefix("tribe:") for tag in source_tags.intersection(target_tags) if tag.startswith("tribe:"))
    for tribe in shared:
        facts.append(
            SynergyFact(
                source_id=source_id,
                source_name=source_name,
                target_id=target.id,
                target_name=target.name,
                score=6.0,
                reason=f"同属 {tribe} 体系，保留阵容一致性。",
                tags=(f"tribe:{tribe}",),
            )
        )
    return facts


def _normalize_tag(tag: str) -> str:
    return TAG_ALIASES.get(tag, tag)


def _source_id(source) -> str:
    return getattr(source, "card_id", getattr(source, "id", "unknown"))


def _card_like(source) -> Card:
    if isinstance(source, Card):
        return source
    return Card(
        id=source.card_id,
        name=source.name,
        tier=1,
        attack=source.attack,
        health=source.health,
        tribes=list(source.tribes),
        text=source.text,
        tags=list(getattr(source, "enchantments", [])),
    )
