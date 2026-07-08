from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schemas import GameState
from .synergy import infer_card_tags


@dataclass(frozen=True, slots=True)
class CompositionGuide:
    id: str
    title: str
    source_image: str = ""
    row: int | None = None
    tribes: tuple[str, ...] = ()
    core_cards: tuple[str, ...] = ()
    key_cards: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    plan_lines: tuple[str, ...] = ()
    playstyle: str = ""
    score: float = 0.0
    match_reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "source_image": self.source_image,
            "row": self.row,
            "tribes": list(self.tribes),
            "core_cards": list(self.core_cards),
            "key_cards": list(self.key_cards),
            "tags": list(self.tags),
            "playstyle": self.playstyle,
            "plan_lines": list(self.plan_lines),
            "match_reasons": list(self.match_reasons),
        }


def load_composition_guides(path: str | Path | None = None) -> list[CompositionGuide]:
    data_path = _resolve_composition_path(path)
    if data_path is None or not data_path.exists():
        return []
    try:
        data = json.loads(data_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    guides: list[CompositionGuide] = []
    for item in data if isinstance(data, list) else []:
        guides.extend(_guides_from_item(item))
    return guides


def relevant_compositions(state: GameState, guides: list[CompositionGuide] | None = None, limit: int = 5) -> list[CompositionGuide]:
    candidates = guides if guides is not None else load_composition_guides()
    if not candidates:
        return []

    state_tribes = {tribe for minion in state.player.board for tribe in minion.tribes}
    state_tribes.update(tribe for card in state.shop.cards + state.player.hand for tribe in card.tribes)
    state_tags = set()
    state_names = set()
    for card in state.shop.cards + state.player.hand:
        state_tags.update(infer_card_tags(card))
        state_names.add(card.name)
    for minion in state.player.board:
        state_names.add(minion.name)
        state_tribes.update(minion.tribes)
        state_tags.update(tag for tag in minion.enchantments)

    scored: list[CompositionGuide] = []
    for guide in candidates:
        score = 0.0
        reasons: list[str] = []
        tribe_overlap = state_tribes.intersection(guide.tribes)
        if tribe_overlap:
            score += 12.0 + 2.0 * len(tribe_overlap)
            reasons.append(f"种族匹配: {', '.join(sorted(tribe_overlap))}")
        tag_overlap = state_tags.intersection(guide.tags)
        if tag_overlap:
            score += 5.0 + len(tag_overlap)
            reasons.append(f"机制匹配: {', '.join(sorted(tag_overlap)[:4])}")
        card_overlap = state_names.intersection(set(guide.core_cards) | set(guide.key_cards))
        if card_overlap:
            score += 18.0 + 3.0 * len(card_overlap)
            reasons.append(f"核心牌出现: {', '.join(sorted(card_overlap)[:4])}")
        if score > 0:
            scored.append(_replace_score(guide, score, tuple(reasons)))
    return sorted(scored, key=lambda item: item.score, reverse=True)[:limit]


def _guides_from_item(item: dict[str, Any]) -> list[CompositionGuide]:
    image = str(item.get("image") or "")
    image_id = str(item.get("id") or Path(image).stem or "composition")
    root_title = str(item.get("title") or "")
    guides: list[CompositionGuide] = []
    rows = item.get("rows")
    if isinstance(rows, list) and rows:
        for row in rows:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or root_title or image_id)
            plan_lines = _string_tuple(row.get("recommended_play_lines") or row.get("description_lines"))
            guides.append(
                CompositionGuide(
                    id=f"{image_id}:{row.get('row', len(guides) + 1)}",
                    title=title,
                    source_image=image,
                    row=_int_or_none(row.get("row")),
                    tribes=_string_tuple(row.get("tribes")),
                    core_cards=_card_names(row.get("core_cards") or row.get("cards")),
                    key_cards=_card_names(row.get("key_cards") or row.get("cards")),
                    tags=_normalize_tags(row.get("tags") or plan_lines),
                    plan_lines=plan_lines,
                    playstyle=str(row.get("playstyle") or ""),
                )
            )
        return guides

    legacy_plan_lines = _string_tuple(item.get("plan_lines"))
    if not legacy_plan_lines and item.get("plan"):
        legacy_plan_lines = (str(item.get("plan")),)
    guides.append(
        CompositionGuide(
            id=image_id,
            title=str(item.get("name") or root_title or image_id),
            source_image=image,
            tribes=_string_tuple(item.get("tribes")),
            core_cards=_string_tuple(item.get("core_cards")),
            key_cards=_string_tuple(item.get("key_cards")),
            tags=_normalize_tags(item.get("core_tags") or item.get("tags")),
            plan_lines=legacy_plan_lines,
            playstyle=str(item.get("playstyle") or item.get("plan") or ""),
        )
    )
    return guides


def _replace_score(guide: CompositionGuide, score: float, reasons: tuple[str, ...]) -> CompositionGuide:
    return CompositionGuide(
        id=guide.id,
        title=guide.title,
        source_image=guide.source_image,
        row=guide.row,
        tribes=guide.tribes,
        core_cards=guide.core_cards,
        key_cards=guide.key_cards,
        tags=guide.tags,
        plan_lines=guide.plan_lines,
        playstyle=guide.playstyle,
        score=score,
        match_reasons=reasons,
    )


def _resolve_composition_path(path: str | Path | None) -> Path | None:
    if path:
        return Path(path)
    root = Path(__file__).resolve().parents[2]
    for candidate in (
        root / "examples" / "data" / "compositions.vision.json",
        root / "examples" / "data" / "compositions.json",
    ):
        if candidate.exists():
            return candidate
    return None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


def _card_names(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return _string_tuple(value)
    names: list[str] = []
    for item in value:
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("target_name") or "").strip()
        else:
            name = str(item).strip()
        if name:
            names.append(name)
    return tuple(names)


def _normalize_tags(value: Any) -> tuple[str, ...]:
    text = " ".join(_string_tuple(value)).lower()
    tags = set(_string_tuple(value))
    keyword_tags = {
        "battlecry": ("战吼", "battlecry"),
        "deathrattle": ("亡语", "deathrattle"),
        "summon": ("召唤", "衍生", "summon"),
        "buff": ("成长", "增益", "+", "buff"),
        "economy": ("经济", "铸币", "金币", "economy"),
        "scaling": ("永久", "养", "成长", "scaling"),
        "spell": ("法术", "spell"),
        "tavern_spell": ("酒馆法术",),
        "divine_shield": ("圣盾",),
        "taunt": ("嘲讽",),
        "venomous": ("烈毒", "剧毒"),
        "reborn": ("复生",),
    }
    for tag, keywords in keyword_tags.items():
        if any(keyword.lower() in text for keyword in keywords):
            tags.add(tag)
    return tuple(sorted(str(tag) for tag in tags if str(tag).strip()))


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
