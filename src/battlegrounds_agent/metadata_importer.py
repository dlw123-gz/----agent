from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


TRIBE_ALIASES = {
    "BEAST": "野兽",
    "DEMON": "恶魔",
    "DRAGON": "龙",
    "ELEMENTAL": "元素",
    "MECHANICAL": "机械",
    "MECH": "机械",
    "MURLOC": "鱼人",
    "NAGA": "纳迦",
    "PIRATE": "海盗",
    "QUILBOAR": "野猪人",
    "UNDEAD": "亡灵",
    "ALL": "全部",
}

TAG_KEYWORDS = {
    "battlecry": ("战吼", "Battlecry"),
    "deathrattle": ("亡语", "Deathrattle"),
    "discover": ("发现", "Discover"),
    "summon": ("召唤", "Summon"),
    "buff": ("获得+", "获得 +", "使一个", "使其获得", "give", "gain", "+"),
    "economy": ("铸币", "金币", "出售", "coin", "gold", "sell"),
    "scaling": ("每当", "每回合", "永久", "额外获得", "whenever", "after you", "permanently"),
    "spell": ("法术", "Spell"),
    "tavern_spell": ("酒馆法术", "Tavern spell", "Tavern Spell"),
    "magnetic": ("磁力", "Magnetic"),
    "reborn": ("复生", "Reborn"),
    "divine_shield": ("圣盾", "Divine Shield"),
    "taunt": ("嘲讽", "Taunt"),
    "venomous": ("剧毒", "烈毒", "Venomous", "Poisonous"),
    "windfury": ("风怒", "Windfury"),
}


def merge_metadata(index_path: str | Path, source_path: str | Path, output_path: str | Path) -> dict[str, int]:
    index_rows = _read_json_list(index_path)
    source_rows = _read_json_list(source_path)
    source_by_id = {_normalize_id(str(row.get("id", ""))): row for row in source_rows if row.get("id")}

    matched = 0
    missing = 0
    for row in index_rows:
        card_id = _normalize_id(str(row.get("id", "")))
        source = source_by_id.get(card_id)
        if source is None:
            row["needs_metadata"] = True
            missing += 1
            continue
        _merge_row(row, source)
        matched += 1

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(index_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"total": len(index_rows), "matched": matched, "missing": missing}


def _merge_row(row: dict[str, Any], source: dict[str, Any]) -> None:
    row["name"] = _first_text(source, "name", "name_zhCN", "name_zhTW", "name_enUS") or row.get("name") or row["id"]
    row["tier"] = _tier(source)
    row["attack"] = _int_or_none(source.get("attack"))
    row["health"] = _int_or_none(source.get("health"))
    row["tribes"] = _tribes(source)
    row["text"] = _clean_text(_first_text(source, "text", "text_zhCN", "text_zhTW", "text_enUS"))
    row["tags"] = sorted(set(row.get("tags") or []) | _infer_tags(row["text"], row["tribes"]))
    row["needs_metadata"] = not bool(row["name"] and row["text"])
    row["metadata_source"] = "hearthstonejson"


def _tier(source: dict[str, Any]) -> int | None:
    battlegrounds = source.get("battlegrounds")
    if isinstance(battlegrounds, dict):
        for key in ("tier", "techLevel"):
            value = _int_or_none(battlegrounds.get(key))
            if value is not None:
                return value
    for key in ("techLevel", "tier"):
        value = _int_or_none(source.get(key))
        if value is not None:
            return value
    return None


def _tribes(source: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    for key in ("races", "race", "minionTypes", "minionType"):
        value = source.get(key)
        if isinstance(value, list):
            values.extend(value)
        elif value:
            values.append(value)
    tribes = []
    for value in values:
        tribe = TRIBE_ALIASES.get(str(value), str(value))
        if tribe and tribe not in tribes:
            tribes.append(tribe)
    return tribes


def _infer_tags(text: str, tribes: list[str]) -> set[str]:
    tags = {f"tribe:{tribe}" for tribe in tribes}
    for tag, keywords in TAG_KEYWORDS.items():
        if any(keyword.lower() in text.lower() for keyword in keywords):
            tags.add(tag)
    if "战吼触发两次" in text or "Battlecries trigger twice" in text:
        tags.add("double_battlecry")
    if "亡语触发两次" in text or "Deathrattles trigger twice" in text:
        tags.add("deathrattle_multiplier")
    return tags


def _first_text(source: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text or "")
    return text.replace("\\n", " ").replace("\n", " ").strip()


def _normalize_id(card_id: str) -> str:
    return card_id.removesuffix("_battlegroundsImage").strip()


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_json_list(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Merge Hearthstone card metadata into generated image indexes.")
    parser.add_argument("--index", default="examples/data/cards.generated.json", help="Generated card/trinket index JSON.")
    parser.add_argument("--source", required=True, help="Local HearthstoneJSON cards.json path.")
    parser.add_argument("--output", default="examples/data/cards.enriched.json", help="Merged output JSON.")
    args = parser.parse_args()

    summary = merge_metadata(args.index, args.source, args.output)
    summary["output"] = args.output
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
