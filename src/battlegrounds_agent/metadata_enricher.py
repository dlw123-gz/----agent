from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .llm import OpenAICompatibleClient
from .synergy import infer_card_tags


CARD_PROMPT = """
Read this Hearthstone Battlegrounds card image.
Return only JSON with:
id, name, tier, attack, health, tribes, text, tags.
Use Chinese for name/text when visible. tags should be English mechanism labels such as:
battlecry, deathrattle, discover, summon, buff, economy, scaling, spell, tavern_spell,
dragon_buff, double_battlecry, deathrattle_multiplier, summon_payoff, spell_payoff.
Use null for unreadable numeric fields, [] for unknown tribes/tags.
Do not invent rules that are not visible on the card.
""".strip()

TRINKET_PROMPT = """
Read this Hearthstone Battlegrounds trinket image.
Return only JSON with:
id, name, text, tier, tribes, tags.
Use Chinese for name/text when visible. tier should be lesser, greater, or null.
tags should be English mechanism labels such as:
battlecry, deathrattle, discover, summon, buff, economy, scaling, spell, tavern_spell,
dragon_buff, double_battlecry, deathrattle_multiplier, summon_payoff, spell_payoff.
Use [] for unknown tribes/tags. Do not invent rules that are not visible.
""".strip()


def enrich_assets(
    input_path: str | Path,
    output_path: str | Path,
    kind: str,
    limit: int | None,
    start_index: int,
    client: OpenAICompatibleClient,
) -> dict:
    source = Path(input_path)
    rows = json.loads(source.read_text(encoding="utf-8"))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    enriched = []
    count = 0
    for index, row in enumerate(rows):
        if index < start_index:
            enriched.append(row)
            continue
        if limit is not None and count >= limit:
            enriched.append(row)
            continue
        if not row.get("needs_metadata", True):
            enriched.append(row)
            continue
        image = row.get("image")
        if not image or not Path(image).exists():
            row["metadata_error"] = "image missing"
            enriched.append(row)
            continue
        try:
            data = _read_metadata(row, kind, client)
            merged = _merge_metadata(row, data, kind)
            enriched.append(merged)
            count += 1
            output.write_text(json.dumps(enriched + rows[index + 1 :], ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            row["metadata_error"] = str(exc)
            enriched.append(row)
            output.write_text(json.dumps(enriched + rows[index + 1 :], ensure_ascii=False, indent=2), encoding="utf-8")

    output.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "input": str(input_path),
        "output": str(output_path),
        "kind": kind,
        "start_index": start_index,
        "enriched_count": count,
        "total": len(rows),
    }


def _read_metadata(row: dict, kind: str, client: OpenAICompatibleClient) -> dict:
    prompt = CARD_PROMPT if kind == "card" else TRINKET_PROMPT
    prompt = f"{prompt}\nThe asset id is {row['id']}. Include this exact id in the JSON."
    raw = client.complete_with_image(prompt, row["image"])
    return _parse_json(raw)


def _merge_metadata(row: dict, data: dict, kind: str) -> dict:
    merged = dict(row)
    merged["id"] = row["id"]
    merged["name"] = data.get("name") or row.get("name") or row["id"]
    merged["text"] = data.get("text") or row.get("text") or ""
    merged["tribes"] = list(data.get("tribes") or row.get("tribes") or [])
    merged["tags"] = sorted(set(list(data.get("tags") or []) + list(row.get("tags") or [])))
    if kind == "card":
        merged["tier"] = _int_or_none(data.get("tier")) or row.get("tier")
        merged["attack"] = _int_or_none(data.get("attack")) if data.get("attack") is not None else row.get("attack")
        merged["health"] = _int_or_none(data.get("health")) if data.get("health") is not None else row.get("health")
        from .schemas import Card

        card = Card(
            id=merged["id"],
            name=merged["name"],
            tier=int(merged.get("tier") or 1),
            attack=int(merged.get("attack") or 0),
            health=int(merged.get("health") or 0),
            tribes=merged["tribes"],
            text=merged["text"],
            tags=merged["tags"],
        )
        merged["tags"] = sorted(infer_card_tags(card))
    else:
        merged["tier"] = data.get("tier") or row.get("tier")
    merged["needs_metadata"] = not bool(merged.get("name") and merged.get("text"))
    merged.pop("metadata_error", None)
    return merged


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise
        return json.loads(text[start : end + 1])


def _int_or_none(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Fill card/trinket JSON metadata from asset images using a vision model.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kind", choices=["card", "trinket"], required=True)
    parser.add_argument("--limit", type=int, help="Only enrich the first N missing rows.")
    parser.add_argument("--start-index", type=int, default=0, help="Start enriching from this zero-based row index.")
    parser.add_argument("--env-file")
    args = parser.parse_args()

    client = OpenAICompatibleClient.from_env(args.env_file)
    result = enrich_assets(args.input, args.output, args.kind, args.limit, args.start_index, client)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
