from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .llm import OpenAICompatibleClient


SUPPORTED_EXTENSIONS = {".png", ".webp", ".jpg", ".jpeg"}

REGIONS = {
    "tier": (0.04, 0.05, 0.31, 0.27),
    "name": (0.20, 0.49, 0.80, 0.61),
    "text": (0.15, 0.61, 0.85, 0.84),
    "tribe": (0.27, 0.83, 0.73, 0.91),
    "attack": (0.02, 0.78, 0.23, 0.98),
    "health": (0.77, 0.78, 0.98, 0.98),
}

OCR_PROMPT = """
Read the Hearthstone Battlegrounds card regions in this debug panel.
The panel contains crops labeled: name, text, tribe, attack, health.
Return only valid JSON with:
name, text, tribe, attack, health.
Use Chinese exactly as visible. If the tribe crop has no visible tribe text, return tribe as "".
attack and health must be integers or null.
Do not infer hidden information from game knowledge; read only the visible crops.
""".strip()

TAG_KEYWORDS = {
    "battlecry": ("战吼", "Battlecry"),
    "deathrattle": ("亡语", "Deathrattle"),
    "discover": ("发现", "Discover"),
    "summon": ("召唤", "Summon"),
    "buff": ("获得+", "获得 +", "使一个", "使其获得", "+", "gain", "give"),
    "economy": ("铸币", "金币", "出售", "买入", "sell", "gold", "coin"),
    "scaling": ("每当", "每回合", "永久", "额外获得", "whenever", "permanently"),
    "spell": ("法术", "Spell"),
    "tavern_spell": ("酒馆法术", "Tavern Spell", "Tavern spell"),
    "magnetic": ("磁力", "Magnetic"),
    "reborn": ("复生", "Reborn"),
    "divine_shield": ("圣盾", "Divine Shield"),
    "taunt": ("嘲讽", "Taunt"),
    "venomous": ("剧毒", "烈毒", "Venomous", "Poisonous"),
    "windfury": ("风怒", "Windfury"),
}


def enrich_cards_from_images(
    input_json: str | Path,
    output_json: str | Path,
    *,
    limit: int | None = None,
    start_index: int = 0,
    debug_dir: str | Path | None = None,
    use_vision: bool = False,
    client: OpenAICompatibleClient | None = None,
) -> dict[str, Any]:
    rows = _read_json_list(input_json)
    output = Path(output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    changed = 0
    processed = 0

    for index, row in enumerate(rows):
        if index < start_index:
            continue
        if limit is not None and processed >= limit:
            break
        if not row.get("needs_metadata", True):
            continue
        image_path = row.get("image")
        if not image_path or not Path(image_path).exists():
            row["ocr_error"] = "image missing"
            continue

        processed += 1
        try:
            result = read_card_image(
                image_path,
                card_id=str(row.get("id") or Path(image_path).stem),
                debug_dir=Path(debug_dir) / str(row.get("id")) if debug_dir else None,
                use_vision=use_vision,
                client=client,
            )
            _merge_row(row, result)
            changed += 1
            output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            row["ocr_error"] = str(exc)
            output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "input": str(input_json),
        "output": str(output_json),
        "processed": processed,
        "changed": changed,
        "total": len(rows),
    }


def read_card_image(
    image_path: str | Path,
    *,
    card_id: str | None = None,
    debug_dir: str | Path | None = None,
    use_vision: bool = False,
    client: OpenAICompatibleClient | None = None,
) -> dict[str, Any]:
    path = Path(image_path)
    image = Image.open(path).convert("RGBA")
    crops = crop_card_regions(image)
    tier = count_tier_stars(crops["tier"])

    panel_path = None
    if debug_dir:
        panel_path = write_debug_crops(crops, debug_dir, path.stem)

    data: dict[str, Any] = {
        "id": card_id or path.stem.removesuffix("_battlegroundsImage"),
        "tier": tier,
        "name": "",
        "text": "",
        "tribes": [],
        "attack": None,
        "health": None,
        "tags": [],
    }
    if use_vision:
        if client is None:
            raise ValueError("client is required when use_vision=True")
        if panel_path is None:
            panel_path = write_debug_crops(crops, Path.cwd() / "work" / "card_ocr_tmp", path.stem)
        ocr = _parse_json(client.complete_with_image(OCR_PROMPT, panel_path))
        data.update(_normalize_ocr_result(ocr))

    data["tags"] = _tags_for(data)
    return data


def crop_card_regions(image: Image.Image) -> dict[str, Image.Image]:
    width, height = image.size
    crops = {}
    for name, box in REGIONS.items():
        left, top, right, bottom = box
        crops[name] = image.crop(
            (
                int(width * left),
                int(height * top),
                int(width * right),
                int(height * bottom),
            )
        )
    return crops


def count_tier_stars(tier_crop: Image.Image) -> int | None:
    image = tier_crop.convert("RGB")
    width, height = image.size
    pixels = image.load()
    yellow_pixels: set[tuple[int, int]] = set()
    for y in range(int(height * 0.05), int(height * 0.72)):
        for x in range(width):
            r, g, b = pixels[x, y]
            if r >= 165 and g >= 115 and b <= 105 and r >= g * 0.85:
                yellow_pixels.add((x, y))

    components = _connected_components(yellow_pixels)
    star_components = []
    for points in components:
        if len(points) < 20:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        box_width = max(xs) - min(xs) + 1
        box_height = max(ys) - min(ys) + 1
        if box_width >= 5 and box_height >= 5:
            star_components.append(points)
    if not star_components:
        return None
    return len(star_components)


def write_debug_crops(crops: dict[str, Image.Image], output_dir: str | Path, stem: str) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, crop in crops.items():
        crop.save(output / f"{stem}_{name}.png")
    panel = _make_debug_panel(crops)
    panel_path = output / f"{stem}_ocr_panel.png"
    panel.save(panel_path)
    return panel_path


def _make_debug_panel(crops: dict[str, Image.Image]) -> Image.Image:
    font = ImageFont.load_default()
    rows = []
    for name in ("name", "text", "tribe", "attack", "health"):
        crop = _scale_for_ocr(crops[name])
        label_width = 70
        row = Image.new("RGB", (label_width + crop.width + 20, crop.height + 18), "white")
        draw = ImageDraw.Draw(row)
        draw.text((8, 6), name, fill="black", font=font)
        row.paste(crop.convert("RGB"), (label_width, 9))
        rows.append(row)
    panel_width = max(row.width for row in rows)
    panel_height = sum(row.height for row in rows) + 8 * (len(rows) + 1)
    panel = Image.new("RGB", (panel_width + 16, panel_height), "white")
    y = 8
    for row in rows:
        panel.paste(row, (8, y))
        y += row.height + 8
    return panel


def _scale_for_ocr(image: Image.Image) -> Image.Image:
    scale = 3
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.LANCZOS)


def _connected_components(points: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []
    for point in list(points):
        if point in seen:
            continue
        stack = [point]
        seen.add(point)
        component = []
        while stack:
            x, y = stack.pop()
            component.append((x, y))
            for nx in (x - 1, x, x + 1):
                for ny in (y - 1, y, y + 1):
                    neighbor = (nx, ny)
                    if neighbor in points and neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
        components.append(component)
    return components


def _normalize_ocr_result(raw: dict[str, Any]) -> dict[str, Any]:
    tribe = str(raw.get("tribe") or "").strip()
    text = _clean_text(str(raw.get("text") or ""))
    return {
        "name": str(raw.get("name") or "").strip(),
        "text": text,
        "tribes": [tribe] if tribe else [],
        "attack": _int_or_none(raw.get("attack")),
        "health": _int_or_none(raw.get("health")),
    }


def _merge_row(row: dict[str, Any], result: dict[str, Any]) -> None:
    for key in ("name", "text", "tribes", "tags"):
        value = result.get(key)
        if value:
            row[key] = value
    for key in ("tier", "attack", "health"):
        if result.get(key) is not None:
            row[key] = result[key]
    row["needs_metadata"] = not bool(row.get("name") and row.get("text"))
    row["metadata_source"] = "card_image_ocr"
    row.pop("ocr_error", None)
    row.pop("metadata_error", None)


def _tags_for(data: dict[str, Any]) -> list[str]:
    text = str(data.get("text") or "")
    tags = {f"tribe:{tribe}" for tribe in list(data.get("tribes") or [])}
    lower_text = text.lower()
    for tag, keywords in TAG_KEYWORDS.items():
        if any(keyword.lower() in lower_text for keyword in keywords):
            tags.add(tag)
    if "战吼触发两次" in text or "Battlecries trigger twice" in text:
        tags.add("double_battlecry")
    if "亡语触发两次" in text or "Deathrattles trigger twice" in text:
        tags.add("deathrattle_multiplier")
    return sorted(tags)


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise
        return json.loads(text[start : end + 1])


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(str(value).strip())
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
    parser = argparse.ArgumentParser(description="Crop Hearthstone Battlegrounds card images, OCR visible fields, and write JSON.")
    parser.add_argument("--image", help="Read one card image and print JSON.")
    parser.add_argument("--input-json", help="Generated card index JSON to enrich.")
    parser.add_argument("--output-json", help="Output enriched JSON.")
    parser.add_argument("--debug-dir", default="work/card_ocr", help="Directory for region crops and OCR panels.")
    parser.add_argument("--use-vision", action="store_true", help="Call the configured OpenAI-compatible vision model for OCR.")
    parser.add_argument("--limit", type=int, help="Only process N missing rows when using --input-json.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--env-file")
    args = parser.parse_args()

    if not args.image and not args.input_json:
        parser.error("Provide --image or --input-json.")
    if args.input_json and not args.output_json:
        parser.error("--output-json is required with --input-json.")

    client = OpenAICompatibleClient.from_env(args.env_file) if args.use_vision else None
    if args.image:
        result = read_card_image(
            args.image,
            debug_dir=args.debug_dir,
            use_vision=args.use_vision,
            client=client,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    result = enrich_cards_from_images(
        args.input_json,
        args.output_json,
        limit=args.limit,
        start_index=args.start_index,
        debug_dir=args.debug_dir,
        use_vision=args.use_vision,
        client=client,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
