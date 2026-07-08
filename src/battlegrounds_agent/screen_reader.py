from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .card_slot_analyzer import analyze_card_slot
from .image_recognizer import ImageKind, ImageRecognizer, RecognitionResult


@dataclass(frozen=True, slots=True)
class CropRegion:
    id: str
    kind: ImageKind
    box: tuple[float, float, float, float]
    note: str = ""


@dataclass(frozen=True, slots=True)
class SlotRecognition:
    slot_id: str
    kind: ImageKind
    box_pixels: tuple[int, int, int, int]
    crop_path: str | None
    matches: list[RecognitionResult]
    features: dict[str, Any] | None = None


def load_crop_profile(path: str | Path) -> list[CropRegion]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        CropRegion(
            id=item["id"],
            kind=item["kind"],
            box=tuple(item["box"]),
            note=item.get("note", ""),
        )
        for item in data["regions"]
    ]


def read_screen(
    screenshot_path: str | Path,
    profile_path: str | Path,
    recognizer: ImageRecognizer,
    top_k: int = 3,
    output_dir: str | Path | None = None,
    max_shop_tier: int | None = None,
) -> list[SlotRecognition]:
    screenshot = Image.open(screenshot_path).convert("RGB")
    regions = load_crop_profile(profile_path)
    output_path = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

    results: list[SlotRecognition] = []
    for region in regions:
        box_pixels = _resolve_box(region.box, screenshot.size)
        crop = screenshot.crop(box_pixels)
        crop_path = None
        if output_path:
            crop_file = output_path / f"{region.id}.png"
            crop.save(crop_file)
            crop_path = str(crop_file)
        features = analyze_card_slot(crop).to_dict() if region.kind == "card" else None
        if region.kind == "card" and region.id.startswith("board_") and features is not None:
            features["presence_score"] = _board_minion_presence_score(crop)
        if region.kind == "card" and features and _looks_like_empty_card_slot(region.id, features):
            results.append(
                SlotRecognition(
                    slot_id=region.id,
                    kind=region.kind,
                    box_pixels=box_pixels,
                    crop_path=crop_path,
                    matches=[],
                    features=features,
                )
            )
            continue
        if region.id.startswith("shop_") and features and features.get("slot_type") == "tavern_spell":
            results.append(
                SlotRecognition(
                    slot_id=region.id,
                    kind=region.kind,
                    box_pixels=box_pixels,
                    crop_path=crop_path,
                    matches=[],
                    features=features,
                )
            )
            continue
        slot_tier = features.get("tier") if features and isinstance(features.get("tier"), int) else None
        tier_confidence = float(features.get("tier_confidence") or 0.0) if features else 0.0
        slot_max_shop_tier = max_shop_tier
        if region.id.startswith("shop_"):
            # Shop badge OCR is noisy on tiny tavern minions. Use the HUD tier
            # as an upper bound, but do not exact-filter by the slot badge. If
            # a visible shop badge confidently shows a higher valid tier than
            # the fallback HUD value, allow that tier so a low fallback does not
            # remove the correct card from the candidate database.
            if _valid_shop_tier(slot_tier) and tier_confidence >= 0.65:
                slot_max_shop_tier = max(slot_max_shop_tier or 0, slot_tier)
            slot_tier = None
        elif region.id.startswith("board_"):
            # Board minions are rendered with buffs, golden frames, deathrattle
            # icons, and combat effects. Their star/stat reads are too noisy to
            # use as hard database filters, so match by art only.
            slot_tier = None
        if tier_confidence < 0.65 or (max_shop_tier is not None and slot_tier is not None and slot_tier > max_shop_tier):
            slot_tier = None
        matches = recognizer.recognize_image(
            crop,
            kind=region.kind,
            top_k=top_k,
            card_tier=slot_tier,
            max_card_tier=slot_max_shop_tier if region.id.startswith("shop_") else None,
        )
        results.append(
            SlotRecognition(
                slot_id=region.id,
                kind=region.kind,
                box_pixels=box_pixels,
                crop_path=crop_path,
                matches=matches,
                features=features,
            )
        )
    return results


def _resolve_box(box: tuple[float, float, float, float], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    left, top, right, bottom = box
    if max(box) <= 1.0:
        return (
            int(left * width),
            int(top * height),
            int(right * width),
            int(bottom * height),
        )
    return tuple(int(value) for value in box)


def _looks_like_empty_card_slot(slot_id: str, features: dict[str, Any]) -> bool:
    confidence = float(features.get("confidence") or 0.0)
    if slot_id.startswith("board_"):
        presence_score = _float_or_zero(features.get("presence_score"))
        if presence_score > 0.18:
            return False
        return presence_score <= 0.18
    return (
        slot_id.startswith("shop_")
        and
        confidence < 0.40
        and features.get("attack") is None
        and features.get("health") is None
    )


def _valid_shop_tier(value: Any) -> bool:
    try:
        tier = int(value)
    except (TypeError, ValueError):
        return False
    return 1 <= tier <= 6


def _board_minion_presence_score(image: Image.Image) -> float:
    crop = _crop_ratio(image.convert("RGB"), (0.12, 0.04, 0.88, 0.82))
    width, height = crop.size
    pixels = crop.load()
    total = 0
    wood = 0
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    radius_x = width * 0.43
    radius_y = height * 0.48
    for y in range(height):
        for x in range(width):
            dx = (x - center_x) / radius_x
            dy = (y - center_y) / radius_y
            if dx * dx + dy * dy > 1.0:
                continue
            total += 1
            r, g, b = pixels[x, y]
            if 65 <= r <= 185 and 35 <= g <= 135 and 25 <= b <= 125 and r >= g + 12 and g >= b - 8:
                wood += 1
    if total == 0:
        return 0.0
    return round(1.0 - (wood / total), 4)


def _crop_ratio(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    width, height = image.size
    left, top, right, bottom = box
    return image.crop((int(left * width), int(top * height), int(right * width), int(bottom * height)))


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_jsonable(results: list[SlotRecognition]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for result in results:
        item = asdict(result)
        item["matches"] = [asdict(match) for match in result.matches]
        payload.append(item)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop a game screenshot and recognize card/trinket slots.")
    parser.add_argument("--screenshot", required=True, help="Full game screenshot.")
    parser.add_argument("--profile", required=True, help="Crop profile JSON.")
    parser.add_argument("--output-dir", help="Optional directory for cropped slot images.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--card-dir", default=r"C:\Users\dlw\Documents\Codex\2026-07-05\gen\outputs\data card")
    parser.add_argument("--trinket-dir", default=r"C:\Users\dlw\Documents\Codex\2026-07-05\gen\outputs\data sp")
    parser.add_argument("--team-dir", default=r"C:\Users\dlw\Documents\Codex\2026-07-05\gen\outputs\data team")
    args = parser.parse_args()

    recognizer = ImageRecognizer.from_directories(args.card_dir, args.trinket_dir, args.team_dir)
    results = read_screen(
        screenshot_path=args.screenshot,
        profile_path=args.profile,
        recognizer=recognizer,
        top_k=args.top_k,
        output_dir=args.output_dir,
    )
    print(json.dumps(_to_jsonable(results), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
