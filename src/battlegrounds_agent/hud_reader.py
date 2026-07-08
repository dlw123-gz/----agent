from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .llm import OpenAICompatibleClient


@dataclass(frozen=True, slots=True)
class HudReading:
    tavern_tier: int | None = None
    level_cost: int | None = None
    gold: int | None = None
    health: int | None = None
    armor: int | None = None
    turn: int | None = None
    confidence: float | None = None
    notes: str = ""


HUD_REGIONS = {
    "top_shop_controls": (0.285, 0.000, 0.780, 0.220),
    "level_button": (0.290, 0.035, 0.425, 0.215),
    "tavern_badge": (0.400, 0.110, 0.470, 0.210),
    "tavern_tier": (0.400, 0.110, 0.470, 0.210),
    "level_cost": (0.290, 0.035, 0.425, 0.215),
    "timer": (0.895, 0.370, 0.990, 0.520),
    "hero_health": (0.455, 0.735, 0.650, 0.990),
    "gold": (0.600, 0.735, 0.860, 0.990),
}

TEXT_SUBREGIONS = {
    "gold_text": (0.43, 0.66, 0.60, 0.79),
    "health_text": (0.34, 0.25, 0.62, 0.43),
    "armor_text": (0.34, 0.00, 0.60, 0.22),
    "tavern_text": (0.30, 0.20, 0.76, 0.70),
    "level_text": (0.25, 0.04, 0.78, 0.46),
    "level_cost_text": (0.26, 0.05, 0.72, 0.45),
}


def crop_hud_regions(image_path: str | Path, output_dir: str | Path) -> dict[str, str]:
    image = Image.open(image_path).convert("RGB")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name, box in HUD_REGIONS.items():
        crop = image.crop(_resolve_box(box, image.size))
        path = output / f"{name}.png"
        crop.save(path)
        paths[name] = str(path)
    return paths


def read_hud_with_templates(image_path: str | Path, output_dir: str | Path | None = None) -> HudReading:
    output = Path(output_dir) if output_dir else Path.cwd() / "work" / "hud_template"
    crops = crop_hud_regions(image_path, output)
    notes: list[str] = []

    gold_text = _crop_named(crops["gold"], "gold_text")
    health_text = _crop_named(crops["hero_health"], "health_text")
    armor_text = _crop_named(crops["hero_health"], "armor_text")
    tavern_text = Image.open(crops["tavern_badge"]).convert("RGB")
    level_text = _crop_named(crops["level_button"], "level_cost_text")

    _save_debug_crop(gold_text, output / "gold_text.png")
    _save_debug_crop(health_text, output / "health_text.png")
    _save_debug_crop(armor_text, output / "armor_text.png")
    _save_debug_crop(tavern_text, output / "tavern_text.png")
    _save_debug_crop(level_text, output / "level_text.png")

    gold, gold_conf = _best_gold_match(gold_text)
    health, health_conf = _best_hud_number_match(health_text, range(1, 61))
    armor, armor_conf = _best_armor_match(armor_text)

    tavern = _count_tavern_badge_stars(tavern_text)
    tavern_conf = 0.78 if tavern is not None else 0.0
    # The top upgrade-cost number is small and stylized; expose it only when the
    # template score is strong enough to avoid overwriting state with a reroll cost.
    level_cost_text_value, level_conf = _best_text_match(level_text, [str(value) for value in range(0, 12)])
    level_cost = _int_or_none(level_cost_text_value) if level_conf >= 0.45 else None

    for label, value, conf in (
        ("gold", gold, gold_conf),
        ("health", health, health_conf),
        ("armor", armor, armor_conf),
        ("tavern_tier", tavern, tavern_conf),
        ("level_cost", level_cost, level_conf),
    ):
        if value is None or conf < 0.25:
            notes.append(f"{label}: uncertain local template match confidence={conf:.2f}")
    confidences = [gold_conf, health_conf, armor_conf]
    overall_confidence = sum(confidences) / len(confidences)
    # If the normal shop HUD is not visible, decorative stars elsewhere on the
    # board can look like the tavern badge. Do not let that override fallback
    # values when the rest of the HUD read clearly failed.
    if overall_confidence < 0.35:
        if tavern is not None:
            notes.append(f"tavern_tier: ignored because shop HUD confidence={overall_confidence:.2f}")
        tavern = None
    return HudReading(
        tavern_tier=tavern if tavern_conf >= 0.65 else None,
        level_cost=level_cost,
        gold=gold if gold_conf >= 0.55 else None,
        health=health if health_conf >= 0.55 else None,
        armor=armor if armor_conf >= 0.45 else None,
        confidence=overall_confidence,
        notes="; ".join(notes),
    )


def read_hud_with_vision(image_path: str | Path, client: OpenAICompatibleClient) -> HudReading:
    prompt = """
You are reading a Hearthstone Battlegrounds shop-phase screenshot.
Return only compact JSON with these keys:
tavern_tier, level_cost, gold, health, armor, turn, confidence, notes.
Use null when a value is not visible. Do not guess hidden values.
Focus on the player's HUD: tavern tier at top, upgrade cost near it, gold near bottom/right, hero health near bottom center.
""".strip()
    raw = client.complete_with_image(prompt, image_path)
    data = _parse_json_object(raw)
    return HudReading(
        tavern_tier=_int_or_none(data.get("tavern_tier")),
        level_cost=_int_or_none(data.get("level_cost")),
        gold=_int_or_none(data.get("gold")),
        health=_int_or_none(data.get("health")),
        armor=_int_or_none(data.get("armor")),
        turn=_int_or_none(data.get("turn")),
        confidence=_float_or_none(data.get("confidence")),
        notes=str(data.get("notes") or ""),
    )


def merge_hud_options(options, hud: HudReading):
    from .state_builder import BuildStateOptions

    return BuildStateOptions(
        tavern_tier=hud.tavern_tier if hud.tavern_tier is not None else options.tavern_tier,
        health=hud.health if hud.health is not None else options.health,
        armor=hud.armor if hud.armor is not None else options.armor,
        gold=hud.gold if hud.gold is not None else options.gold,
        turn=hud.turn if hud.turn is not None else options.turn,
        available_tribes=options.available_tribes,
        roll_cost=options.roll_cost,
        level_cost=hud.level_cost if hud.level_cost is not None else options.level_cost,
        min_score=options.min_score,
        low_confidence_score=getattr(options, "low_confidence_score", 0.68),
    )


def _resolve_box(box: tuple[float, float, float, float], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    left, top, right, bottom = box
    return (int(left * width), int(top * height), int(right * width), int(bottom * height))


def _crop_named(path: str | Path, name: str) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image.crop(_resolve_box(TEXT_SUBREGIONS[name], image.size))


def _save_debug_crop(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def _count_tavern_badge_stars(image: Image.Image) -> int | None:
    components = _yellow_star_components(image)
    stars = []
    height = image.height
    for points in components:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        width = max(xs) - min(xs) + 1
        comp_height = max(ys) - min(ys) + 1
        area = len(points)
        if 5 <= width <= 14 and 5 <= comp_height <= 14 and 14 <= area <= 80 and min(ys) <= height * 0.78:
            stars.append(points)
    if not stars:
        return None
    count = len(stars)
    return count if 1 <= count <= 7 else None


def _yellow_star_components(image: Image.Image) -> list[list[tuple[int, int]]]:
    image = image.convert("RGB")
    width, height = image.size
    pixels = image.load()
    yellow_pixels: set[tuple[int, int]] = set()
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            if r >= 145 and g >= 95 and b <= 130 and r >= g * 0.85:
                yellow_pixels.add((x, y))

    seen: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []
    for point in list(yellow_pixels):
        if point in seen:
            continue
        stack = [point]
        seen.add(point)
        component: list[tuple[int, int]] = []
        while stack:
            x, y = stack.pop()
            component.append((x, y))
            for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if neighbor in yellow_pixels and neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        if len(component) >= 3:
            components.append(component)
    return components


def _best_number_match(image: Image.Image, numbers) -> tuple[int | None, float]:
    text, confidence = _best_text_match(image, [str(value) for value in numbers])
    return (_int_or_none(text), confidence)


def _best_hud_number_match(image: Image.Image, numbers) -> tuple[int | None, float]:
    text, confidence = _best_hud_text_match(image, [str(value) for value in numbers])
    return (_int_or_none(text), confidence)


def _best_armor_match(image: Image.Image) -> tuple[int | None, float]:
    text, confidence = _best_hud_text_match(image, [str(value) for value in range(0, 31)])
    value = _int_or_none(text)
    if value == 10:
        return value, confidence

    # The armor shield background often dominates the mask. On the common 10 armor
    # badge the rendered candidates 15/18/19 score only slightly above 10, so prefer
    # 10 when it is visually plausible instead of returning a misleading value.
    ten_score = _score_hud_text(image, "10")
    if ten_score >= 0.42 and confidence - ten_score <= 0.09:
        return 10, max(ten_score, 0.50)
    if confidence >= 0.58:
        return value, confidence
    return None, confidence


def _best_gold_match(image: Image.Image) -> tuple[int | None, float]:
    cv2 = _cv2()
    if cv2 is None:
        return None, 0.0
    mask = _text_mask(image)
    scores: list[tuple[float, str]] = []
    for maximum in range(1, 11):
        for current in range(0, maximum + 1):
            text = f"{current}/{maximum}"
            scores.append((_score_hud_text_mask(mask, text), text))
    ranked = sorted(scores, reverse=True)[:5]
    if not ranked:
        return None, 0.0
    votes: dict[int, int] = {}
    for _, text in ranked:
        current = _int_or_none(text.split("/", 1)[0])
        if current is not None:
            votes[current] = votes.get(current, 0) + 1
    if not votes:
        return None, 0.0
    current = max(votes.items(), key=lambda item: (item[1], item[0]))[0]
    agreement = votes[current] / len(ranked)
    return current, ranked[0][0] * agreement


def _read_component_text(image: Image.Image, allow_slash: bool = False) -> tuple[str | None, float]:
    cv2 = _cv2()
    np = _np()
    if cv2 is None:
        return None, 0.0
    mask = _text_mask(image)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    components = []
    height, width = mask.shape
    for index in range(1, n):
        x, y, w, h, area = [int(value) for value in stats[index]]
        if area < 12 or h < max(7, height * 0.30) or w < 2:
            continue
        if y > height * 0.70:
            continue
        if w > width * 0.55 or h > height * 0.90:
            continue
        components.append((x, y, w, h, area))
    components = _remove_contained_components(sorted(components, key=lambda item: item[0]))
    components = _keep_primary_text_line(components)
    armor_special = _try_read_armor_ten(components)
    if armor_special is not None and not allow_slash:
        return armor_special, 0.9
    if not components:
        return None, 0.0

    chars: list[str] = []
    scores: list[float] = []
    for x, y, w, h, _ in components:
        component = mask[max(0, y - 1) : min(height, y + h + 1), max(0, x - 1) : min(width, x + w + 1)]
        if allow_slash and _looks_like_slash(component):
            chars.append("/")
            scores.append(0.9)
            continue
        digit, score = _classify_digit(component)
        chars.append(digit)
        scores.append(score)
    text = "".join(chars)
    if allow_slash and "/" not in text and len(text) >= 2:
        text = text[0] + "/" + text[1:]
    text = re.sub(r"[^0-9/]", "", text)
    if not text:
        return None, 0.0
    return text, sum(scores) / len(scores)


def _try_read_armor_ten(components: list[tuple[int, int, int, int, int]]) -> str | None:
    if len(components) < 2:
        return None
    ordered = sorted(components, key=lambda item: item[0])
    left = ordered[0]
    right = ordered[1]
    _, _, lw, lh, _ = left
    _, _, rw, rh, _ = right
    if lw <= max(8, lh * 0.55) and rw >= lw * 1.2 and rh >= lh * 0.75:
        return "10"
    return None


def _remove_contained_components(components: list[tuple[int, int, int, int, int]]) -> list[tuple[int, int, int, int, int]]:
    result = []
    for comp in components:
        x, y, w, h, _ = comp
        contained = False
        for other in components:
            if comp is other:
                continue
            ox, oy, ow, oh, _ = other
            if x >= ox and y >= oy and x + w <= ox + ow and y + h <= oy + oh and (w * h) < (ow * oh):
                contained = True
                break
        if not contained:
            result.append(comp)
    return result


def _keep_primary_text_line(components: list[tuple[int, int, int, int, int]]) -> list[tuple[int, int, int, int, int]]:
    if len(components) <= 2:
        return components
    best = []
    best_score = -1
    for comp in components:
        _, y, _, h, _ = comp
        center = y + h / 2
        line = [item for item in components if abs((item[1] + item[3] / 2) - center) <= max(5, h * 0.55)]
        score = sum(item[4] for item in line)
        if score > best_score:
            best = line
            best_score = score
    return sorted(best, key=lambda item: item[0])


def _looks_like_slash(component) -> bool:
    h, w = component.shape
    if w > max(5, h * 0.55):
        return False
    ys, xs = _np().where(component > 0)
    if len(xs) < 4:
        return False
    if xs.max() == xs.min():
        return True
    slope = (ys.max() - ys.min()) / max(1, xs.max() - xs.min())
    return slope > 1.2


def _classify_digit(component) -> tuple[str, float]:
    cv2 = _cv2()
    np = _np()
    normalized = cv2.resize(component, (24, 32), interpolation=cv2.INTER_AREA)
    _, normalized = cv2.threshold(normalized, 20, 255, cv2.THRESH_BINARY)
    # Shape rules avoid common template mistakes on small Hearthstone HUD digits.
    h, w = component.shape
    if w <= h * 0.45:
        return "1", 0.85
    best_digit = "0"
    best_score = -1.0
    for digit in "0123456789":
        template = _render_digit_template(digit)
        overlap = np.logical_and(normalized > 0, template > 0).sum()
        union = np.logical_or(normalized > 0, template > 0).sum()
        iou = overlap / max(1, union)
        corr = cv2.matchTemplate(normalized, template, cv2.TM_CCOEFF_NORMED)[0][0]
        score = (float(iou) * 0.65) + (float(corr) * 0.35)
        if score > best_score:
            best_digit = digit
            best_score = score
    return best_digit, max(0.0, best_score)


def _render_digit_template(digit: str):
    cv2 = _cv2()
    np = _np()
    canvas = Image.new("L", (24, 32), 0)
    font = _font(27)
    bbox = _text_bbox(digit, font)
    draw = ImageDraw.Draw(canvas)
    x = (24 - (bbox[2] - bbox[0])) // 2 - bbox[0]
    y = (32 - (bbox[3] - bbox[1])) // 2 - bbox[1]
    draw.text((x, y), digit, font=font, fill=255, stroke_width=1, stroke_fill=255)
    arr = np.array(canvas)
    _, arr = cv2.threshold(arr, 20, 255, cv2.THRESH_BINARY)
    return arr


def _best_text_match(image: Image.Image, candidates: list[str]) -> tuple[str | None, float]:
    cv2 = _cv2()
    if cv2 is None:
        return None, 0.0
    mask = _text_mask(image)
    best_text = None
    best_score = -1.0
    for text in candidates:
        for template in _render_templates(text, mask.shape):
            result = cv2.matchTemplate(mask, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, _ = cv2.minMaxLoc(result)
            if score > best_score:
                best_text = text
            best_score = float(score)
    return best_text, max(0.0, best_score)


def _best_hud_text_match(image: Image.Image, candidates: list[str]) -> tuple[str | None, float]:
    cv2 = _cv2()
    if cv2 is None:
        return None, 0.0
    mask = _text_mask(image)
    best_text = None
    best_score = -1.0
    for text in candidates:
        score = _score_hud_text_mask(mask, text)
        if score > best_score:
            best_text = text
            best_score = score
    return best_text, max(0.0, best_score)


def _score_hud_text(image: Image.Image, text: str) -> float:
    cv2 = _cv2()
    if cv2 is None:
        return 0.0
    return _score_hud_text_mask(_text_mask(image), text)


def _score_hud_text_mask(mask, text: str) -> float:
    cv2 = _cv2()
    best_score = -1.0
    for template in _render_aligned_templates(text, mask.shape):
        result = cv2.matchTemplate(mask, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(result)
        best_score = max(best_score, float(score))
    return max(0.0, best_score)


def _text_mask(image: Image.Image):
    cv2 = _cv2()
    np = _np()
    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    bright = cv2.inRange(gray, 150, 255)
    kernel = np.ones((2, 2), np.uint8)
    return cv2.dilate(bright, kernel, iterations=1)


def _render_templates(text: str, target_shape) -> list:
    cv2 = _cv2()
    np = _np()
    height, width = target_shape
    templates = []
    for font_size in range(12, 42, 2):
        font = _font(font_size)
        bbox = _text_bbox(text, font)
        text_width = max(1, bbox[2] - bbox[0])
        text_height = max(1, bbox[3] - bbox[1])
        canvas_width = min(width, max(text_width + 10, 8))
        canvas_height = min(height, max(text_height + 10, 8))
        if canvas_width > width or canvas_height > height:
            continue
        canvas = Image.new("L", (canvas_width, canvas_height), 0)
        draw = ImageDraw.Draw(canvas)
        draw.text((5 - bbox[0], 5 - bbox[1]), text, font=font, fill=255, stroke_width=1, stroke_fill=255)
        arr = np.array(canvas)
        _, arr = cv2.threshold(arr, 20, 255, cv2.THRESH_BINARY)
        if arr.shape[0] <= height and arr.shape[1] <= width:
            templates.append(arr)
    return templates


def _render_aligned_templates(text: str, target_shape) -> list:
    cv2 = _cv2()
    np = _np()
    height, width = target_shape
    templates = []
    for font_path in _hud_font_paths():
        for font_size in range(14, 36, 3):
            if not Path(font_path).exists():
                continue
            font = ImageFont.truetype(font_path, size=font_size)
            bbox = _text_bbox(text, font)
            text_width = max(1, bbox[2] - bbox[0])
            text_height = max(1, bbox[3] - bbox[1])
            if text_width > width * 1.15 or text_height > height * 1.15:
                continue
            canvas = Image.new("L", (width, height), 0)
            draw = ImageDraw.Draw(canvas)
            x = (width - text_width) // 2 - bbox[0]
            y = (height - text_height) // 2 - bbox[1]
            draw.text((x, y), text, font=font, fill=255, stroke_width=1, stroke_fill=255)
            arr = np.array(canvas)
            _, arr = cv2.threshold(arr, 20, 255, cv2.THRESH_BINARY)
            templates.append(arr)
    return templates


def _hud_font_paths() -> tuple[str, ...]:
    return (
        r"C:\Windows\Fonts\calibrib.ttf",
        r"C:\Windows\Fonts\arialbd.ttf",
    )


def _font_paths() -> tuple[str, ...]:
    return (
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\cambriab.ttf",
        r"C:\Windows\Fonts\georgiab.ttf",
    )


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _font_paths():
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _text_bbox(text: str, font) -> tuple[int, int, int, int]:
    image = Image.new("L", (200, 80), 0)
    draw = ImageDraw.Draw(image)
    return draw.textbbox((0, 0), text, font=font, stroke_width=1)


def _cv2():
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    return cv2


def _np():
    import numpy as np  # type: ignore

    return np


def _parse_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise ValueError(f"Vision response did not contain JSON: {text}") from None
        return json.loads(match.group(0))


def _int_or_none(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Crop and optionally read Battlegrounds HUD values.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="work/hud")
    parser.add_argument("--use-vision", action="store_true")
    parser.add_argument("--use-template", action="store_true")
    parser.add_argument("--env-file")
    args = parser.parse_args()

    crops = crop_hud_regions(args.image, args.output_dir)
    payload = {"image": args.image, "crops": crops, "reading": None}
    if args.use_template:
        payload["reading"] = asdict(read_hud_with_templates(args.image, args.output_dir))
    elif args.use_vision:
        client = OpenAICompatibleClient.from_env(args.env_file)
        payload["reading"] = asdict(read_hud_with_vision(args.image, client))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
