from __future__ import annotations

from dataclasses import dataclass, asdict

from PIL import Image

from .hud_reader import _int_or_none, _read_component_text


@dataclass(frozen=True, slots=True)
class CardSlotFeatures:
    tier: int | None = None
    attack: int | None = None
    health: int | None = None
    slot_type: str = "minion"
    tier_confidence: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> dict[str, int | float | str | None]:
        return asdict(self)


def analyze_card_slot(image: Image.Image) -> CardSlotFeatures:
    image = image.convert("RGB")
    tier, tier_conf = _read_tier(image)
    slot_type = _detect_slot_type(image)
    attack, attack_conf = _read_best_stat(
        image,
        (
            (0.00, 0.42, 0.45, 0.78),
            (0.00, 0.50, 0.45, 0.86),
            (0.00, 0.58, 0.42, 0.92),
            (0.00, 0.68, 0.42, 1.00),
        ),
    )
    health, health_conf = _read_best_stat(
        image,
        (
            (0.55, 0.42, 1.00, 0.78),
            (0.55, 0.50, 1.00, 0.86),
            (0.58, 0.58, 1.00, 0.92),
            (0.58, 0.68, 1.00, 1.00),
        ),
    )
    confidence = (tier_conf + attack_conf + health_conf) / 3.0
    return CardSlotFeatures(
        tier=tier if tier_conf >= 0.40 else None,
        attack=attack if attack_conf >= 0.42 else None,
        health=health if health_conf >= 0.42 else None,
        slot_type=slot_type,
        tier_confidence=tier_conf,
        confidence=confidence,
    )


def _read_tier(image: Image.Image) -> tuple[int | None, float]:
    for box in (
        (0.25, 0.00, 0.62, 0.24),
        (0.18, 0.10, 0.82, 0.42),
        (0.24, 0.16, 0.76, 0.48),
    ):
        star_count = _count_card_badge_stars(_crop_ratio(image, box))
        if star_count is not None:
            return star_count, 0.72
    # The purple tavern-star badge is small and decorative. Fall back to the
    # older component reader only when color-based star counting fails.
    return _read_stat(image, (0.04, 0.00, 0.56, 0.32))


def _read_stat(image: Image.Image, box: tuple[float, float, float, float]) -> tuple[int | None, float]:
    crop = _crop_ratio(image, box)
    text, confidence = _read_component_text(crop)
    return _int_or_none(text), confidence


def _read_best_stat(
    image: Image.Image,
    boxes: tuple[tuple[float, float, float, float], ...],
) -> tuple[int | None, float]:
    readings = [_read_stat(image, box) for box in boxes]
    valid = [(value, confidence) for value, confidence in readings if value is not None]
    if not valid:
        return None, max((confidence for _, confidence in readings), default=0.0)
    # Prefer compact one- or two-digit readings; huge values usually come from
    # glowing artwork or overlapping effects, not minion stats.
    plausible = [(value, confidence) for value, confidence in valid if 0 <= value <= 200]
    if plausible:
        return max(plausible, key=lambda item: item[1])
    return max(valid, key=lambda item: item[1])


def _detect_slot_type(image: Image.Image) -> str:
    lower = _crop_ratio(image, (0.00, 0.42, 1.00, 0.82))
    pixels = list(lower.getdata())
    total = max(len(pixels), 1)
    white = sum(1 for r, g, b in pixels if r > 185 and g > 185 and b > 185) / total
    green = sum(1 for r, g, b in pixels if g > 140 and r < 140 and b < 120) / total
    red = sum(1 for r, g, b in pixels if r > 150 and g < 110 and b < 110) / total
    orange = sum(1 for r, g, b in pixels if r > 170 and 70 < g < 170 and b < 90) / total
    stat_signal = white + green
    spell_signal = red + orange
    if stat_signal < 0.02 and spell_signal > 0.15:
        return "tavern_spell"
    return "minion"


def _crop_ratio(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    width, height = image.size
    left, top, right, bottom = box
    return image.crop((int(left * width), int(top * height), int(right * width), int(bottom * height)))


def _count_card_badge_stars(image: Image.Image) -> int | None:
    components = _yellow_star_components(image)
    stars = []
    for points in components:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        width = max(xs) - min(xs) + 1
        height = max(ys) - min(ys) + 1
        area = len(points)
        if 4 <= width <= 14 and 4 <= height <= 14 and 8 <= area <= 80:
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
            if r >= 145 and g >= 95 and b <= 135 and r >= g * 0.85:
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
