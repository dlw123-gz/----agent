from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True, slots=True)
class GameArea:
    left: int
    top: int
    right: int
    bottom: int
    score: float

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def detect_game_area(image_path: str | Path, min_area_ratio: float = 0.08) -> GameArea:
    image = Image.open(image_path).convert("RGB")
    return detect_game_area_image(image, min_area_ratio=min_area_ratio)


def detect_game_area_image(image: Image.Image, min_area_ratio: float = 0.08) -> GameArea:
    original_width, original_height = image.size
    scale = min(1.0, 420 / max(original_width, original_height))
    small = image.resize((int(original_width * scale), int(original_height * scale)), Image.Resampling.BILINEAR)
    mask = _content_mask(small)
    candidates = _components(mask, small.size)
    if not candidates:
        return GameArea(0, 0, original_width, original_height, 0.0)

    min_pixels = int(small.width * small.height * min_area_ratio)
    scored: list[tuple[float, tuple[int, int, int, int]]] = []
    for box, pixels in candidates:
        left, top, right, bottom = box
        width = right - left
        height = bottom - top
        if pixels < min_pixels or width < small.width * 0.25 or height < small.height * 0.25:
            continue
        aspect = width / max(height, 1)
        aspect_score = max(0.0, 1.0 - abs(aspect - 16 / 9) / 1.2)
        fill = pixels / max(width * height, 1)
        area_score = (width * height) / (small.width * small.height)
        score = area_score * 0.65 + aspect_score * 0.25 + min(fill, 1.0) * 0.10
        scored.append((score, box))

    if not scored:
        return GameArea(0, 0, original_width, original_height, 0.0)

    score, box = max(scored, key=lambda item: item[0])
    left, top, right, bottom = box
    inv = 1 / scale
    return GameArea(
        left=max(0, int(left * inv)),
        top=max(0, int(top * inv)),
        right=min(original_width, int(right * inv)),
        bottom=min(original_height, int(bottom * inv)),
        score=round(score, 4),
    )


def crop_game_area(image_path: str | Path, output_path: str | Path, min_area_ratio: float = 0.08) -> GameArea:
    image = Image.open(image_path).convert("RGB")
    area = detect_game_area_image(image, min_area_ratio=min_area_ratio)
    crop = image.crop((area.left, area.top, area.right, area.bottom))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output)
    return area


def _content_mask(image: Image.Image) -> list[bool]:
    mask: list[bool] = []
    for r, g, b in image.getdata():
        high = max(r, g, b)
        low = min(r, g, b)
        brightness = (r + g + b) / 3
        saturation = (high - low) / max(high, 1)
        mask.append((brightness < 235 and saturation > 0.07) or brightness < 95)
    return mask


def _components(mask: list[bool], size: tuple[int, int]) -> list[tuple[tuple[int, int, int, int], int]]:
    width, height = size
    seen = bytearray(width * height)
    components: list[tuple[tuple[int, int, int, int], int]] = []
    for start, active in enumerate(mask):
        if not active or seen[start]:
            continue
        stack = [start]
        seen[start] = 1
        left = right = start % width
        top = bottom = start // width
        pixels = 0
        while stack:
            index = stack.pop()
            pixels += 1
            x = index % width
            y = index // width
            left = min(left, x)
            right = max(right, x + 1)
            top = min(top, y)
            bottom = max(bottom, y + 1)
            for neighbor in _neighbors(index, x, y, width, height):
                if mask[neighbor] and not seen[neighbor]:
                    seen[neighbor] = 1
                    stack.append(neighbor)
        components.append(((left, top, right, bottom), pixels))
    return components


def _neighbors(index: int, x: int, y: int, width: int, height: int):
    if x > 0:
        yield index - 1
    if x + 1 < width:
        yield index + 1
    if y > 0:
        yield index - width
    if y + 1 < height:
        yield index + width


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Detect and crop the likely Hearthstone game area inside a screenshot.")
    parser.add_argument("--image", required=True, help="Window or full-screen screenshot.")
    parser.add_argument("--output", help="Optional crop output path.")
    parser.add_argument("--min-area-ratio", type=float, default=0.08)
    args = parser.parse_args()

    if args.output:
        area = crop_game_area(args.image, args.output, min_area_ratio=args.min_area_ratio)
    else:
        area = detect_game_area(args.image, min_area_ratio=args.min_area_ratio)
    print(json.dumps(asdict(area), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
