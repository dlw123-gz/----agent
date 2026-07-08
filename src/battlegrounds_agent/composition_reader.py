from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from .image_recognizer import ImageRecognizer
from .llm import OpenAICompatibleClient


SUPPORTED_EXTENSIONS = {".png", ".webp", ".jpg", ".jpeg"}
ROW_BOXES = (
    (0.040, 0.225, 0.960, 0.365),
    (0.040, 0.385, 0.960, 0.525),
    (0.040, 0.545, 0.960, 0.685),
    (0.040, 0.705, 0.960, 0.845),
    (0.040, 0.865, 0.960, 0.985),
)
TEXT_BOX = (0.000, 0.000, 0.285, 1.000)
CARDS_BOX = (0.285, 0.000, 1.000, 1.000)


def read_composition_image(
    image_path: str | Path,
    recognizer: ImageRecognizer,
    output_dir: str | Path,
    top_k: int = 3,
) -> dict:
    image = Image.open(image_path).convert("RGB")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, row_box in enumerate(ROW_BOXES, start=1):
        row_image = image.crop(_resolve_box(row_box, image.size))
        row_dir = output / f"row_{index}"
        row_dir.mkdir(parents=True, exist_ok=True)
        row_path = row_dir / "row.png"
        text_path = row_dir / "text.png"
        row_image.save(row_path)
        row_image.crop(_resolve_box(TEXT_BOX, row_image.size)).save(text_path)
        card_results = _recognize_row_cards(row_image, recognizer, row_dir, top_k)
        rows.append(
            {
                "row": index,
                "title": "",
                "description_lines": [],
                "tribes": [],
                "text_crop": str(text_path),
                "row_crop": str(row_path),
                "cards": card_results,
                "needs_text_ocr": True,
            }
        )
    return {
        "id": Path(image_path).stem,
        "image": str(image_path),
        "rows": rows,
    }


def extract_composition_text_with_vision(image_path: str | Path, client: OpenAICompatibleClient) -> dict:
    prompt = """
Read this Hearthstone Battlegrounds composition guide image.
The image contains several composition rows. Each row has a Chinese composition title, short play instructions, tribe label, and core card pictures.
Return only JSON:
{
  "title": string,
  "season": string | null,
  "rows": [
    {
      "row": number,
      "title": string,
      "description_lines": [string],
      "tribes": [string],
      "core_cards": [string],
      "key_cards": [string],
      "tags": [string],
      "playstyle": string,
      "recommended_play_lines": [string]
    }
  ]
}
Preserve each short Chinese guide line separately.
Use Chinese card names when visible; do not invent card ids.
Tags should be concise mechanics such as summon, deathrattle, battlecry, buff, scaling, economy, tavern_spell, divine_shield, taunt, venomous.
""".strip()
    raw = client.complete_with_image(prompt, image_path)
    return _parse_json(raw)


def merge_vision_text(payload: dict, vision: dict) -> dict:
    payload["title"] = vision.get("title", payload.get("title", ""))
    payload["season"] = vision.get("season", payload.get("season", ""))
    by_row = {int(row.get("row", 0)): row for row in vision.get("rows", []) if row.get("row") is not None}
    for row in payload["rows"]:
        text_row = by_row.get(row["row"])
        if not text_row:
            continue
        row["title"] = text_row.get("title", row["title"])
        row["description_lines"] = text_row.get("description_lines", row["description_lines"])
        row["tribes"] = text_row.get("tribes", row["tribes"])
        row["core_cards"] = text_row.get("core_cards", [])
        row["key_cards"] = text_row.get("key_cards", [])
        row["tags"] = text_row.get("tags", [])
        row["playstyle"] = text_row.get("playstyle", "")
        row["recommended_play_lines"] = text_row.get("recommended_play_lines", [])
        row["needs_text_ocr"] = False
    return payload


def read_composition_directory(
    input_dir: str | Path,
    recognizer: ImageRecognizer,
    output_dir: str | Path,
    top_k: int = 3,
    client: OpenAICompatibleClient | None = None,
) -> list[dict]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payloads: list[dict] = []
    for image_path in _image_files(input_dir):
        item_output = output / image_path.stem
        payload = read_composition_image(image_path, recognizer, item_output, top_k=top_k)
        if client is not None:
            payload = merge_vision_text(payload, extract_composition_text_with_vision(image_path, client))
        payloads.append(payload)
    return payloads


def _recognize_row_cards(row_image: Image.Image, recognizer: ImageRecognizer, output_dir: Path, top_k: int) -> list[dict]:
    cards_area = row_image.crop(_resolve_box(CARDS_BOX, row_image.size))
    width, height = cards_area.size
    slot_count = 6
    results = []
    for index in range(slot_count):
        left = int(width * index / slot_count)
        right = int(width * (index + 1) / slot_count)
        crop = cards_area.crop((left, 0, right, height))
        crop_path = output_dir / f"card_{index + 1}.png"
        crop.save(crop_path)
        matches = recognizer.recognize_image(crop, kind="card", top_k=top_k)
        results.append(
            {
                "slot": index + 1,
                "crop": str(crop_path),
                "matches": [asdict(match) for match in matches],
            }
        )
    return results


def _resolve_box(box: tuple[float, float, float, float], image_size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = image_size
    left, top, right, bottom = box
    return (int(left * width), int(top * height), int(right * width), int(bottom * height))


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise
        return json.loads(text[start : end + 1])


def _image_files(directory: str | Path) -> list[Path]:
    root = Path(directory)
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Extract composition guide rows and card matches from team images.")
    parser.add_argument("--image")
    parser.add_argument("--input-dir")
    parser.add_argument("--output-dir", default="work/composition_reader")
    parser.add_argument("--output-json")
    parser.add_argument("--card-dir", default=r"C:\Users\dlw\Documents\Codex\2026-07-05\gen\outputs\data card")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--use-vision-text", action="store_true")
    parser.add_argument("--env-file")
    args = parser.parse_args()

    recognizer = ImageRecognizer.from_directories(card_dir=args.card_dir)
    client = OpenAICompatibleClient.from_env(args.env_file) if args.use_vision_text else None
    if args.input_dir:
        payloads = read_composition_directory(args.input_dir, recognizer, args.output_dir, top_k=args.top_k, client=client)
        output_path = Path(args.output_json) if args.output_json else Path(args.output_dir) / "compositions.vision.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payloads, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(output_path), "images": len(payloads), "rows": sum(len(item["rows"]) for item in payloads)}, ensure_ascii=False, indent=2))
        return

    if not args.image:
        raise SystemExit("Either --image or --input-dir is required.")
    payload = read_composition_image(args.image, recognizer, args.output_dir, top_k=args.top_k)
    if client is not None:
        payload = merge_vision_text(payload, extract_composition_text_with_vision(args.image, client))
    output_path = Path(args.output_json) if args.output_json else Path(args.output_dir) / "composition.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "rows": len(payload["rows"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
