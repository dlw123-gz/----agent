from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image


SUPPORTED_EXTENSIONS = {".png", ".webp", ".jpg", ".jpeg"}


def build_card_index(card_dir: str | Path) -> list[dict]:
    rows = []
    for path in _image_files(card_dir):
        rows.append(
            {
                "id": _image_id(path),
                "name": "",
                "tier": None,
                "attack": None,
                "health": None,
                "tribes": [],
                "text": "",
                "tags": [],
                "image": str(path),
                "source_filename": path.name,
                "needs_metadata": True,
            }
        )
    return rows


def build_trinket_index(trinket_dir: str | Path) -> list[dict]:
    rows = []
    for path in _image_files(trinket_dir):
        rows.append(
            {
                "id": _image_id(path),
                "name": "",
                "text": "",
                "tier": None,
                "tribes": [],
                "tags": [],
                "image": str(path),
                "source_filename": path.name,
                "needs_metadata": True,
            }
        )
    return rows


def build_composition_image_index(team_dir: str | Path) -> list[dict]:
    rows = []
    for path in _image_files(team_dir):
        width, height = Image.open(path).size
        rows.append(
            {
                "id": path.stem,
                "title": "",
                "season": "",
                "image": str(path),
                "source_filename": path.name,
                "width": width,
                "height": height,
                "rows": [],
                "needs_vision_extraction": True,
            }
        )
    return rows


def write_json(path: str | Path, payload: object) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _image_files(directory: str | Path) -> list[Path]:
    root = Path(directory)
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def _image_id(path: Path) -> str:
    return path.stem.removesuffix("_battlegroundsImage")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Build JSON indexes from local Battlegrounds image assets.")
    parser.add_argument("--card-dir", default=r"C:\Users\dlw\Documents\Codex\2026-07-05\gen\outputs\data card")
    parser.add_argument("--trinket-dir", default=r"C:\Users\dlw\Documents\Codex\2026-07-05\gen\outputs\data sp")
    parser.add_argument("--team-dir", default=r"C:\Users\dlw\Documents\Codex\2026-07-05\gen\outputs\data team")
    parser.add_argument("--output-dir", default="examples/data")
    args = parser.parse_args()

    output = Path(args.output_dir)
    cards = build_card_index(args.card_dir)
    trinkets = build_trinket_index(args.trinket_dir)
    compositions = build_composition_image_index(args.team_dir)
    write_json(output / "cards.generated.json", cards)
    write_json(output / "trinkets.generated.json", trinkets)
    write_json(output / "compositions.generated.json", compositions)
    print(
        json.dumps(
            {
                "cards": len(cards),
                "trinkets": len(trinkets),
                "composition_images": len(compositions),
                "output_dir": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
