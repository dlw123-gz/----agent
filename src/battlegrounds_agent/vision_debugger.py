from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .crop_profiles import resolve_profile
from .database import CardDatabase
from .game_area_detector import crop_game_area
from .hud_reader import crop_hud_regions, read_hud_with_vision
from .image_recognizer import ImageRecognizer
from .llm import OpenAICompatibleClient
from .screen_reader import read_screen
from .state_builder import BuildStateOptions, build_state_from_slots
from .state_summary import summarize_recognition, summarize_state
from .window_capture import capture_screen, capture_window


def main() -> None:
    _configure_stdout()
    parser = argparse.ArgumentParser(description="Debug game-area detection, slot cropping, and image matching.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--screenshot", help="Use an existing screenshot.")
    source.add_argument("--window-title", help="Capture a visible window by title substring.")
    source.add_argument("--screen", action="store_true", help="Capture the full screen.")
    parser.add_argument("--profile", help="Crop profile JSON.")
    parser.add_argument("--phase", choices=["shop", "shop-buy", "shop-buy-16x9", "shop-buy-right", "trinket-fullscreen", "trinket-video", "triple-discover"], help="Use a built-in crop profile for a known phase.")
    parser.add_argument("--detect-game-area", action="store_true", help="Crop likely game area before slot reading.")
    parser.add_argument("--output-dir", default="work/vision_debug", help="Directory for crops and report.json.")
    parser.add_argument("--card-dir", default=r"C:\Users\dlw\Documents\Codex\2026-07-05\gen\outputs\data card")
    parser.add_argument("--trinket-dir", default=r"C:\Users\dlw\Documents\Codex\2026-07-05\gen\outputs\data sp")
    parser.add_argument("--team-dir", default=r"C:\Users\dlw\Documents\Codex\2026-07-05\gen\outputs\data team")
    parser.add_argument("--card-json", default="examples/data/cards.enriched.json")
    parser.add_argument("--trinket-json", default="examples/data/trinkets.json")
    parser.add_argument("--min-score", type=float, default=0.58)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--use-vision-hud", action="store_true", help="Use the configured vision model to read HUD values.")
    parser.add_argument("--env-file")
    args = parser.parse_args()
    profile = resolve_profile(args.profile, args.phase)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.screenshot:
        screenshot = Path(args.screenshot)
    elif args.screen:
        screenshot = capture_screen(output_dir / "screen_capture.png")
    else:
        screenshot = capture_window(args.window_title, output_dir / "window_capture.png")
    input_image = screenshot
    game_area = None
    if args.detect_game_area:
        input_image = output_dir / "detected_game_area.png"
        game_area = crop_game_area(screenshot, input_image)
    hud_crops = crop_hud_regions(input_image, output_dir / "hud")
    hud_reading = None
    if args.use_vision_hud:
        client = OpenAICompatibleClient.from_env(args.env_file)
        hud_reading = read_hud_with_vision(input_image, client)

    recognizer = ImageRecognizer.from_directories(args.card_dir, args.trinket_dir, args.team_dir)
    slots = read_screen(
        screenshot_path=input_image,
        profile_path=profile,
        recognizer=recognizer,
        top_k=args.top_k,
        output_dir=output_dir / "slots",
    )
    database = CardDatabase.load(args.card_json, args.trinket_json)
    state = build_state_from_slots(slots, database, BuildStateOptions(min_score=args.min_score))
    report = {
        "source_screenshot": str(screenshot),
        "input_image": str(input_image),
        "profile": str(profile),
        "game_area": asdict(game_area) if game_area else None,
        "hud": {
            "crops": hud_crops,
            "reading": asdict(hud_reading) if hud_reading else None,
        },
        "slot_count": len(slots),
        "slots": _slots_report(slots),
        "recognition_summary": summarize_recognition(slots, min_score=args.min_score),
        "state_summary": summarize_state(state),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "input_image": str(input_image)}, ensure_ascii=False, indent=2))


def _slots_report(slots) -> list[dict]:
    rows = []
    for slot in slots:
        rows.append(
            {
                "slot_id": slot.slot_id,
                "kind": slot.kind,
                "box_pixels": slot.box_pixels,
                "crop_path": slot.crop_path,
                "matches": [asdict(match) for match in slot.matches],
            }
        )
    return rows


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


if __name__ == "__main__":
    main()
