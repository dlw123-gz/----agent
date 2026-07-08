from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

from PIL import Image, ImageDraw

from .crop_profiles import resolve_profile
from .database import CardDatabase
from .game_area_detector import crop_game_area
from .hud_reader import crop_hud_regions, merge_hud_options, read_hud_with_templates, read_hud_with_vision
from .image_recognizer import ImageRecognizer
from .llm import OpenAICompatibleClient
from .planner import BattlegroundsAgent
from .screen_reader import read_screen
from .schemas import ActionPlan, ActionType, CandidateAction
from .state_builder import BuildStateOptions, build_state_from_slots
from .vision_shop_reader import read_shop_with_vision
from .window_capture import capture_screen, capture_window


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture/read a Battlegrounds screen and print an action plan.")
    parser.add_argument("--profile", help="Crop profile JSON.")
    parser.add_argument("--phase", choices=["shop", "shop-buy", "shop-buy-16x9", "shop-buy-right", "trinket-fullscreen", "trinket-video", "triple-discover"], help="Use a built-in crop profile for a known phase.")
    parser.add_argument("--screenshot", help="Use an existing screenshot instead of capturing a window.")
    parser.add_argument("--screen", action="store_true", help="Capture the full screen instead of a window.")
    parser.add_argument("--window-title", default="炉石传说", help="Substring of the game window title.")
    parser.add_argument("--detect-game-area", action="store_true", help="Crop the likely game area inside the window/screenshot before reading slots.")
    parser.add_argument("--watch", action="store_true", help="Keep capturing and planning.")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between watch iterations.")
    parser.add_argument("--work-dir", default="work/live", help="Directory for captured screenshots and crops.")
    parser.add_argument("--card-json", default="examples/data/cards.enriched.json")
    parser.add_argument("--trinket-json", default="examples/data/trinkets.json")
    parser.add_argument("--card-dir", default=r"C:\Users\dlw\Documents\Codex\2026-07-05\gen\outputs\data card")
    parser.add_argument("--trinket-dir", default=r"C:\Users\dlw\Documents\Codex\2026-07-05\gen\outputs\data sp")
    parser.add_argument("--team-dir", default=r"C:\Users\dlw\Documents\Codex\2026-07-05\gen\outputs\data team")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--min-score", type=float, default=0.58)
    parser.add_argument("--tavern-tier", type=int, default=1)
    parser.add_argument("--health", type=int, default=40)
    parser.add_argument("--armor", type=int, default=0)
    parser.add_argument("--gold", type=int, default=3)
    parser.add_argument("--turn", type=int, default=1)
    parser.add_argument("--roll-cost", type=int, default=1)
    parser.add_argument("--level-cost", type=int)
    parser.add_argument("--available-tribes", default="", help="Comma-separated tribe names.")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--use-template-hud", action="store_true", help="Use local template matching to read health, armor, and gold.")
    parser.add_argument("--use-vision-hud", action="store_true", help="Use the configured vision model to read health, gold, tavern tier, and level cost.")
    parser.add_argument("--use-vision-shop", action="store_true", help="Use the configured vision model to read shop slots and override local image matching.")
    parser.add_argument("--env-file", help="Optional .env file for LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL.")
    args = parser.parse_args()
    profile = resolve_profile(args.profile, args.phase)

    database = CardDatabase.load(args.card_json, args.trinket_json)
    recognizer = ImageRecognizer.from_directories(args.card_dir, args.trinket_dir, args.team_dir)
    llm = OpenAICompatibleClient.from_env(args.env_file) if args.use_llm or args.use_vision_hud or args.use_vision_shop else None
    agent = BattlegroundsAgent(llm_client=llm)
    options = BuildStateOptions(
        tavern_tier=args.tavern_tier,
        health=args.health,
        armor=args.armor,
        gold=args.gold,
        turn=args.turn,
        available_tribes=tuple(_split_csv(args.available_tribes)),
        roll_cost=args.roll_cost,
        level_cost=args.level_cost,
        min_score=args.min_score,
    )

    while True:
        payload = _run_once(args, profile, database, recognizer, agent, options, llm)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        if not args.watch:
            break
        time.sleep(args.interval)


def _run_once(args, profile: Path, database: CardDatabase, recognizer: ImageRecognizer, agent: BattlegroundsAgent, options: BuildStateOptions, llm) -> dict:
    work_dir = Path(args.work_dir)
    if args.screenshot:
        screenshot = Path(args.screenshot)
    elif args.screen:
        screenshot = capture_screen(work_dir / "current_screenshot.png")
    else:
        screenshot = capture_window(args.window_title, work_dir / "current_screenshot.png")
    input_image = screenshot
    game_area = None
    if args.detect_game_area:
        input_image = work_dir / "detected_game_area.png"
        game_area = crop_game_area(screenshot, input_image)
    hud_source = input_image
    hud_crops = crop_hud_regions(hud_source, work_dir / "hud")
    hud_reading = None
    effective_options = options
    if getattr(args, "use_template_hud", False):
        hud_reading = read_hud_with_templates(hud_source, work_dir / "hud")
        effective_options = merge_hud_options(options, hud_reading)
    if args.use_vision_hud:
        if llm is None:
            raise RuntimeError("--use-vision-hud requires a configured LLM client")
        hud_reading = read_hud_with_vision(input_image, llm)
        effective_options = merge_hud_options(options, hud_reading)
    slots = read_screen(
        screenshot_path=input_image,
        profile_path=profile,
        recognizer=recognizer,
        top_k=args.top_k,
        output_dir=work_dir / "crops",
        max_shop_tier=_shop_tier_limit(hud_reading, effective_options),
    )
    phase_status = _phase_status(profile, hud_reading, slots)
    effective_options = _apply_shop_tier_inference(effective_options, hud_reading, slots)
    build_slots = slots
    if phase_status["phase"] != "shop":
        # In combat/spectator screens the shop crop boxes overlap board minions.
        # Never let those raw crops become shop cards or LLM buy targets.
        build_slots = [slot for slot in slots if not slot.slot_id.startswith("shop_")]
    state = build_state_from_slots(build_slots, database, effective_options)
    if hud_reading and hud_reading.tavern_tier is None:
        source = _tavern_tier_source(hud_reading, slots, effective_options)
        state.notes = _join_notes(
            state.notes,
            f"HUD tavern tier was not trusted; using {source} Tavern={effective_options.tavern_tier}.",
        )
    vision_shop = None
    if getattr(args, "use_vision_shop", False):
        if llm is None:
            raise RuntimeError("--use-vision-shop requires a configured LLM client")
        vision_shop = read_shop_with_vision(input_image, llm, database, slots)
        state.shop.cards = vision_shop.cards
        if vision_shop.notes:
            state.notes = _join_notes(state.notes, vision_shop.notes)
    if phase_status["phase"] != "shop":
        state.shop.cards = []
        state.notes = _join_notes(
            state.notes,
            "Current screen does not look like a shop-buy phase; shop recognition and buy recommendations were skipped.",
        )
        plan = _non_shop_plan()
    else:
        plan = agent.plan_turn(state, use_llm=args.use_llm)
    recognition_debug = _recognition_debug(slots, database, effective_options, hud_reading)
    _write_recognition_debug(work_dir, recognition_debug)
    return {
        "screenshot": str(screenshot),
        "input_image": str(input_image),
        "game_area": asdict(game_area) if game_area else None,
        "phase_status": phase_status,
        "recognition_debug": recognition_debug,
        "vision_shop": {
            "slots": vision_shop.slots,
            "notes": vision_shop.notes,
        } if vision_shop else None,
        "hud": {
            "crops": hud_crops,
            "reading": asdict(hud_reading) if hud_reading else None,
        },
        "state": asdict(state),
        "plan": asdict(plan),
    }


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _shop_tier_limit(hud_reading, options: BuildStateOptions) -> int | None:
    if hud_reading and hud_reading.tavern_tier is not None:
        return hud_reading.tavern_tier
    return options.tavern_tier


def _apply_shop_tier_inference(options: BuildStateOptions, hud_reading, slots) -> BuildStateOptions:
    if hud_reading and hud_reading.tavern_tier is not None:
        return options
    inferred = _infer_tavern_tier_from_shop_badges(slots)
    if inferred is None or inferred <= options.tavern_tier:
        return options
    return replace(options, tavern_tier=inferred)


def _infer_tavern_tier_from_shop_badges(slots) -> int | None:
    tiers = []
    for slot in slots:
        if not getattr(slot, "slot_id", "").startswith("shop_"):
            continue
        features = slot.features or {}
        confidence = _float_or_zero(features.get("tier_confidence"))
        if confidence < 0.65:
            continue
        try:
            tier = int(features.get("tier"))
        except (TypeError, ValueError):
            continue
        if 1 <= tier <= 6:
            tiers.append(tier)
    return max(tiers) if tiers else None


def _recognition_debug(slots, database: CardDatabase, options: BuildStateOptions, hud_reading) -> dict:
    return {
        "tavern_tier": {
            "final": options.tavern_tier,
            "hud": hud_reading.tavern_tier if hud_reading else None,
            "source": _tavern_tier_source(hud_reading, slots, options),
            "shop_badge_inferred": _infer_tavern_tier_from_shop_badges(slots),
            "shop_candidate_max_tier": _shop_tier_limit(hud_reading, options),
        },
        "shop_slots": [_slot_debug(slot, database) for slot in slots if slot.slot_id.startswith("shop_")],
        "board_slots": [_slot_debug(slot, database) for slot in slots if slot.slot_id.startswith("board_")],
    }


def _tavern_tier_source(hud_reading, slots, options: BuildStateOptions) -> str:
    if hud_reading and hud_reading.tavern_tier is not None:
        return "hud"
    inferred = _infer_tavern_tier_from_shop_badges(slots)
    if inferred is not None and inferred == options.tavern_tier:
        return "shop_badges"
    return "fallback_input"


def _slot_debug(slot, database: CardDatabase) -> dict:
    candidates = []
    for match in slot.matches[:8]:
        card = database.card_by_id(match.id)
        candidates.append(
            {
                "id": match.id,
                "name": card.name if card else match.id,
                "tier": card.tier if card else match.tier,
                "score": match.score,
                "image": card.image if card else match.path,
            }
        )
    return {
        "slot_id": slot.slot_id,
        "crop_path": slot.crop_path,
        "features": slot.features,
        "candidates": candidates,
    }


def _write_recognition_debug(work_dir: Path, payload: dict) -> None:
    path = work_dir / "recognition_debug.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_candidate_sheet(work_dir, payload, "shop_slots", "recognition_debug_shop_candidates.png")
    _write_candidate_sheet(work_dir, payload, "board_slots", "recognition_debug_board_candidates.png")


def _write_candidate_sheet(work_dir: Path, payload: dict, slot_key: str, output_name: str) -> None:
    slots = payload.get(slot_key) or []
    rows = []
    for slot in slots:
        crop_path = slot.get("crop_path")
        candidates = slot.get("candidates") or []
        if not crop_path or not Path(crop_path).exists():
            continue
        rows.append((slot, candidates[:4]))
    if not rows:
        return

    row_height = 150
    crop_width = 82
    candidate_width = 98
    label_height = 26
    width = crop_width + candidate_width * 4 + 24
    height = row_height * len(rows)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    for row_index, (slot, candidates) in enumerate(rows):
        top = row_index * row_height
        draw.text((4, top + 4), str(slot.get("slot_id")), fill=(0, 0, 0))
        crop = Image.open(slot["crop_path"]).convert("RGB")
        crop.thumbnail((crop_width - 8, row_height - label_height - 8), Image.Resampling.LANCZOS)
        sheet.paste(crop, (4, top + label_height))
        x = crop_width + 8
        for candidate in candidates:
            image_path = candidate.get("image")
            if image_path and Path(image_path).exists():
                image = Image.open(image_path).convert("RGB")
                image.thumbnail((candidate_width - 12, 104), Image.Resampling.LANCZOS)
                sheet.paste(image, (x + 4, top + label_height))
            label = f"{candidate.get('name') or candidate.get('id')} {candidate.get('score')}"
            draw.text((x + 4, top + 4), label[:14], fill=(0, 0, 0))
            x += candidate_width

    output = work_dir / output_name
    sheet.save(output)


def _phase_status(profile: Path, hud_reading, slots=None) -> dict:
    profile_name = profile.stem.lower()
    shop_profile = "shop" in profile_name and "trinket" not in profile_name
    if not shop_profile:
        return {"phase": "shop", "confidence": None, "reason": "profile does not require shop HUD guard"}
    shop_evidence = _shop_slot_evidence(slots or [])
    if shop_evidence["strong_slots"] >= 3:
        return {
            "phase": "shop",
            "confidence": max(_hud_confidence(hud_reading), shop_evidence["confidence"]),
            "reason": f"shop minion row visible ({shop_evidence['strong_slots']} shop-like slots)",
        }
    if hud_reading is None or hud_reading.confidence is None:
        return {"phase": "shop", "confidence": None, "reason": "no HUD confidence available"}
    if hud_reading.confidence < 0.35:
        return {
            "phase": "non_shop_or_uncertain",
            "confidence": hud_reading.confidence,
            "reason": "shop HUD was not confidently visible",
        }
    return {"phase": "shop", "confidence": hud_reading.confidence, "reason": "shop HUD visible"}


def _shop_slot_evidence(slots) -> dict:
    strong_slots = 0
    total_score = 0.0
    for slot in slots:
        if not getattr(slot, "slot_id", "").startswith("shop_"):
            continue
        features = slot.features or {}
        tier_confidence = _float_or_zero(features.get("tier_confidence"))
        has_badge = tier_confidence >= 0.55 and _valid_tier(features.get("tier"))
        has_stats = features.get("attack") is not None or features.get("health") is not None
        match_score = slot.matches[0].score if slot.matches else 0.0
        if has_badge or (has_stats and match_score >= 0.45):
            strong_slots += 1
            total_score += max(tier_confidence, match_score)
    confidence = min(total_score / max(strong_slots, 1), 1.0) if strong_slots else 0.0
    return {"strong_slots": strong_slots, "confidence": confidence}


def _valid_tier(value) -> bool:
    try:
        tier = int(value)
    except (TypeError, ValueError):
        return False
    return 1 <= tier <= 6


def _hud_confidence(hud_reading) -> float:
    if hud_reading is None or hud_reading.confidence is None:
        return 0.0
    return _float_or_zero(hud_reading.confidence)


def _float_or_zero(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _non_shop_plan() -> ActionPlan:
    return ActionPlan(
        summary="当前画面不像商店阶段，先不要按这个截图做购买/升本判断。",
        actions=[
            CandidateAction(
                action_type=ActionType.HOLD,
                score=1.0,
                reason="商店、金币或升级按钮不可见，继续识别会把战斗随从误当商店牌。",
            )
        ],
        composition_goal="等进入商店阶段后再点击 Capture & Decide；战斗阶段只适合看阵容强度，不适合判断买牌。",
        buy_recommendation="当前不是商店阶段，无法给购买目标。",
        level_or_roll_recommendation="等商店阶段且金币/升级按钮可见后再判断升本或刷新。",
        potential_synergy_cards=[],
        risk_level="medium",
    )


def _join_notes(*values: str) -> str:
    return "; ".join(value for value in values if value)


if __name__ == "__main__":
    main()
