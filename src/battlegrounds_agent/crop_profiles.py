from __future__ import annotations

from pathlib import Path
from typing import Literal


PhaseName = Literal["shop", "shop-buy", "shop-buy-16x9", "shop-buy-right", "trinket-fullscreen", "trinket-video", "triple-discover"]


PROFILE_FILES: dict[str, str] = {
    "shop": "doc_game_image3.json",
    "shop-buy": "shop_buy_fullscreen.json",
    "shop-buy-16x9": "shop_buy_fullscreen_16x9.json",
    "shop-buy-right": "shop_buy_video_right_crop.json",
    "trinket-fullscreen": "trinket_shop_fullscreen.json",
    "trinket-video": "trinket_shop_video_crop.json",
    "triple-discover": "triple_discover_video_crop.json",
}


def resolve_profile(profile: str | None = None, phase: str | None = None) -> Path:
    if profile:
        return Path(profile)
    if not phase:
        raise ValueError("Either --profile or --phase is required.")
    try:
        filename = PROFILE_FILES[phase]
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILE_FILES))
        raise ValueError(f"Unknown phase {phase!r}. Choose one of: {choices}") from exc
    return Path(__file__).resolve().parents[2] / "examples" / "crop_profiles" / filename
