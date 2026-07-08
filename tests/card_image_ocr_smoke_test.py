from __future__ import annotations

from pathlib import Path

from PIL import Image

from battlegrounds_agent.card_image_ocr import count_tier_stars, crop_card_regions, read_card_image


SAMPLE_CARD = Path(r"C:\Users\dlw\Documents\Codex\2026-07-05\gen\outputs\data card\BG22_202_battlegroundsImage.png")


def test_card_region_crops_and_tier_count() -> None:
    if not SAMPLE_CARD.exists():
        print("sample card image missing, skipped")
        return

    image = Image.open(SAMPLE_CARD).convert("RGBA")
    crops = crop_card_regions(image)
    assert set(crops) == {"tier", "name", "text", "tribe", "attack", "health"}
    assert count_tier_stars(crops["tier"]) == 2

    result = read_card_image(SAMPLE_CARD)
    assert result["id"] == "BG22_202"
    assert result["tier"] == 2


if __name__ == "__main__":
    test_card_region_crops_and_tier_count()
    print("card_image_ocr_smoke_test ok")
