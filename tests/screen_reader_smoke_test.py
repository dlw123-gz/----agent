from __future__ import annotations

from pathlib import Path

from battlegrounds_agent.image_recognizer import ImageRecognizer
from battlegrounds_agent.screen_reader import read_screen


def test_doc_screenshot_can_be_cropped_and_scored() -> None:
    root = Path(__file__).resolve().parents[1]
    outputs = root.parent
    screenshot = outputs.parent / "work" / "docx_media" / "image3.png"
    profile = root / "examples" / "crop_profiles" / "doc_game_image3.json"

    recognizer = ImageRecognizer.from_directories(
        card_dir=outputs / "data card",
        trinket_dir=outputs / "data sp",
    )
    results = read_screen(screenshot, profile, recognizer, top_k=1)

    assert len(results) == 5
    assert all(result.matches for result in results)


if __name__ == "__main__":
    test_doc_screenshot_can_be_cropped_and_scored()
    print("screen reader smoke test passed")
