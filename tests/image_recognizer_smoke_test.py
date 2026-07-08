from __future__ import annotations

from pathlib import Path

from battlegrounds_agent.image_recognizer import ImageRecognizer


def test_known_card_matches_itself() -> None:
    outputs = Path(__file__).resolve().parents[2]
    card_dir = outputs / "data card"
    trinket_dir = outputs / "data sp"
    query = card_dir / "BG20_100_battlegroundsImage.png"

    recognizer = ImageRecognizer.from_directories(card_dir=card_dir, trinket_dir=trinket_dir)
    result = recognizer.recognize(query, kind="card", top_k=1)[0]

    assert result.id == "BG20_100"
    assert result.score == 1.0


if __name__ == "__main__":
    test_known_card_matches_itself()
    print("image recognizer smoke test passed")
