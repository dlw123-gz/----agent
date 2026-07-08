from __future__ import annotations

import json
import tempfile
from pathlib import Path

from battlegrounds_agent.metadata_importer import merge_metadata


def test_merge_hearthstonejson_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        index_path = root / "cards.generated.json"
        source_path = root / "cards.source.json"
        output_path = root / "cards.enriched.json"

        index_path.write_text(
            json.dumps(
                [
                    {
                        "id": "BG99_001",
                        "name": "",
                        "tier": None,
                        "attack": None,
                        "health": None,
                        "tribes": [],
                        "text": "",
                        "tags": [],
                        "image": "BG99_001_battlegroundsImage.png",
                        "needs_metadata": True,
                    }
                ]
            ),
            encoding="utf-8",
        )
        source_path.write_text(
            json.dumps(
                [
                    {
                        "id": "BG99_001",
                        "name": "测试龙",
                        "text": "<b>战吼：</b>发现一张龙牌。",
                        "attack": 3,
                        "health": 4,
                        "races": ["DRAGON"],
                        "battlegrounds": {"tier": 2},
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        summary = merge_metadata(index_path, source_path, output_path)
        rows = json.loads(output_path.read_text(encoding="utf-8"))

    assert summary == {"total": 1, "matched": 1, "missing": 0}
    assert rows[0]["name"] == "测试龙"
    assert rows[0]["tier"] == 2
    assert rows[0]["attack"] == 3
    assert rows[0]["health"] == 4
    assert rows[0]["tribes"] == ["龙"]
    assert "battlecry" in rows[0]["tags"]
    assert "discover" in rows[0]["tags"]
    assert "tribe:龙" in rows[0]["tags"]
    assert rows[0]["needs_metadata"] is False


if __name__ == "__main__":
    test_merge_hearthstonejson_metadata()
    print("metadata_importer_smoke_test ok")
