from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageFilter, ImageOps

ImageKind = Literal["card", "trinket", "team"]

SUPPORTED_EXTENSIONS = {".png", ".webp", ".jpg", ".jpeg"}

CARD_BOXES = (
    (0.14, 0.05, 0.86, 0.46),
    (0.17, 0.41, 0.83, 0.58),
    (0.17, 0.55, 0.83, 0.80),
    (0.20, 0.76, 0.80, 0.91),
)
TRINKET_BOXES = (
    (0.14, 0.04, 0.86, 0.46),
    (0.17, 0.41, 0.83, 0.58),
    (0.17, 0.55, 0.83, 0.87),
)
TEAM_BOXES = ((0.0, 0.0, 1.0, 1.0),)


@dataclass(frozen=True, slots=True)
class RecognitionEntry:
    id: str
    kind: ImageKind
    path: str
    width: int
    height: int
    tier: int | None
    hashes: tuple[int, ...]
    histogram: tuple[float, ...]
    compact_hashes: tuple[int, ...] = ()
    compact_histogram: tuple[float, ...] = ()
    portrait_hashes: tuple[int, ...] = ()
    portrait_histogram: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class RecognitionResult:
    id: str
    kind: ImageKind
    path: str
    score: float
    hash_distance: float
    histogram_distance: float
    cv_score: float | None = None
    tier: int | None = None


class ImageRecognizer:
    def __init__(self, entries: list[RecognitionEntry], use_cv: bool = True) -> None:
        self.entries = entries
        self._cv2 = None
        self._np = None
        self._orb = None
        self._compact_cv_descriptors: dict[str, object] = {}
        self._portrait_vectors: dict[str, tuple[float, ...]] = {}
        self._oval_vectors: dict[str, tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]] = {}
        self._init_portrait_vectors()
        if use_cv:
            self._init_optional_cv()

    @classmethod
    def from_directories(
        cls,
        card_dir: str | Path | None = None,
        trinket_dir: str | Path | None = None,
        team_dir: str | Path | None = None,
        card_json: str | Path | None = None,
        use_cv: bool = True,
    ) -> "ImageRecognizer":
        entries: list[RecognitionEntry] = []
        tier_by_id = _load_card_tiers(card_json)
        for kind, directory in (
            ("card", card_dir),
            ("trinket", trinket_dir),
            ("team", team_dir),
        ):
            if directory is None:
                continue
            entries.extend(_load_entries(Path(directory), kind, tier_by_id=tier_by_id))
        if not entries:
            raise ValueError("No image entries loaded. Check the database directories.")
        return cls(entries, use_cv=use_cv)

    @classmethod
    def from_json_index(cls, index_path: str | Path, use_cv: bool = True) -> "ImageRecognizer":
        data = json.loads(Path(index_path).read_text(encoding="utf-8"))
        entries = [
            _entry_from_json_item(item)
            for item in data
        ]
        return cls(entries, use_cv=use_cv)

    def write_json_index(self, index_path: str | Path) -> None:
        output = Path(index_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_json_index(), ensure_ascii=False, indent=2), encoding="utf-8")

    def recognize(
        self,
        image_path: str | Path,
        kind: ImageKind | None = None,
        top_k: int = 5,
        card_tier: int | None = None,
        max_card_tier: int | None = None,
    ) -> list[RecognitionResult]:
        return self.recognize_image(_open_rgb(Path(image_path)), kind=kind, top_k=top_k, card_tier=card_tier, max_card_tier=max_card_tier)

    def recognize_image(
        self,
        query_image: Image.Image,
        kind: ImageKind | None = None,
        top_k: int = 5,
        card_tier: int | None = None,
        max_card_tier: int | None = None,
    ) -> list[RecognitionResult]:
        query_image = _flatten_rgb(query_image)
        candidates = [entry for entry in self.entries if kind is None or entry.kind == kind]
        candidates = _filter_card_candidates(candidates, card_tier=card_tier, max_card_tier=max_card_tier)
        if not candidates:
            raise ValueError(f"No candidates available for kind={kind!r}")

        results: list[RecognitionResult] = []
        use_compact_card = kind == "card" and _looks_like_compact_card(query_image)
        use_board_minion_card = use_compact_card and _looks_like_board_minion_card(query_image)
        if use_compact_card:
            query_hashes = _compact_query_card_hashes(query_image)
            query_histogram = _compact_query_card_histogram(query_image)
            query_portrait_hashes = _query_card_portrait_hashes(query_image)
            query_portrait_histogram = _query_card_portrait_histogram(query_image)
            query_portrait_vector = _portrait_vector(_query_card_portrait_image(query_image))
            query_art_profiles = _query_card_art_profiles(query_image)
            query_oval_profiles = _query_card_oval_profiles(query_image)
            query_descriptor = self._cv_descriptor(_compact_query_card_image(query_image))
        else:
            query_art_profiles = []
            query_oval_profiles = []
            query_descriptor = None
        for entry in candidates:
            cv_score = None
            portrait_similarity = 0.0
            if use_compact_card and entry.kind == "card" and entry.compact_hashes:
                compact_hash_distance = _average_hash_distance(query_hashes, entry.compact_hashes)
                compact_histogram_distance = _histogram_distance(query_histogram, entry.compact_histogram)
                if entry.portrait_hashes:
                    portrait_hash_distance = _average_hash_distance(query_portrait_hashes, entry.portrait_hashes)
                    portrait_histogram_distance = _histogram_distance(query_portrait_histogram, entry.portrait_histogram)
                    hash_distance = (portrait_hash_distance * 0.78) + (compact_hash_distance * 0.22)
                    histogram_distance = (portrait_histogram_distance * 0.78) + (compact_histogram_distance * 0.22)
                else:
                    hash_distance = compact_hash_distance
                    histogram_distance = compact_histogram_distance
                portrait_similarity = _vector_similarity(query_portrait_vector, self._portrait_vectors.get(entry.path))
                cv_score = self._cv_match_score(query_descriptor, self._compact_cv_descriptors.get(entry.path))
            else:
                resized = query_image.resize((entry.width, entry.height), Image.Resampling.LANCZOS)
                hashes = _profile_hashes(resized, entry.kind)
                histogram = _color_histogram(resized)
                hash_distance = _average_hash_distance(hashes, entry.hashes)
                histogram_distance = _histogram_distance(histogram, entry.histogram)
            score = max(0.0, 1.0 - (hash_distance * 0.72 + histogram_distance * 0.28))
            if use_compact_card and entry.kind == "card":
                art_score = _compact_art_variant_score(query_art_profiles, entry, self._portrait_vectors.get(entry.path))
                oval_score = _compact_oval_score(query_oval_profiles, self._oval_vectors.get(entry.path))
                score = max(score, (score * 0.50) + (portrait_similarity * 0.50), art_score, oval_score)
            if cv_score is not None:
                score = max(score, (score * 0.62) + (cv_score * 0.38))
                if use_board_minion_card and cv_score >= 0.45:
                    # Board minions lose the full-card layout but keep many
                    # distinctive local art details. Let strong ORB matches
                    # break ties between otherwise similar blue/gold portraits.
                    score = max(score, score + 0.025 + ((cv_score - 0.45) * 0.55))
            results.append(
                RecognitionResult(
                    id=entry.id,
                    kind=entry.kind,
                    path=entry.path,
                    tier=entry.tier,
                    score=round(score, 4),
                    hash_distance=round(hash_distance, 4),
                    histogram_distance=round(histogram_distance, 4),
                    cv_score=round(cv_score, 4) if cv_score is not None else None,
                )
            )

        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]

    def to_json_index(self) -> list[dict]:
        return [asdict(entry) for entry in self.entries]

    def _init_portrait_vectors(self) -> None:
        for entry in self.entries:
            if entry.kind != "card":
                continue
            try:
                image = _open_rgb(Path(entry.path))
                self._portrait_vectors[entry.path] = _portrait_vector(_database_card_portrait_image(image))
                self._oval_vectors[entry.path] = _oval_feature_vectors(_database_card_oval_image(image))
            except Exception:
                continue

    def _init_optional_cv(self) -> None:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except Exception:
            return
        self._cv2 = cv2
        self._np = np
        self._orb = cv2.ORB_create(nfeatures=700)
        for entry in self.entries:
            if entry.kind == "card":
                image = _open_rgb(Path(entry.path))
                self._compact_cv_descriptors[entry.path] = self._cv_descriptor(_compact_database_card_image(image))

    def _cv_descriptor(self, image: Image.Image) -> object | None:
        if self._cv2 is None or self._np is None or self._orb is None:
            return None
        gray = self._np.array(image.convert("L"))
        _, descriptor = self._orb.detectAndCompute(gray, None)
        return descriptor

    def _cv_match_score(self, left: object | None, right: object | None) -> float | None:
        if self._cv2 is None or left is None or right is None:
            return None
        matcher = self._cv2.BFMatcher(self._cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(left, right)
        if not matches:
            return 0.0
        good = [match for match in matches if match.distance <= 72]
        if not good:
            return 0.0
        avg_distance = sum(match.distance for match in good) / len(good)
        count_score = min(len(good) / 22.0, 1.0)
        distance_score = max(0.0, 1.0 - (avg_distance / 96.0))
        return count_score * distance_score


def _load_entries(directory: Path, kind: ImageKind, tier_by_id: dict[str, int] | None = None) -> list[RecognitionEntry]:
    entries: list[RecognitionEntry] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            image = _open_rgb(path)
            entries.append(
                RecognitionEntry(
                    id=_image_id(path),
                    kind=kind,
                    path=str(path),
                    width=image.width,
                    height=image.height,
                    tier=(tier_by_id or {}).get(_image_id(path)) if kind == "card" else None,
                    hashes=_profile_hashes(image, kind),
                    histogram=_color_histogram(image),
                    compact_hashes=_compact_database_card_hashes(image) if kind == "card" else (),
                    compact_histogram=_compact_database_card_histogram(image) if kind == "card" else (),
                    portrait_hashes=_database_card_portrait_hashes(image) if kind == "card" else (),
                    portrait_histogram=_database_card_portrait_histogram(image) if kind == "card" else (),
                )
            )
    return entries


def _entry_from_json_item(item: dict) -> RecognitionEntry:
    kind = item["kind"]
    path = item["path"]
    tier_by_id = _load_card_tiers(None) if kind == "card" else {}
    portrait_hashes = tuple(item.get("portrait_hashes") or ())
    portrait_histogram = tuple(item.get("portrait_histogram") or ())
    if kind == "card" and (not portrait_hashes or not portrait_histogram):
        try:
            image = _open_rgb(Path(path))
            portrait_hashes = _database_card_portrait_hashes(image)
            portrait_histogram = _database_card_portrait_histogram(image)
        except Exception:
            portrait_hashes = ()
            portrait_histogram = ()
    return RecognitionEntry(
        id=item["id"],
        kind=kind,
        path=path,
        width=int(item["width"]),
        height=int(item["height"]),
        tier=_int_or_none(item.get("tier")) if "tier" in item else tier_by_id.get(item["id"]),
        hashes=tuple(item["hashes"]),
        histogram=tuple(item["histogram"]),
        compact_hashes=tuple(item.get("compact_hashes") or ()),
        compact_histogram=tuple(item.get("compact_histogram") or ()),
        portrait_hashes=portrait_hashes,
        portrait_histogram=portrait_histogram,
    )


def _image_id(path: Path) -> str:
    stem = path.stem
    return stem.removesuffix("_battlegroundsImage")


def _filter_card_candidates(
    candidates: list[RecognitionEntry],
    card_tier: int | None = None,
    max_card_tier: int | None = None,
) -> list[RecognitionEntry]:
    if card_tier is not None:
        exact = [entry for entry in candidates if entry.kind != "card" or entry.tier == card_tier]
        if exact:
            return exact
    if max_card_tier is not None:
        bounded = [entry for entry in candidates if entry.kind != "card" or entry.tier is None or entry.tier <= max_card_tier]
        if bounded:
            return bounded
    return candidates


def _load_card_tiers(card_json: str | Path | None) -> dict[str, int]:
    paths: list[Path] = []
    if card_json:
        paths.append(Path(card_json))
    paths.extend(
        [
            Path.cwd() / "examples" / "data" / "cards.enriched.json",
            Path(__file__).resolve().parents[2] / "examples" / "data" / "cards.enriched.json",
        ]
    )
    for path in paths:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        result: dict[str, int] = {}
        for item in data:
            card_id = str(item.get("id") or "")
            tier = _int_or_none(item.get("tier"))
            if card_id and tier is not None:
                result[card_id] = tier
        return result
    return {}


def _int_or_none(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _open_rgb(path: Path) -> Image.Image:
    return _flatten_rgb(Image.open(path))


def _flatten_rgb(image: Image.Image) -> Image.Image:
    image = image.convert("RGBA")
    background = Image.new("RGBA", image.size, (0, 0, 0, 255))
    return Image.alpha_composite(background, image).convert("RGB")


def _profile_hashes(image: Image.Image, kind: ImageKind) -> tuple[int, ...]:
    boxes = {"card": CARD_BOXES, "trinket": TRINKET_BOXES, "team": TEAM_BOXES}[kind]
    return tuple(_difference_hash(_crop_ratio(image, box)) for box in boxes)


def _looks_like_compact_card(image: Image.Image) -> bool:
    width, height = image.size
    return width < 230 or height < 360


def _looks_like_board_minion_card(image: Image.Image) -> bool:
    width, height = image.size
    if width <= 0:
        return False
    return width < 230 and height < 340 and (height / width) < 1.9


def _compact_database_card_hashes(image: Image.Image) -> tuple[int, ...]:
    boxes = (
        (0.20, 0.07, 0.82, 0.45),
        (0.12, 0.04, 0.35, 0.22),
    )
    return tuple(_difference_hash(_crop_ratio(image, box)) for box in boxes)


def _compact_database_card_histogram(image: Image.Image) -> tuple[float, ...]:
    return _color_histogram(_compact_database_card_image(image))


def _compact_database_card_image(image: Image.Image) -> Image.Image:
    return _crop_ratio(image, (0.20, 0.07, 0.82, 0.45))


def _database_card_portrait_hashes(image: Image.Image) -> tuple[int, ...]:
    boxes = (
        (0.27, 0.10, 0.78, 0.43),
        (0.31, 0.14, 0.73, 0.39),
        (0.25, 0.13, 0.80, 0.50),
    )
    return tuple(_difference_hash(_crop_ratio(image, box)) for box in boxes)


def _database_card_portrait_histogram(image: Image.Image) -> tuple[float, ...]:
    return _color_histogram(_crop_ratio(image, (0.27, 0.10, 0.78, 0.43)))


def _database_card_portrait_image(image: Image.Image) -> Image.Image:
    return _crop_ratio(image, (0.27, 0.10, 0.78, 0.43))


def _database_card_oval_image(image: Image.Image) -> Image.Image:
    return _crop_ratio(image, (0.18, 0.05, 0.82, 0.52))


def _query_card_oval_profiles(image: Image.Image) -> list[tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]]:
    boxes = (
        (0.08, 0.04, 0.92, 0.82),
        (0.14, 0.06, 0.86, 0.76),
        (0.08, 0.28, 0.92, 0.84),
        (0.14, 0.32, 0.86, 0.78),
        (0.20, 0.36, 0.80, 0.72),
        (0.04, 0.24, 0.96, 0.88),
    )
    return [_oval_feature_vectors(_crop_ratio(image, box)) for box in boxes]


def _oval_feature_vectors(image: Image.Image) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    return (
        _oval_image_vector(image, mode="rgb"),
        _oval_image_vector(image, mode="gray"),
        _oval_image_vector(image, mode="edge"),
    )


def _compact_oval_score(
    query_profiles: list[tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]],
    entry_profile: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]] | None,
) -> float:
    if not query_profiles or entry_profile is None:
        return 0.0
    best = 0.0
    for query_profile in query_profiles:
        rgb = _vector_similarity(query_profile[0], entry_profile[0])
        gray = _vector_similarity(query_profile[1], entry_profile[1])
        edge = _vector_similarity(query_profile[2], entry_profile[2])
        best = max(best, rgb * 0.55 + gray * 0.30 + edge * 0.15)
    # Cosine-style image vectors have a compressed range for Hearthstone art.
    # Calibrate them into the same score band as hash scores so obvious oval
    # matches can dominate compact shop-minion recognition.
    return max(0.0, min(1.0, 0.50 + (best - 0.50) * 1.75))


def _compact_query_card_hashes(image: Image.Image) -> tuple[int, ...]:
    boxes = (
        (0.10, 0.08, 0.90, 0.72),
        (0.00, 0.00, 0.38, 0.28),
    )
    return tuple(_difference_hash(_crop_ratio(image, box)) for box in boxes)


def _compact_query_card_histogram(image: Image.Image) -> tuple[float, ...]:
    return _color_histogram(_compact_query_card_image(image))


def _compact_query_card_image(image: Image.Image) -> Image.Image:
    return _crop_ratio(image, (0.10, 0.08, 0.90, 0.72))


def _query_card_portrait_hashes(image: Image.Image) -> tuple[int, ...]:
    boxes = (
        (0.16, 0.28, 0.88, 0.70),
        (0.22, 0.30, 0.82, 0.68),
        (0.10, 0.30, 0.92, 0.78),
    )
    return tuple(_difference_hash(_crop_ratio(image, box)) for box in boxes)


def _query_card_portrait_histogram(image: Image.Image) -> tuple[float, ...]:
    return _color_histogram(_crop_ratio(image, (0.16, 0.28, 0.88, 0.70)))


def _query_card_portrait_image(image: Image.Image) -> Image.Image:
    return _crop_ratio(image, (0.14, 0.32, 0.86, 0.78))


def _query_card_art_profiles(image: Image.Image) -> list[tuple[int, tuple[float, ...], tuple[float, ...]]]:
    # Tavern minions are rendered as board pieces, not full cards. Try several
    # inner art crops and let the database portrait choose the best overlap.
    boxes = (
        (0.08, 0.02, 0.92, 0.82),
        (0.14, 0.06, 0.86, 0.76),
        (0.06, 0.05, 0.94, 0.70),
        (0.10, 0.12, 0.90, 0.66),
        (0.16, 0.20, 0.86, 0.62),
        (0.04, 0.22, 0.96, 0.78),
        (0.18, 0.30, 0.82, 0.70),
    )
    profiles = []
    for box in boxes:
        crop = _crop_ratio(image, box)
        profiles.append((_difference_hash(crop), _color_histogram(crop), _portrait_vector(crop)))
    return profiles


def _compact_art_variant_score(
    query_profiles: list[tuple[int, tuple[float, ...], tuple[float, ...]]],
    entry: RecognitionEntry,
    entry_vector: tuple[float, ...] | None,
) -> float:
    if not query_profiles:
        return 0.0
    db_hashes = entry.portrait_hashes or entry.compact_hashes or entry.hashes
    db_histogram = entry.portrait_histogram or entry.compact_histogram or entry.histogram
    best = 0.0
    for query_hash, query_histogram, query_vector in query_profiles:
        hash_distance = min((_hash_distance(query_hash, db_hash) for db_hash in db_hashes), default=1.0)
        histogram_distance = _histogram_distance(query_histogram, db_histogram)
        base_score = max(0.0, 1.0 - (hash_distance * 0.48 + histogram_distance * 0.52))
        vector_score = _vector_similarity(query_vector, entry_vector)
        best = max(best, base_score, (base_score * 0.65) + (vector_score * 0.35))
    return best


def _crop_ratio(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    width, height = image.size
    left, top, right, bottom = box
    return image.crop((int(left * width), int(top * height), int(right * width), int(bottom * height)))


def _difference_hash(image: Image.Image, hash_size: int = 16) -> int:
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(resized.getdata())
    value = 0
    for row in range(hash_size):
        row_start = row * (hash_size + 1)
        for col in range(hash_size):
            value <<= 1
            if pixels[row_start + col] > pixels[row_start + col + 1]:
                value |= 1
    return value


def _color_histogram(image: Image.Image, bins: int = 16) -> tuple[float, ...]:
    small = image.resize((96, 96), Image.Resampling.BILINEAR)
    channels = small.split()
    values: list[float] = []
    total = float(small.width * small.height)
    for channel in channels:
        hist = channel.histogram()
        for start in range(0, 256, 256 // bins):
            values.append(sum(hist[start : start + (256 // bins)]) / total)
    return tuple(values)


def _image_vector(image: Image.Image, mode: str = "rgb", size: int = 64) -> tuple[float, ...]:
    resized = image.resize((size, size), Image.Resampling.BILINEAR)
    if mode == "edge":
        converted = ImageOps.autocontrast(resized.convert("L")).filter(ImageFilter.FIND_EDGES)
        values = [pixel / 255.0 for pixel in converted.getdata()]
    elif mode == "gray":
        converted = ImageOps.autocontrast(resized.convert("L"))
        values = [pixel / 255.0 for pixel in converted.getdata()]
    else:
        converted = resized.convert("RGB")
        values = []
        for r, g, b in converted.getdata():
            values.extend((r / 255.0, g / 255.0, b / 255.0))
    mean = sum(values) / len(values)
    centered = [value - mean for value in values]
    norm = sum(value * value for value in centered) ** 0.5
    if norm <= 1e-9:
        return tuple(0.0 for _ in centered)
    return tuple(value / norm for value in centered)


def _oval_image_vector(image: Image.Image, mode: str = "rgb", size: int = 56) -> tuple[float, ...]:
    resized = image.resize((size, size), Image.Resampling.BILINEAR)
    if mode == "edge":
        converted = ImageOps.autocontrast(resized.convert("L")).filter(ImageFilter.FIND_EDGES)
    elif mode == "gray":
        converted = ImageOps.autocontrast(resized.convert("L"))
    else:
        converted = resized.convert("RGB")

    values: list[float] = []
    center = (size - 1) / 2.0
    radius_x = size * 0.43
    radius_y = size * 0.47
    for y in range(size):
        for x in range(size):
            dx = (x - center) / radius_x
            dy = (y - center) / radius_y
            if dx * dx + dy * dy > 1.0:
                continue
            pixel = converted.getpixel((x, y))
            if mode == "rgb":
                r, g, b = pixel
                values.extend((r / 255.0, g / 255.0, b / 255.0))
            else:
                values.append(pixel / 255.0)
    if not values:
        return ()
    mean = sum(values) / len(values)
    centered = [value - mean for value in values]
    norm = sum(value * value for value in centered) ** 0.5
    if norm <= 1e-9:
        return tuple(0.0 for _ in centered)
    return tuple(value / norm for value in centered)


def _portrait_vector(image: Image.Image, size: int = 40) -> tuple[float, ...]:
    gray = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    values = [pixel / 255.0 for pixel in gray.getdata()]
    mean = sum(values) / len(values)
    centered = [value - mean for value in values]
    norm = sum(value * value for value in centered) ** 0.5
    if norm <= 1e-9:
        return tuple(0.0 for _ in centered)
    return tuple(value / norm for value in centered)


def _vector_similarity(left: tuple[float, ...], right: tuple[float, ...] | None) -> float:
    if not right or len(left) != len(right):
        return 0.0
    cosine = sum(a * b for a, b in zip(left, right))
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def _average_hash_distance(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("Hash profiles must have the same length")
    distances = [_hash_distance(a, b) for a, b in zip(left, right)]
    return sum(distances) / len(distances)


def _hash_distance(left: int, right: int) -> float:
    return (left ^ right).bit_count() / float(16 * 16)


def _histogram_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(abs(a - b) for a, b in zip(left, right)) / 6.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize card/trinket/team images from local image databases.")
    parser.add_argument("--image", required=True, help="Image to recognize. Use a cropped card/trinket image first.")
    parser.add_argument("--kind", choices=["card", "trinket", "team"], help="Limit matching to one database type.")
    parser.add_argument("--card-dir", default=r"C:\Users\dlw\Documents\Codex\2026-07-05\gen\outputs\data card")
    parser.add_argument("--trinket-dir", default=r"C:\Users\dlw\Documents\Codex\2026-07-05\gen\outputs\data sp")
    parser.add_argument("--team-dir", default=r"C:\Users\dlw\Documents\Codex\2026-07-05\gen\outputs\data team")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    recognizer = ImageRecognizer.from_directories(args.card_dir, args.trinket_dir, args.team_dir)
    results = recognizer.recognize(args.image, kind=args.kind, top_k=args.top_k)
    print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
