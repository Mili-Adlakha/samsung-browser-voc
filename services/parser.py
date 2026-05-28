"""Parse raw Play Store review text into structured ReviewRecord objects."""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import re
from datetime import datetime, timezone

from models.schemas import ReviewRecord

logger = logging.getLogger(__name__)

MIN_REVIEW_LENGTH = 15

_HELPFUL_RE = re.compile(
    r"(\d+)\s+people?\s+found this review helpful",
    re.IGNORECASE,
)
_NOISE_LINE_RE = re.compile(
    r"^(?:Did you find this helpful\??\s*(?:Yes\s*No)?|"
    r"\d+\s+people?\s+found this review helpful|"
    r"\d+\s+person\s+found this review helpful)$",
    re.IGNORECASE,
)
_DATE_LINE_RE = re.compile(
    r"^("
    r"\d{1,2}\s+"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{4}"
    r"|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2},?\s+\d{4}"
    r"|\d{4}-\d{2}-\d{2}"
    r")\s*$",
    re.IGNORECASE,
)
_HELPFUL_SPLIT_RE = re.compile(
    r"Did you find this helpful\?\s*(?:Yes\s*No)?\s*",
    re.IGNORECASE,
)
_NUMBERED_HEADER_RE = re.compile(
    r"^\s*\d+\.\s*Rating:\s*(\d+)\s*/\s*5\s*\|\s*(.+?)\s*\|\s*(.+?)\s*$",
    re.IGNORECASE,
)
_NUMBERED_SPLIT_RE = re.compile(
    r"(?=^\s*\d+\.\s*Rating:)",
    re.MULTILINE | re.IGNORECASE,
)
_UPVOTES_LINE_RE = re.compile(r"^Upvotes:\s*(\d+)\s*$", re.IGNORECASE)
_CSV_HEADER_MARKERS = ("reviewid", "authorname", "reviewtext")
_NUMERIC_RATING_RE = re.compile(
    r"^(\d)\s*(?:/\s*5|out\s+of\s+5|stars?)?\s*$",
    re.IGNORECASE,
)
_REVIEW_BLOCK_START_RE = re.compile(
    r"(?=^(?:[^\n★⭐]{2,60})\n[★⭐][★⭐\s☆]*\s*$)",
    re.MULTILINE,
)

# Play Console / third-party CSV column aliases (lowercase keys)
_CSV_COLUMN_MAP = {
    "authorname": "author",
    "author name": "author",
    "reviewer name": "author",
    "name": "author",
    "reviewtext": "text",
    "review text": "text",
    "comment": "text",
    "body": "text",
    "starrating": "rating",
    "star rating": "rating",
    "rating": "rating",
    "score": "rating",
    "thumbsupcount": "upvotes",
    "thumbs up count": "upvotes",
    "helpful": "upvotes",
    "at": "date",
    "review date": "date",
    "date": "date",
    "review submit date and time": "date",
}


def detect_format(raw_text: str) -> str:
    """Return 'csv', 'numbered', or 'playstore' based on the first 200 characters."""
    first_line = raw_text.strip().splitlines()[0] if raw_text.strip() else ""
    first_lower = first_line.lower()

    if "," in first_line:
        csv_signals = (
            "reviewid" in first_lower and "authorname" in first_lower,
            "review text" in first_lower and "star rating" in first_lower,
            "reviewtext" in first_lower and "starrating" in first_lower,
            "reviewer name" in first_lower and "star rating" in first_lower,
        )
        if any(csv_signals):
            return "csv"
    if re.search(r"^\s*\d+\.\s*rating:\s*\d", raw_text[:200], re.MULTILINE | re.IGNORECASE):
        return "numbered"
    return "playstore"


def parse_play_store_text(raw_text: str, app_version: str) -> list[ReviewRecord]:
    """Parse Play Store paste text (formats A/B/C) into ReviewRecord list."""
    if not raw_text or not raw_text.strip():
        return []

    fmt = detect_format(raw_text)
    if fmt == "csv":
        return _parse_csv(raw_text, app_version)
    if fmt == "numbered":
        return _parse_numbered(raw_text, app_version)
    return _parse_format_a(raw_text, app_version)


def _split_format_a_blocks(normalized: str) -> list[str]:
    """Split paste text into review blocks (handles single or double newlines)."""
    helpful_blocks = [
        b.strip() for b in _HELPFUL_SPLIT_RE.split(normalized) if b.strip()
    ]
    if len(helpful_blocks) > 1:
        return helpful_blocks

    blocks = [b.strip() for b in re.split(r"\n\s*\n+", normalized) if b.strip()]
    if len(blocks) > 1:
        return blocks

    lookahead_blocks = [
        b.strip()
        for b in _REVIEW_BLOCK_START_RE.split(normalized)
        if b.strip()
    ]
    if len(lookahead_blocks) > 1:
        return lookahead_blocks

    return [normalized] if normalized else []


def _parse_format_a(raw_text: str, app_version: str) -> list[ReviewRecord]:
    normalized = raw_text.replace("\r\n", "\n").strip()
    blocks = _split_format_a_blocks(normalized)
    records: list[ReviewRecord] = []
    for block in blocks:
        parsed = _parse_format_a_block(block)
        if not parsed:
            continue
        record = _make_record(app_version, len(records), **parsed)
        if record:
            records.append(record)
    return records


def _parse_format_a_block(block: str) -> dict | None:
    lines = [line.strip() for line in block.split("\n") if line.strip()]
    if not lines:
        return None

    star_idx: int | None = None
    rating = 0
    for i, line in enumerate(lines):
        if _is_star_line(line):
            star_idx = i
            rating = _count_stars(line)
            break
        numeric_match = _NUMERIC_RATING_RE.match(line)
        if numeric_match:
            star_idx = i
            rating = int(numeric_match.group(1))
            break
        slash_match = re.search(r"(\d)\s*/\s*5", line)
        if slash_match and len(line) < 20:
            star_idx = i
            rating = int(slash_match.group(1))
            break

    if star_idx is None:
        return _parse_date_author_block(block)

    if star_idx == 0:
        author_name = "Play Store User"
    else:
        author_name = lines[star_idx - 1]
        if _is_star_line(author_name) or _is_noise_line(author_name):
            author_name = "Play Store User"

    rating = max(0, min(5, rating))

    upvotes = 0
    match = _HELPFUL_RE.search(block)
    if match:
        upvotes = int(match.group(1))

    body_lines: list[str] = []
    for line in lines[star_idx + 1 :]:
        if _is_noise_line(line) or _HELPFUL_RE.search(line):
            continue
        body_lines.append(line)

    review_text = " ".join(body_lines).strip()
    if not review_text:
        logger.warning("Skipping block: empty review body for author %s", author_name)
        return None

    return {
        "author_name": author_name,
        "rating": rating,
        "review_text": review_text,
        "thumbs_up_count": upvotes,
        "review_date": "",
    }


def _parse_date_author_block(block: str) -> dict | None:
    """
    Play Store web paste without star lines:
    Author Name -> Date -> Review text -> Did you find this helpful?
    """
    lines = [line.strip() for line in block.split("\n") if line.strip()]
    lines = [line for line in lines if not _is_noise_line(line)]
    if len(lines) < 2:
        return None

    author_name = lines[0]
    if _is_date_line(author_name) or _is_ui_chrome_line(author_name):
        return None

    review_date = ""
    body_start = 1
    if len(lines) > 1 and _is_date_line(lines[1]):
        review_date = lines[1]
        body_start = 2

    body_lines = [
        line for line in lines[body_start:] if not _is_date_line(line)
    ]
    review_text = " ".join(body_lines).strip()
    if not review_text:
        logger.warning("Skipping block: empty body for %s", author_name)
        return None

    upvotes = 0
    match = _HELPFUL_RE.search(block)
    if match:
        upvotes = int(match.group(1))

    return {
        "author_name": author_name,
        "rating": 0,
        "review_text": review_text,
        "thumbs_up_count": upvotes,
        "review_date": review_date,
    }


def _is_date_line(line: str) -> bool:
    return bool(_DATE_LINE_RE.match(line.strip()))


def _is_ui_chrome_line(line: str) -> bool:
    lowered = line.lower()
    chrome_phrases = (
        "see all reviews",
        "was this review helpful",
        "flag as inappropriate",
        "reply from developer",
    )
    return any(phrase in lowered for phrase in chrome_phrases)


def _normalize_csv_row(row: dict[str, str | None]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for key, value in row.items():
        if key is None:
            continue
        canonical = _CSV_COLUMN_MAP.get(key.strip().lower(), key.strip().lower())
        if value is not None and str(value).strip():
            normalized[canonical] = str(value).strip()
    return normalized


def _parse_csv(raw_text: str, app_version: str) -> list[ReviewRecord]:
    normalized = raw_text.strip().lstrip("\ufeff")
    reader = csv.DictReader(io.StringIO(normalized))
    if not reader.fieldnames:
        return []

    records: list[ReviewRecord] = []
    for raw_row in reader:
        row = _normalize_csv_row(raw_row)
        author = (
            row.get("author")
            or raw_row.get("authorName")
            or raw_row.get("author_name")
            or ""
        ).strip()
        text = (
            row.get("text")
            or raw_row.get("reviewText")
            or raw_row.get("review_text")
            or ""
        ).strip()
        rating_raw = (
            row.get("rating")
            or raw_row.get("starRating")
            or raw_row.get("star_rating")
            or "0"
        ).strip()
        upvotes_raw = (
            row.get("upvotes")
            or raw_row.get("thumbsUpCount")
            or raw_row.get("thumbs_up_count")
            or "0"
        ).strip()
        review_date = (
            row.get("date") or raw_row.get("at") or raw_row.get("review_date") or ""
        ).strip()

        try:
            rating = int(float(rating_raw))
        except ValueError:
            rating = 0
        rating = max(0, min(5, rating))

        try:
            upvotes = int(float(upvotes_raw))
        except ValueError:
            upvotes = 0

        if not author and not text:
            continue

        record = _make_record(
            app_version,
            len(records),
            author_name=author or "Unknown",
            rating=rating,
            review_text=text,
            thumbs_up_count=max(0, upvotes),
            review_date=review_date,
        )
        if record:
            records.append(record)
    return records


def _parse_numbered(raw_text: str, app_version: str) -> list[ReviewRecord]:
    normalized = raw_text.replace("\r\n", "\n").strip()
    blocks = [b.strip() for b in _NUMBERED_SPLIT_RE.split(normalized) if b.strip()]
    records: list[ReviewRecord] = []
    for block in blocks:
        parsed = _parse_numbered_block(block)
        if not parsed:
            continue
        record = _make_record(app_version, len(records), **parsed)
        if record:
            records.append(record)
    return records


def _parse_numbered_block(block: str) -> dict | None:
    lines = [line.strip() for line in block.split("\n") if line.strip()]
    if not lines:
        return None

    header_match = _NUMBERED_HEADER_RE.match(lines[0])
    if not header_match:
        logger.warning("Skipping numbered block: invalid header")
        return None

    rating = int(header_match.group(1))
    author_name = header_match.group(2).strip()
    review_date = header_match.group(3).strip()

    upvotes = 0
    body_lines: list[str] = []
    for line in lines[1:]:
        upvote_match = _UPVOTES_LINE_RE.match(line)
        if upvote_match:
            upvotes = int(upvote_match.group(1))
            continue
        if _is_noise_line(line):
            continue
        body_lines.append(line)

    review_text = " ".join(body_lines).strip()
    if not review_text:
        logger.warning("Skipping numbered block: empty body for %s", author_name)
        return None

    return {
        "author_name": author_name,
        "rating": max(0, min(5, rating)),
        "review_text": review_text,
        "thumbs_up_count": upvotes,
        "review_date": review_date,
    }


def _make_record(
    app_version: str,
    index: int,
    *,
    author_name: str,
    rating: int,
    review_text: str,
    thumbs_up_count: int,
    review_date: str = "",
) -> ReviewRecord | None:
    cleaned = _clean_review_text(review_text)
    if len(cleaned) < MIN_REVIEW_LENGTH:
        logger.warning(
            "Skipping review from %s: body too short (%d chars)",
            author_name,
            len(cleaned),
        )
        return None

    review_id = (
        f"v{app_version}_{hashlib.md5(cleaned.encode()).hexdigest()[:8]}_{index}"
    )
    ingested_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    return ReviewRecord(
        review_id=review_id,
        app_version=app_version,
        author_name=author_name,
        rating=rating,
        review_text=cleaned,
        thumbs_up_count=thumbs_up_count,
        review_date=review_date,
        ingested_at=ingested_at,
    )


def _clean_review_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = _HELPFUL_RE.sub("", cleaned)
    cleaned = re.sub(
        r"Did you find this helpful\?\s*(?:Yes\s*No)?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", cleaned).strip()


def _is_star_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    star_chars = sum(1 for ch in stripped if ch in "★⭐")
    if star_chars == 0:
        return False
    non_star = re.sub(r"[★⭐☆\s]", "", stripped)
    return len(non_star) == 0


def _count_stars(line: str) -> int:
    count = line.count("★") + line.count("⭐")
    return min(count, 5) if count else 0


def _is_noise_line(line: str) -> bool:
    return bool(_NOISE_LINE_RE.match(line.strip()))
